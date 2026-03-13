import rasterio
import numpy as np
import joblib
import json
import pandas as pd
import argparse

from rasterio.windows import Window
from rasterio.features import geometry_mask
from shapely.geometry import shape, Polygon, LinearRing
from pathlib import Path


parser = argparse.ArgumentParser(description="Predict deforestation raster with quantity adjustment")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--site-only",
    action="store_true",
    help="Predict only inside Odisha sites",
)
group.add_argument(
    "--full-raster",
    action="store_true",
    help="Predict on entire raster (default)",
)

args = parser.parse_args()
SITE_ONLY = args.site_only


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "images"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs" / "predictions"

OUT_DIR.mkdir(parents=True, exist_ok=True)

predict_year = "2010_15"
ex_ante = False
data_preprocessing = "smote"   # must match training

if ex_ante:
    train_year = (
        "20"
        + str(int(predict_year[2:4]) - 5).zfill(2)
        + "_"
        + str(int(predict_year[5:7]) - 5).zfill(2)
    )
    model_train_year = (
        "20"
        + str(int(predict_year[2:4]) - 10).zfill(2)
        + "_"
        + str(int(predict_year[5:7]) - 10).zfill(2)
    )
    model_predict_year = (
        "20"
        + str(int(predict_year[2:4]) - 5).zfill(2)
        + "_"
        + str(int(predict_year[5:7]) - 5).zfill(2)
    )
else:
    train_year = predict_year[:4]

suffix = "ex_ante" if ex_ante else "ex_post"

PRED_TIF = DATA_DIR / f"training_data_x_{train_year}_y_{predict_year}.tif"
SITES_CSV = DATA_DIR / "Odisha_sites.csv"

if ex_ante:
    print(f"rf_{data_preprocessing}_x_{model_train_year}_y_{model_predict_year}_{suffix}.joblib")
else:
    print(f"rf_{data_preprocessing}_x_{train_year}_y_{predict_year}_{suffix}.joblib")
if ex_ante:
    MODEL_PATH = (
        MODEL_DIR
        / f"rf_{data_preprocessing}_x_{model_train_year}_y_{model_predict_year}_{suffix}.joblib"
    )
else:
    MODEL_PATH = (
        MODEL_DIR
        / f"rf_{data_preprocessing}_x_{train_year}_y_{predict_year}_{suffix}.joblib"
    )



out_suffix = "odisha_sites" if SITE_ONLY else "full"

OUT_PROB = OUT_DIR / f"deforestation_prob_{predict_year}_{out_suffix}_{suffix}.tif"
OUT_CLASS = OUT_DIR / f"deforestation_class_{predict_year}_{out_suffix}_{suffix}.tif"
OUT_DENSITY = OUT_DIR / f"deforestation_density_{predict_year}_{out_suffix}_{suffix}.tif"
OUT_ADJUSTED = OUT_DIR / f"deforestation_adjusted_{predict_year}_{out_suffix}_{suffix}.tif"

FEATURE_BANDS = [
    "0_Rainfall_norm",
    "1_Mean_Rainfall",
    "2_NDVI",
    "3_EVI",
    "4_ground_temp",
    "5_edge_distance",
    "6_elevation",
    "7_slope",
    "8_deforestation_density",
]

GROUND_TRUTH_BAND = "9_deforestation"  # Band index for ground truth

WINDOW_SIZE = 1024
PROB_THRESHOLD = 0.5


print("Prediction year:", predict_year)
print("Training year:", train_year)
print("Ex-ante:", ex_ante)
print("Site-only:", SITE_ONLY)
print("Model path:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


model = joblib.load(MODEL_PATH)


site_geoms = None
if SITE_ONLY:
    sites_df = pd.read_csv(SITES_CSV)

    site_geoms = []
    for g in sites_df[".geo"]:
        geom = shape(json.loads(g))

        if isinstance(geom, LinearRing):
            geom = Polygon(geom)
            print("Converted LinearRing to Polygon")

        if not geom.is_valid:
            geom = geom.buffer(0)

        site_geoms.append(geom)

    print(f"Loaded {len(site_geoms)} Odisha site polygons")


print("\n" + "="*60)
print("STEP 1: Initial Prediction (No Adjustment)")
print("="*60)

with rasterio.open(PRED_TIF) as src:
    band_names = list(src.descriptions)
    feature_idxs = [band_names.index(b) for b in FEATURE_BANDS]
    ground_truth_idx = band_names.index(GROUND_TRUTH_BAND)
    
    transform = src.transform
    pixel_width = abs(transform[0])  # in meters
    pixel_height = abs(transform[4])  # in meters
    pixel_area_m2 = pixel_width * pixel_height
    pixel_area_m2 = 900
    pixel_area_ha = pixel_area_m2 / 10000  # convert to hectares

    
    print(f"Pixel resolution: {pixel_width}m x {pixel_height}m")
    print(f"Pixel area: {pixel_area_ha:.6f} hectares")

    profile = src.profile.copy()
    profile.update(
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="lzw",
    )

    prob_map = np.full((src.height, src.width), np.nan, dtype="float32")
    
    with rasterio.open(OUT_PROB, "w", **profile) as dst_prob, \
         rasterio.open(OUT_CLASS, "w", **profile) as dst_cls, \
         rasterio.open(OUT_DENSITY, "w", **profile) as dst_density:

        for row in range(0, src.height, WINDOW_SIZE):
            for col in range(0, src.width, WINDOW_SIZE):

                window = Window(
                    col,
                    row,
                    min(WINDOW_SIZE, src.width - col),
                    min(WINDOW_SIZE, src.height - row),
                )

                chunk = src.read(window=window)
                h, w = chunk.shape[1], chunk.shape[2]

                pixels = chunk.reshape(chunk.shape[0], -1).T
                X = pixels[:, feature_idxs]

                valid = ~np.isnan(X).any(axis=1)

                if SITE_ONLY:
                    site_mask = geometry_mask(
                        site_geoms,
                        transform=src.window_transform(window),
                        invert=True,      # True = inside polygons
                        out_shape=(h, w),
                    )
                    valid &= site_mask.flatten()

                prob = np.full(len(X), np.nan, dtype="float32")
                cls = np.full(len(X), np.nan, dtype="float32")
                density = np.full(len(X), np.nan, dtype="float32")

                if valid.any():
                    prob_valid = model.predict_proba(X[valid])[:, 1]
                    prob[valid] = prob_valid
                    cls[valid] = (prob_valid >= PROB_THRESHOLD).astype("float32")
                    density[valid] = prob_valid * pixel_area_ha

                prob_reshaped = prob.reshape(h, w)
                
                prob_map[row:row+h, col:col+w] = prob_reshaped
                
                dst_prob.write(prob_reshaped, 1, window=window)
                dst_cls.write(cls.reshape(h, w), 1, window=window)
                dst_density.write(density.reshape(h, w), 1, window=window)

            print(f"Processed rows {row}–{min(row + WINDOW_SIZE, src.height)}")

print(f"\nSaved initial probability raster: {OUT_PROB}")
print(f"Saved initial class raster: {OUT_CLASS}")
print(f"Saved initial density raster: {OUT_DENSITY}")


print("\n" + "="*60)
print("STEP 2: Calculate Expected Deforestation (ED)")
print("="*60)

with rasterio.open(PRED_TIF) as src:
    ground_truth = src.read(ground_truth_idx + 1)  # rasterio uses 1-based indexing
    
    if SITE_ONLY:
        site_mask_full = geometry_mask(
            site_geoms,
            transform=src.transform,
            invert=True,
            out_shape=(src.height, src.width),
        )
        ground_truth = np.where(site_mask_full, ground_truth, np.nan)
    
    valid_gt = ~np.isnan(ground_truth)
    deforested_pixels = np.sum(ground_truth[valid_gt] == 1)
    
    ED = deforested_pixels * pixel_area_ha
    
    print(f"Deforested pixels: {deforested_pixels:,}")
    print(f"Expected Deforestation (ED): {ED:.2f} hectares")


print("\n" + "="*60)
print("STEP 3: Calculate Modeled Deforestation (MD)")
print("="*60)

with rasterio.open(OUT_DENSITY) as src:
    initial_density = src.read(1)
    
    valid_density = ~np.isnan(initial_density)
    MD = np.sum(initial_density[valid_density])
    
    print(f"Modeled Deforestation (MD): {MD:.2f} hectares")


print("\n" + "="*60)
print("STEP 4: Calculate Adjustment Ratio (AR)")
print("="*60)

AR = ED / MD if MD > 0 else 1.0

print(f"Adjustment Ratio (AR): {AR:.6f}")

if AR > 1:
    print("→ Model underpredicted, scaling UP")
elif AR < 1:
    print("→ Model overpredicted, scaling DOWN")
else:
    print("→ Model prediction matches ground truth")


print("\n" + "="*60)
print("STEP 5: Apply Quantity Adjustment")
print("="*60)

with rasterio.open(OUT_DENSITY) as src:
    adjusted_density = src.read(1).copy()
    profile = src.profile.copy()

iteration = 0
max_iterations = 100

while True:
    iteration += 1
    
    valid_mask = ~np.isnan(adjusted_density)
    adjusted_density[valid_mask] *= AR
    
    exceeds_max = adjusted_density > pixel_area_ha
    n_exceeded = np.sum(exceeds_max & valid_mask)
    
    if n_exceeded > 0:
        print(f"Iteration {iteration}: {n_exceeded} pixels exceed max density, capping and recalculating...")
        
        adjusted_density[exceeds_max] = pixel_area_ha
        
        MD = np.sum(adjusted_density[valid_mask])
        AR = ED / MD if MD > 0 else 1.0
        
        print(f"  New AR: {AR:.6f}")
        
        if AR <= 1.00001:
            print(f"  Converged after {iteration} iterations (AR ≤ 1.00001)")
            break
            
        if iteration >= max_iterations:
            print(f"  WARNING: Reached max iterations ({max_iterations})")
            break
    else:
        print(f"No pixels exceeded maximum density")
        break

final_MD = np.sum(adjusted_density[valid_mask])
final_AR = ED / final_MD if final_MD > 0 else 1.0

print(f"\nFinal Modeled Deforestation: {final_MD:.2f} hectares")
print(f"Final Adjustment Ratio: {final_AR:.6f}")
print(f"Adjustment error: {abs(final_MD - ED):.2f} hectares ({abs(final_MD - ED)/ED*100:.4f}%)")


print("\n" + "="*60)
print("STEP 6: Save Adjusted Density Map")
print("="*60)

with rasterio.open(OUT_ADJUSTED, "w", **profile) as dst:
    dst.write(adjusted_density, 1)

print(f"Saved adjusted density raster: {OUT_ADJUSTED}")


print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

valid_adjusted = ~np.isnan(adjusted_density) & (adjusted_density > 0)

if np.any(valid_adjusted):
    min_density = np.min(adjusted_density[valid_adjusted])
    max_density = np.max(adjusted_density[valid_adjusted])
    mean_density = np.mean(adjusted_density[valid_adjusted])
    median_density = np.median(adjusted_density[valid_adjusted])
    
    min_prob = (min_density / pixel_area_ha) * 100
    max_prob = (max_density / pixel_area_ha) * 100
    mean_prob = (mean_density / pixel_area_ha) * 100
    median_prob = (median_density / pixel_area_ha) * 100
    
    print(f"\nAdjusted Density (ha/pixel):")
    print(f"  Min:    {min_density:.8f} ha/pixel ({min_prob:.4f}%)")
    print(f"  Max:    {max_density:.8f} ha/pixel ({max_prob:.4f}%)")
    print(f"  Mean:   {mean_density:.8f} ha/pixel ({mean_prob:.4f}%)")
    print(f"  Median: {median_density:.8f} ha/pixel ({median_prob:.4f}%)")
    
    print(f"\nTotal Predicted Deforestation: {final_MD:.2f} hectares")
    print(f"Expected Deforestation (Ground Truth): {ED:.2f} hectares")

print("\n" + "="*60)
print("PREDICTION COMPLETE")
print("="*60)



print("\n" + "="*60)
print("STEP 7: Convert Adjusted Density to Probability")
print("="*60)

OUT_ADJUSTED_PROB = OUT_DIR / f"deforestation_adjusted_prob_{predict_year}_{out_suffix}_{suffix}.tif"

adjusted_probability = np.where(
    ~np.isnan(adjusted_density),
    adjusted_density / pixel_area_ha,
    np.nan
)

valid_probs = adjusted_probability[~np.isnan(adjusted_probability)]
if np.any(valid_probs > 1.0):
    print(f"WARNING: {np.sum(valid_probs > 1.0)} pixels have probability > 1.0!")

with rasterio.open(OUT_ADJUSTED_PROB, "w", **profile) as dst:
    dst.write(adjusted_probability.astype('float32'), 1)

print(f"Saved adjusted probability raster: {OUT_ADJUSTED_PROB}")
print(f"\nAdjusted Probability Range: [{np.min(valid_probs):.6f}, {np.max(valid_probs):.6f}]")
print(f"Adjusted Probability Range: [{np.min(valid_probs)*100:.4f}%, {np.max(valid_probs)*100:.4f}%]")