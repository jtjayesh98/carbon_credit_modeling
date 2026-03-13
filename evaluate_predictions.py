import rasterio
import numpy as np
import json
import pandas as pd
import argparse

from rasterio.windows import Window
from rasterio.features import geometry_mask
from shapely.geometry import shape, Polygon, LinearRing
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)

from rasterio.warp import reproject, Resampling


parser = argparse.ArgumentParser(
    description="Evaluate deforestation predictions"
)

group = parser.add_mutually_exclusive_group(required=True)

group.add_argument(
    "--odisha-sites",
    action="store_true",
    help="Evaluate only inside Odisha site polygons",
)

group.add_argument(
    "--sampled-pixels",
    action="store_true",
    help="Evaluate only at randomized sampled pixel locations",
)

args = parser.parse_args()

EVAL_ODISHA = args.odisha_sites
EVAL_SAMPLED = args.sampled_pixels


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "images"
PRED_DIR = BASE_DIR / "outputs" / "predictions"

predict_year = "2010_15"
ex_ante = True
data_preprocessing = "smote"

if ex_ante:
    train_year = (
        "20"
        + str(int(predict_year[2:4]) - 5).zfill(2)
        + "_"
        + str(int(predict_year[5:7]) - 5).zfill(2)
    )
else:
    train_year = predict_year[:4]

suffix = "ex_ante" if ex_ante else "ex_post"

counterfactual_tag = False
udef_arp_tag = False

if counterfactual_tag:
    PRED_TIF = (
        PRED_DIR
        / f"counterfactuals_prediction_{predict_year}_{suffix}.tif"
    )
elif udef_arp_tag:
    PRED_TIF = PRED_DIR / "Acre_Adjucted_Density_Map_VP.tif"
else:
    PRED_TIF = (
        PRED_DIR
        / f"deforestation_class_{predict_year}_full_{suffix}.tif"
    )

GT_TIF = DATA_DIR / f"training_data_x_{train_year}_y_{predict_year}.tif"

SITES_CSV = DATA_DIR / "Odisha_sites.csv"
SAMPLED_PIXELS_CSV = BASE_DIR / "sampled_ground_truth_pixels.csv"

LABEL_BAND = "9_deforestation"
WINDOW_SIZE = 1024


print("Prediction year:", predict_year)
print("Training year:", train_year)
print("Ex-ante:", ex_ante)
print("Evaluation mode:",
      "Odisha sites" if EVAL_ODISHA else "Sampled pixels")
print("Prediction raster:", PRED_TIF)
print("Ground truth raster:", GT_TIF)


with rasterio.open(GT_TIF) as gt_src:
    band_names = list(gt_src.descriptions)
    label_idx = band_names.index(LABEL_BAND)


site_geoms = None

if EVAL_ODISHA:
    sites_df = pd.read_csv(SITES_CSV)

    site_geoms = []
    for g in sites_df[".geo"]:
        geom = shape(json.loads(g))

        if isinstance(geom, LinearRing):
            geom = Polygon(geom)

        if not geom.is_valid:
            geom = geom.buffer(0)

        site_geoms.append(geom)

    print(f"Loaded {len(site_geoms)} Odisha site polygons")


sampled_rows = sampled_cols = sampled_gt = None

if EVAL_SAMPLED:
    samples_df = pd.read_csv(SAMPLED_PIXELS_CSV)

    sampled_rows = samples_df["row"].values.astype(int)
    sampled_cols = samples_df["col"].values.astype(int)
    sampled_gt = samples_df["ground_truth"].values.astype(int)

    print(f"Loaded {len(samples_df)} sampled validation pixels")


y_true_all = []
y_pred_all = []

with rasterio.open(GT_TIF) as gt_src, rasterio.open(PRED_TIF) as pred_src:

    if udef_arp_tag:
        aligned_pred = np.empty(
            (gt_src.height, gt_src.width),
            dtype=np.float32
        )

        reproject(
            source=rasterio.band(pred_src, 1),
            destination=aligned_pred,
            src_transform=pred_src.transform,
            src_crs=pred_src.crs,
            dst_transform=gt_src.transform,
            dst_crs=gt_src.crs,
            resampling=Resampling.nearest
        )

        pixel_area_ha = (30 * 30) / 10000.0
        aligned_pred[aligned_pred < 0] = 0

        rel_prob = aligned_pred / pixel_area_ha
        rel_prob = rel_prob / np.nanmax(rel_prob)

        pred_bin = (rel_prob >= 0.5).astype(np.uint8)

    else:
        pred_bin = pred_src.read(1).astype(np.uint8)

    gt_full = gt_src.read(label_idx + 1).astype(np.float32)

    if EVAL_ODISHA:

        for row in range(0, gt_src.height, WINDOW_SIZE):
            for col in range(0, gt_src.width, WINDOW_SIZE):

                window = Window(
                    col,
                    row,
                    min(WINDOW_SIZE, gt_src.width - col),
                    min(WINDOW_SIZE, gt_src.height - row),
                )

                gt = gt_src.read(label_idx + 1, window=window)
                pred = pred_bin[
                    row:row + window.height,
                    col:col + window.width
                ]

                mask = (~np.isnan(gt)) & (~np.isnan(pred))

                site_mask = geometry_mask(
                    site_geoms,
                    transform=gt_src.window_transform(window),
                    invert=True,
                    out_shape=gt.shape,
                )

                mask &= site_mask

                if mask.any():
                    y_true_all.append(gt[mask].ravel())
                    y_pred_all.append(pred[mask].ravel())

        print("Finished Odisha site evaluation")

    if EVAL_SAMPLED:

        for r, c, gt_val in zip(
            sampled_rows,
            sampled_cols,
            sampled_gt
        ):
            if np.isnan(gt_full[r, c]):
                continue

            y_true_all.append(gt_val)
            y_pred_all.append(pred_bin[r, c])

        print("Finished sampled pixel evaluation")


if not y_true_all:
    raise RuntimeError("No valid pixels found for evaluation.")

y_true = np.array(y_true_all).astype(int)
y_pred = np.array(y_pred_all).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
kappa = cohen_kappa_score(y_true, y_pred)


print("\n=== Evaluation Results ===")
print(f"Prediction year: {predict_year}")
print(f"Training year:   {train_year}")
print(f"Evaluation mode: {'Odisha sites' if EVAL_ODISHA else 'Sampled pixels'}")
print(f"Accuracy:        {accuracy:.4f}")
print(f"Kappa:           {kappa:.4f}")
print(f"Precision:       {precision:.4f}")
print(f"Recall:          {recall:.4f}")
print(f"F1 Score:        {f1:.4f}")
print("\nConfusion Matrix:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")
