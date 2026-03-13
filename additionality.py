import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from rasterio.warp import reproject, Resampling

from pyproj import Transformer


# Reproject x,y coordinates from source CRS to destination CRS
def reproject_xy(x, y, src_crs, dst_crs):
    if src_crs == dst_crs:
        return x, y

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(x, y)

# Sample values from a numpy array at specified points, handling CRS reprojection
def sample_array_at_points(
    array,
    transform,
    src_crs,
    df,
    xy_cols=("x", "y"),
    df_crs_col="crs"
):
    vals = []

    for _, r in df.iterrows():
        x, y = r[xy_cols[0]], r[xy_cols[1]]
        point_crs = r[df_crs_col]

        x_r, y_r = reproject_xy(x, y, point_crs, src_crs)
        col, row = rowcol(transform, x_r, y_r)

        if row < 0 or col < 0 or row >= array.shape[0] or col >= array.shape[1]:
            vals.append(np.nan)
        else:
            vals.append(array[row, col])

    return np.array(vals)



# Sample raster values at given points, reprojecting coordinates if necessary
def sample_raster_at_points(
    src,
    df,
    band=1,
    xy_cols=("x", "y"),
    df_crs_col="crs"
):
    """
    CRS-safe raster sampling.
    Uses x/y coordinates, reprojects if needed,
    then converts to row/col using raster transform.
    """
    vals = []
    raster_crs = src.crs

    for _, r in df.iterrows():
        x, y = r[xy_cols[0]], r[xy_cols[1]]
        point_crs = r[df_crs_col]

        x_r, y_r = reproject_xy(x, y, point_crs, raster_crs)

        col, row = rowcol(src.transform, x_r, y_r)

        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            vals.append(np.nan)
        else:
            vals.append(
                src.read(band, window=((row, row + 1), (col, col + 1)))[0, 0]
            )

    return np.array(vals)



# Reproject truth raster band to prediction grid using nearest neighbor resampling
def reproject_truth_to_prediction(truth_path, pred_src, truth_band=9):
    """
    Reprojects truth band to prediction grid using nearest neighbor.
    """
    with rasterio.open(truth_path) as truth_src:
        truth_data = truth_src.read(truth_band)

        aligned = np.empty(
            (pred_src.height, pred_src.width),
            dtype=truth_data.dtype
        )

        reproject(
            source=truth_data,
            destination=aligned,
            src_transform=truth_src.transform,
            src_crs=truth_src.crs,
            dst_transform=pred_src.transform,
            dst_crs=pred_src.crs,
            resampling=Resampling.nearest
        )

    return aligned


from rasterio.crs import CRS

# Compute the area of a pixel in square meters for a given raster source
def compute_pixel_area(src, lat_ref=None):
    transform = src.transform
    crs = src.crs

    if crs and crs.is_projected:
        return abs(transform.a * transform.e)

    elif crs and crs.is_geographic:
        if lat_ref is None:
            raise ValueError("lat_ref required for geographic CRS")

        R = 6378137
        deg2rad = np.pi / 180

        pixel_width_m = (
            transform.a * deg2rad * R * np.cos(lat_ref * deg2rad)
        )
        pixel_height_m = abs(transform.e) * deg2rad * R

        return pixel_width_m * pixel_height_m

    else:
        raise ValueError("Unknown CRS type")


# Get the band index (1-based) for a given band name in the raster
def get_band_index_by_name(src, band_name):
    if src.descriptions is None:
        raise ValueError("Raster has no band descriptions")

    try:
        return src.descriptions.index(band_name) + 1  # rasterio is 1-based
    except ValueError:
        raise ValueError(
            f"Band '{band_name}' not found. "
            f"Available bands: {src.descriptions}"
        )
    
# Get truth data aligned to the prediction raster grid
def get_truth_on_prediction_grid(
    truth_path,
    pred_src,
    truth_band_name="9_deforestation"
):
    with rasterio.open(truth_path) as truth_src:

        truth_band = get_band_index_by_name(
            truth_src,
            truth_band_name
        )

        if (
            truth_src.crs == pred_src.crs and
            truth_src.transform == pred_src.transform and
            truth_src.width == pred_src.width and
            truth_src.height == pred_src.height
        ):
            return truth_src.read(truth_band)

        truth_data = truth_src.read(truth_band)

        aligned = np.full(
            (pred_src.height, pred_src.width),
            np.nan,
            dtype=np.float32
        )

        reproject(
            source=truth_data,
            destination=aligned,
            src_transform=truth_src.transform,
            src_crs=truth_src.crs,
            dst_transform=pred_src.transform,
            dst_crs=pred_src.crs,
            resampling=Resampling.nearest
        )

        return aligned


# Compute additionality metrics by comparing predicted probabilities with ground truth at sampled points
def compute_additionality(points_csv, prob_raster, truth_raster):
    df = pd.read_csv(points_csv)

    with rasterio.open(prob_raster) as prob_src:

        lat_ref = df["y"].mean()
        pixel_area = compute_pixel_area(prob_src, lat_ref)

        print(f"Pixel area (m²): {pixel_area:.3f}")

        truth_aligned = get_truth_on_prediction_grid(
            truth_raster,
            prob_src,
            truth_band_name="9_deforestation"
        )

        df["pred_prob"] = sample_raster_at_points(
            prob_src,
            df,
            band=1
        )


        df["truth"] = sample_array_at_points(
            truth_aligned,
            prob_src.transform,
            prob_src.crs,
            df
        )
    df = df.dropna(subset=["pred_prob", "truth"])

    assert np.isscalar(df["truth"].iloc[0]), "Truth must be scalar per row"
    udefarp = True
    print(df["truth"].sum())
    print(df["pred_prob"].sum())
    print(df.head())
    if udefarp:
        df["additionality"] = 0 - (df["truth"] * pixel_area/10000 - (df["pred_prob"]))
    else:
        df["additionality"] =  0 -((df["truth"] - df["pred_prob"]) * pixel_area)/10000



    return df




POINTS = "./data/images/sampled_ground_truth_pixels.csv"
PROB_TIF = "./outputs/predictions/deforestation_adjusted_prob_2010_15_full_ex_post.tif"
PROB_TIF = "./outputs/predictions/Acre_Adjucted_Density_Map_VP.tif"
TRUTH_TIF = "./data/images/training_data_x_2005_10_y_2010_15.tif"

df_add = compute_additionality(
    points_csv=POINTS,
    prob_raster=PROB_TIF,
    truth_raster=TRUTH_TIF
)



print("Additionality summary:")
print(df_add["additionality"].describe())
print("Total additionality (ha):", df_add["additionality"].sum())
