import json
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd

from shapely.geometry import Polygon
from rasterio.features import geometry_mask
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


predict_year = "2010_15"
ex_ante = True          # toggle ex-ante / ex-post here




K = 1
EPS = 1e-8


if ex_ante:
    train_year = (
        "20"
        + str(int(predict_year[2:4]) - 5).zfill(2)
        + "_"
        + str(int(predict_year[5:7]) - 5).zfill(2)
    )
    suffix = "ex_ante"
else:
    train_year = predict_year[:4]
    suffix = "ex_post"

TIF_PATH = f"./data/images/training_data_x_{train_year}_y_{predict_year}.tif"
SITES_CSV = "./data/images/Odisha_sites.csv"

OUTPUT_TIF = (
    f"./outputs/predictions/"
    f"counterfactuals_prediction_{predict_year}_{suffix}.tif"
)


with rasterio.open(TIF_PATH) as src:
    data = src.read()                # (bands, H, W)
    profile = src.profile
    transform = src.transform
    raster_crs = src.crs

X_all = data[:8]                     # feature bands
y_all = data[8]                      # deforestation label
H, W = y_all.shape

print(data[:8])
print("Raster shape:", H, W)


df = pd.read_csv(SITES_CSV)

def linearring_json_to_polygon(s):
    geom = json.loads(s)
    if geom["type"] != "LinearRing":
        raise ValueError("Expected LinearRing geometry")
    return Polygon(geom["coordinates"])

df["geometry"] = df[".geo"].apply(linearring_json_to_polygon)

gdf = gpd.GeoDataFrame(
    df,
    geometry="geometry",
    crs="EPSG:4326"
).to_crs(raster_crs)

regions_union = gdf.geometry.unary_union





train_mask = geometry_mask(
    geometries=[regions_union],
    transform=transform,
    invert=True,            # True = inside Odisha
    out_shape=(H, W)
)


X_train = X_all[:, train_mask].T     # (N_train, 8)
y_train = y_all[train_mask]

valid_train = ~np.isnan(X_train).any(axis=1)
X_train = X_train[valid_train]
y_train = y_train[valid_train]

print("Training pixels (Odisha):", len(y_train))


scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)


nn = NearestNeighbors(
    n_neighbors=K,
    metric="euclidean"
)
nn.fit(X_train_std)


X_full = X_all.reshape(X_all.shape[0], -1).T   # (H*W, 8)
valid_full = ~np.isnan(X_full).any(axis=1)

X_full_valid = X_full[valid_full]
X_full_std = scaler.transform(X_full_valid)

print("Prediction pixels (full raster):", X_full_valid.shape[0])


dist, idx = nn.kneighbors(X_full_std)

weights = 1.0 / (dist + EPS)
y_pred_full = (
    np.sum(weights * y_train[idx], axis=1)
    / np.sum(weights, axis=1)
)


prediction_map = np.full(H * W, np.nan, dtype=np.float32)
prediction_map[valid_full] = y_pred_full
prediction_map = prediction_map.reshape(H, W)

profile.update(
    count=1,
    dtype="float32",
    nodata=np.nan
)

with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
    dst.write(prediction_map, 1)


print("✅ Counterfactual raster written to:")
print("   ", OUTPUT_TIF)

print("✅ Training support: Odisha sites only")
print("✅ Prediction domain: FULL raster")

nan_frac = np.isnan(prediction_map).mean()
print(f"NaN fraction in output: {nan_frac:.4f}")
