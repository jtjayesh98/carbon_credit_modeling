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
ex_ante = False

if ex_ante:
    train_year = "20" + str(int(predict_year[2:4]) - 5).zfill(2) + "_" + \
                 str(int(predict_year[5:7]) - 5).zfill(2)
    results = "ante"
else:
    train_year = predict_year[:4]
    results = "post"


TIF_PATH = f"./data/images/training_data_x_{train_year}_y_{predict_year}.tif"
CSV_PATH = f"./data/images/Odisha_sites.csv"
OUTPUT_TIF = f"./outputs/predictions/counterfactuals_prediction_{predict_year}_ex_{results}.tif"

K = 1
EPS = 1e-8


with rasterio.open(TIF_PATH) as src:
    data = src.read()                # (bands, H, W)
    profile = src.profile
    transform = src.transform
    raster_crs = src.crs

X_all = data[:8]                     # features
y_all = data[8]                      # label
H, W = y_all.shape


df = pd.read_csv(CSV_PATH)

def linearring_json_to_polygon(s):
    """
    Converts GeoJSON LinearRing string to shapely Polygon
    """
    geom = json.loads(s)
    if geom["type"] != "LinearRing":
        raise ValueError("Expected LinearRing geometry")
    return Polygon(geom["coordinates"])

df[".geo"] = df[".geo"].apply(linearring_json_to_polygon)

gdf = gpd.GeoDataFrame(
    df,
    geometry=".geo",
    crs="EPSG:4326"   # LinearRing coordinates are lon/lat
)

gdf = gdf.to_crs(raster_crs)

regions_union = gdf.geometry.unary_union


region_mask = geometry_mask(
    geometries=[regions_union],
    transform=transform,
    invert=True,          # True = inside polygon
    out_shape=(H, W)
)


X = X_all[:, region_mask].T          # (N, 8)
y = y_all[region_mask]               # (N,)

valid = ~np.isnan(X).any(axis=1)
X = X[valid]
y = y[valid]


scaler = StandardScaler()
X_std = scaler.fit_transform(X)


nn = NearestNeighbors(
    n_neighbors=K,
    metric="euclidean"
)
nn.fit(X_std)

dist, idx = nn.kneighbors(X_std)

print(dist.shape)


weights = 1.0 / (dist + EPS)
y_pred = np.sum(weights * y[idx], axis=1) / np.sum(weights, axis=1)


prediction_map = np.full((H, W), np.nan, dtype=np.float32)

flat_mask_indices = np.flatnonzero(region_mask)
prediction_map_flat = prediction_map.flatten()
prediction_map_flat[flat_mask_indices[valid]] = y_pred
prediction_map = prediction_map_flat.reshape(H, W)

profile.update(
    count=1,
    dtype="float32",
    nodata=np.nan
)

with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
    dst.write(prediction_map, 1)

print("✅ Euclidean weighted prediction written to:", OUTPUT_TIF)
print("✅ Total pixels used:", len(y_pred))
