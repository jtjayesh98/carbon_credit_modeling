import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
from rasterio.features import rasterize
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import json
from rasterio.warp import reproject, Resampling
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Path to prediction raster
# prediction_path = "./outputs/predictions/Acre_Modeling_Region_HRP.tif"
# prediction_path = "./outputs/predictions/Acre_Adjucted_Density_Map_VP.tif"
prediction_path = "./outputs/predictions/counterfactual_prediction_FULL_2010_15_ex_ante.tif"
# prediction_path = "./outputs/predictions/deforestation_adjusted_2010_15_full_ex_post.tif"

# Determine band to read
band = 2 if 'counterfactual' in prediction_path else 1
# Read Odisha sites CSV
df = pd.read_csv('./data/images/Odisha_sites.csv')
def parse_geo(geo_str):
    data = json.loads(geo_str)
    if data['type'] == 'LinearRing':
        data['type'] = 'Polygon'
        data['coordinates'] = [data['coordinates']]
    return shape(data)
df['geometry'] = df['.geo'].apply(parse_geo)
gdf = gpd.GeoDataFrame(df, geometry='geometry')

# Set CRS to WGS84
gdf.crs = 'EPSG:4326'

# Target CRS: WGS84
target_crs = 'EPSG:4326'

# Load prediction data
with rasterio.open(prediction_path) as pred_src:
    pred = pred_src.read(band)
    pred_transform = pred_src.transform
    pred_crs = pred_src.crs
    nodata = pred_src.nodata

if nodata is not None:
    pred = np.where(pred == nodata, np.nan, pred)

# For each site
for idx, row in gdf.iterrows():
    site_name = row['Name']
    geom = row['geometry']

    # Get bounds of the geometry in WGS84
    minx, miny, maxx, maxy = geom.bounds

    # Define resolution (degrees, approx 10m)
    res = 0.0001

    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)

    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    # Reproject prediction to this WGS84 grid
    pred_reproj = np.zeros((height, width), dtype=np.float32)

    reproject(
        source=pred,
        destination=pred_reproj,
        src_transform=pred_transform,
        src_crs=pred_crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest
    )

    # Now, mask the reprojected prediction with the geometry
    # Since the grid is exactly the bounds, and geom is inside, but to mask exactly to the polygon
    mask_array = np.zeros((height, width), dtype=np.uint8)
    # But easier, since reproject fills with nan outside? No, reproject fills with 0 or something.

    # Actually, to mask, I can use rasterio.features.rasterize to create a mask from geom
    from rasterio.features import rasterize

    mask_array = rasterize([geom], out_shape=(height, width), transform=transform, fill=0, default_value=1, dtype=np.uint8)

    # Apply mask
    pred_masked = np.where(mask_array == 1, pred_reproj, np.nan)

    # Now, pred_masked is the final image
    out_image = pred_masked
    out_transform = transform

    # Get extent
    bounds = rasterio.transform.array_bounds(out_image.shape[0], out_image.shape[1], out_transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        # out_image/1000,
        out_image,
        cmap="viridis",
        extent=extent,
        origin="upper"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    map_type = "Counterfactual" if 'counterfactual' in prediction_path else "Prediction"
    ax.set_title(f"{map_type} Density Map - {site_name}", fontsize=12)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Prediction Score")

    plt.tight_layout()

    # Save figures
    suffix = "cf" if 'counterfactual' in prediction_path else "model"
    plt.savefig(f"./figures/{suffix}_prediction_ex_ante_{site_name}.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(f"./figures/{suffix}_prediction_ex_ante_{site_name}.jpg", dpi=600, bbox_inches="tight")

    plt.close()

print("Prediction images generated for all Odisha sites with proper alignment.")