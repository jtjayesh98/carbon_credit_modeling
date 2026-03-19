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
import os

# Path to ground truth raster
gt_path = "./data/images/training_data_x_2010_y_2010_15.tif"

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

# Load ground truth data
with rasterio.open(gt_path) as gt_src:
    descriptions = list(gt_src.descriptions)
    if "9_deforestation" in descriptions:
        band_idx = descriptions.index("9_deforestation") + 1
    else:
        band_idx = 1
    gt = gt_src.read(band_idx)
    gt_transform = gt_src.transform
    gt_crs = gt_src.crs
    nodata = gt_src.nodata

if nodata is not None:
    gt = np.where(gt == nodata, np.nan, gt)

# Ensure figures folder exists
os.makedirs("./figures", exist_ok=True)

# For each site
for idx, row in gdf.iterrows():
    site_name = row['Name']
    geom = row['geometry']

    # Count deforested pixels accurately from original gt
    from rasterio.features import geometry_mask
    gt_mask = geometry_mask([geom], transform=gt_transform, invert=True, out_shape=gt.shape)
    gt_masked_orig = np.where(gt_mask, gt, np.nan)
    deforested_pixels = np.sum(gt_masked_orig == 1)
    deforested_area_ha = deforested_pixels * 0.09  # Assuming 0.09 ha per pixel
    print(f"{site_name}: Deforested pixels = {deforested_pixels}, Deforested area = {deforested_area_ha:.2f} ha")

    # Get bounds of the geometry in WGS84
    minx, miny, maxx, maxy = geom.bounds

    # Define resolution (degrees, approx 10m)
    res = 0.0001

    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)

    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    # Reproject gt to this WGS84 grid
    gt_reproj = np.zeros((height, width), dtype=np.float32)

    reproject(
        source=gt,
        destination=gt_reproj,
        src_transform=gt_transform,
        src_crs=gt_crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest
    )

    # Mask with the geometry
    mask_array = rasterize([geom], out_shape=(height, width), transform=transform, fill=0, default_value=1, dtype=np.uint8)

    # Apply mask
    gt_masked = np.where(mask_array == 1, gt_reproj, np.nan)

    # Now, gt_masked is the final image
    out_image = gt_masked
    out_transform = transform

    # Get extent
    bounds = rasterio.transform.array_bounds(out_image.shape[0], out_image.shape[1], out_transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        out_image,
        cmap="Reds",  # Binary colormap for deforestation
        extent=extent,
        origin="upper",
        vmin=0, vmax=1
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Ground Truth Deforestation - {site_name}", fontsize=12)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=[0, 1])
    cbar.set_label("Deforestation (0=No, 1=Yes)")

    plt.tight_layout()

    # Save figures
    plt.savefig(f"./figures/gt_deforestation_{site_name}.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(f"./figures/gt_deforestation_{site_name}.jpg", dpi=600, bbox_inches="tight")

    plt.close()

print("Ground truth deforestation images generated for all Odisha sites.")