import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import json
from rasterio.warp import reproject, Resampling
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Path to forest cover raster
forest_cover_path = "./data/images/Dhenkanal_2010.tif"

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

# Get forest cover CRS
with rasterio.open(forest_cover_path) as fc_src:
    fc_crs = fc_src.crs

# Reproject geometries to forest cover CRS
gdf = gdf.to_crs(fc_crs)

# For each site
for idx, row in gdf.iterrows():
    site_name = row['Name']
    geom = row['geometry']

    # Mask the forest cover to the site geometry
    with rasterio.open(forest_cover_path) as src:
        out_image, out_transform = mask(src, [geom], crop=True, nodata=src.nodata)

    # Convert to float and set nodata to nan
    out_image = out_image.astype(np.float32)
    if src.nodata is not None:
        out_image = np.where(out_image == src.nodata, np.nan, out_image)

    # Remove singleton dimension if present
    if out_image.ndim == 3:
        out_image = out_image[0]

    # Get extent for plotting
    bounds = rasterio.transform.array_bounds(out_image.shape[0], out_image.shape[1], out_transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]  # left, right, bottom, top

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        out_image,
        cmap="Greens",
        extent=extent,
        origin="upper"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Forest Cover Map - {site_name}", fontsize=12)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Forest Cover")

    plt.tight_layout()

    # Save figures
    plt.savefig(f"./figures/forest_cover_{site_name}.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(f"./figures/forest_cover_{site_name}.jpg", dpi=600, bbox_inches="tight")

    plt.close()  # Close to avoid memory issues

print("Forest cover images generated for all Odisha sites.")