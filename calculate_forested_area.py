import rasterio
import numpy as np
import pandas as pd
import json
from shapely.geometry import shape
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "images"

# Load sites
sites_df = pd.read_csv(DATA_DIR / "Odisha_sites.csv")
sites_df['geometry'] = sites_df['.geo'].apply(lambda x: shape(json.loads(x)))
# Convert LinearRing to Polygon if needed
from shapely.geometry import Polygon, LinearRing
sites_df['geometry'] = sites_df['geometry'].apply(lambda g: Polygon(g) if isinstance(g, LinearRing) else g)

# Function to compute area like in additionality.py
def compute_polygon_area_ha(geom):
    target_crs = 'EPSG:4326'
    minx, miny, maxx, maxy = geom.bounds
    res = 0.0001  # degrees, approx 10m
    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    
    from rasterio.features import rasterize
    mask_array = rasterize([geom], out_shape=(height, width), transform=transform, fill=0, default_value=1, dtype=np.uint8)
    
    total_pixels = np.sum(mask_array == 1)
    
    # Compute pixel area
    lat_ref = geom.centroid.y
    R = 6378137
    deg2rad = np.pi / 180
    pixel_width_m = res * deg2rad * R * np.cos(lat_ref * deg2rad)
    pixel_height_m = res * deg2rad * R
    pixel_area_m2 = pixel_width_m * pixel_height_m
    area_ha = total_pixels * pixel_area_m2 / 10000
    return area_ha

# Add area to df
sites_df['total_area_ha'] = sites_df['geometry'].apply(compute_polygon_area_ha)

# Load training tif for valid mask and transform
gt_tif = DATA_DIR / "training_data_x_2010_y_2010_15.tif"
with rasterio.open(gt_tif) as gt_src:
    # Read the "9_deforestation" band
    descriptions = list(gt_src.descriptions)
    if "9_deforestation" in descriptions:
        band_idx = descriptions.index("9_deforestation") + 1
    else:
        band_idx = 1  # fallback
    gt_deforestation = gt_src.read(band_idx)
    valid_mask_gt = ~np.isnan(gt_deforestation)
    transform = gt_src.transform

# Load Dhenkanal_2010.tif and reproject to training grid
dhenkanal_tif = DATA_DIR / "Dhenkanal_2010.tif"
with rasterio.open(dhenkanal_tif) as src:
    dhenkanal_data = src.read(1)
    dhenkanal_aligned = np.full_like(gt_deforestation, np.nan, dtype=dhenkanal_data.dtype)
    reproject(
        source=dhenkanal_data,
        destination=dhenkanal_aligned,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=gt_src.crs,
        resampling=Resampling.nearest
    )
    # Create valid mask for dhenkanal (non-NaN pixels)
    valid_mask_dhenkanal = ~np.isnan(dhenkanal_aligned)

results = []

for idx, row in sites_df.iterrows():
    name = row['Name']
    geom = row['geometry']

    # Create geometry mask on training grid
    geom_mask = geometry_mask([geom], transform=transform, invert=True, out_shape=dhenkanal_aligned.shape)

    # Combined mask for dhenkanal (valid pixels in dhenkanal within geometry)
    combined_mask_dhenkanal = valid_mask_dhenkanal & geom_mask

    # Total area = all pixels (0 and 1) in Dhenkanal_2010 within geometry
    total_pixels = np.sum(combined_mask_dhenkanal)
    total_area_ha = total_pixels * 0.09

    # Forested area = pixels with value 1 in Dhenkanal_2010 within geometry
    forested_mask = (dhenkanal_aligned == 1) & combined_mask_dhenkanal
    forested_pixels = np.sum(forested_mask)
    forested_area_ha = forested_pixels * 0.09

    # Deforested area = pixels with value 1 in ground truth deforestation within geometry
    combined_mask_gt = valid_mask_gt & geom_mask
    deforested_mask = (gt_deforestation == 1) & combined_mask_gt
    deforested_pixels = np.sum(deforested_mask)
    deforested_area_ha = deforested_pixels * 0.09

    results.append({
        'Region': name,
        'Total_Area_ha': total_area_ha,
        'Forested_Area_ha': forested_area_ha,
        'Deforested_Area_ha': deforested_area_ha
    })

    print(f"{name}: Total {total_area_ha:.2f} ha, Forested {forested_area_ha:.2f} ha, Deforested {deforested_area_ha:.2f} ha")

# Save to csv
output_df = pd.DataFrame(results)
output_df.to_csv(BASE_DIR / "forested_and_deforested_area_by_region.csv", index=False)