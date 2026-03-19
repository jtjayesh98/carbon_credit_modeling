import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import shape
import json


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


# Compute additionality metrics for an entire area defined by a geometry
def compute_additionality_for_area(geom, prob_raster_path, truth_raster_path, site_name, map_type):
    target_crs = 'EPSG:4326'
    
    # Get bounds of the geometry in WGS84
    minx, miny, maxx, maxy = geom.bounds

    # Define resolution (degrees, approx 10m)
    res = 0.0001

    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)

    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    # Reproject prob to this WGS84 grid
    with rasterio.open(prob_raster_path) as prob_src:
        band = 2 if map_type.startswith('cf') else 1
        prob_reproj = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=prob_src.read(band),
            destination=prob_reproj,
            src_transform=prob_src.transform,
            src_crs=prob_src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

    # Reproject truth to the same grid
    with rasterio.open(truth_raster_path) as truth_src:
        truth_band = get_band_index_by_name(truth_src, "9_deforestation")
        truth_reproj = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=truth_src.read(truth_band),
            destination=truth_reproj,
            src_transform=truth_src.transform,
            src_crs=truth_src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

    # Now, mask both with the geometry (geom is in WGS84)
    from rasterio.features import rasterize
    mask_array = rasterize([geom], out_shape=(height, width), transform=transform, fill=0, default_value=1, dtype=np.uint8)

    # Apply mask
    prob_masked = np.where(mask_array == 1, prob_reproj, np.nan)
    truth_masked = np.where(mask_array == 1, truth_reproj, np.nan)

    # Compute pixel area (in m², using approximate lat)
    lat_ref = geom.centroid.y
    R = 6378137
    deg2rad = np.pi / 180
    pixel_width_m = res * deg2rad * R * np.cos(lat_ref * deg2rad)
    pixel_height_m = res * deg2rad * R
    pixel_area = pixel_width_m * pixel_height_m

    print(f"Pixel area (m²): {pixel_area:.3f} for {site_name}, {map_type}")

    # Compute additionality
    valid = ~np.isnan(prob_masked) & ~np.isnan(truth_masked)
    if site_name == "Pangatira":
        print(f"Mean pred for {site_name}, {map_type}: {np.mean(prob_masked[valid]):.6f}")
        print(f"Mean truth for {site_name}, {map_type}: {np.mean(truth_masked[valid]):.6f}")
    if map_type == 'udef_arp':
        additionality = np.sum(0 - (truth_masked[valid] * pixel_area / 10000 - prob_masked[valid]))
    else:
        additionality = np.sum(0 - ((truth_masked[valid] - prob_masked[valid]) * pixel_area) / 10000)

    return additionality




# Placeholders for the 5 different maps
maps = {
    'cf_ex_ante': {'path': None, 'scale': 'm2'},  # Placeholder, set to actual path
    'cf_ex_post': {'path': None, 'scale': 'm2'},
    'model_ex_ante': {'path': None, 'scale': 'm2'},
    'model_ex_post': {'path': None, 'scale': 'm2'},
    'udef_arp': {'path': None, 'scale': 'other'},
}

# For now, set example paths (update as needed)
maps['cf_ex_ante']['path'] = "./outputs/predictions/counterfactual_prediction_FULL_2010_15_ex_ante.tif"  # Placeholder
maps['cf_ex_post']['path'] = "./outputs/predictions/counterfactual_prediction_FULL_2010_15_ex_post.tif"
maps['model_ex_ante']['path'] = "./outputs/predictions/deforestation_prob_2010_15_full_ex_ante.tif"
maps['model_ex_post']['path'] = "./outputs/predictions/deforestation_prob_2010_15_full_ex_post.tif"
maps['udef_arp']['path'] = "./outputs/predictions/Acre_Adjucted_Density_Map_VP.tif"

# Load Odisha sites
df_sites = pd.read_csv('./data/images/Odisha_sites.csv')
def parse_geo(geo_str):
    data = json.loads(geo_str)
    if data['type'] == 'LinearRing':
        data['type'] = 'Polygon'
        data['coordinates'] = [data['coordinates']]
    return shape(data)
df_sites['geometry'] = df_sites['.geo'].apply(parse_geo)
gdf_sites = gpd.GeoDataFrame(df_sites, geometry='geometry')
gdf_sites.crs = 'EPSG:4326'

# Compute areas in hectares
gdf_projected = gdf_sites.to_crs('EPSG:3857')  # Web Mercator for area calculation
gdf_sites['area_ha'] = gdf_projected.area / 10000  # m² to ha

TRUTH_TIF = "./data/images/training_data_x_2005_10_y_2010_15.tif"

# Compute for each site and each map
results = {}
for idx, row in gdf_sites.iterrows():
    site_name = row['Name']
    geom = row['geometry']
    area_ha = row['area_ha']
    results[site_name] = {'area_ha': area_ha}
    for map_type, map_info in maps.items():
        if map_info['path'] is None:
            print(f"Skipping {map_type} for {site_name}: path not set")
            continue
        additionality = compute_additionality_for_area(geom, map_info['path'], TRUTH_TIF, site_name, map_type)
        results[site_name][map_type] = additionality
        print(f"Additionality for {site_name}, {map_type}: {additionality}")

# Optionally, save results
import json
with open('./additionality_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Create and display table
import pandas as pd
df_results = pd.DataFrame(results).T  # Sites as rows, maps as columns
print("\nAdditionality Table:")
print(df_results)
df_results.to_csv('./additionality_table.csv')
print("Table saved to ./additionality_table.csv")



# print("Additionality summary:")
# print(df_sites["additionality"].describe())
# print("Total additionality (ha):", df_sites["additionality"].sum())
