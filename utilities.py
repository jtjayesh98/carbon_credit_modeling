import json
import warnings
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd

from shapely.geometry import shape, Polygon, LinearRing
from rasterio.features import rasterize



# Convert GeoTIFF to DataFrame with treatment assignment based on polygons
def tif_to_dataframe_with_treatment(
    tif_path: str,
    polygon_csv_path: str,
    geo_column: str = ".geo",
    drop_all_nan: bool = True,
    all_touched: bool = False,
):
    """
    Convert a GeoTIFF into a pixel-level DataFrame and assign a Treatment
    indicator based on polygon geometries stored as LinearRings.

    The returned DataFrame retains raster + geometry metadata in df.attrs
    so rasters can always be reconstructed later.
    """

    with rasterio.open(tif_path) as src:
        img = src.read(masked=False)
        transform = src.transform
        raster_crs = src.crs
        height, width = src.height, src.width
        band_names = src.descriptions

    if raster_crs is None:
        raise ValueError("Raster CRS is None — cannot align geometries.")

    bands = img.shape[0]

    X = img.reshape(bands, -1).T

    if band_names is None or all(bn is None for bn in band_names):
        columns = [f"band_{i+1}" for i in range(bands)]
    else:
        columns = [
            bn if bn is not None else f"band_{i+1}"
            for i, bn in enumerate(band_names)
        ]

    df = pd.DataFrame(X, columns=columns)

    if drop_all_nan:
        df = df.dropna(how="all")

    poly_df = pd.read_csv(polygon_csv_path)

    if geo_column not in poly_df.columns:
        raise ValueError(f"Column '{geo_column}' not found in polygon CSV.")

    geometries = []

    for idx, geo_str in enumerate(poly_df[geo_column]):
        geom = shape(json.loads(geo_str))

        if isinstance(geom, LinearRing):
            geom = Polygon(geom)

        if not isinstance(geom, Polygon):
            raise TypeError(
                f"Geometry at row {idx} is not a Polygon after conversion."
            )

        if not geom.is_valid:
            geom = geom.buffer(0)

        geometries.append(geom)

    gdf = gpd.GeoDataFrame(
        poly_df,
        geometry=geometries,
        crs="EPSG:4326"
    )

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    treatment_mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )

    df["Treatment"] = treatment_mask.reshape(-1)[df.index]

    treated_ratio = df["Treatment"].mean()

    if treated_ratio == 0.0:
        warnings.warn(
            "No treated pixels detected — check CRS or geometry alignment."
        )

    if treated_ratio > 0.5:
        warnings.warn(
            f"High treated ratio ({treated_ratio:.2f}). "
            "Polygons may be too large or misaligned."
        )


    rows, cols = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing="ij"
    )

    df["row"] = rows.reshape(-1)[df.index]
    df["col"] = cols.reshape(-1)[df.index]

    return df



# Reconstruct multi-band GeoTIFF from DataFrame with row/col indices
def dataframe_to_multiband_tif(
    df,
    value_cols,
    output_tif,
    height,
    width,
    transform,
    crs,
    dtype="float32",
    nodata=np.nan,
):
    """
    Reconstruct a multi-band GeoTIFF from a DataFrame with row/col indices.

    value_cols : list of column names to export as bands
    """

    bands = []

    for col in value_cols:
        band = np.full((height, width), nodata, dtype=dtype)
        band[
            df["row"].values,
            df["col"].values
        ] = df[col].values
        bands.append(band)

    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(value_cols),
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        for i, (band, name) in enumerate(zip(bands, value_cols), start=1):
            dst.write(band, i)
            dst.set_band_description(i, name)

    return output_tif