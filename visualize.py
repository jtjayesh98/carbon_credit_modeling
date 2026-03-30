# ============================================
# ZOOM-INSET FIGURE (FINAL CLEAN VERSION)
# ============================================

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import shape, Polygon
import rasterio
from rasterio.warp import reproject, Resampling
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ============================================
# FILE PATHS
# ============================================

prediction_path = "./outputs/predictions/Acre_Modeling_Region_HRP.tif"
mask_path = "./data/images/Dhenkanal_2010.tif"
csv_path = "./data/images/Odisha_sites.csv"
site_name = "Pangatira"

# ============================================
# STEP 1 — LOAD MASK (REFERENCE GRID)
# ============================================

with rasterio.open(mask_path) as mask_src:
    mask = mask_src.read(1)
    mask_transform = mask_src.transform
    mask_crs = mask_src.crs
    mask_height = mask_src.height
    mask_width = mask_src.width

# ============================================
# STEP 2 — LOAD PREDICTION
# ============================================

with rasterio.open(prediction_path) as pred_src:
    pred = pred_src.read(1)
    pred_transform = pred_src.transform
    pred_crs = pred_src.crs
    nodata = pred_src.nodata

if nodata is not None:
    pred = np.where(pred == nodata, np.nan, pred)

# ============================================
# STEP 3 — ALIGN RASTER (CRITICAL)
# ============================================

pred_aligned = np.zeros((mask_height, mask_width), dtype=np.float32)

reproject(
    source=pred,
    destination=pred_aligned,
    src_transform=pred_transform,
    src_crs=pred_crs,
    dst_transform=mask_transform,
    dst_crs=mask_crs,
    resampling=Resampling.nearest
)

pred_aligned = np.where(mask == 0, np.nan, pred_aligned)

# ============================================
# STEP 4 — COMPUTE EXTENT
# ============================================

left = mask_transform.c
right = left + mask_width * mask_transform.a
top = mask_transform.f
bottom = top + mask_height * mask_transform.e

extent = [left, right, bottom, top]

# ============================================
# STEP 5 — LOAD SITE GEOMETRY
# ============================================

df = pd.read_csv(csv_path)
row = df[df['Name'] == site_name].iloc[0]

geo_json = json.loads(row['.geo'])
geom_raw = shape(geo_json)

# FIX: .geo entries are stored as "LinearRing" (not "Polygon").
# shape() returns a LinearRing object, which has no .exterior attribute
# and never matches geom_type == "Polygon" — so nothing was drawn.
# Wrap it in a Polygon explicitly.
if geom_raw.geom_type == "LinearRing":
    site_geom = Polygon(geom_raw)
else:
    site_geom = geom_raw  # already Polygon or MultiPolygon

gdf = gpd.GeoDataFrame(index=[0], geometry=[site_geom], crs="EPSG:4326")
gdf = gdf.to_crs(mask_crs)

minx, miny, maxx, maxy = gdf.total_bounds

# ============================================
# HELPER: geo → pixel coordinate
# ============================================

def geo_to_pixel(x_geo, y_geo):
    """Convert geographic coordinates to pixel indices in the full raster."""
    col = (x_geo - left) / mask_transform.a
    row_idx = (top - y_geo) / abs(mask_transform.e)
    return col, row_idx

# ============================================
# STEP 6 — CREATE FIGURE
# ============================================

fig, ax = plt.subplots(figsize=(9, 6))

im = ax.imshow(
    pred_aligned / 1000,
    cmap="viridis",
    extent=extent,
    origin="upper"
)

# --------------------------------------------
# DRAW SITE POLYGON ON MAIN MAP
# --------------------------------------------

for geom_item in gdf.geometry:
    if geom_item.geom_type == "Polygon":
        x, y = geom_item.exterior.xy
        ax.plot(x, y, color='red', linewidth=1.5, zorder=5)
    elif geom_item.geom_type == "MultiPolygon":
        for part in geom_item.geoms:
            x, y = part.exterior.xy
            ax.plot(x, y, color='red', linewidth=1.5, zorder=5)

# --------------------------------------------
# DRAW BOUNDING BOX ON MAIN MAP
# Pangatira is ~0.006 deg wide vs the ~0.9 deg wide raster, so a
# 100% buffer (buf = site_width) makes the box ~3x the site size
# and clearly visible at the full-raster zoom level.
# --------------------------------------------

buf = (maxx - minx) * 1.0  # 100% buffer

rect = mpatches.Rectangle(
    (minx - buf, miny - buf),
    (maxx - minx) + 2 * buf,
    (maxy - miny) + 2 * buf,
    linewidth=2,
    edgecolor='white',
    facecolor='none',
    linestyle='--',
    zorder=6
)
ax.add_patch(rect)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Adjusted Prediction Density Map with Site Zoom", fontsize=12)

# ============================================
# STEP 7 — INSET (INSIDE MAIN MAP, LOWER-RIGHT)
# ============================================

axins = inset_axes(
    ax,
    width="35%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(0, 0, 0.97, 0.97),
    bbox_transform=ax.transAxes,
    borderpad=0
)

# Pixel bounds for cropping (same buf as the rectangle above)
col_min = max(0,           int((minx - buf - left)  / mask_transform.a))
col_max = min(mask_width,  int((maxx + buf - left)  / mask_transform.a))
row_min = max(0,           int((top  - maxy - buf)  / abs(mask_transform.e)))
row_max = min(mask_height, int((top  - miny + buf)  / abs(mask_transform.e)))

cropped = pred_aligned[row_min:row_max, col_min:col_max]

axins.imshow(
    cropped / 1000,
    cmap="viridis",
    origin="upper"
)

# --------------------------------------------
# DRAW PANGATIRA BOUNDARY IN INSET
# Shift full-raster pixel coords by crop origin (col_min, row_min).
# --------------------------------------------

for geom_item in gdf.geometry:
    if geom_item.geom_type == "Polygon":
        coords = list(geom_item.exterior.coords)
        xs = [geo_to_pixel(c[0], c[1])[0] - col_min for c in coords]
        ys = [geo_to_pixel(c[0], c[1])[1] - row_min for c in coords]
        axins.plot(xs, ys, color='red', linewidth=1.5, zorder=5)
    elif geom_item.geom_type == "MultiPolygon":
        for part in geom_item.geoms:
            coords = list(part.exterior.coords)
            xs = [geo_to_pixel(c[0], c[1])[0] - col_min for c in coords]
            ys = [geo_to_pixel(c[0], c[1])[1] - row_min for c in coords]
            axins.plot(xs, ys, color='red', linewidth=1.5, zorder=5)

axins.set_xticks([])
axins.set_yticks([])
axins.set_title(site_name, fontsize=7, color='white', pad=2)

# ============================================
# STEP 7b — INSET BORDER + CONNECTOR LINES
# ============================================

# Black border around the inset panel
for spine in axins.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Connector lines: two right corners of the bbox on the main map
# to the two left corners of the inset panel.
# ConnectionPatch bridges two axes using their own coordinate systems:
#   coordsA = ax.transData  (geographic degrees)
#   coordsB = axins.transData (pixel indices in cropped array)
# inset_h = number of rows in cropped array = bottom y in axins (origin='upper')

inset_h = row_max - row_min

from matplotlib.patches import ConnectionPatch

# Top connector: bbox top-right -> inset top-left
con_top = ConnectionPatch(
    xyA=(maxx + buf, maxy + buf),
    coordsA=ax.transData,
    xyB=(0, 0),
    coordsB=axins.transData,
    color='black',
    linewidth=1.2,
    linestyle='--',
    zorder=10
)
fig.add_artist(con_top)

# Bottom connector: bbox bottom-right -> inset bottom-left
con_bot = ConnectionPatch(
    xyA=(maxx + buf, miny - buf),
    coordsA=ax.transData,
    xyB=(0, inset_h),
    coordsB=axins.transData,
    color='black',
    linewidth=1.2,
    linestyle='--',
    zorder=10
)
fig.add_artist(con_bot)

# ============================================
# STEP 8 — COLORBAR
# ============================================

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(im, cax=cax)
cbar.set_label("Deforestation Vulnerability Classes (UDef-ARP)")

# ============================================
# SAVE
# ============================================

plt.tight_layout()
plt.savefig("./figures/zoom_figure.pdf", dpi=600, bbox_inches="tight")
plt.savefig("./figures/zoom_figure.jpg", dpi=600, bbox_inches="tight")

plt.show()

# ============================================
# DONE
# ============================================