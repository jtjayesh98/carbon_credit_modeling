import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from mpl_toolkits.axes_grid1 import make_axes_locatable

prediction_path = "./outputs/predictions/Acre_Adjucted_Density_Map_VP.tif"
# prediction_path = "./outputs/predictions/Acre_Modeling_Region_HRP.tif"
mask_path = "./data/images/Dhenkanal_2010.tif"

with rasterio.open(mask_path) as mask_src:
    mask = mask_src.read(1)
    mask_transform = mask_src.transform
    mask_crs = mask_src.crs
    mask_height = mask_src.height
    mask_width = mask_src.width

with rasterio.open(prediction_path) as pred_src:
    pred = pred_src.read(1)
    pred_transform = pred_src.transform
    pred_crs = pred_src.crs
    nodata = pred_src.nodata

if nodata is not None:
    pred = np.where(pred == nodata, np.nan, pred)

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

valid_mask = ~np.isnan(pred_aligned)

rows = np.any(valid_mask, axis=1)
cols = np.any(valid_mask, axis=0)

row_min, row_max = np.where(rows)[0][[0, -1]]
col_min, col_max = np.where(cols)[0][[0, -1]]

pred_cropped = pred_aligned[row_min:row_max+1, col_min:col_max+1]

left = mask_transform.c + col_min * mask_transform.a
right = mask_transform.c + (col_max + 1) * mask_transform.a
top = mask_transform.f + row_min * mask_transform.e
bottom = mask_transform.f + (row_max + 1) * mask_transform.e

extent = [left, right, bottom, top]

fig, ax = plt.subplots(figsize=(7,5))

im = ax.imshow(
    pred_cropped/1000,
    cmap="viridis",
    extent=extent,
    origin="upper"
)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Adjusted Prediction Density Map", fontsize=12)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("Deforestation Vulnerability Classes (UDef-ARP)")
# cbar.set_label("Prediction Score")

plt.tight_layout()

plt.savefig("./figures/region.pdf", dpi=600, bbox_inches="tight")
plt.savefig("./figures/region.jpg", dpi=600, bbox_inches="tight")

plt.show()