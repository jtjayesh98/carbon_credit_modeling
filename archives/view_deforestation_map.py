import geemap
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "images"
PRED_DIR = BASE_DIR / "outputs" / "predictions"

PERIOD = "2005_10"

GT_TIF = DATA_DIR / f"training_data_x_{PERIOD}_y_{PERIOD}.tif"
PRED_TIF = PRED_DIR / f"deforestation_class_{PERIOD}.tif"

Map = geemap.Map()

PALETTE = ["#00FF00", "#FF0000"]

Map.add_raster(
    str(GT_TIF),
    band=10,
    palette=PALETTE,
    vmin=0,
    vmax=1,
    layer_name="Ground Truth"
)

Map.add_raster(
    str(PRED_TIF),
    palette=PALETTE,
    vmin=0,
    vmax=1,
    layer_name="Prediction"
)

Map.add_layer_control()

Map.to_html("deforestation_map.html", title="Deforestation Map")
print("Map saved to deforestation_map.html")
