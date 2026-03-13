import numpy as np
import pandas as pd
import rasterio
from scipy.stats import wasserstein_distance
from pathlib import Path
import argparse


parser = argparse.ArgumentParser(
    description="Compute Wasserstein distance between actual and counterfactual rasters"
)

parser.add_argument(
    "--actual-tif",
    type=str,
    required=True,
    help="Path to actual / baseline prediction raster",
)

parser.add_argument(
    "--counterfactual-tif",
    type=str,
    required=True,
    help="Path to counterfactual prediction raster",
)

parser.add_argument(
    "--sampled-csv",
    type=str,
    default=None,
    help="Optional CSV of sampled pixels (row, col). If omitted, uses full raster.",
)

parser.add_argument(
    "--bootstrap",
    action="store_true",
    help="Compute bootstrap confidence intervals",
)

parser.add_argument(
    "--n-boot",
    type=int,
    default=1000,
)

parser.add_argument(
    "--output-csv",
    type=str,
    default="wasserstein_results.csv",
)

args = parser.parse_args()


with rasterio.open(args.actual_tif) as src:
    actual = src.read(1).astype(np.float32)

with rasterio.open(args.counterfactual_tif) as src:
    cf = src.read(1).astype(np.float32)

assert actual.shape == cf.shape, "Raster shapes do not match"

H, W = actual.shape
print("Raster shape:", H, W)


if args.sampled_csv:
    samples = pd.read_csv(args.sampled_csv)

    rows = samples["row"].values.astype(int)
    cols = samples["col"].values.astype(int)

    y_actual = actual[rows, cols]
    y_cf = cf[rows, cols]

    print("Using sampled pixels:", len(rows))

else:
    y_actual = actual.flatten()
    y_cf = cf.flatten()

    print("Using full raster")


mask = (
    ~np.isnan(y_actual)
    & ~np.isnan(y_cf)
)

y_actual = y_actual[mask]
y_cf = y_cf[mask]

print("Pixels on shared support:", len(y_actual))

if len(y_actual) == 0:
    raise RuntimeError("No overlapping valid pixels found.")


wd = wasserstein_distance(y_actual, y_cf)

results = {
    "wasserstein": wd,
    "n_eval": len(y_actual),
}


if args.bootstrap:
    rng = np.random.default_rng(42)
    n = len(y_actual)

    stats = []
    for _ in range(args.n_boot):
        idx = rng.choice(n, n, replace=True)
        stats.append(
            wasserstein_distance(y_actual[idx], y_cf[idx])
        )

    stats = np.array(stats)

    results.update({
        "bootstrap_mean": stats.mean(),
        "ci_low": np.percentile(stats, 2.5),
        "ci_high": np.percentile(stats, 97.5),
    })


out_df = pd.DataFrame([results])
out_path = Path(args.output_csv)

out_df.to_csv(out_path, index=False)

print("\n=== Wasserstein Results ===")
print(out_df)
print(f"\nSaved to: {out_path.resolve()}")
