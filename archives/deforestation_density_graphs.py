import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

csv_path = Path("./data/export_three_maps_10.csv")  # <-- change if needed
df = pd.read_csv(csv_path)


feature_col = "deforestation_density_2005_2010"
period_cols = ["deforestation_2005_2010", "deforestation_2010_2015"]

long_df = df.melt(
    id_vars=[feature_col],
    value_vars=period_cols,
    var_name="period",
    value_name="deforestation"
)

period_name_map = {
    "deforestation_2005_2010": "2005–2010",
    "deforestation_2010_2015": "2010–2015",
}
long_df["period"] = long_df["period"].map(period_name_map)

long_df["deforestation"] = long_df["deforestation"].astype(int).astype("category")

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.violinplot(
    data=long_df,
    x="period",
    y=feature_col,
    hue="deforestation",
    split=True,
    inner="quart",
)
plt.title("Distribution of deforestation density by class and period")
plt.ylabel("Deforestation density (radius 5 px)")
plt.xlabel("Time period")
plt.legend(title="Deforestation (0 = no, 1 = yes)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=long_df,
    x="period",
    y=feature_col,
    hue="deforestation",
)
plt.title("Boxplot of deforestation density by class and period")
plt.ylabel("Deforestation density (radius 5 px)")
plt.xlabel("Time period")
plt.legend(title="Deforestation (0 = no, 1 = yes)")
plt.tight_layout()
plt.show()

g = sns.FacetGrid(
    long_df,
    col="period",
    hue="deforestation",
    sharex=True,
    sharey=True,
    height=4,
    aspect=1.3
)
g.map(sns.kdeplot, feature_col, common_norm=False, fill=True, alpha=0.5)
g.add_legend(title="Deforestation")
g.set_axis_labels("Deforestation density (radius 5 px)", "Density")
g.fig.suptitle("KDE of deforestation density by class and period", y=1.05)
plt.show()

g = sns.FacetGrid(
    long_df,
    col="period",
    hue="deforestation",
    sharex=True,
    sharey=True,
    height=4,
    aspect=1.3
)
g.map(plt.hist, feature_col, bins=30, alpha=0.5, density=True)
g.add_legend(title="Deforestation")
g.set_axis_labels("Deforestation density (radius 5 px)", "Density")
g.fig.suptitle("Histogram of deforestation density by class and period", y=1.05)
plt.show()

n_bins = 10  # adjust as you like
feature_min = long_df[feature_col].min()
feature_max = long_df[feature_col].max()

bins = np.linspace(feature_min, feature_max, n_bins + 1)
long_df["density_bin"] = pd.cut(long_df[feature_col], bins=bins, include_lowest=True)

grouped = (
    long_df
    .groupby(["period", "density_bin"])["deforestation"]
    .apply(lambda x: (x.astype(int).mean()))
    .reset_index(name="p_deforestation")
)

grouped["density_bin_str"] = grouped["density_bin"].astype(str)

heatmap_data = grouped.pivot(
    index="period",
    columns="density_bin_str",
    values="p_deforestation"
)

plt.figure(figsize=(12, 4))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    cbar_kws={"label": "P(deforestation = 1)"}
)
plt.title("Probability of deforestation vs. density bin and period")
plt.xlabel("Deforestation density bin (radius 5 px)")
plt.ylabel("Period")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
