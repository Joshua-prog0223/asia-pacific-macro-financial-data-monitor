
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --------------------------------------------------
# 1. Paths and data
# --------------------------------------------------

project_dir = Path(__file__).resolve().parent.parent

input_path = (
    project_dir
    / "data"
    / "processed"
    / "credit_gap_dsr_private_panel.csv"
)

output_path = (
    project_dir
    / "figures"
    / "credit_gap_dsr_diagnostics.png"
)

df = pd.read_csv(input_path)
df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
df["date"] = df["quarter"].dt.to_timestamp()

countries = ["AU", "JP", "KR"]

country_names = {
    "AU": "Australia",
    "JP": "Japan",
    "KR": "Korea",
}


# --------------------------------------------------
# 2. Validate the input
# --------------------------------------------------

required_columns = {
    "country_code",
    "quarter",
    "credit_gap",
    "dsr",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

if df.duplicated(["country_code", "quarter"]).any():
    raise ValueError("Duplicate country-quarter observations found.")

if df[["credit_gap", "dsr"]].isna().any().any():
    raise ValueError("Missing values found in the indicators.")

if set(df["country_code"].unique()) != set(countries):
    raise ValueError(
        "Unexpected country coverage. "
        f"Found: {sorted(df['country_code'].unique())}"
    )


# --------------------------------------------------
# 3. Prepare changes and standardized series
# --------------------------------------------------

prepared = {}

for country in countries:
    group = (
        df.loc[df["country_code"] == country]
        .sort_values("quarter")
        .copy()
    )

    # Quarter-to-quarter changes.
    group["credit_gap_change"] = group["credit_gap"].diff()
    group["dsr_change"] = group["dsr"].diff()

    # Standardized levels: mean = 0, standard deviation = 1.
    group["credit_gap_z"] = (
        group["credit_gap"] - group["credit_gap"].mean()
    ) / group["credit_gap"].std(ddof=0)

    group["dsr_z"] = (
        group["dsr"] - group["dsr"].mean()
    ) / group["dsr"].std(ddof=0)

    # Standardize quarterly changes for visual comparison.
    group["credit_gap_change_z"] = (
        group["credit_gap_change"]
        - group["credit_gap_change"].mean()
    ) / group["credit_gap_change"].std(ddof=0)

    group["dsr_change_z"] = (
        group["dsr_change"]
        - group["dsr_change"].mean()
    ) / group["dsr_change"].std(ddof=0)

    prepared[country] = group


# --------------------------------------------------
# 4. Create a 3 x 3 diagnostic figure
# --------------------------------------------------

fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(17, 13),
)

fig.suptitle(
    "Credit-to-GDP Gap and Private-Sector Debt Service Ratio",
    fontsize=16,
    fontweight="bold",
    y=0.99,
)

column_titles = [
    "Standardized levels",
    "Standardized quarterly changes",
    "Scatter: quarterly changes",
]

for col, title in enumerate(column_titles):
    axes[0, col].set_title(title, fontsize=12, pad=12)


# --------------------------------------------------
# 5. Plot each country
# --------------------------------------------------

for row, country in enumerate(countries):
    group = prepared[country]

    # First column: standardized levels over time.
    ax_levels = axes[row, 0]

    ax_levels.plot(
        group["date"],
        group["credit_gap_z"],
        label="Credit gap",
        linewidth=1.7,
    )

    ax_levels.plot(
        group["date"],
        group["dsr_z"],
        label="Private-sector DSR",
        linewidth=1.7,
    )

    ax_levels.axhline(0, linewidth=0.8, linestyle="--")
    ax_levels.set_ylabel(
        f"{country_names[country]}\nStandard deviations"
    )
    ax_levels.grid(True, alpha=0.25)

    if row == 0:
        ax_levels.legend(fontsize=9, loc="upper left")

    # Second column: standardized quarterly changes.
    ax_changes = axes[row, 1]

    ax_changes.plot(
        group["date"],
        group["credit_gap_change_z"],
        label="Credit gap change",
        linewidth=1.2,
    )

    ax_changes.plot(
        group["date"],
        group["dsr_change_z"],
        label="DSR change",
        linewidth=1.2,
    )

    ax_changes.axhline(0, linewidth=0.8, linestyle="--")
    ax_changes.set_ylabel("Standard deviations")
    ax_changes.grid(True, alpha=0.25)

    if row == 0:
        ax_changes.legend(fontsize=9, loc="upper left")

    # Third column: scatter of quarterly changes.
    ax_scatter = axes[row, 2]

    changes = group[
        ["credit_gap_change", "dsr_change"]
    ].dropna()

    x = changes["credit_gap_change"].to_numpy()
    y = changes["dsr_change"].to_numpy()

    ax_scatter.scatter(
        x,
        y,
        alpha=0.65,
        s=25,
    )

    # Add a linear trend line for visual guidance only.
    if len(changes) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        ax_scatter.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=1.5,
        )

    pearson_r = changes["credit_gap_change"].corr(
        changes["dsr_change"],
        method="pearson",
    )

    spearman_r = changes["credit_gap_change"].corr(
        changes["dsr_change"],
        method="spearman",
    )

    ax_scatter.set_title(
        f"Pearson r = {pearson_r:.3f} | "
        f"Spearman r = {spearman_r:.3f}",
        fontsize=10,
    )

    ax_scatter.set_xlabel("Quarterly change in credit gap")
    ax_scatter.set_ylabel("Quarterly change in DSR")
    ax_scatter.grid(True, alpha=0.25)


# --------------------------------------------------
# 6. Format time axes and save
# --------------------------------------------------

for row in range(3):
    for col in [0, 1]:
        axes[row, col].xaxis.set_major_locator(
            mdates.YearLocator(5)
        )
        axes[row, col].xaxis.set_major_formatter(
            mdates.DateFormatter("%Y")
        )
        axes[row, col].tick_params(
            axis="x",
            labelrotation=35,
        )

fig.text(
    0.5,
    0.015,
    (
        "Note: standardized series are used for visual comparison. "
        "Scatter plots show quarterly changes. "
        "Correlations are descriptive, not causal."
    ),
    ha="center",
    fontsize=9,
)

fig.tight_layout(rect=[0, 0.04, 1, 0.96])

output_path.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print("\nFigure saved to:")
print(output_path)
