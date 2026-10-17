
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "credit_gap_dsr_sector_panel.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"

WINDOW = 20  # Exactly 20 quarterly changes per window

OUTPUT_FILE = (
    OUTPUT_DIR / "credit_gap_dsr_rolling_correlations.csv"
)

FIGURE_FILE = (
    FIGURES_DIR / "rolling_correlations_credit_gap_dsr.png"
)


# --------------------------------------------------
# 2. Load and validate the panel
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "country_code",
    "quarter",
    "sector_code",
    "credit_gap",
    "dsr",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns: {sorted(missing_columns)}"
    )

df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")

df = df.sort_values(
    ["country_code", "sector_code", "quarter"]
).copy()

keys = ["country_code", "sector_code", "quarter"]

if df.duplicated(keys).any():
    raise ValueError("Duplicate country-sector-quarter keys.")

if df[["credit_gap", "dsr"]].isna().any().any():
    raise ValueError("Missing values in credit gap or DSR.")

group_columns = ["country_code", "sector_code"]

# Calculate changes before constructing rolling windows.
df["credit_gap_change"] = (
    df.groupby(group_columns)["credit_gap"].diff()
)

df["dsr_change"] = (
    df.groupby(group_columns)["dsr"].diff()
)


# --------------------------------------------------
# 3. Calculate rolling correlations
# --------------------------------------------------

results = []

for (country, sector), group in df.groupby(group_columns):

    group = (
        group.sort_values("quarter")
        .reset_index(drop=True)
    )

    # Confirm consecutive quarterly observations.
    quarter_steps = (
        group["quarter"].astype("int64").diff().dropna()
    )

    if not (quarter_steps == 1).all():
        raise ValueError(
            f"Non-consecutive quarters in {country}-{sector}."
        )

    # Each window contains exactly 20 consecutive changes.
    changes = group[
        [
            "quarter",
            "credit_gap_change",
            "dsr_change",
        ]
    ].dropna().reset_index(drop=True)

    if len(changes) < WINDOW:
        raise ValueError(
            f"Insufficient changes for {country}-{sector}."
        )

    for end in range(WINDOW, len(changes) + 1):

        window_data = changes.iloc[end - WINDOW:end]

        x = window_data["credit_gap_change"]
        y = window_data["dsr_change"]

        results.append(
            {
                "country_code": country,
                "sector_code": sector,
                "window_start": str(
                    window_data.iloc[0]["quarter"]
                ),
                "window_end": str(
                    window_data.iloc[-1]["quarter"]
                ),
                "observations": len(window_data),
                "pearson_changes": x.corr(
                    y, method="pearson"
                ),
                "spearman_changes": x.corr(
                    y, method="spearman"
                ),
            }
        )

rolling_df = pd.DataFrame(results)

if rolling_df.empty:
    raise ValueError("No rolling correlations were calculated.")

if not (rolling_df["observations"] == WINDOW).all():
    raise ValueError("A window does not contain exactly 20 changes.")


# --------------------------------------------------
# 4. Summarize results
# --------------------------------------------------

summary = (
    rolling_df.groupby(["country_code", "sector_code"])
    .agg(
        windows=("pearson_changes", "count"),
        pearson_mean=("pearson_changes", "mean"),
        pearson_median=("pearson_changes", "median"),
        pearson_min=("pearson_changes", "min"),
        pearson_max=("pearson_changes", "max"),
        positive_window_share=(
            "pearson_changes",
            lambda values: (values > 0).mean(),
        ),
        spearman_median=("spearman_changes", "median"),
    )
    .reset_index()
)


# --------------------------------------------------
# 5. Save the results
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

rolling_df.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# 6. Plot rolling Pearson correlations
# --------------------------------------------------

country_names = {
    "AU": "Australia",
    "JP": "Japan",
    "KR": "Korea",
}

sector_names = {
    "H": "Households and NPISHs",
    "N": "Non-financial corporations",
}

fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(14, 12),
    sharey=True,
)

for row_index, country in enumerate(["AU", "JP", "KR"]):
    for col_index, sector in enumerate(["H", "N"]):

        ax = axes[row_index, col_index]

        subset = rolling_df[
            (rolling_df["country_code"] == country)
            & (rolling_df["sector_code"] == sector)
        ].copy()

        subset["window_end"] = pd.PeriodIndex(
            subset["window_end"], freq="Q"
        ).to_timestamp(how="end")

        ax.plot(
            subset["window_end"],
            subset["pearson_changes"],
        )

        ax.axhline(0, linewidth=1)
        ax.set_title(
            f"{country_names[country]} — "
            f"{sector_names[sector]}"
        )
        ax.set_ylabel("Pearson correlation")
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)

fig.suptitle(
    "Rolling correlations: credit gap and DSR changes\n"
    "20 consecutive quarterly changes",
    fontsize=14,
)

fig.tight_layout()
fig.savefig(FIGURE_FILE, dpi=200, bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------
# 7. Report results
# --------------------------------------------------

print("Corrected rolling correlation analysis")
print("======================================")
print(f"Window length: {WINDOW} quarterly changes")
print(f"Total windows: {len(rolling_df)}")
print(
    "Observations per window:",
    sorted(rolling_df["observations"].unique()),
)

print("\nSummary by country and sector")
print("-----------------------------")
print(summary.round(3).to_string(index=False))

print("\nSaved files:")
print(OUTPUT_FILE)
print(FIGURE_FILE)
