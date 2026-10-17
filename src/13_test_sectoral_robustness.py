
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "credit_gap_dsr_sector_panel.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"

ROBUSTNESS_FILE = OUTPUT_DIR / "credit_gap_dsr_sector_robustness.csv"
INFLUENCE_FILE = OUTPUT_DIR / "credit_gap_dsr_sector_influence.csv"


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
        f"Missing required columns: {sorted(missing_columns)}"
    )

df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")

df = df.sort_values(
    ["country_code", "sector_code", "quarter"]
).copy()

if df.duplicated(
    ["country_code", "sector_code", "quarter"]
).any():
    raise ValueError("Duplicate country-sector-quarter keys found.")

if df[["credit_gap", "dsr"]].isna().any().any():
    raise ValueError("Missing values found in analysis variables.")

print("Sectoral robustness audit")
print("=========================")
print(f"Rows loaded: {len(df)}")


# --------------------------------------------------
# 3. Compute changes within each country-sector group
# --------------------------------------------------

group_columns = ["country_code", "sector_code"]

df["credit_gap_change"] = (
    df.groupby(group_columns)["credit_gap"].diff()
)

df["dsr_change"] = (
    df.groupby(group_columns)["dsr"].diff()
)


# --------------------------------------------------
# 4. Robustness analysis by country and sector
# --------------------------------------------------

robustness_results = []
influence_results = []

for (country, sector), group in df.groupby(group_columns):

    changes = (
        group[
            [
                "quarter",
                "credit_gap_change",
                "dsr_change",
            ]
        ]
        .dropna()
        .reset_index(drop=True)
    )

    x = changes["credit_gap_change"]
    y = changes["dsr_change"]

    if len(changes) < 10:
        raise ValueError(
            f"Too few observations for {country}-{sector}."
        )

    baseline_pearson = x.corr(y, method="pearson")
    baseline_spearman = x.corr(y, method="spearman")

    # Flag statistical extremes separately for each variable.
    x_threshold = x.abs().quantile(0.95)
    y_threshold = y.abs().quantile(0.95)

    flagged = (
        (x.abs() > x_threshold)
        | (y.abs() > y_threshold)
    )

    filtered = changes.loc[~flagged]

    filtered_pearson = filtered[
        "credit_gap_change"
    ].corr(
        filtered["dsr_change"],
        method="pearson",
    )

    filtered_spearman = filtered[
        "credit_gap_change"
    ].corr(
        filtered["dsr_change"],
        method="spearman",
    )

    robustness_results.append(
        {
            "country_code": country,
            "sector_code": sector,
            "observations": len(changes),
            "baseline_pearson": baseline_pearson,
            "baseline_spearman": baseline_spearman,
            "flagged_observations": int(flagged.sum()),
            "pearson_without_flagged": filtered_pearson,
            "spearman_without_flagged": filtered_spearman,
            "pearson_change": (
                filtered_pearson - baseline_pearson
            ),
            "spearman_change": (
                filtered_spearman - baseline_spearman
            ),
        }
    )

    # Leave-one-out: remove one quarterly observation at a time.
    for i, row in changes.iterrows():

        reduced = changes.drop(index=i)

        reduced_pearson = reduced[
            "credit_gap_change"
        ].corr(
            reduced["dsr_change"],
            method="pearson",
        )

        influence_results.append(
            {
                "country_code": country,
                "sector_code": sector,
                "quarter": str(row["quarter"]),
                "baseline_pearson": baseline_pearson,
                "pearson_without_quarter": reduced_pearson,
                "correlation_change": (
                    reduced_pearson - baseline_pearson
                ),
            }
        )


# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

robustness_df = pd.DataFrame(robustness_results)

influence_df = pd.DataFrame(influence_results)

influence_df["absolute_correlation_change"] = (
    influence_df["correlation_change"].abs()
)

influence_df = influence_df.sort_values(
    ["country_code", "sector_code", "absolute_correlation_change"],
    ascending=[True, True, False],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

robustness_df.to_csv(ROBUSTNESS_FILE, index=False)
influence_df.to_csv(INFLUENCE_FILE, index=False)


# --------------------------------------------------
# 6. Report results
# --------------------------------------------------

print("\nRobustness results")
print("------------------")

print(
    robustness_df.round(3).to_string(index=False)
)

print("\nMost influential quarters by group")
print("-----------------------------------")

for (country, sector), group in influence_df.groupby(
    ["country_code", "sector_code"]
):
    print(f"\n{country} - sector {sector}")

    print(
        group.head(5)[
            [
                "quarter",
                "baseline_pearson",
                "pearson_without_quarter",
                "correlation_change",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

print("\nSaved files:")
print(ROBUSTNESS_FILE)
print(INFLUENCE_FILE)
