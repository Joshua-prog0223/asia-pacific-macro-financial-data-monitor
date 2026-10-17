
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Paths and input data
# --------------------------------------------------

project_dir = Path(__file__).resolve().parent.parent

credit_path = (
    project_dir
    / "data"
    / "processed"
    / "credit_gap_asia_pacific.csv"
)

dsr_path = (
    project_dir
    / "data"
    / "processed"
    / "dsr_asia_pacific.csv"
)

output_path = (
    project_dir
    / "data"
    / "processed"
    / "credit_gap_dsr_private_panel.csv"
)

stats_path = (
    project_dir
    / "data"
    / "processed"
    / "credit_gap_dsr_correlations.csv"
)


# --------------------------------------------------
# 2. Load and prepare the datasets
# --------------------------------------------------

credit = pd.read_csv(credit_path)
dsr = pd.read_csv(dsr_path)

print("CREDIT DATA")
print("Shape:", credit.shape)
print("Columns:", credit.columns.tolist())

print("\nDSR DATA")
print("Shape:", dsr.shape)
print("Columns:", dsr.columns.tolist())

# Convert quarter labels into quarterly periods.
credit["quarter"] = pd.PeriodIndex(
    credit["quarter"], freq="Q"
)

dsr["quarter"] = pd.PeriodIndex(
    dsr["quarter"], freq="Q"
)


# --------------------------------------------------
# 3. Select the indicators of interest
# --------------------------------------------------

# Series C = credit-to-GDP gap.
credit_gap = credit.loc[
    credit["series_code"] == "C",
    ["country_code", "quarter", "value"],
].copy()

credit_gap = credit_gap.rename(
    columns={"value": "credit_gap"}
)

# Borrower P = private non-financial sector.
dsr_private = dsr.loc[
    dsr["borrower_code"] == "P",
    ["country_code", "quarter", "value"],
].copy()

dsr_private = dsr_private.rename(
    columns={"value": "dsr"}
)

print("\nSELECTED DATA")
print("Credit-gap observations:", len(credit_gap))
print("Private-sector DSR observations:", len(dsr_private))

print(
    "Countries in credit gap:",
    sorted(credit_gap["country_code"].unique()),
)

print(
    "Countries in DSR:",
    sorted(dsr_private["country_code"].unique()),
)


# --------------------------------------------------
# 4. Check keys before merging
# --------------------------------------------------

key_columns = ["country_code", "quarter"]

credit_duplicates = credit_gap.duplicated(
    key_columns
).sum()

dsr_duplicates = dsr_private.duplicated(
    key_columns
).sum()

print("\nDUPLICATE CHECK")
print("Credit-gap duplicate keys:", credit_duplicates)
print("DSR duplicate keys:", dsr_duplicates)

if credit_duplicates > 0 or dsr_duplicates > 0:
    raise ValueError(
        "Duplicate country-quarter keys detected. "
        "Investigate them before merging."
    )


# --------------------------------------------------
# 5. Merge on country and quarter
# --------------------------------------------------

panel = pd.merge(
    credit_gap,
    dsr_private,
    on=key_columns,
    how="inner",
    validate="one_to_one",
)

panel = panel.sort_values(
    ["country_code", "quarter"]
).reset_index(drop=True)

print("\nMERGED PANEL")
print("Shape:", panel.shape)
print("Countries:", sorted(panel["country_code"].unique()))

print("\nObservations per country:")
print(
    panel.groupby("country_code")
    .size()
    .rename("observations")
)


# --------------------------------------------------
# 6. Check coverage and data quality
# --------------------------------------------------

coverage = panel.groupby("country_code").agg(
    first_quarter=("quarter", "min"),
    last_quarter=("quarter", "max"),
    observations=("quarter", "count"),
    missing_credit_gap=("credit_gap", lambda x: x.isna().sum()),
    missing_dsr=("dsr", lambda x: x.isna().sum()),
)

print("\nCOUNTRY COVERAGE")
print(coverage)

print("\nMissing values in merged panel:")
print(panel[["credit_gap", "dsr"]].isna().sum())

if panel[["credit_gap", "dsr"]].isna().any().any():
    raise ValueError(
        "Missing indicator values detected in the merged panel."
    )

print("\nCommon quarterly coverage by country:")
for country, group in panel.groupby("country_code"):
    print(
        f"{country}: "
        f"{group['quarter'].min()} to "
        f"{group['quarter'].max()}"
    )


# --------------------------------------------------
# 7. Calculate country-specific correlations
# --------------------------------------------------

results = []

for country, group in panel.groupby("country_code"):
    group = group.sort_values("quarter").copy()

    # Quarter-to-quarter changes, in indicator units.
    group["credit_gap_change"] = group["credit_gap"].diff()
    group["dsr_change"] = group["dsr"].diff()

    level_data = group[["credit_gap", "dsr"]].dropna()

    change_data = group[
        ["credit_gap_change", "dsr_change"]
    ].dropna()

    results.append(
        {
            "country_code": country,
            "observations_levels": len(level_data),
            "pearson_levels": level_data["credit_gap"].corr(
                level_data["dsr"], method="pearson"
            ),
            "spearman_levels": level_data["credit_gap"].corr(
                level_data["dsr"], method="spearman"
            ),
            "observations_changes": len(change_data),
            "pearson_changes": change_data[
                "credit_gap_change"
            ].corr(
                change_data["dsr_change"],
                method="pearson",
            ),
            "spearman_changes": change_data[
                "credit_gap_change"
            ].corr(
                change_data["dsr_change"],
                method="spearman",
            ),
        }
    )

stats = pd.DataFrame(results)


# --------------------------------------------------
# 8. Save outputs
# --------------------------------------------------

# Save the merged panel with quarterly periods as strings.
panel_to_save = panel.copy()
panel_to_save["quarter"] = panel_to_save["quarter"].astype(str)
panel_to_save.to_csv(output_path, index=False)

stats.to_csv(stats_path, index=False)

print("\nCORRELATION RESULTS")
print(stats.round(3).to_string(index=False))

print("\nSaved merged panel to:")
print(output_path)

print("\nSaved correlation results to:")
print(stats_path)

print("\nAnalysis completed successfully.")
