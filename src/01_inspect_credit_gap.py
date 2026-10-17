
from pathlib import Path

import pandas as pd


# 1. Define the input and output paths
project_dir = Path(__file__).resolve().parent.parent

input_path = (
    project_dir
    / "data"
    / "raw"
    / "WS_CREDIT_GAP_csv_flat (1).csv"
)

output_dir = project_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)


# 2. Load the BIS dataset
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())


# 3. Select the four target economies
country_column = "BORROWERS_CTY:Borrowers' country"
series_column = "CG_DTYPE:Credit gap data type"
period_column = "TIME_PERIOD:Time period or range"
value_column = "OBS_VALUE:Observation Value"

target_codes = ["AU", "JP", "KR", "SG"]

df["country_code"] = (
    df[country_column].str.split(":").str[0].str.strip()
)

df["series_code"] = (
    df[series_column].str.split(":").str[0].str.strip()
)

df["series_name"] = (
    df[series_column].str.split(": ", n=1).str[-1]
)

df["quarter"] = df[period_column].str.replace(
    "-", "", regex=False
)

df["quarter"] = pd.PeriodIndex(
    df["quarter"], freq="Q"
)

df["value"] = pd.to_numeric(
    df[value_column], errors="coerce"
)

selected = df.loc[
    df["country_code"].isin(target_codes),
    [
        "country_code",
        "series_code",
        "series_name",
        "quarter",
        "value",
    ],
].copy()

# Inspect series definitions and measurement metadata
metadata_columns = [
    country_column,
    series_column,
    "UNIT_MEASURE:Unit of measure",
    "UNIT_MULT:Unit Multiplier",
    "DECIMALS:Decimals",
]

metadata = df.loc[
    df["country_code"].isin(target_codes),
    metadata_columns,
].drop_duplicates()

print("\nSeries definitions and measurement metadata:")
print(metadata.to_string(index=False))


# 4. Run basic data-quality checks
print("\nSelected dataset shape:", selected.shape)

print("\nObservations by country and series:")
print(
    selected.groupby(
        ["country_code", "series_code"]
    ).agg(
        observations=("value", "size"),
        first_quarter=("quarter", "min"),
        last_quarter=("quarter", "max"),
        missing_values=("value", lambda x: x.isna().sum()),
    )
)

duplicate_count = selected.duplicated(
    ["country_code", "series_code", "quarter"]
).sum()

print("\nDuplicate country-series-quarter keys:", duplicate_count)
print("Missing numeric values:", selected["value"].isna().sum())


# 5. Save the selected dataset
output_path = output_dir / "credit_gap_asia_pacific.csv"

selected.to_csv(output_path, index=False)

print("\nProcessed dataset saved to:", output_path)