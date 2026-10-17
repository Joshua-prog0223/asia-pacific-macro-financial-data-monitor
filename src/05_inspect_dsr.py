
from pathlib import Path
import pandas as pd

# 1. Locate the project
project_dir = Path(__file__).resolve().parent.parent

input_path = (
    project_dir
    / "data"
    / "raw"
    / "WS_DSR_csv_flat (1).csv"
)

output_dir = project_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Load the raw data
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# 3. Extract country and borrower-sector codes
country_column = "BORROWERS_CTY:Borrowers' country"
borrower_column = "DSR_BORROWERS:Borrowers"
period_column = "TIME_PERIOD:Time period or range"
value_column = "OBS_VALUE:Observation Value"

df["country_code"] = (
    df[country_column].str.split(":").str[0].str.strip()
)

df["borrower_code"] = (
    df[borrower_column].str.split(":").str[0].str.strip()
)

df["quarter"] = pd.PeriodIndex(
    df[period_column].str.replace("-", "", regex=False),
    freq="Q",
)

df["value"] = pd.to_numeric(
    df[value_column],
    errors="coerce",
)

# Inspect official series definitions and measurement metadata
metadata_columns = [
    country_column,
    borrower_column,
    "UNIT_MEASURE:Unit of measure",
    "UNIT_MULT:Unit Multiplier",
    "FREQ:Frequency",
]

available_metadata = [
    col for col in metadata_columns if col in df.columns
]

print("\nOfficial DSR metadata:")
print(
    df.loc[
        df[country_column].str.split(":").str[0].str.strip().isin(
            ["AU", "JP", "KR"]
        ),
        available_metadata,
    ]
    .drop_duplicates()
    .to_string(index=False)
)

# 4. Select our target economies
target_countries = ["AU", "JP", "KR", "SG"]

selected = df.loc[
    df["country_code"].isin(target_countries),
    [
        "country_code",
        "borrower_code",
        "quarter",
        "value",
    ],
].copy()

# 5. Audit coverage and missing values
print("\nSelected dataset shape:", selected.shape)

coverage = (
    selected.groupby(["country_code", "borrower_code"])
    .agg(
        observations=("value", "size"),
        first_quarter=("quarter", "min"),
        last_quarter=("quarter", "max"),
        missing_values=("value", lambda x: x.isna().sum()),
    )
)

print("\nCoverage by country and sector:")
print(coverage.to_string())

# 6. Check duplicate country-sector-quarter keys
duplicate_count = selected.duplicated(
    ["country_code", "borrower_code", "quarter"]
).sum()

print("\nDuplicate keys:", duplicate_count)
print("Missing numeric values:", selected["value"].isna().sum())

# 7. Save the selected data
output_path = output_dir / "dsr_asia_pacific.csv"
selected.to_csv(output_path, index=False)

print("\nSelected data saved to:", output_path)
