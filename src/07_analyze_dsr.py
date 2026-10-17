
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent

input_path = (
    project_dir
    / "data"
    / "processed"
    / "dsr_asia_pacific.csv"
)

output_dir = project_dir / "data" / "processed"
df = pd.read_csv(input_path)

df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")

country_names = {
    "AU": "Australia",
    "JP": "Japan",
    "KR": "Korea",
}

sector_names = {
    "H": "Households and NPISHs",
    "N": "Non-financial corporations",
    "P": "Private non-financial sector",
}

# 1. Descriptive statistics by country and sector
summary = (
    df.groupby(["country_code", "borrower_code"])["value"]
    .agg(["count", "mean", "std", "min", "max"])
    .reset_index()
)

summary["country"] = summary["country_code"].map(country_names)
summary["sector"] = summary["borrower_code"].map(sector_names)

summary = summary[
    ["country", "sector", "count", "mean", "std", "min", "max"]
]

summary.to_csv(
    output_dir / "dsr_descriptive_statistics.csv",
    index=False,
)

print("\nDESCRIPTIVE STATISTICS")
print(summary.round(2).to_string(index=False))

# 2. Identify the quarters of the minimum and maximum
print("\nMINIMUM AND MAXIMUM QUARTERS")

for (country, sector), group in df.groupby(
    ["country_code", "borrower_code"]
):
    min_row = group.loc[group["value"].idxmin()]
    max_row = group.loc[group["value"].idxmax()]

    print(
        f"{country_names[country]} | {sector_names[sector]}: "
        f"min={min_row['value']:.2f} ({min_row['quarter']}), "
        f"max={max_row['value']:.2f} ({max_row['quarter']})"
    )

# 3. Correlations between quarterly changes across countries
print("\nCORRELATIONS OF QUARTERLY CHANGES")

for sector_code, sector_name in sector_names.items():
    sector_df = df[df["borrower_code"] == sector_code]

    levels = sector_df.pivot(
        index="quarter",
        columns="country_code",
        values="value",
    ).sort_index()

    changes = levels.diff()

    print(f"\n{sector_name}")
    print(changes.corr().round(2).to_string())

print("\nSaved: dsr_descriptive_statistics.csv")
