
from pathlib import Path

import pandas as pd


# 1. Load the processed dataset
project_dir = Path(__file__).resolve().parent.parent

input_path = (
    project_dir
    / "data"
    / "processed"
    / "credit_gap_asia_pacific.csv"
)

df = pd.read_csv(input_path)

df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")

print("Loaded dataset shape:", df.shape)


# 2. Check the observation keys
key_columns = ["country_code", "series_code", "quarter"]

duplicate_count = df.duplicated(key_columns).sum()

print("\nDuplicate keys:", duplicate_count)


# 3. Inspect the common period for series C
gap_data = df.loc[df["series_code"] == "C"].copy()

coverage = gap_data.groupby("country_code").agg(
    first_quarter=("quarter", "min"),
    last_quarter=("quarter", "max"),
    observations=("value", "count"),
)

print("\nCredit-to-GDP gap coverage:")
print(coverage)


common_start = coverage["first_quarter"].max()
common_end = coverage["last_quarter"].min()

print("\nCommon comparison period:")
print("Start:", common_start)
print("End:", common_end)


# 4. Reshape the dataset: one row per country and quarter
wide = df.pivot(
    index=["country_code", "quarter"],
    columns="series_code",
    values="value",
)

print("\nAvailable series columns:", wide.columns.tolist())

required_series = {"A", "B", "C"}

if required_series.issubset(wide.columns):
    check = wide.dropna(subset=["A", "B", "C"]).copy()

    check["calculated_gap"] = check["A"] - check["B"]
    check["difference"] = check["C"] - check["calculated_gap"]
    check["absolute_difference"] = check["difference"].abs()

    print("\nArithmetic consistency: C compared with A - B")
    print("Complete triplets:", len(check))
    print(
        "Exact matches:",
        (check["absolute_difference"] == 0).sum(),
    )
    print(
        "Maximum absolute difference:",
        check["absolute_difference"].max(),
    )
    print(
        "Mean absolute difference:",
        check["absolute_difference"].mean(),
    )

    print("\nLargest discrepancies:")
    print(
        check.sort_values(
            "absolute_difference",
            ascending=False,
        )[
            ["A", "B", "C", "calculated_gap", "difference"]
        ].head(10)
    )
else:
    print(
        "\nWARNING: At least one required series "
        "is missing from the dataset."
    )
