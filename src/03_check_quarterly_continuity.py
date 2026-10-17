
from pathlib import Path

import pandas as pd


project_dir = Path(__file__).resolve().parent.parent

input_path = (
    project_dir
    / "data"
    / "processed"
    / "credit_gap_asia_pacific.csv"
)

df = pd.read_csv(input_path)
df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")


# 1. Check missing quarters within each country-series
print("QUARTERLY CONTINUITY BY COUNTRY AND SERIES")

continuity_results = []

for (country, series), group in df.groupby(
    ["country_code", "series_code"]
):
    observed = pd.PeriodIndex(
        group["quarter"].sort_values().unique(),
        freq="Q",
    )

    expected = pd.period_range(
        start=observed.min(),
        end=observed.max(),
        freq="Q",
    )

    missing = expected.difference(observed)

    continuity_results.append(
        {
            "country": country,
            "series": series,
            "expected_quarters": len(expected),
            "observed_quarters": len(observed),
            "missing_quarters": len(missing),
            "missing_examples": ", ".join(
                str(q) for q in missing[:5]
            ),
        }
    )

continuity = pd.DataFrame(continuity_results)

print(continuity.to_string(index=False))


# 2. Build the common comparison sample for series C
common_start = pd.Period("1980Q4", freq="Q")
common_end = pd.Period("2026Q1", freq="Q")

countries = ["AU", "JP", "KR", "SG"]

common = df.loc[
    (df["series_code"] == "C")
    & (df["country_code"].isin(countries))
    & (df["quarter"] >= common_start)
    & (df["quarter"] <= common_end)
].copy()


# 3. Check the expected balanced panel
expected_quarters = pd.period_range(
    start=common_start,
    end=common_end,
    freq="Q",
)

expected_rows = len(countries) * len(expected_quarters)

print("\nCOMMON SAMPLE AUDIT")
print("Expected quarters per country:", len(expected_quarters))
print("Expected total observations:", expected_rows)
print("Actual observations:", len(common))

counts = common.groupby("country_code")["quarter"].nunique()

print("\nQuarters per country:")
print(counts)

expected_index = pd.MultiIndex.from_product(
    [countries, expected_quarters],
    names=["country_code", "quarter"],
)

actual_index = pd.MultiIndex.from_frame(
    common[["country_code", "quarter"]].drop_duplicates()
)

missing_keys = expected_index.difference(actual_index)

print("\nMissing country-quarter combinations:", len(missing_keys))

if len(missing_keys) > 0:
    print(missing_keys[:20].tolist())


# 4. Save the common comparison sample
output_path = (
    project_dir
    / "data"
    / "processed"
    / "credit_gap_common_sample.csv"
)

common.sort_values(
    ["country_code", "quarter"]
).to_csv(output_path, index=False)

print("\nCommon sample saved to:", output_path)
