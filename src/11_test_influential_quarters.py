
from pathlib import Path

import pandas as pd


project_dir = Path(__file__).resolve().parent.parent

input_path = (
    project_dir
    / "data"
    / "processed"
    / "credit_gap_dsr_private_panel.csv"
)

df = pd.read_csv(input_path)
df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")

for country in ["AU", "JP", "KR"]:
    group = (
        df[df["country_code"] == country]
        .sort_values("quarter")
        .copy()
    )

    group["gap_change"] = group["credit_gap"].diff()
    group["dsr_change"] = group["dsr"].diff()

    changes = group[
        ["quarter", "gap_change", "dsr_change"]
    ].dropna().reset_index(drop=True)

    baseline = changes["gap_change"].corr(
        changes["dsr_change"]
    )

    # Leave-one-out: recompute the correlation after removing
    # each quarter's paired change, one observation at a time.
    loo_results = []

    for i in range(len(changes)):
        reduced = changes.drop(index=i)

        correlation = reduced["gap_change"].corr(
            reduced["dsr_change"]
        )

        loo_results.append(
            {
                "quarter": str(changes.loc[i, "quarter"]),
                "correlation_without_quarter": correlation,
                "change_vs_baseline": correlation - baseline,
            }
        )

    loo = pd.DataFrame(loo_results)

    print(f"\n{'=' * 55}")
    print(f"{country}: baseline Pearson correlation = {baseline:.3f}")

    print("\nQuarters with greatest influence on the correlation:")
    print(
        loo.reindex(
            loo["change_vs_baseline"].abs()
            .sort_values(ascending=False)
            .index
        )
        .head(8)
        .round(3)
        .to_string(index=False)
    )

    target_quarters = {
        "AU": [],
        "JP": ["2020Q2"],
        "KR": ["1999Q2", "1999Q3", "1999Q4"],
    }[country]

    for quarter in target_quarters:
        reduced = changes[changes["quarter"].astype(str) != quarter]

        if len(reduced) < len(changes):
            new_corr = reduced["gap_change"].corr(
                reduced["dsr_change"]
            )
            print(
                f"\nExcluding {quarter}: "
                f"Pearson r = {new_corr:.3f} "
                f"(baseline {baseline:.3f})"
            )
