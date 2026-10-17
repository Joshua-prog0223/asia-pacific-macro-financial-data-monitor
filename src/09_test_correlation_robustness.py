
from pathlib import Path

import pandas as pd


project_dir = Path(__file__).resolve().parent.parent

input_path = (
    project_dir
    / "data"
    / "processed"
    / "credit_gap_dsr_private_panel.csv"
)

output_path = (
    project_dir
    / "data"
    / "processed"
    / "credit_gap_dsr_robustness.csv"
)

df = pd.read_csv(input_path)
df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")

results = []

for country, group in df.groupby("country_code"):
    group = group.sort_values("quarter").copy()

    group["credit_gap_change"] = group["credit_gap"].diff()
    group["dsr_change"] = group["dsr"].diff()

    changes = group[
        ["quarter", "credit_gap_change", "dsr_change"]
    ].dropna().copy()

    # Baseline correlation using all quarterly changes.
    baseline_pearson = changes["credit_gap_change"].corr(
        changes["dsr_change"], method="pearson"
    )
    baseline_spearman = changes["credit_gap_change"].corr(
        changes["dsr_change"], method="spearman"
    )

    # Flag quarters with unusually large changes in either indicator.
    # The 95th percentile is a diagnostic threshold, not a deletion rule.
    credit_threshold = changes["credit_gap_change"].abs().quantile(0.95)
    dsr_threshold = changes["dsr_change"].abs().quantile(0.95)

    changes["extreme_change"] = (
        (changes["credit_gap_change"].abs() > credit_threshold)
        | (changes["dsr_change"].abs() > dsr_threshold)
    )

    flagged = changes.loc[changes["extreme_change"]]
    remaining = changes.loc[~changes["extreme_change"]]

    results.append(
        {
            "country_code": country,
            "n_changes": len(changes),
            "baseline_pearson": baseline_pearson,
            "baseline_spearman": baseline_spearman,
            "extreme_quarters_flagged": len(flagged),
            "pearson_without_flagged": remaining[
                "credit_gap_change"
            ].corr(remaining["dsr_change"], method="pearson"),
            "spearman_without_flagged": remaining[
                "credit_gap_change"
            ].corr(remaining["dsr_change"], method="spearman"),
        }
    )

    print(f"\n--- {country} ---")
    print("Largest flagged changes:")
    print(
        flagged[
            ["quarter", "credit_gap_change", "dsr_change"]
        ].to_string(index=False)
    )

summary = pd.DataFrame(results)
summary.to_csv(output_path, index=False)

print("\nROBUSTNESS COMPARISON")
print(summary.round(3).to_string(index=False))

print("\nSaved results to:")
print(output_path)
