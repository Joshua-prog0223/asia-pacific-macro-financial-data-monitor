from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "credit_gap_dsr_sector_difference.csv"
OUTPUT = ROOT / "figures" / "sectoral_correlation_comparison.png"

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT}")

df = pd.read_csv(INPUT)

required = {
    "country_code",
    "block_length",
    "household_pearson",
    "corporate_pearson",
    "difference_corporate_minus_household",
    "difference_ci_lower",
    "difference_ci_upper",
}
missing = required.difference(df.columns)

if missing:
    raise ValueError(
        f"Missing columns in {INPUT.name}: {sorted(missing)}\n"
        f"Available columns: {list(df.columns)}"
    )

# Use the four-quarter block bootstrap results for the main figure.
plot_df = df.loc[df["block_length"] == 4].copy()

country_order = ["AU", "JP", "KR"]
plot_df["country_code"] = pd.Categorical(
    plot_df["country_code"],
    categories=country_order,
    ordered=True,
)
plot_df = plot_df.sort_values("country_code")

if plot_df["country_code"].nunique() != 3:
    raise ValueError(
        "Expected results for AU, JP and KR at block_length == 4."
    )

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

panels = [
    (
        "household_pearson",
        "A. Household DSR",
        "Correlation with credit-gap changes",
    ),
    (
        "corporate_pearson",
        "B. Corporate DSR",
        "Correlation with credit-gap changes",
    ),
]

for ax, (column, title, ylabel) in zip(axes[:2], panels):
    values = plot_df[column]
    ax.bar(plot_df["country_code"].astype(str), values)
    ax.axhline(0, linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.3, 1.0)
    ax.grid(axis="y", alpha=0.25)

ax = axes[2]
differences = plot_df["difference_corporate_minus_household"]
lower = plot_df["difference_ci_lower"]
upper = plot_df["difference_ci_upper"]

yerr = [
    (differences - lower).clip(lower=0),
    (upper - differences).clip(lower=0),
]

ax.errorbar(
    plot_df["country_code"].astype(str),
    differences,
    yerr=yerr,
    fmt="o",
    capsize=5,
)
ax.axhline(0, linewidth=0.8)
ax.set_title("C. Difference between correlations")
ax.set_ylabel("Corporate minus household correlation")
ax.grid(axis="y", alpha=0.25)

fig.suptitle(
    "Credit-gap changes and debt-service ratios, 1999Q2-2026Q1",
    fontsize=13,
)
fig.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Figure saved to: {OUTPUT}")
print("\nData used in the figure:")
print(
    plot_df[
        [
            "country_code",
            "household_pearson",
            "corporate_pearson",
            "difference_corporate_minus_household",
            "difference_ci_lower",
            "difference_ci_upper",
        ]
    ].to_string(index=False)
)