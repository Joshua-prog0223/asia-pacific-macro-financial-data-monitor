
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# 1. Define project paths
project_dir = Path(__file__).resolve().parent.parent

input_path = (
    project_dir
    / "data"
    / "processed"
    / "credit_gap_common_sample.csv"
)

output_dir = project_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)


# 2. Load and prepare the common sample
df = pd.read_csv(input_path)

df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
df["date"] = df["quarter"].dt.to_timestamp(how="end")

country_names = {
    "AU": "Australia",
    "JP": "Japan",
    "KR": "Korea",
    "SG": "Singapore",
}

df["country"] = df["country_code"].map(country_names)

df = df.sort_values(["country_code", "quarter"])


# 3. Plot the credit-to-GDP gap
fig, ax = plt.subplots(figsize=(13, 7))

for country_code, group in df.groupby("country_code"):
    country_name = country_names[country_code]

    ax.plot(
        group["date"],
        group["value"],
        label=country_name,
        linewidth=1.5,
    )

# Zero means the observed credit-to-GDP ratio equals its trend
ax.axhline(y=0, linestyle="--", linewidth=1)

ax.set_title(
    "Credit-to-GDP Gaps in Selected Asia-Pacific Economies"
)
ax.set_xlabel("Quarter")
ax.set_ylabel("Percentage points of GDP")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()

output_path = output_dir / "credit_gap_four_countries.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print("Graph saved to:", output_path)

plt.show()
