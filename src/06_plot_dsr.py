
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent

input_path = (
    project_dir
    / "data"
    / "processed"
    / "dsr_asia_pacific.csv"
)

output_dir = project_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)
df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
df["date"] = df["quarter"].dt.to_timestamp(how="end")

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

df["country"] = df["country_code"].map(country_names)
df["sector"] = df["borrower_code"].map(sector_names)

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(13, 12),
    sharex=True,
)

for ax, (sector_code, sector_name) in zip(
    axes,
    sector_names.items(),
):
    sector_df = df[df["borrower_code"] == sector_code]

    for country_code, group in sector_df.groupby("country_code"):
        group = group.sort_values("quarter")

        ax.plot(
            group["date"],
            group["value"],
            label=country_names[country_code],
            linewidth=1.5,
        )

    ax.set_title(sector_name)
    ax.set_ylabel("DSR (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Quarter")

fig.suptitle(
    "Debt Service Ratios in Selected Asia-Pacific Economies",
    fontsize=15,
)
fig.tight_layout()

output_path = output_dir / "dsr_three_countries_three_sectors.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print("Graph saved to:", output_path)

plt.show()
