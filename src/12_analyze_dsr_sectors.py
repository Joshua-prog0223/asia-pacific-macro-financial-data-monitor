
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CREDIT_FILE = ROOT / "data" / "processed" / "credit_gap_asia_pacific.csv"
DSR_FILE = ROOT / "data" / "processed" / "dsr_asia_pacific.csv"

OUTPUT_PANEL = (
    ROOT / "data" / "processed" / "credit_gap_dsr_sector_panel.csv"
)
OUTPUT_RESULTS = (
    ROOT / "data" / "processed" / "credit_gap_dsr_sector_correlations.csv"
)

SECTORS = {
    "H": "Households and NPISHs",
    "N": "Non-financial corporations",
}


# --------------------------------------------------
# 2. Load and select the relevant series
# --------------------------------------------------

def load_data():
    credit = pd.read_csv(CREDIT_FILE)
    dsr = pd.read_csv(DSR_FILE)

    required_credit = {
        "country_code", "series_code", "quarter", "value"
    }
    required_dsr = {
        "country_code", "borrower_code", "quarter", "value"
    }

    if not required_credit.issubset(credit.columns):
        raise ValueError(
            f"Unexpected credit columns: {credit.columns.tolist()}"
        )

    if not required_dsr.issubset(dsr.columns):
        raise ValueError(
            f"Unexpected DSR columns: {dsr.columns.tolist()}"
        )

    # Credit gap = series C
    credit = credit.loc[
        credit["series_code"] == "C",
        ["country_code", "quarter", "value"],
    ].copy()

    credit = credit.rename(columns={"value": "credit_gap"})

    # H = households and NPISHs; N = non-financial corporations
    dsr = dsr.loc[
        dsr["borrower_code"].isin(SECTORS),
        ["country_code", "borrower_code", "quarter", "value"],
    ].copy()

    dsr = dsr.rename(columns={"value": "dsr"})
    dsr = dsr.rename(columns={"borrower_code": "sector_code"})

    return credit, dsr


# --------------------------------------------------
# 3. Validate keys and merge
# --------------------------------------------------

def build_panel(credit, dsr):
    credit_key = ["country_code", "quarter"]
    dsr_key = ["country_code", "sector_code", "quarter"]

    if credit.duplicated(credit_key).any():
        raise ValueError("Duplicate country-quarter keys in credit data.")

    if dsr.duplicated(dsr_key).any():
        raise ValueError("Duplicate country-sector-quarter keys in DSR data.")

    if credit["credit_gap"].isna().any():
        raise ValueError("Missing credit-gap values.")

    if dsr["dsr"].isna().any():
        raise ValueError("Missing DSR values.")

    panel = dsr.merge(
        credit,
        on=["country_code", "quarter"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = panel["_merge"] != "both"

    if unmatched.any():
        examples = panel.loc[
            unmatched,
            ["country_code", "sector_code", "quarter"],
        ].head()

        raise ValueError(
            "Some DSR observations have no matching credit gap:\n"
            f"{examples}"
        )

    panel = panel.drop(columns="_merge")
    panel["sector_name"] = panel["sector_code"].map(SECTORS)

    panel = panel.sort_values(
        ["country_code", "sector_code", "quarter"]
    ).reset_index(drop=True)

    # Compute changes separately within each country-sector pair.
    panel["credit_gap_change"] = (
        panel.groupby(
            ["country_code", "sector_code"]
        )["credit_gap"].diff()
    )

    panel["dsr_change"] = (
        panel.groupby(
            ["country_code", "sector_code"]
        )["dsr"].diff()
    )

    return panel


# --------------------------------------------------
# 4. Calculate descriptive correlations
# --------------------------------------------------

def calculate_correlations(panel):
    results = []

    group_columns = [
        "country_code",
        "sector_code",
        "sector_name",
    ]

    for (country, sector, sector_name), group in panel.groupby(
        group_columns
    ):
        # Correlations in levels
        levels = group[["credit_gap", "dsr"]].dropna()

        # Correlations in quarter-to-quarter changes
        changes = group[
            ["credit_gap_change", "dsr_change"]
        ].dropna()

        row = {
            "country_code": country,
            "sector_code": sector,
            "sector_name": sector_name,
            "observations_levels": len(levels),
            "pearson_levels": float("nan"),
            "spearman_levels": float("nan"),
            "observations_changes": len(changes),
            "pearson_changes": float("nan"),
            "spearman_changes": float("nan"),
        }

        if len(levels) >= 3:
            row["pearson_levels"] = pearsonr(
                levels["credit_gap"], levels["dsr"]
            ).statistic

            row["spearman_levels"] = spearmanr(
                levels["credit_gap"], levels["dsr"]
            ).statistic

        if len(changes) >= 3:
            row["pearson_changes"] = pearsonr(
                changes["credit_gap_change"],
                changes["dsr_change"],
            ).statistic

            row["spearman_changes"] = spearmanr(
                changes["credit_gap_change"],
                changes["dsr_change"],
            ).statistic

        results.append(row)

    return pd.DataFrame(results)


# --------------------------------------------------
# 5. Main
# --------------------------------------------------

def main():
    credit, dsr = load_data()

    print("Credit-gap observations:", len(credit))
    print("DSR sector observations:", len(dsr))

    panel = build_panel(credit, dsr)
    results = calculate_correlations(panel)

    OUTPUT_PANEL.parent.mkdir(parents=True, exist_ok=True)

    panel.to_csv(OUTPUT_PANEL, index=False)
    results.to_csv(OUTPUT_RESULTS, index=False)

    print("\nPanel validation")
    print("----------------")
    print("Rows:", len(panel))
    print("Countries:", sorted(panel["country_code"].unique()))
    print("Sectors:", sorted(panel["sector_code"].unique()))
    print("Quarter range:", panel["quarter"].min(),
          "to", panel["quarter"].max())
    print("Missing credit gap:", panel["credit_gap"].isna().sum())
    print("Missing DSR:", panel["dsr"].isna().sum())

    print("\nSectoral correlation results")
    print("----------------------------")
    print(results.round(3).to_string(index=False))

    print("\nSaved files:")
    print(OUTPUT_PANEL)
    print(OUTPUT_RESULTS)


if __name__ == "__main__":
    main()
