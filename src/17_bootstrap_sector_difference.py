
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    ROOT / "data" / "processed" / "credit_gap_dsr_sector_panel.csv"
)
OUTPUT_FILE = (
    ROOT / "data" / "processed" / "credit_gap_dsr_sector_difference.csv"
)

N_BOOTSTRAP = 2000
BLOCK_LENGTHS = [2, 4, 8]
RANDOM_SEED = 20261017
CONFIDENCE_LEVEL = 0.95


def pearson_correlation(x, y):
    """Calculate Pearson correlation; return NaN for constant data."""
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])



def moving_block_indices(n, block_length, rng):
    """Sample contiguous, non-circular blocks until n indices are collected."""

    if n < 1:
        raise ValueError("Sample size must be at least 1.")

    if block_length < 1:
        raise ValueError("Block length must be at least 1.")

    if block_length > n:
        raise ValueError("Block length cannot exceed the sample size.")

    sampled = []

    while len(sampled) < n:
        start = rng.integers(0, n - block_length + 1)
        sampled.extend(range(start, start + block_length))

    return np.asarray(sampled[:n], dtype=int)


def main():
    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "country_code",
        "sector_code",
        "quarter",
        "credit_gap_change",
        "dsr_change",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    results = []

    for country in sorted(df["country_code"].unique()):
        country_df = df[df["country_code"] == country]

        household = (
            country_df[country_df["sector_code"] == "H"][
                ["quarter", "credit_gap_change", "dsr_change"]
            ]
            .rename(columns={"dsr_change": "household_dsr_change"})
        )

        corporate = (
            country_df[country_df["sector_code"] == "N"][
                ["quarter", "credit_gap_change", "dsr_change"]
            ]
            .rename(columns={"dsr_change": "corporate_dsr_change"})
        )

        # Align household and corporate observations on the same quarter.
        paired = household.merge(
            corporate,
            on=["quarter", "credit_gap_change"],
            how="inner",
            validate="one_to_one",
        )

        paired = (
            paired.dropna()
            .sort_values("quarter")
            .reset_index(drop=True)
        )

        x = paired["credit_gap_change"].to_numpy()
        y_household = paired["household_dsr_change"].to_numpy()
        y_corporate = paired["corporate_dsr_change"].to_numpy()

        n = len(paired)

        if n < 20:
            raise ValueError(
                f"Too few aligned observations for {country}: {n}"
            )

        r_household = pearson_correlation(x, y_household)
        r_corporate = pearson_correlation(x, y_corporate)
        observed_difference = r_corporate - r_household

        for block_length in BLOCK_LENGTHS:
            rng = np.random.default_rng(
                RANDOM_SEED
                + sum(ord(char) for char in country)
                + block_length
            )

            bootstrap_differences = []

            for _ in range(N_BOOTSTRAP):
                idx = moving_block_indices(n, block_length, rng)

                r_h = pearson_correlation(
                    x[idx], y_household[idx]
                )
                r_n = pearson_correlation(
                    x[idx], y_corporate[idx]
                )

                if np.isfinite(r_h) and np.isfinite(r_n):
                    bootstrap_differences.append(r_n - r_h)

            if len(bootstrap_differences) < 0.95 * N_BOOTSTRAP:
                raise RuntimeError(
                    f"Too many invalid bootstrap samples: {country}, "
                    f"block length {block_length}"
                )

            alpha = 1 - CONFIDENCE_LEVEL
            lower = np.quantile(bootstrap_differences, alpha / 2)
            upper = np.quantile(
                bootstrap_differences, 1 - alpha / 2
            )

            results.append(
                {
                    "country_code": country,
                    "observations": n,
                    "block_length": block_length,
                    "household_pearson": r_household,
                    "corporate_pearson": r_corporate,
                    "difference_corporate_minus_household": (
                        observed_difference
                    ),
                    "difference_ci_lower": lower,
                    "difference_ci_upper": upper,
                    "bootstrap_repetitions": len(bootstrap_differences),
                    "confidence_level": CONFIDENCE_LEVEL,
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("Paired block-bootstrap comparison: corporate vs household")
    print("=" * 66)
    print(f"Bootstrap repetitions requested: {N_BOOTSTRAP}")
    print(f"Block lengths: {BLOCK_LENGTHS}")
    print()
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print()
    print(f"Saved file:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
