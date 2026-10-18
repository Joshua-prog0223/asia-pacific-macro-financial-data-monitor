
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "credit_gap_dsr_sector_panel.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "credit_gap_dsr_bootstrap_inference.csv"
)

BLOCK_LENGTH = 4
N_BOOTSTRAP = 2000
RANDOM_SEED = 20261015
CONFIDENCE_LEVEL = 0.95


# --------------------------------------------------
# 2. Moving-block bootstrap
# --------------------------------------------------

def moving_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample consecutive blocks of indices with replacement."""

    if block_length < 1 or block_length > n:
        raise ValueError("Invalid block length.")

    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length

    starts = rng.integers(
        low=0,
        high=max_start + 1,
        size=n_blocks,
    )

    indices = np.concatenate(
        [
            np.arange(start, start + block_length)
            for start in starts
        ]
    )

    return indices[:n]



def bootstrap_correlations(
    x: np.ndarray,
    y: np.ndarray,
    block_length: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bootstrap distributions of Pearson and Spearman."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")

    n = len(x)

    if len(y) != n:
        raise ValueError("x and y must have the same length.")

    if n < 1:
        raise ValueError("Sample size must be at least 1.")

    if block_length < 1 or block_length > n:
        raise ValueError("Invalid block length.")

    if n < 2 * block_length:
        raise ValueError("Too few observations for block bootstrap.")

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must contain only finite values.")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    pearson_values = []
    spearman_values = []

    for _ in range(n_bootstrap):
        indices = moving_block_indices(
            n=n,
            block_length=block_length,
            rng=rng,
        )

        xb = x[indices]
        yb = y[indices]

        if np.std(xb) == 0 or np.std(yb) == 0:
            continue

        pearson = pd.Series(xb).corr(
            pd.Series(yb), method="pearson"
        )
        spearman = pd.Series(xb).corr(
            pd.Series(yb), method="spearman"
        )

        if np.isfinite(pearson) and np.isfinite(spearman):
            pearson_values.append(pearson)
            spearman_values.append(spearman)

    return (
        np.asarray(pearson_values),
        np.asarray(spearman_values),
    )


# --------------------------------------------------


def main():
    # 3. Load and validate the panel
    # --------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "country_code",
        "quarter",
        "sector_code",
        "credit_gap",
        "dsr",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")

    df = df.sort_values(
        ["country_code", "sector_code", "quarter"]
    ).copy()

    keys = ["country_code", "sector_code", "quarter"]

    if df.duplicated(keys).any():
        raise ValueError("Duplicate country-sector-quarter keys.")

    if df[["credit_gap", "dsr"]].isna().any().any():
        raise ValueError("Missing values found.")

    group_columns = ["country_code", "sector_code"]

    df["credit_gap_change"] = (
        df.groupby(group_columns)["credit_gap"].diff()
    )

    df["dsr_change"] = (
        df.groupby(group_columns)["dsr"].diff()
    )


    # --------------------------------------------------
    # 4. Bootstrap confidence intervals
    # --------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    alpha = 1 - CONFIDENCE_LEVEL
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    results = []

    for (country, sector), group in df.groupby(group_columns):

        changes = (
            group[
                [
                    "quarter",
                    "credit_gap_change",
                    "dsr_change",
                ]
            ]
            .dropna()
            .reset_index(drop=True)
        )

        x = changes["credit_gap_change"].to_numpy(dtype=float)
        y = changes["dsr_change"].to_numpy(dtype=float)

        if len(x) < 2 * BLOCK_LENGTH:
            raise ValueError(
                f"Too few observations for {country}-{sector}."
            )

        observed_pearson = pd.Series(x).corr(
            pd.Series(y), method="pearson"
        )

        observed_spearman = pd.Series(x).corr(
            pd.Series(y), method="spearman"
        )

        pearson_boot, spearman_boot = bootstrap_correlations(
            x=x,
            y=y,
            block_length=BLOCK_LENGTH,
            n_bootstrap=N_BOOTSTRAP,
            rng=rng,
        )


        if (
            len(pearson_boot) < 0.95 * N_BOOTSTRAP
            or len(spearman_boot) < 0.95 * N_BOOTSTRAP
            or len(pearson_boot) == 0
            or len(spearman_boot) == 0
        ):
            raise RuntimeError(
                f"Too many invalid bootstrap samples for {country}-{sector}: "
                f"Pearson={len(pearson_boot)}, "
                f"Spearman={len(spearman_boot)}."
            )

        if len(pearson_boot) != len(spearman_boot):
            raise RuntimeError(
                f"Bootstrap distribution lengths differ for {country}-{sector}."
            )


        results.append(
            {
                "country_code": country,
                "sector_code": sector,
                "observations": len(x),
                "block_length": BLOCK_LENGTH,
                "bootstrap_repetitions": len(pearson_boot),
                "pearson_observed": observed_pearson,
                "pearson_ci_lower": np.quantile(
                    pearson_boot, lower_q
                ),
                "pearson_ci_upper": np.quantile(
                    pearson_boot, upper_q
                ),
                "spearman_observed": observed_spearman,
                "spearman_ci_lower": np.quantile(
                    spearman_boot, lower_q
                ),
                "spearman_ci_upper": np.quantile(
                    spearman_boot, upper_q
                ),
            }
        )


    # --------------------------------------------------
    # 5. Save and report
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("Block-bootstrap correlation inference")
    print("=====================================")
    print(f"Block length: {BLOCK_LENGTH} quarters")
    print(f"Requested repetitions: {N_BOOTSTRAP}")
    print(f"Confidence level: {CONFIDENCE_LEVEL:.0%}")

    print("\nResults")
    print("-------")
    print(results_df.round(3).to_string(index=False))

    print("\nSaved file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
