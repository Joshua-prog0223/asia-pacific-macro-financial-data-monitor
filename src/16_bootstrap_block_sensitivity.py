
from pathlib import Path

import numpy as np
import pandas as pd


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
    / "credit_gap_dsr_block_sensitivity.csv"
)

BLOCK_LENGTHS = [2, 4, 8]
N_BOOTSTRAP = 2000
SEED = 20261016


def moving_block_indices(n, block_length, rng):
    """Sample consecutive, non-circular blocks of indices."""

    if n < 1:
        raise ValueError("Sample size must be at least 1.")

    if block_length < 1:
        raise ValueError("Block length must be at least 1.")

    if block_length > n:
        raise ValueError("Block length cannot exceed sample size.")

    n_blocks = int(np.ceil(n / block_length))

    starts = rng.integers(
        0,
        n - block_length + 1,
        size=n_blocks,
    )

    indices = np.concatenate([
        np.arange(start, start + block_length)
        for start in starts
    ])

    return indices[:n]



def bootstrap_correlations(
    x,
    y,
    block_length,
    rng,
    n_bootstrap=None,
):
    """Return bootstrap distributions of Pearson and Spearman."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    n = len(x)

    if n < 1:
        raise ValueError("Sample size must be at least 1.")

    if block_length < 1 or block_length > n:
        raise ValueError("Invalid block length.")

    if n < 2 * block_length:
        raise ValueError("Too few observations for block bootstrap.")

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must contain only finite values.")

    if n_bootstrap is None:
        n_bootstrap = N_BOOTSTRAP

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    pearson_values = []
    spearman_values = []

    for _ in range(n_bootstrap):
        indices = moving_block_indices(n, block_length, rng)

        xb = x[indices]
        yb = y[indices]

        if np.std(xb) == 0 or np.std(yb) == 0:
            continue

        pearson_values.append(
            pd.Series(xb).corr(pd.Series(yb), method="pearson")
        )

        spearman_values.append(
            pd.Series(xb).corr(pd.Series(yb), method="spearman")
        )

    return np.asarray(pearson_values), np.asarray(spearman_values)


def main():
    df = pd.read_csv(INPUT_FILE)
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    
    group_columns = ["country_code", "sector_code"]
    
    df = df.sort_values(group_columns + ["quarter"]).copy()
    
    if df.duplicated(
        group_columns + ["quarter"]
    ).any():
        raise ValueError("Duplicate country-sector-quarter keys.")
    
    if df[["credit_gap", "dsr"]].isna().any().any():
        raise ValueError("Missing analysis values.")
    
    df["gap_change"] = df.groupby(group_columns)["credit_gap"].diff()
    df["dsr_change"] = df.groupby(group_columns)["dsr"].diff()
    
    results = []
    
    for (country, sector), group in df.groupby(group_columns):
        changes = group[
            ["gap_change", "dsr_change"]
        ].dropna()
    
        x = changes["gap_change"].to_numpy(dtype=float)
        y = changes["dsr_change"].to_numpy(dtype=float)
    
        pearson_observed = pd.Series(x).corr(pd.Series(y))
        spearman_observed = pd.Series(x).corr(
            pd.Series(y), method="spearman"
        )
    
        for block_length in BLOCK_LENGTHS:
            rng = np.random.default_rng(
                SEED + block_length
            )
    
            pearson_boot, spearman_boot = bootstrap_correlations(
                x, y, block_length, rng
            )
    
            if len(pearson_boot) < 0.95 * N_BOOTSTRAP:
                raise RuntimeError(
                    f"Too many degenerate samples: {country}-{sector}, "
                    f"block length {block_length}"
                )
    
            results.append({
                "country_code": country,
                "sector_code": sector,
                "observations": len(x),
                "block_length": block_length,
                "bootstrap_repetitions": len(pearson_boot),
                "pearson_observed": pearson_observed,
                "pearson_ci_lower": np.quantile(pearson_boot, 0.025),
                "pearson_ci_upper": np.quantile(pearson_boot, 0.975),
                "spearman_observed": spearman_observed,
                "spearman_ci_lower": np.quantile(spearman_boot, 0.025),
                "spearman_ci_upper": np.quantile(spearman_boot, 0.975),
            })
    
    results_df = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    print("Bootstrap block-length sensitivity")
    print("==================================")
    print(f"Requested repetitions per group and length: {N_BOOTSTRAP}")
    print(f"Block lengths: {BLOCK_LENGTHS}")
    
    print("\nPearson confidence intervals")
    print("----------------------------")
    print(
        results_df[
            [
                "country_code",
                "sector_code",
                "block_length",
                "pearson_observed",
                "pearson_ci_lower",
                "pearson_ci_upper",
            ]
        ].round(3).to_string(index=False)
    )
    
    print("\nSaved file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
