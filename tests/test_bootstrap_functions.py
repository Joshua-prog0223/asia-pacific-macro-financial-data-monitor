import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "17_bootstrap_sector_difference.py"
)

SPEC = importlib.util.spec_from_file_location(
    "bootstrap_sector_difference",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pearson_perfect_positive_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])

    result = MODULE.pearson_correlation(x, y)

    assert result == pytest.approx(1.0)


def test_pearson_perfect_negative_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([8.0, 6.0, 4.0, 2.0])

    result = MODULE.pearson_correlation(x, y)

    assert result == pytest.approx(-1.0)


def test_pearson_constant_series_returns_nan():
    x = np.array([2.0, 2.0, 2.0, 2.0])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    result = MODULE.pearson_correlation(x, y)

    assert np.isnan(result)


def test_pearson_result_is_between_minus_one_and_one():
    x = np.array([2.0, 5.0, 1.0, 7.0, 4.0])
    y = np.array([3.0, 1.0, 6.0, 2.0, 5.0])

    result = MODULE.pearson_correlation(x, y)

    assert -1.0 <= result <= 1.0


def test_moving_block_indices_has_requested_length():
    rng = np.random.default_rng(123)

    indices = MODULE.moving_block_indices(
        n=20,
        block_length=4,
        rng=rng,
    )

    assert len(indices) == 20


def test_moving_block_indices_stays_within_bounds():
    rng = np.random.default_rng(123)

    indices = MODULE.moving_block_indices(
        n=20,
        block_length=4,
        rng=rng,
    )

    assert np.all(indices >= 0)
    assert np.all(indices < 20)


def test_moving_block_indices_is_reproducible():
    indices_a = MODULE.moving_block_indices(
        20, 4, np.random.default_rng(123)
    )
    indices_b = MODULE.moving_block_indices(
        20, 4, np.random.default_rng(123)
    )

    np.testing.assert_array_equal(indices_a, indices_b)


def test_moving_blocks_preserve_consecutive_order():
    block_length = 4
    indices = MODULE.moving_block_indices(
        20,
        block_length,
        np.random.default_rng(123),
    )

    for start in range(0, len(indices) - block_length + 1, block_length):
        block = indices[start:start + block_length]

        assert np.all(np.diff(block) == 1)


def test_block_length_larger_than_sample_raises_error():
    with pytest.raises(ValueError):
        MODULE.moving_block_indices(
            5, 6, np.random.default_rng(123)
        )


def test_zero_block_length_raises_error():
    with pytest.raises(ValueError, match="at least 1"):
        MODULE.moving_block_indices(
            10, 0, np.random.default_rng(123)
        )


def test_zero_sample_size_raises_error():
    with pytest.raises(ValueError, match="at least 1"):
        MODULE.moving_block_indices(
            0, 1, np.random.default_rng(123)
        )


def load_bootstrap_inference_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "15_bootstrap_correlation_inference.py"
    )

    spec = importlib.util.spec_from_file_location(
        "bootstrap_correlation_inference",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_rejects_different_series_lengths():
    module = load_bootstrap_inference_module()

    with pytest.raises(ValueError, match="same length"):
        module.bootstrap_correlations(
            x=np.array([1.0, 2.0, 3.0, 4.0]),
            y=np.array([1.0, 2.0, 3.0]),
            block_length=2,
            n_bootstrap=10,
            rng=np.random.default_rng(123),
        )


def test_bootstrap_rejects_too_few_observations():
    module = load_bootstrap_inference_module()

    with pytest.raises(ValueError, match="Too few observations"):
        module.bootstrap_correlations(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([2.0, 4.0, 6.0]),
            block_length=2,
            n_bootstrap=10,
            rng=np.random.default_rng(123),
        )


def test_bootstrap_correlations_are_reproducible():
    module = load_bootstrap_inference_module()
    x = np.arange(1.0, 21.0)
    y = 2.0 * x + np.sin(x)

    result_a = module.bootstrap_correlations(
        x, y, 4, 100, np.random.default_rng(42)
    )
    result_b = module.bootstrap_correlations(
        x, y, 4, 100, np.random.default_rng(42)
    )

    np.testing.assert_array_equal(result_a[0], result_b[0])
    np.testing.assert_array_equal(result_a[1], result_b[1])


def test_bootstrap_correlations_are_bounded():
    module = load_bootstrap_inference_module()
    x = np.arange(1.0, 31.0)
    y = 3.0 * x + np.cos(x)

    pearson, spearman = module.bootstrap_correlations(
        x, y, 4, 100, np.random.default_rng(7)
    )

    assert len(pearson) > 0
    assert len(spearman) > 0
    assert np.all(np.abs(pearson) <= 1.0)
    assert np.all(np.abs(spearman) <= 1.0)


def load_block_sensitivity_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "16_bootstrap_block_sensitivity.py"
    )

    spec = importlib.util.spec_from_file_location(
        "bootstrap_block_sensitivity",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sensitivity_indices_have_requested_length():
    module = load_block_sensitivity_module()

    indices = module.moving_block_indices(
        20, 4, np.random.default_rng(123)
    )

    assert len(indices) == 20


def test_sensitivity_indices_stay_within_bounds():
    module = load_block_sensitivity_module()

    indices = module.moving_block_indices(
        20, 4, np.random.default_rng(123)
    )

    assert np.all(indices >= 0)
    assert np.all(indices < 20)


def test_sensitivity_indices_are_reproducible():
    module = load_block_sensitivity_module()

    a = module.moving_block_indices(
        20, 4, np.random.default_rng(123)
    )
    b = module.moving_block_indices(
        20, 4, np.random.default_rng(123)
    )

    np.testing.assert_array_equal(a, b)



def test_sensitivity_bootstrap_correlations_are_bounded():
    module = load_block_sensitivity_module()
    x = np.arange(1.0, 31.0)
    y = 2.0 * x + np.sin(x)

    pearson, spearman = module.bootstrap_correlations(
        x,
        y,
        4,
        np.random.default_rng(123),
        n_bootstrap=100,
    )

    assert len(pearson) > 0
    assert len(spearman) > 0
    assert np.all(np.abs(pearson) <= 1.0)
    assert np.all(np.abs(spearman) <= 1.0)


def test_sensitivity_rejects_zero_block_length():
    module = load_block_sensitivity_module()

    with pytest.raises(ValueError, match="Invalid block length"):
        module.bootstrap_correlations(
            np.arange(1.0, 11.0),
            np.arange(2.0, 12.0),
            0,
            np.random.default_rng(123),
            n_bootstrap=10,
        )


def test_sensitivity_rejects_different_series_lengths():
    module = load_block_sensitivity_module()

    with pytest.raises(ValueError, match="same length"):
        module.bootstrap_correlations(
            np.arange(1.0, 11.0),
            np.arange(1.0, 10.0),
            2,
            np.random.default_rng(123),
            n_bootstrap=10,
        )


def test_sensitivity_rejects_non_finite_values():
    module = load_block_sensitivity_module()
    x = np.arange(1.0, 11.0)
    y = np.arange(1.0, 11.0)
    y[4] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        module.bootstrap_correlations(
            x,
            y,
            2,
            np.random.default_rng(123),
            n_bootstrap=10,
        )

