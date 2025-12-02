"""Tests for Chapter 16 regression lab."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.psych_ch16_regression import (
    fit_multiple_regression,
    fit_simple_regression,
    simulate_psych_regression_dataset,
)


def test_simulated_dataset_shape_and_columns() -> None:
    df = simulate_psych_regression_dataset(n=150, random_state=1)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (150, 5)

    expected_cols = {
        "stress",
        "sleep_hours",
        "study_hours",
        "motivation",
        "exam_score",
    }
    assert set(df.columns) == expected_cols

    # Basic sanity checks: more study and sleep tend to help
    corr_matrix = df.corr(numeric_only=True)
    assert corr_matrix.loc["study_hours", "exam_score"] > 0.3
    assert corr_matrix.loc["sleep_hours", "exam_score"] > 0.1
    assert corr_matrix.loc["stress", "exam_score"] < -0.1


def test_simple_regression_returns_reasonable_values() -> None:
    df = simulate_psych_regression_dataset(n=200, random_state=123)
    result = fit_simple_regression(df)

    # Required keys
    for key in ("slope", "intercept", "r", "r_squared", "se_est", "n"):
        assert key in result

    # Exam score should increase with study hours
    assert result["slope"] > 0

    # Correlation should be moderate to strong and consistent with R²
    assert 0.3 < result["r"] < 1.0
    assert np.isclose(result["r_squared"], result["r"] ** 2, atol=1e-6)

    # Standard error of estimate should be positive
    assert result["se_est"] > 0


def test_multiple_regression_improves_r2_over_simple() -> None:
    df = simulate_psych_regression_dataset(n=200, random_state=123)

    simple = fit_simple_regression(df)
    multi = fit_multiple_regression(
        df,
        outcome="exam_score",
        predictors=("study_hours", "sleep_hours", "stress"),
    )

    # Basic structure checks
    summary = multi["summary"]
    assert isinstance(summary, pd.DataFrame)
    assert {"names", "coef", "se", "T", "pval", "r2", "adj_r2"}.issubset(
        summary.columns
    )

    # R² and adjusted R² should be between 0 and 1
    assert 0.0 <= multi["r2"] <= 1.0
    assert 0.0 <= multi["adj_r2"] <= 1.0

    # Multiple regression should explain at least as much variance as simple
    assert multi["r2"] >= simple["r_squared"] - 1e-6
