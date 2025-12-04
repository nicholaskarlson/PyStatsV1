"""Tests for Psychology Track B – Chapter 19 non-parametric lab."""

from scripts.psych_ch19_nonparametrics import (
    run_chi_square_gof,
    run_chi_square_independence,
    simulate_survey_gof_data,
    simulate_survey_independence_data,
)


def test_gof_dataset_has_expected_structure() -> None:
    """Simulated GOF survey should have the right size and column."""
    df = simulate_survey_gof_data(n=300, random_state=42)
    assert len(df) == 300
    assert "coping_strategy" in df.columns
    # We should see at least three distinct strategies in this sample
    assert df["coping_strategy"].nunique() >= 3


def test_gof_detects_deviation_from_uniform_null() -> None:
    """GOF test should usually detect that preferences are not uniform."""
    df = simulate_survey_gof_data(n=400, random_state=7)
    result = run_chi_square_gof(df)

    # Strong deviation from uniform, so p should be comfortably below .05
    assert result.p_value < 0.05
    assert result.dof == result.counts.size - 1


def test_independence_detects_association_between_therapy_and_improvement() -> None:
    """Independence test should detect association and yield non-trivial Cramér's V."""
    df = simulate_survey_independence_data(n=300, random_state=99)
    result = run_chi_square_independence(df)

    # Engineered effect: treatment should matter for improvement.
    assert result.p_value < 0.05

    # Cramér's V should be in a "small-to-medium" range, not near zero.
    assert 0.10 <= result.cramer_v <= 0.60
