"""
Deterministic tests for Chapter 14 – Repeated-Measures ANOVA helpers.

These tests focus on:

* Balanced design: every participant has one observation at each time level.
* Sums-of-squares decomposition:
    SS_Total ≈ SS_Subjects + SS_Time + SS_Residual
* Degrees-of-freedom relationship:
    df_Total = df_Subjects + df_Time + df_Residual
* A non-trivial Time effect (F > 0 and a reasonable p-value).
"""

from __future__ import annotations

import math


from scripts.psych_ch14_repeated_measures_anova import (
    TIME_LEVELS,
    RepeatedMeasuresANOVAResult,
    compute_repeated_measures_anova,
    simulate_repeated_measures_data,
)


def test_simulation_is_balanced() -> None:
    """
    Every participant should have exactly one observation at each time level.
    """
    df = simulate_repeated_measures_data(random_seed=14)

    # Number of unique participants and time levels.
    n_subjects = df["participant_id"].nunique()
    n_time_levels = df["time"].nunique()

    assert n_time_levels == len(TIME_LEVELS)

    # For each subject, check that they appear once at each time level.
    counts = (
        df.groupby("participant_id")["time"]
        .nunique()
        .reset_index(name="n_levels")
    )
    assert len(counts) == n_subjects
    assert (counts["n_levels"] == n_time_levels).all()


def test_anova_sums_of_squares_decompose_correctly() -> None:
    """
    Check that:

        SS_Total ≈ SS_Subjects + SS_Time + SS_Residual

    and that degrees of freedom match the textbook relationships.
    """
    df = simulate_repeated_measures_data(random_seed=14)
    anova: RepeatedMeasuresANOVAResult = compute_repeated_measures_anova(df)

    lhs = anova.ss_total
    rhs = anova.ss_subjects + anova.ss_time + anova.ss_residual
    assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-6)

    df_lhs = anova.df_total
    df_rhs = anova.df_subjects + anova.df_time + anova.df_residual
    assert df_lhs == df_rhs


def test_time_effect_is_non_trivial() -> None:
    """
    Given the simulation settings (training reduces stress over time),
    we expect a meaningful Time effect:

    * F_Time should be > 0
    * p_Time should be between 0 and 1 and typically < 0.05.
    """
    df = simulate_repeated_measures_data(random_seed=14)
    anova = compute_repeated_measures_anova(df)

    assert anova.f_time > 0.0
    assert 0.0 < anova.p_time < 1.0

    # With the current simulation parameters, the effect should be
    # clearly significant at alpha = 0.05. If this ever fails due to
    # minor changes in the simulator, it can be relaxed.
    assert anova.p_time < 0.05
