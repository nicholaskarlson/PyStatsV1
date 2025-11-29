"""Chapter 12: One-way ANOVA utilities for Track B.

This module provides:

* A deterministic simulator for a three-group stress-management study:
    - simulate_one_way_stress_study

* A one-way ANOVA built from raw scores:
    - one_way_anova

* Bonferroni-corrected pairwise t-tests:
    - bonferroni_pairwise_t

The goal is to keep the code explicit and reproducible so students can see
how ANOVA is implemented "under the hood" and cross-check results against
SciPy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

Number = Union[int, float]
RandomStateLike = Union[int, np.random.Generator, None]


@dataclass
class OneWayAnovaResult:
    """Container for one-way ANOVA results."""

    k: int
    N: int
    df_between: int
    df_within: int
    df_total: int
    ss_between: float
    ss_within: float
    ss_total: float
    ms_between: float
    ms_within: float
    F: float
    p_value: float
    eta_sq: float


@dataclass
class PairwiseResult:
    """Container for a single pairwise comparison (pooled-variance t-test)."""

    group1: str
    group2: str
    n1: int
    n2: int
    t_stat: float
    df: int
    p_uncorrected: float
    p_bonferroni: float


def _get_rng(random_state: RandomStateLike) -> np.random.Generator:
    """Return a NumPy Generator from an int seed or an existing Generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def simulate_one_way_stress_study(
    n_per_group: int = 30,
    group_names: Iterable[str] = ("control", "cbt", "mindfulness"),
    means: Iterable[Number] = (18.0, 15.0, 12.0),
    sds: Iterable[Number] = (8.0, 8.0, 8.0),
    random_state: RandomStateLike = 12,
) -> pd.DataFrame:
    """Simulate a simple one-way between-subjects stress study.

    Parameters
    ----------
    n_per_group
        Number of participants per group (assumed equal here).
    group_names
        Iterable of group labels (e.g., ["control", "cbt", "mindfulness"]).
    means
        Iterable of population means for stress_score in each group.
    sds
        Iterable of population standard deviations for stress_score in each group.
    random_state
        Integer seed, NumPy Generator, or None for randomness.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
        - participant_id
        - group
        - stress_score
    """
    rng = _get_rng(random_state)

    group_names = list(group_names)
    means = list(means)
    sds = list(sds)

    if not (len(group_names) == len(means) == len(sds)):
        raise ValueError("group_names, means, and sds must have the same length.")

    rows = []
    participant_id = 1
    for g, mu, sigma in zip(group_names, means, sds):
        scores = rng.normal(loc=mu, scale=sigma, size=n_per_group)
        for x in scores:
            rows.append(
                {
                    "participant_id": participant_id,
                    "group": g,
                    "stress_score": float(x),
                }
            )
            participant_id += 1

    return pd.DataFrame(rows)


def one_way_anova(
    df: pd.DataFrame,
    *,
    group_col: str = "group",
    dv_col: str = "stress_score",
) -> OneWayAnovaResult:
    """Compute a one-way ANOVA from a long-format DataFrame.

    Parameters
    ----------
    df
        DataFrame with at least a grouping column and a numeric DV column.
    group_col
        Name of the grouping (factor) column.
    dv_col
        Name of the dependent variable column.

    Returns
    -------
    OneWayAnovaResult
        Dataclass with sums of squares, mean squares, F, p, and eta^2.
    """
    # Drop rows with missing values in the relevant columns, if any.
    sub = df[[group_col, dv_col]].dropna()

    groups = sub[group_col].unique()
    k = len(groups)

    # Group sizes and means
    group_stats = sub.groupby(group_col)[dv_col].agg(["size", "mean"])
    n_j = group_stats["size"].to_numpy(dtype=float)
    mean_j = group_stats["mean"].to_numpy(dtype=float)

    # Overall stats
    N = int(n_j.sum())
    grand_mean = float(sub[dv_col].mean())

    # Sums of squares
    ss_between = float(np.sum(n_j * (mean_j - grand_mean) ** 2))

    # For SS_within we subtract each score from its group mean.
    # Use groupby/transform to broadcast group means.
    centered = sub[dv_col] - sub.groupby(group_col)[dv_col].transform("mean")
    ss_within = float(np.sum(centered**2))

    ss_total = ss_between + ss_within

    df_between = k - 1
    df_within = N - k
    df_total = N - 1

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    F = ms_between / ms_within
    p_value = float(stats.f.sf(F, df_between, df_within))

    eta_sq = ss_between / ss_total if ss_total > 0.0 else 0.0

    return OneWayAnovaResult(
        k=k,
        N=N,
        df_between=df_between,
        df_within=df_within,
        df_total=df_total,
        ss_between=ss_between,
        ss_within=ss_within,
        ss_total=ss_total,
        ms_between=ms_between,
        ms_within=ms_within,
        F=float(F),
        p_value=p_value,
        eta_sq=float(eta_sq),
    )


def bonferroni_pairwise_t(
    df: pd.DataFrame,
    *,
    group_col: str = "group",
    dv_col: str = "stress_score",
) -> list[PairwiseResult]:
    """Compute Bonferroni-corrected pairwise t-tests between all groups.

    This uses pooled-variance independent-samples t-tests (equal_var=True),
    matching the classical Student's t-test in Chapter 10.

    Parameters
    ----------
    df
        DataFrame with group and DV columns.
    group_col
        Name of the grouping column.
    dv_col
        Name of the dependent variable column.

    Returns
    -------
    list[PairwiseResult]
        One result per unique pair of groups.
    """
    sub = df[[group_col, dv_col]].dropna()
    groups = sorted(sub[group_col].unique())
    m = len(groups)

    # Number of pairwise comparisons.
    num_pairs = m * (m - 1) // 2

    results: list[PairwiseResult] = []

    for i in range(m):
        for j in range(i + 1, m):
            g1, g2 = groups[i], groups[j]
            x1 = sub.loc[sub[group_col] == g1, dv_col].to_numpy()
            x2 = sub.loc[sub[group_col] == g2, dv_col].to_numpy()

            n1 = x1.size
            n2 = x2.size

            # Pooled-variance independent-samples t-test.
            t_stat, p_unc = stats.ttest_ind(x1, x2, equal_var=True)
            dfree = n1 + n2 - 2

            # Bonferroni correction.
            p_bonf = min(p_unc * num_pairs, 1.0)

            results.append(
                PairwiseResult(
                    group1=str(g1),
                    group2=str(g2),
                    n1=n1,
                    n2=n2,
                    t_stat=float(t_stat),
                    df=dfree,
                    p_uncorrected=float(p_unc),
                    p_bonferroni=float(p_bonf),
                )
            )

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for Chapter 12."""
    parser = argparse.ArgumentParser(
        description=(
            "Simulate and analyze a one-way stress-management study "
            "using explicit ANOVA calculations and Bonferroni post-hocs."
        )
    )
    parser.add_argument(
        "--n-per-group",
        type=int,
        default=30,
        help="Number of participants per group (default: 30).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for the simulator (default: 2025).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "Optional path to write the simulated dataset as CSV. "
            "Parent directories are created if necessary."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Command-line entry point for Chapter 12."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    df = simulate_one_way_stress_study(
        n_per_group=args.n_per_group,
        random_state=args.seed,
    )

    if args.csv:
        import pathlib

        path = pathlib.Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    # Compute ANOVA
    anova_res = one_way_anova(df)

    # SciPy safety check.
    grouped = [grp["stress_score"].to_numpy() for _, grp in df.groupby("group")]
    scipy_F, scipy_p = stats.f_oneway(*grouped)

    # Pairwise tests
    pairwise = bonferroni_pairwise_t(df)

    # Pretty printing
    print("One-way ANOVA on stress scores (control vs CBT vs mindfulness)")
    print("--------------------------------------------------------------")

    group_stats = df.groupby("group")["stress_score"].agg(["size", "mean"])
    n_per_group = group_stats["size"].iloc[0]
    equal_ns = group_stats["size"].nunique() == 1

    if equal_ns:
        print(f"Group means (n per group = {int(n_per_group)}):")
    else:
        print("Group means (unequal ns):")

    for g, row in group_stats.iterrows():
        print(f"  {g:<11} mean = {row['mean']:.2f} (n = {int(row['size'])})")

    print("\nANOVA table:")
    print(
        f"  SS_between = {anova_res.ss_between:.2f}, "
        f"df_between = {anova_res.df_between}, "
        f"MS_between = {anova_res.ms_between:.2f}"
    )
    print(
        f"  SS_within  = {anova_res.ss_within:.2f}, "
        f"df_within  = {anova_res.df_within}, "
        f"MS_within  = {anova_res.ms_within:.2f}"
    )
    print(
        f"  SS_total   = {anova_res.ss_total:.2f}, "
        f"df_total   = {anova_res.df_total}"
    )
    print(
        f"  F({anova_res.df_between}, {anova_res.df_within}) = "
        f"{anova_res.F:.2f}, p = {anova_res.p_value:.4f}"
    )
    print(f"  eta^2 = {anova_res.eta_sq:.2f}")

    print("\nPairwise comparisons (Bonferroni-corrected p-values):")
    for res in pairwise:
        print(
            f"  {res.group1} vs {res.group2}:  "
            f"t({res.df}) = {res.t_stat:.2f}, "
            f"p_unc = {res.p_uncorrected:.3f}, "
            f"p_bonf = {res.p_bonferroni:.3f}"
        )

    print("\nSciPy check: f_oneway "
          f"F = {scipy_F:.2f}, p = {scipy_p:.4f}")

    # Basic sanity check: our F should match SciPy's up to tiny rounding error.
    if not np.allclose(anova_res.F, scipy_F, rtol=1e-10, atol=1e-10):
        print(
            "WARNING: manual ANOVA F does not match SciPy f_oneway "
            "within numerical tolerance."
        )


if __name__ == "__main__":
    main()
