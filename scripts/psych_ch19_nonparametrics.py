"""Psychology Track B – Chapter 19 non-parametric statistics lab.

This script demonstrates chi-square tests on simulated survey-style data:

1. Chi-square goodness-of-fit test on coping strategy preferences.
2. Chi-square test of independence for therapy type × improvement.

Outputs:
- data/synthetic/psych_ch19_survey_gof.csv
- data/synthetic/psych_ch19_survey_independence.csv
- outputs/track_b/ch19_gof_table.csv
- outputs/track_b/ch19_independence_table.csv
- outputs/track_b/ch19_gof_barplot.png
- outputs/track_b/ch19_stacked_bar.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import chi2_contingency, chisquare

# Directory setup (parallel to other Track B scripts)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic"
OUTPUTS_DIR = ROOT / "outputs" / "track_b"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ChiSquareGOFResult:
    """Container for chi-square goodness-of-fit results."""

    counts: pd.Series
    expected: np.ndarray
    chi2: float
    p_value: float
    dof: int


@dataclass
class ChiSquareIndependenceResult:
    """Container for chi-square independence results."""

    contingency: pd.DataFrame
    expected: np.ndarray
    chi2: float
    p_value: float
    dof: int
    cramer_v: float
    stats_table: pd.DataFrame


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def simulate_survey_gof_data(
    n: int = 200,
    random_state: int | None = 123,
) -> pd.DataFrame:
    """Simulate a coping-strategy preference survey for GOF test.

    Participants choose their primary coping strategy from four options.
    The *true* probabilities are not uniform, so the chi-square GOF test
    should usually detect a deviation from a 25/25/25/25 null.

    Parameters
    ----------
    n:
        Number of participants.
    random_state:
        Optional random seed for reproducibility.

    Returns
    -------
    DataFrame
        Single column 'coping_strategy' with n rows.
    """
    rng = np.random.default_rng(random_state)
    categories = np.array(["exercise", "therapy", "mindfulness", "social"])
    # True preferences: slightly more exercise and mindfulness
    true_probs = np.array([0.35, 0.20, 0.30, 0.15])
    choices = rng.choice(categories, size=n, p=true_probs)

    return pd.DataFrame({"coping_strategy": choices})


def simulate_survey_independence_data(
    n: int = 240,
    random_state: int | None = 456,
) -> pd.DataFrame:
    """Simulate therapy type × improvement data for independence test.

    We assign each participant to one of three therapy conditions and then
    generate an improvement outcome with different probabilities by group.

    Parameters
    ----------
    n:
        Number of participants.
    random_state:
        Optional random seed.

    Returns
    -------
    DataFrame
        Columns: 'therapy', 'improvement'.
    """
    rng = np.random.default_rng(random_state)
    therapy = rng.choice(["control", "cbt", "mindfulness"], size=n)

    improvement_labels: list[str] = []
    for t in therapy:
        if t == "control":
            p_improve = 0.45
        elif t == "cbt":
            p_improve = 0.70
        else:  # mindfulness
            p_improve = 0.65

        improved = rng.random() < p_improve
        improvement_labels.append("improved" if improved else "no_change")

    return pd.DataFrame({"therapy": therapy, "improvement": improvement_labels})


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def run_chi_square_gof(
    df: pd.DataFrame,
    category_col: str = "coping_strategy",
    null_probs: np.ndarray | None = None,
) -> ChiSquareGOFResult:
    """Run a chi-square goodness-of-fit test on category counts.

    Parameters
    ----------
    df:
        DataFrame containing the categorical variable.
    category_col:
        Column name containing the categories.
    null_probs:
        Optional vector of expected probabilities under H0. If None,
        a uniform distribution is assumed.

    Returns
    -------
    ChiSquareGOFResult
        Counts, expected counts, chi-square statistic, p-value, and df.
    """
    counts = df[category_col].value_counts().sort_index()
    observed = counts.to_numpy()
    k = observed.size
    n_obs = observed.sum()

    if null_probs is None:
        expected = np.full(k, n_obs / k)
    else:
        probs = np.asarray(null_probs, dtype=float)
        if probs.size != k:
            msg = "null_probs must have same length as number of categories"
            raise ValueError(msg)
        probs = probs / probs.sum()
        expected = n_obs * probs

    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    dof = k - 1

    return ChiSquareGOFResult(
        counts=counts,
        expected=expected,
        chi2=float(chi2_stat),
        p_value=float(p_value),
        dof=dof,
    )


def run_chi_square_independence(df: pd.DataFrame) -> ChiSquareIndependenceResult:
    """Run a chi-square test of independence on therapy × improvement.

    Uses both SciPy and Pingouin:

    * SciPy's :func:`chi2_contingency` for the classical chi-square test.
    * Pingouin's :func:`chi2_independence` for effect sizes (Cramér's V).

    Parameters
    ----------
    df:
        DataFrame with 'therapy' and 'improvement' columns.

    Returns
    -------
    ChiSquareIndependenceResult
        Contingency table, expected counts, chi-square statistic, p-value,
        df, Cramér's V, and full Pingouin stats table.
    """
    contingency = pd.crosstab(df["therapy"], df["improvement"])

    chi2_stat, p_value, dof, expected = chi2_contingency(
        contingency.to_numpy(),
        correction=False,
    )

    expected_df, observed_df, stats = pg.chi2_independence(
        data=df,
        x="therapy",
        y="improvement",
        correction=False,
    )

    _ = expected_df  # unused, but kept for clarity
    _ = observed_df

    stats_pearson = stats.loc[stats["test"] == "pearson"].iloc[0]
    cramer_v = float(stats_pearson["cramer"])

    return ChiSquareIndependenceResult(
        contingency=contingency,
        expected=expected,
        chi2=float(chi2_stat),
        p_value=float(p_value),
        dof=int(dof),
        cramer_v=cramer_v,
        stats_table=stats,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_gof_bar(
    counts: pd.Series,
    expected: np.ndarray,
    output_path: Path,
) -> None:
    """Create a side-by-side bar chart of observed vs expected counts."""
    categories = list(counts.index)
    observed = counts.to_numpy()
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, observed, width, label="Observed")
    ax.bar(x + width / 2, expected, width, label="Expected")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45)
    ax.set_ylabel("Count")
    ax.set_title("Chi-square goodness-of-fit: coping strategies")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_stacked_bar(
    contingency: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a stacked bar plot of improvement proportions by therapy."""
    proportions = contingency.div(contingency.sum(axis=1), axis=0)

    ax = proportions.plot(kind="bar", stacked=True)
    ax.set_ylabel("Proportion within therapy type")
    ax.set_title("Improvement by therapy type")

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full Chapter 19 non-parametric lab demo."""
    print("Simulating non-parametric datasets for Chapter 19...\n")

    # -----------------------------
    # 1. Goodness-of-fit example
    # -----------------------------
    gof_df = simulate_survey_gof_data()
    print("First 6 rows (GOF survey):")
    print(gof_df.head(), "\n")

    gof_result = run_chi_square_gof(gof_df)

    print("Chi-square goodness-of-fit on coping_strategy")
    print("Counts:")
    print(gof_result.counts)
    print()
    print(
        f"Chi²({gof_result.dof}) = {gof_result.chi2:.3f}, "
        f"p = {gof_result.p_value:.4f}",
    )
    print()

    gof_data_path = DATA_DIR / "psych_ch19_survey_gof.csv"
    gof_table_path = OUTPUTS_DIR / "ch19_gof_table.csv"
    gof_plot_path = OUTPUTS_DIR / "ch19_gof_barplot.png"

    gof_df.to_csv(gof_data_path, index=False)
    gof_summary_df = pd.DataFrame(
        {
            "category": gof_result.counts.index.to_list(),
            "observed": gof_result.counts.to_numpy(),
            "expected": gof_result.expected,
        },
    )
    gof_summary_df.to_csv(gof_table_path, index=False)
    plot_gof_bar(gof_result.counts, gof_result.expected, gof_plot_path)

    print(f"GOF data saved to: {gof_data_path}")
    print(f"GOF summary table saved to: {gof_table_path}")
    print(f"GOF bar plot saved to: {gof_plot_path}\n")

    # -----------------------------
    # 2. Independence example
    # -----------------------------
    indep_df = simulate_survey_independence_data()
    print("First 6 rows (independence survey):")
    print(indep_df.head(), "\n")

    indep_result = run_chi_square_independence(indep_df)

    print("Chi-square test of independence (therapy × improvement)")
    print("Contingency table:")
    print(indep_result.contingency)
    print()
    print(
        f"Chi²({indep_result.dof}) = {indep_result.chi2:.3f}, "
        f"p = {indep_result.p_value:.6f}, "
        f"Cramer's V = {indep_result.cramer_v:.3f}",
    )
    print()

    indep_data_path = DATA_DIR / "psych_ch19_survey_independence.csv"
    indep_stats_path = OUTPUTS_DIR / "ch19_independence_table.csv"
    indep_plot_path = OUTPUTS_DIR / "ch19_stacked_bar.png"

    indep_df.to_csv(indep_data_path, index=False)
    indep_result.stats_table.to_csv(indep_stats_path, index=False)
    plot_stacked_bar(indep_result.contingency, indep_plot_path)

    print(f"Independence data saved to: {indep_data_path}")
    print(f"Independence stats saved to: {indep_stats_path}")
    print(f"Stacked bar plot saved to: {indep_plot_path}\n")

    print("Chapter 19 non-parametric lab complete.")


if __name__ == "__main__":
    main()
