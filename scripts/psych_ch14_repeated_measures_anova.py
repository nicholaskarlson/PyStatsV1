"""
PyStatsV1 Track B – Chapter 14
Repeated-Measures ANOVA on Stress Scores (Pre, Post, Follow-Up)

This script simulates a simple 1-factor repeated-measures design where the same
participants are measured at three time points:

    - pre:      before training
    - post:     immediately after training
    - followup: some time after training

It then:

* computes the repeated-measures ANOVA "by hand" using sums of squares that
  match the Chapter 14 mini-book exposition:
    SS_Total      = SS_Subjects + SS_Within
    SS_Within     = SS_Time + SS_Residual
* reports F, p, and eta-squared for the Time effect
* runs paired-samples t-tests as simple follow-ups
* (optionally) cross-checks the ANOVA using pingouin.rm_anova if pingouin is
  installed
* (optionally) writes the simulated data to CSV and saves a simple time-course
  plot

The design is balanced (every participant has a score at each time point).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

# Default time levels for the Chapter 14 example
TIME_LEVELS: List[str] = ["pre", "post", "followup"]


@dataclass
class RepeatedMeasuresANOVAResult:
    """Container for the key ANOVA components for the Time factor."""

    n_subjects: int
    n_time_levels: int

    ss_total: float
    df_total: int

    ss_subjects: float
    df_subjects: int

    ss_time: float
    df_time: int
    ms_time: float

    ss_residual: float
    df_residual: int
    ms_residual: float

    f_time: float
    p_time: float
    eta2_time: float


def simulate_repeated_measures_data(
    n_subjects: int = 40,
    time_levels: Sequence[str] | None = None,
    random_seed: int = 14,
) -> pd.DataFrame:
    """
    Simulate a balanced repeated-measures dataset with a Time effect.

    Parameters
    ----------
    n_subjects:
        Number of participants.
    time_levels:
        Ordered labels for the repeated-measures factor (e.g., ["pre", "post", "followup"]).
    random_seed:
        Seed for reproducibility.

    Returns
    -------
    df : pandas.DataFrame
        Long-format data with columns:
            - "participant_id"
            - "time"
            - "stress_score"
    """
    if time_levels is None:
        time_levels = TIME_LEVELS

    rng = np.random.default_rng(random_seed)

    # Grand mean stress level and time-specific improvements.
    grand_mean = 20.0
    # Time effects: training reduces stress from pre -> post -> followup
    time_effects: Dict[str, float] = {
        "pre": 0.0,
        "post": -4.0,
        "followup": -6.0,
    }

    # Random subject intercepts (individual differences) and residual noise.
    subject_sd = 4.0
    residual_sd = 3.0

    subject_offsets = rng.normal(loc=0.0, scale=subject_sd, size=n_subjects)

    rows: List[Dict[str, float | int | str]] = []
    for subj_index in range(n_subjects):
        participant_id = subj_index + 1
        subj_offset = float(subject_offsets[subj_index])

        for t in time_levels:
            mean_score = grand_mean + subj_offset + time_effects.get(t, 0.0)
            observed_score = float(rng.normal(loc=mean_score, scale=residual_sd))
            rows.append(
                {
                    "participant_id": participant_id,
                    "time": str(t),
                    "stress_score": observed_score,
                }
            )

    df = pd.DataFrame(rows)
    return df


def _check_balanced_design(df: pd.DataFrame, subject: str, within: str) -> None:
    """
    Raise a ValueError if the design is not balanced (each subject must have
    the same number of measurements and one per time level).
    """
    counts = df.groupby(subject)[within].nunique().to_numpy()
    unique_counts = np.unique(counts)
    if unique_counts.size != 1:
        raise ValueError(
            f"Unbalanced design: subjects have different numbers of {within} levels. "
            f"Counts per subject: {unique_counts.tolist()}"
        )


def compute_repeated_measures_anova(
    df: pd.DataFrame,
    dv: str = "stress_score",
    subject: str = "participant_id",
    within: str = "time",
) -> RepeatedMeasuresANOVAResult:
    """
    Compute a one-factor repeated-measures ANOVA (within-subjects on `within`)
    using the sums-of-squares decomposition presented in Chapter 14.

        SS_Total      = SS_Subjects + SS_Within
        SS_Within     = SS_Time + SS_Residual

    Assumes a balanced design: each subject has one observation at each level
    of the within-subject factor.

    Parameters
    ----------
    df :
        Long-format DataFrame with columns [subject, within, dv].
    dv :
        Name of the dependent variable column.
    subject :
        Name of the subject identifier column.
    within :
        Name of the within-subject factor column (e.g., "time").

    Returns
    -------
    RepeatedMeasuresANOVAResult
        Dataclass with sums of squares, degrees of freedom, F, p, and eta^2.
    """
    _check_balanced_design(df, subject=subject, within=within)

    # Basic counts
    n_subjects = df[subject].nunique()
    time_levels = df[within].unique()
    n_time_levels = len(time_levels)
    n_total = df.shape[0]

    # Grand mean
    grand_mean = float(df[dv].mean())

    # Means by subject and by time
    subject_means = df.groupby(subject)[dv].mean()
    time_means = df.groupby(within)[dv].mean()

    # SS_Total
    ss_total = float(((df[dv] - grand_mean) ** 2).sum())
    df_total = n_total - 1

    # SS_Subjects: individual differences (participants averaged over time)
    ss_subjects = float(n_time_levels * ((subject_means - grand_mean) ** 2).sum())
    df_subjects = n_subjects - 1

    # SS_Within
    ss_within = ss_total - ss_subjects

    # SS_Time: effect of the within-subject factor (averaging over subjects)
    ss_time = float(n_subjects * ((time_means - grand_mean) ** 2).sum())
    df_time = n_time_levels - 1

    # SS_Residual: leftover within-subject variation
    ss_residual = ss_within - ss_time
    df_residual = (n_subjects - 1) * (n_time_levels - 1)

    ms_time = ss_time / df_time
    ms_residual = ss_residual / df_residual

    # F and p for the Time effect
    f_time = ms_time / ms_residual
    p_time = float(1.0 - stats.f.cdf(f_time, df_time, df_residual))

    # Eta-squared for Time
    eta2_time = ss_time / ss_total

    return RepeatedMeasuresANOVAResult(
        n_subjects=n_subjects,
        n_time_levels=n_time_levels,
        ss_total=ss_total,
        df_total=df_total,
        ss_subjects=ss_subjects,
        df_subjects=df_subjects,
        ss_time=ss_time,
        df_time=df_time,
        ms_time=ms_time,
        ss_residual=ss_residual,
        df_residual=df_residual,
        ms_residual=ms_residual,
        f_time=f_time,
        p_time=p_time,
        eta2_time=eta2_time,
    )


def run_pingouin_rm_anova(
    df: pd.DataFrame,
    dv: str = "stress_score",
    subject: str = "participant_id",
    within: str = "time",
) -> None:
    """
    Optionally cross-check the repeated-measures ANOVA using pingouin.rm_anova.

    If pingouin is not installed, this function prints a short message and
    returns without raising an error.
    """
    try:
        import pingouin as pg  # type: ignore[import]
    except ImportError:
        print(
            "\n[pingouin] Not installed – skipping pingouin.rm_anova check.\n"
            "You can install it with: pip install pingouin\n"
        )
        return

    print("\nPingouin repeated-measures ANOVA (rm_anova):")
    print("--------------------------------------------")
    aov = pg.rm_anova(
        data=df,
        dv=dv,
        within=within,
        subject=subject,
        detailed=True,
    )
    print(aov.to_string(index=False))


def run_pairwise_t_tests(
    df: pd.DataFrame,
    dv: str = "stress_score",
    subject: str = "participant_id",
    within: str = "time",
    alpha: float = 0.05,
) -> None:
    """
    Run paired-samples t-tests between all pairs of time levels as simple
    follow-up analyses. Uses a Bonferroni correction for the reported
    adjusted p-values.

    Prints results to the console.
    """
    # Wide format: one column per time level, one row per participant.
    wide = df.pivot(index=subject, columns=within, values=dv)

    time_levels: List[str] = list(wide.columns)
    n_subjects = wide.shape[0]

    # All unique pairs of time levels
    def _pairs(xs: Sequence[str]) -> Iterable[tuple[str, str]]:
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                yield xs[i], xs[j]

    print("\nPaired-samples t-tests (Time comparisons)")
    print("----------------------------------------")
    m = sum(1 for _ in _pairs(time_levels))  # number of comparisons

    for t1, t2 in _pairs(time_levels):
        scores1 = wide[t1].to_numpy()
        scores2 = wide[t2].to_numpy()
        t_stat, p_val = stats.ttest_rel(scores1, scores2)
        p_bonf = min(p_val * m, 1.0)

        df_t = n_subjects - 1
        print(
            f"{t1} vs {t2}: t({df_t}) = {t_stat: .2f}, "
            f"p_unc = {p_val: .4f}, p_bonf = {p_bonf: .4f}"
        )


def save_data_and_plot(
    df: pd.DataFrame,
    dv: str = "stress_score",
    within: str = "time",
    csv_path: Path | None = None,
    plot_path: Path | None = None,
) -> None:
    """
    Save the simulated dataset to CSV and a simple time-course plot of
    the mean scores across time.

    Parameters
    ----------
    df :
        Long-format DataFrame.
    csv_path :
        Where to write the CSV file (if not None).
    plot_path :
        Where to save the PNG plot (if not None).
    """
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nData saved to: {csv_path}")

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        means = df.groupby(within)[dv].mean()
        order = [lvl for lvl in TIME_LEVELS if lvl in means.index]
        x = list(range(len(order)))
        y = [means[lvl] for lvl in order]

        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xticks(x, order)
        plt.xlabel(within)
        plt.ylabel(dv)
        plt.title("Mean stress_score across time (repeated measures)")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Interaction-style time plot saved to: {plot_path}")


def main() -> None:
    """
    Run the Chapter 14 repeated-measures ANOVA demonstration end-to-end.
    """
    # 1. Simulate data
    df = simulate_repeated_measures_data()

    n_subjects = df["participant_id"].nunique()
    time_levels = list(df["time"].unique())
    print("Repeated-measures ANOVA on stress scores (Time: pre, post, followup)")
    print("------------------------------------------------------------------")
    print(f"Number of participants: {n_subjects}")
    print(f"Time levels: {time_levels}")

    # 2. Print means at each time point
    means = df.groupby("time")["stress_score"].mean()
    print("\nTime means (averaged over participants):")
    for t, m in means.items():
        print(f"  {t:<10} mean = {m: .2f}")

    # 3. Compute ANOVA components
    anova = compute_repeated_measures_anova(df)

    print("\nRepeated-measures ANOVA table (Time factor)")
    print("-------------------------------------------")
    print(
        f"SS_Time      = {anova.ss_time: .2f}, "
        f"df_Time = {anova.df_time}, "
        f"MS_Time = {anova.ms_time: .2f}"
    )
    print(
        f"SS_Subjects  = {anova.ss_subjects: .2f}, "
        f"df_Subjects = {anova.df_subjects}"
    )
    print(
        f"SS_Residual  = {anova.ss_residual: .2f}, "
        f"df_Residual = {anova.df_residual}, "
        f"MS_Residual = {anova.ms_residual: .2f}"
    )
    print(
        f"SS_Total     = {anova.ss_total: .2f}, "
        f"df_Total = {anova.df_total}"
    )
    print(
        f"\nF_Time({anova.df_time}, {anova.df_residual}) = {anova.f_time: .2f}, "
        f"p = {anova.p_time: .4f}"
    )
    print(f"eta^2_Time = {anova.eta2_time: .3f}")

    # 4. Paired-samples t-tests as simple follow-ups
    run_pairwise_t_tests(df)

    # 5. Optional pingouin check
    run_pingouin_rm_anova(df)

    # 6. Save CSV and plot for students/instructors
    data_path = Path("data/synthetic/psych_ch14_repeated_measures_stress.csv")
    plot_path = Path("outputs/track_b/ch14_stress_over_time.png")
    save_data_and_plot(df, csv_path=data_path, plot_path=plot_path)


if __name__ == "__main__":
    main()
