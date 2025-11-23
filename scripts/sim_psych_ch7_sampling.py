# SPDX-License-Identifier: MIT
"""Chapter 7: simulate sampling distributions for a skewed stress-score population.

This module generates:
- a synthetic population of "stress_score" values (skewed, not normal),
- a sampling distribution of the mean via repeated random samples, and
- a plot comparing the population distribution and the distribution of
  sample means.

It is designed to support the Track B Chapter 7 lab.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Default locations and settings
POPULATION_CSV = Path("data/synthetic/psych_ch7_population_stress.csv")
SAMPLE_MEANS_CSV = Path("data/synthetic/psych_ch7_sample_means.csv")
PLOT_PATH = Path("outputs/track_b/ch07_population_vs_sample_means.png")

DEFAULT_POP_N = 50_000
DEFAULT_SAMPLE_N = 25
DEFAULT_REPS = 1_000
DEFAULT_SEED = 123


def generate_population(
    n: int = DEFAULT_POP_N,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate a skewed synthetic population of stress scores.

    We use a Gamma distribution to create a right-skewed variable.

    Parameters
    ----------
    n:
        Number of individuals in the population.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with a single column, 'stress_score'.
    """
    rng = np.random.default_rng(seed)
    # Shape and scale chosen to give a reasonable psych-like range
    stress = rng.gamma(shape=4.0, scale=5.0, size=n)
    return pd.DataFrame({"stress_score": stress})


def write_population_csv(
    path: Path = POPULATION_CSV,
    n: int = DEFAULT_POP_N,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Generate and write the population to CSV."""
    df = generate_population(n=n, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def draw_sample_means(
    population: pd.Series,
    n: int = DEFAULT_SAMPLE_N,
    reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
) -> pd.Series:
    """Draw repeated samples with replacement and return their means.

    Parameters
    ----------
    population:
        Series of population values to sample from.
    n:
        Sample size per replication.
    reps:
        Number of repeated samples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.Series
        Series of length `reps` containing the sample means.
    """
    rng = np.random.default_rng(seed)
    values = population.to_numpy()
    means = []
    for _ in range(reps):
        idx = rng.integers(0, len(values), size=n)
        means.append(values[idx].mean())

    return pd.Series(means, name="sample_mean")


def write_sample_means_csv(
    population: pd.Series,
    path: Path = SAMPLE_MEANS_CSV,
    n: int = DEFAULT_SAMPLE_N,
    reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Draw repeated samples, compute sample means, and write to CSV."""
    sample_means = draw_sample_means(
        population=population,
        n=n,
        reps=reps,
        seed=seed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_means.to_csv(path, index=False)
    return path


def make_population_vs_means_plot(
    population: pd.Series,
    sample_means: pd.Series,
    out_path: Path = PLOT_PATH,
) -> Path:
    """Plot population distribution vs. sampling distribution of the mean."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey=False)

    # Population distribution
    axes[0].hist(population, bins=40, density=True, alpha=0.7)
    axes[0].set_title("Population: stress_score")
    axes[0].set_xlabel("stress_score")
    axes[0].set_ylabel("Density")

    # Sampling distribution of the mean
    axes[1].hist(sample_means, bins=30, density=True, alpha=0.7)
    axes[1].set_title("Sampling distribution of the mean")
    axes[1].set_xlabel("Sample mean (stress_score)")
    axes[1].set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main(
    pop_csv: Path = POPULATION_CSV,
    means_csv: Path = SAMPLE_MEANS_CSV,
    plot_path: Path = PLOT_PATH,
    pop_n: int = DEFAULT_POP_N,
    sample_n: int = DEFAULT_SAMPLE_N,
    reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
) -> int:
    """Entry point for the Chapter 7 lab script.

    Generates a population, draws repeated samples of size `sample_n`, computes
    sample means, writes CSVs, and creates a comparison plot.
    """
    # Generate population and sample means
    population_df = generate_population(n=pop_n, seed=seed)
    population = population_df["stress_score"]

    sample_means = draw_sample_means(
        population=population,
        n=sample_n,
        reps=reps,
        seed=seed + 1,  # use a different seed for sampling
    )

    # Write CSVs
    pop_csv.parent.mkdir(parents=True, exist_ok=True)
    means_csv.parent.mkdir(parents=True, exist_ok=True)
    population_df.to_csv(pop_csv, index=False)
    sample_means.to_csv(means_csv, index=False)

    # Print summary stats
    pop_mean = population.mean()
    pop_sd = population.std(ddof=1)
    means_mean = sample_means.mean()
    means_sd = sample_means.std(ddof=1)
    theoretical_se = pop_sd / np.sqrt(sample_n)

    print(f"Generated population with {len(population):d} individuals")
    print(f"Population mean stress_score = {pop_mean:.2f}")
    print(f"Population SD   stress_score = {pop_sd:.2f}\n")

    print(f"Drew {len(sample_means):d} samples of size n = {sample_n:d}")
    print(f"Sampling distribution mean = {means_mean:.2f}")
    print(f"Sampling distribution SD   = {means_sd:.2f} "
          f"(theoretical SE ≈ {theoretical_se:.2f})\n")

    # Make plot
    make_population_vs_means_plot(
        population=population,
        sample_means=sample_means,
        out_path=plot_path,
    )

    print(f"Wrote population to: {pop_csv}")
    print(f"Wrote sample means to: {means_csv}")
    print(f"Wrote plot to: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
