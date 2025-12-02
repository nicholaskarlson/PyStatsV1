"""
Chapter 16 regression lab for Track B.

Simulate a small psychology dataset and demonstrate:

- simple linear regression (exam_score ~ study_hours)
- multiple regression with several predictors
- standard error of the estimate
- cross-checking our results with pingouin.linear_regression
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pingouin as pg
from matplotlib import pyplot as plt


DATA_DIR = Path("data") / "synthetic"
OUTPUT_DIR = Path("outputs") / "track_b"


def simulate_psych_regression_dataset(
    n: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate a regression-style psychology dataset.

    Columns:
    - stress: perceived stress (higher = more stressed)
    - sleep_hours: average nightly sleep
    - study_hours: weekly study time
    - motivation: latent motivation score
    - exam_score: exam performance determined by a known linear model
    """
    rng = np.random.default_rng(random_state)

    # Latent factors
    motivation = rng.normal(loc=0.0, scale=1.0, size=n)
    baseline_ability = rng.normal(loc=0.0, scale=1.0, size=n)

    # Study hours and sleep correlate with motivation
    study_hours = 10 + 3 * motivation + rng.normal(0, 1.0, size=n)
    sleep_hours = 7 + 0.7 * motivation + rng.normal(0, 0.7, size=n)

    # Stress is higher when motivation is low and sleep is poor
    stress = 50 - 6 * motivation - 2 * (sleep_hours - 7) + rng.normal(0, 5.0, size=n)

    # True linear model for exam_score
    # exam_score = 70
    #              + 4   * study_hours
    #              + 2   * sleep_hours
    #              - 0.6 * stress
    #              + 5   * baseline_ability
    #              + noise
    noise = rng.normal(0, 8.0, size=n)
    exam_score = (
        70
        + 4.0 * study_hours
        + 2.0 * sleep_hours
        - 0.6 * stress
        + 5.0 * baseline_ability
        + noise
    )

    df = pd.DataFrame(
        {
            "stress": stress,
            "sleep_hours": sleep_hours,
            "study_hours": study_hours,
            "motivation": motivation,
            "exam_score": exam_score,
        }
    )

    return df


def fit_simple_regression(df: pd.DataFrame) -> Dict[str, float]:
    """Fit exam_score ~ study_hours via least squares.

    Returns a small dictionary with slope, intercept, r, r_squared, and
    standard error of the estimate (in exam-score units).
    """
    x = df["study_hours"].to_numpy()
    y = df["exam_score"].to_numpy()
    n = y.shape[0]

    # Use polyfit (degree 1) to get slope and intercept
    slope, intercept = np.polyfit(x, y, deg=1)

    y_hat = slope * x + intercept
    residuals = y - y_hat
    sse = float(np.sum(residuals**2))

    # Standard error of estimate
    se_est = float(np.sqrt(sse / (n - 2)))

    # Pearson correlation and R^2
    r = float(np.corrcoef(x, y)[0, 1])
    r_squared = float(r**2)

    return {
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "r_squared": r_squared,
        "se_est": se_est,
        "n": float(n),
    }


def fit_multiple_regression(
    df: pd.DataFrame,
    outcome: str = "exam_score",
    predictors: List[str] | Tuple[str, ...] = (
        "study_hours",
        "sleep_hours",
        "stress",
    ),
) -> Dict[str, object]:
    """Fit a multiple regression model with Pingouin.

    Returns:
        {
            "summary": pingouin regression table (DataFrame),
            "r2": overall R² for the model,
            "adj_r2": adjusted R²,
        }
    """
    # Pingouin expects X as a DataFrame and y as a vector
    X = df.loc[:, list(predictors)]
    y = df[outcome]

    reg_table = pg.linear_regression(X=X, y=y)

    # The model-level R² / adj_R² are typically in the last row
    # (term == 'Intercept') but we can just read them from the table
    # using max, since they are constant across rows.
    r2 = float(reg_table["r2"].max())
    adj_r2 = float(reg_table["adj_r2"].max())

    return {
        "summary": reg_table,
        "r2": r2,
        "adj_r2": adj_r2,
    }


def plot_regression_line(
    df: pd.DataFrame,
    slope: float,
    intercept: float,
    output_path: Path,
) -> None:
    """Make a scatter plot of study_hours vs exam_score with fitted line."""
    x = df["study_hours"].to_numpy()
    y = df["exam_score"].to_numpy()

    x_grid = np.linspace(x.min(), x.max(), 100)
    y_grid = slope * x_grid + intercept

    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.5)
    ax.plot(x_grid, y_grid)
    ax.set_xlabel("Study hours per week")
    ax.set_ylabel("Exam score")
    ax.set_title("Exam score predicted by study hours (simple regression)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the Chapter 16 regression lab."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Simulating psychology regression dataset...")
    df = simulate_psych_regression_dataset(n=200, random_state=123)

    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Simple regression: exam_score ~ study_hours
    print("\nSimple linear regression: exam_score ~ study_hours")
    simple = fit_simple_regression(df)
    print(f"slope (b1):      {simple['slope']:.3f}")
    print(f"intercept (a):   {simple['intercept']:.3f}")
    print(f"r:               {simple['r']:.3f}")
    print(f"R^2:             {simple['r_squared']:.3f}")
    print(f"SE of estimate:  {simple['se_est']:.3f}")

    # Cross-check with Pingouin's linear_regression (single predictor)
    print("\nCross-check with pingouin.linear_regression (single predictor)")
    pg_table = pg.linear_regression(
        X=df[["study_hours"]],
        y=df["exam_score"],
    )
    print(pg_table[["names", "coef", "r2", "adj_r2"]])

    # Multiple regression
    print("\nMultiple regression: exam_score ~ study_hours + sleep_hours + stress")
    multi = fit_multiple_regression(
        df,
        outcome="exam_score",
        predictors=("study_hours", "sleep_hours", "stress"),
    )
    print(multi["summary"][["names", "coef", "se", "T", "pval", "r2", "adj_r2"]])
    print(f"\nModel R^2:        {multi['r2']:.3f}")
    print(f"Model adj R^2:    {multi['adj_r2']:.3f}")

    # Save artifacts
    data_path = DATA_DIR / "psych_ch16_regression.csv"
    table_path = OUTPUT_DIR / "ch16_regression_summary.csv"
    fig_path = OUTPUT_DIR / "ch16_regression_fit.png"

    df.to_csv(data_path, index=False)
    multi["summary"].to_csv(table_path, index=False)
    plot_regression_line(df, simple["slope"], simple["intercept"], fig_path)

    print("\nData saved to:", data_path)
    print("Regression summary saved to:", table_path)
    print("Regression figure saved to:", fig_path)
    print("\nChapter 16 regression lab complete.")


if __name__ == "__main__":
    main()
