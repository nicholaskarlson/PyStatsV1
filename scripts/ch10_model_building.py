# SPDX-License-Identifier: MIT
"""
Chapter 10 — Model Building (R → Python)
- Forward stepwise selection by AIC/BIC
- k-fold CV RMSE
- Predicted vs Actual plot
Usage:
  python scripts/ch10_model_building.py --criterion aic --cvk 5 --save-plot
"""
import argparse
import os
import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_autompg(path: str) -> pd.DataFrame:
    # Robust loader (short/long headers)
    df = pd.read_csv(path, sep=None, engine="python")
    # map possible headers → canonical
    map_try = {
        "mpg": ["mpg"],
        "cylinders": ["cyl", "cylinders"],
        "displacement": ["disp", "displacement"],
        "horsepower": ["hp", "horsepower"],
        "weight": ["wt", "weight"],
        "acceleration": ["acc", "acceleration"],
        "model_year": ["year", "model_year"],
        "origin": ["origin"],
    }
    canon = {}
    for k, opts in map_try.items():
        for o in opts:
            if o in df.columns:
                canon[k] = o
                break
    needed = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year"]
    missing = [k for k in needed if k not in canon]
    if missing:
        raise SystemExit(f"ERROR: expected columns missing: {', '.join(missing)}\nAvailable: {list(df.columns)}")
    cols = [canon[k] for k in needed if k in canon] + ([canon["origin"]] if "origin" in canon else [])
    # keep id-like text column (e.g., first col) only if needed; drop non-numeric dups
    df = df[cols + ["mpg"]] if "mpg" not in cols else df[cols]
    # Coerce numerics
    for k in ["mpg","displacement","horsepower","weight","acceleration","model_year"]:
        df[canon.get(k,k)] = pd.to_numeric(df[canon.get(k,k)], errors="coerce")
    df = df.dropna()
    # rename to canonical names
    out = pd.DataFrame({
        "mpg": df[canon["mpg"]] if "mpg" in canon else df["mpg"],
        "cylinders": df[canon["cylinders"]],
        "displacement": df[canon["displacement"]],
        "horsepower": df[canon["horsepower"]],
        "weight": df[canon["weight"]],
        "acceleration": df[canon["acceleration"]],
        "model_year": df[canon["model_year"]],
    })
    if "origin" in canon:
        out["origin"] = df[canon["origin"]].astype("category")
    out["cylinders"] = out["cylinders"].astype("category")
    return out

def kfold_cv_rmse(df: pd.DataFrame, formula: str, k: int, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    se_sum, n_sum = 0.0, 0
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.setdiff1d(idx, test_idx)
        train = df.iloc[train_idx]
        test  = df.iloc[test_idx]
        try:
            m = smf.ols(formula, data=train).fit()
            yhat = m.predict(test)
            y = test["mpg"].to_numpy()
            se_sum += np.sum((y - yhat)**2)
            n_sum += len(test_idx)
        except Exception:
            # If model fails on a fold (rare), penalize heavily
            se_sum += 1e12
            n_sum += len(test_idx)
    rmse = math.sqrt(se_sum / n_sum)
    return rmse

def forward_stepwise(df: pd.DataFrame, candidates, criterion="aic", kfold=None):
    """Greedy forward selection. criterion ∈ {'aic','bic'}. If kfold set, also track CV RMSE."""
    remaining = list(candidates)
    selected = []
    best_score = float("inf")
    best_model = None

    while remaining:
        scores = []
        for var in remaining:
            formula = "mpg ~ " + " + ".join(selected + [var]) if selected else f"mpg ~ {var}"
            try:
                res = smf.ols(formula, data=df).fit()
            except Exception:
                continue
            score = res.aic if criterion == "aic" else res.bic
            cv = kfold_cv_rmse(df, formula, kfold) if kfold else None
            scores.append((score, cv, var, res, formula))
        if not scores:
            break
        scores.sort(key=lambda t: t[0])  # minimize AIC/BIC
        best_here = scores[0]
        if best_here[0] + 1e-9 < best_score:  # improvement
            best_score = best_here[0]
            best_model = best_here[3]
            selected.append(best_here[2])
            remaining.remove(best_here[2])
        else:
            break
    return selected, best_model, best_score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/autompg.csv")
    p.add_argument("--criterion", choices=["aic","bic"], default="aic")
    p.add_argument("--cvk", type=int, default=5, help="k-folds for RMSE (report only; selection by AIC/BIC)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-plot", action="store_true")
    args = p.parse_args()

    df = load_autompg(args.data)

    # Build candidate list (origin optional)
    candidates = [
        "displacement", "horsepower", "weight", "acceleration", "model_year",
        "C(cylinders)"
    ]
    if "origin" in df.columns:
        candidates.append("C(origin)")

    print("=== Candidates ===")
    print(", ".join(candidates))

    selected, best_model, score = forward_stepwise(df, candidates, criterion=args.criterion, kfold=args.cvk)
    if best_model is None:
        raise SystemExit("Stepwise failed to produce a model.")

    formula = best_model.model.formula
    print("\n=== Selected (forward) ===")
    print("Formula:", formula)
    print(f"{args.criterion.upper()}: {score:.3f}")

    # Report CV RMSE
    cv_rmse = kfold_cv_rmse(df, formula, k=args.cvk, seed=args.seed)
    print(f"CV({args.cvk}) RMSE: {cv_rmse:.4f}")

    # Fit on full data & print brief summary
    res = smf.ols(formula, data=df).fit()
    print("\n=== Coefficients ===")
    for name, est, se, tval, pval in zip(res.params.index, res.params.values, res.bse.values, res.tvalues.values, res.pvalues.values):
        print(f"{name:20s} est={est:9.4f}  SE={se:8.4f}  t={tval:8.3f}  p={pval:.3g}")

    print("\n=== Fit Stats ===")
    print(f"R^2: {res.rsquared:.6f}")
    print(f"Adj R^2: {res.rsquared_adj:.6f}")
    print(f"Sigma (SE): {math.sqrt(res.mse_resid):.6f}")
    print(f"AIC: {res.aic:.3f}  BIC: {res.bic:.3f}")

    if args.save_plot:
        os.makedirs("outputs", exist_ok=True)
        y = df["mpg"].to_numpy()
        yhat = res.fittedvalues.to_numpy()
        plt.figure(figsize=(6,5))
        plt.scatter(y, yhat, alpha=0.65)
        lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
        plt.plot(lims, lims, "--", linewidth=1)
        plt.xlabel("Actual MPG")
        plt.ylabel("Predicted MPG")
        plt.title("Ch10: Predicted vs Actual")
        plt.tight_layout()
        out = "outputs/ch10_pred_vs_actual.png"
        plt.savefig(out, dpi=150)
        print(f"Saved plot -> {out}")

if __name__ == "__main__":
    main()
