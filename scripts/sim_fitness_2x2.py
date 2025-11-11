# SPDX-License-Identifier: MIT
import os, json, time, argparse
import numpy as np, pandas as pd

# globals configured in main()
rng = np.random.default_rng()
N = 80
GROUPS = ["ProgramA","ProgramB"]
DEFAULT_SEED = 7

def simulate(n_per_group=None, seed=DEFAULT_SEED):
    global rng, N
    # Allow overriding globals via args, primarily for testing
    if n_per_group is not None:
        N = n_per_group * len(GROUPS) # Adjust N based on n_per_group
    
    rng = np.random.default_rng(seed)

    os.makedirs("data/synthetic", exist_ok=True)
    subjects = pd.DataFrame({
        "id": np.arange(1, N+1),
        "group": rng.choice(GROUPS, size=N, replace=True),
        "age": rng.integers(18, 45, size=N),
        "sex": rng.choice(["F","M"], size=N, p=[0.5,0.5]),
        "bmi": rng.normal(24, 3.5, size=N).clip(17, 38),
    })

    # baseline strength with small covariate effects
    base = (80
            + 0.8*(subjects["age"]-30)
            + 1.2*(subjects["bmi"]-24)
            + rng.normal(0, 8, size=N))

    # program gains
    gain_A = rng.normal(8, 4, size=N)
    gain_B = rng.normal(12, 4, size=N)
    gain = np.where(subjects["group"].to_numpy()=="ProgramA", gain_A, gain_B)

    long = pd.concat([
        pd.DataFrame({"id": subjects["id"], "time":"pre",
                      "strength": base + rng.normal(0, 3, size=N)}),
        pd.DataFrame({"id": subjects["id"], "time":"post",
                      "strength": base + gain + rng.normal(0, 3, size=N)})
    ], ignore_index=True).merge(subjects, on="id")

    subjects.to_csv("data/synthetic/fitness_subjects.csv", index=False)
    long.to_csv("data/synthetic/fitness_long.csv", index=False)

    meta = dict(
        seed=seed,
        n=int(N),
        design="2x2 mixed (Group between × Time within)",
        programs=GROUPS,
        dv="strength_1RM_like",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    with open("data/synthetic/fitness_meta.json","w") as f:
        json.dump(meta, f, indent=2)

    print("Wrote fitness_subjects.csv, fitness_long.csv, fitness_meta.json")
    # Return for testing convenience
    return subjects, long

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed")
    ap.add_argument("--n-per-group", type=int, default=40, help="Number of subjects per group")
    args = ap.parse_args()
    
    # Pass CLI args to simulate
    simulate(n_per_group=args.n_per_group, seed=args.seed)

if __name__ == "__main__":
    main()