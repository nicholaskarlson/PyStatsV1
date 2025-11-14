# Case Study Template — PyStatsV1

> Copy this file when adding a new chapter (e.g. Ch16) and adapt it.
> Suggested naming convention:
>
> - `docs/ch16_epi_rr_case_study.md`
> - `docs/ch14_tutoring_ab_case_study.md`

---

## 1. Overview

**Chapter:** ChXX – Short title (e.g., “Epidemiology Risk Ratio With Strata”)  
**Primary audience:** (e.g., Psychology undergrads, Public Health students, Sports science)  
**Difficulty:** Intro / Intermediate / Advanced

**What this case study teaches:**

- Bullet point 1
- Bullet point 2
- Bullet point 3

---

## 2. Files in This Case Study

List the core files and what they do:

- `scripts/sim_chXX_<name>.py` – simulator (generates synthetic data)
- `scripts/chXX_<name>_analysis.py` – analyzer (fits model, produces stats/plots)
- Makefile targets:
  - `make chXX`
  - `make chXX-ci`
- Output locations:
  - `data/synthetic/...`
  - `outputs/chXX/...`

If there are any additional helpers (e.g., shared utilities or notebooks), list them here.

---

## 3. How to Run It

### 3.1 Quick start

```bash
make chXX
# or the tiny smoke version:
make chXX-ci
```

Explain briefly what each command does (simulates data, runs analysis, writes outputs).

### 3.2 Direct Python usage (optional)

If helpful, show the underlying Python commands:

```bash
python -m scripts.sim_chXX_<name> --seed 123 --outdir data/synthetic
python -m scripts.chXX_<name>_analysis --datadir data/synthetic --outdir outputs/chXX --seed 123
```

Mention any important CLI flags (e.g., `--n-per-group`, `--n-survey`, `--explain`).

---

## 4. Data Generating Process (DGP)

Describe, in teaching language, how the simulator works.

- What population or experiment is being modeled?
- What are the key parameters (means, variances, correlations, probabilities)?
- Which variables are randomized, and which are fixed?
- If there is stratification, clustering, or repeated measures, explain the structure.

If possible, write out the DGP mathematically, e.g.:

- For each subject *i*, generate a latent score:  
  *Zᵢ ~ N(0, 1)*  
- Generate item responses with a threshold model or linear model  
- Add measurement error with a specific variance

This section is where instructors and advanced students can see how the simulation connects to theory.

---

## 5. Statistical Model & Methods

Describe the statistical model(s) used in the analyzer:

- What test/model is fit? (t-test, ANOVA, mixed model, regression, ICC, etc.)
- What is the estimand? (mean difference, risk ratio, correlation, alpha, etc.)
- What assumptions are being made?
- How are confidence intervals or p-values computed?

If helpful, include model equations, e.g.:

- *Yᵢ = β₀ + β₁·Xᵢ + εᵢ*  
- *log(RR) = log(p₁ / p₀)*, etc.

---

## 6. Interpreting the Output

Explain how to read the key outputs:

- Which JSON file(s) are created?
- Which plots are generated (and what to look for)?
- What does a “typical” run suggest about the effect size or reliability?

Consider adding a short “story”:

> “In a typical run with these parameters, the treatment effect is about X units, and the confidence interval often includes/excludes zero…”

---

## 7. Teaching Notes (Optional)

This section is aimed at instructors or self-learners:

- Suggested questions to ask students:
  - “What happens if we double the sample size?”
  - “What happens if we increase variance or make groups more imbalanced?”
- Suggested exercises:
  - Modify the simulator to violate an assumption (e.g., non-normality, heteroskedasticity).
  - Compare results from two different methods (e.g., t-test vs. nonparametric test).

---

## 8. Extensions & Variants

Ideas for future contributions or student projects:

- Add alternative estimators (e.g., robust regression, different ICC types)
- Add Bayesian versions of the analysis
- Add bootstrap intervals
- Extend to a slightly more complex design

If some of these are already implemented, link to the corresponding scripts or docs.

---

## 9. References & Further Reading

Include any relevant references (textbooks, papers, blog posts):

- Author (Year). *Title.* Publisher.
- Useful blog / vignette links
- Open-source code or tutorials that inspired this case study

---

## 10. Version & Maintenance

- **Introduced in:** PyStatsV1 vX.Y.Z
- **Last updated in:** PyStatsV1 vA.B.C
- **Maintainer(s):** GitHub handle(s), if applicable

If the case study is experimental or subject to change, note that here.
