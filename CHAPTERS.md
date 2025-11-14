# CHAPTERS.md — PyStatsV1 Case Study Overview

This document provides a high‑level overview of the statistical “chapters” implemented in PyStatsV1.

Each chapter includes:

- A **simulator** (`scripts/sim_*`)
- An **analyzer** (`scripts/chXX_*`)
- A **Makefile target** (`make chXX`, `make chXX-ci`)
- A role in replicating applied statistics workflows

The chapters intentionally follow a *“simulator → analysis → artifacts”* pattern so students and contributors can learn generalizable structures.

---

## **Chapter 13 — Intro to Experimental Psychology & Fitness Statistics**
### **13A: Stroop Task (Within‑Subjects)**
**Concepts:**  
- Paired differences  
- Repeated‑measures design  
- Mixed modeling intro  
- Reaction time data

**Files:**  
- `scripts/sim_stroop.py`  
- `scripts/ch13_stroop_within.py`

**Run:**  
```bash
make ch13
make ch13-ci
```

---

### **13B: 2×2 Fitness Study (Mixed Design)**
**Concepts:**  
- Factorial designs  
- Mixed-effects models  
- Confidence intervals & effect sizes  
- Long-format data

**Files:**  
- `scripts/sim_fitness_2x2.py`  
- `scripts/ch13_fitness_mixed.py`

**Run:**  
```bash
make ch13
make ch13-ci
```

---

## **Chapter 14 — A/B Tutoring Experiment (Two‑Sample t‑Test)**
**Concepts:**  
- Independent‑samples t-test  
- Random assignment  
- Group means and confidence intervals  
- Power from sample size

**Files:**  
- `scripts/sim_ch14_tutoring.py`  
- `scripts/ch14_tutoring_ab.py`

**Run:**  
```bash
make ch14
make ch14-ci
```

**Planned enhancements (v0.17+):**  
- 🔍 “Explain Mode” showing calculation steps  
- 📈 Richer summary and visualization

---

## **Chapter 15 — Reliability & Psychometrics**
**Concepts:**  
- Cronbach’s Alpha (internal consistency)  
- ICC (test‑retest reliability)  
- Bland–Altman plots  
- Multivariate normal simulation

**Files:**  
- `scripts/sim_ch15_reliability.py`  
- `scripts/ch15_reliability_analysis.py`

**Run:**  
```bash
make ch15
make ch15-ci
```

**Planned enhancements:**  
- Item–total correlation table  
- Alpha variants (standardized, dropped‑item α)  
- Optional factor‑analytic visualization

---

## Future Chapters (in ROADMAP.md)
- Epidemiology “Risk Ratio with Strata” simulator + analyzer  
- Power analysis modules  
- Confidence interval bootstrapping  
- Regression diagnostics  
- GLMs for count data  
- Bayesian re‑implementations

---

If you'd like to contribute to any chapter, see:  
👉 **[CONTRIBUTING.md](CONTRIBUTING.md)**  
