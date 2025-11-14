# CONTRIBUTING.md
*(PyStatsV1 – Open Statistics Case Studies)*

Welcome! 🎓  
Thank you for your interest in contributing to **PyStatsV1** — an open, teaching-oriented repository that implements real statistical case studies as clean, reproducible Python modules.

This document will show you how the project works, how to get set up locally, how to contribute code, documentation, or examples, and how to open successful pull requests.

PyStatsV1 is intentionally organized like a **factory**:

- Each “chapter” (Ch13, Ch14, Ch15, …) corresponds to a **complete statistical case study**.  
- Each chapter has two core components:  
  1. A **simulator** (`sim_chXX_*.py`) that generates synthetic data  
  2. An **analyzer** (`chXX_*.py`) that performs the statistical analysis  
- Everything is **reproducible**, **seeded**, and **CI-tested**.

If you want to contribute a new “chapter car” or improve an existing one, this guide is for you.

---

# 🔧 **1. Project Goals**

PyStatsV1 aims to:

- Teach **practical statistics** through real, runnable case studies  
- Provide clean, idiomatic, documented **Python implementations**  
- Offer synthetic data generation so learners can run analyses without external datasets  
- Maintain a **rigorous CI pipeline** using Python 3.10 on Windows  
- Keep everything **MIT-licensed, open, and beginner-friendly**

---

# 🏗️ **2. Repository Layout**

```
PyStatsV1/
│
├── scripts/                    # Chapter simulators and analyzers
│   ├── _cli.py                 # Shared CLI helpers (base_parser, apply_seed)
│   ├── sim_ch13_*              # Chapter 13 simulators
│   ├── ch13_*                  # Chapter 13 analyzers
│   ├── sim_ch14_tutoring.py    # Chapter 14 simulator
│   ├── ch14_tutoring_ab.py     # Chapter 14 analyzer
│   ├── sim_ch15_reliability.py # Chapter 15 simulator
│   └── ch15_reliability_analysis.py  # Chapter 15 analyzer
│
├── tests/                      # pytest tests + CLI smoke tests
│   └── test_cli_smoke.py
│
├── data/
│   └── synthetic/              # Generated CSVs (ignored by Git)
│
├── outputs/
│   ├── ch13/ ch14/ ch15/ …     # Generated plots + summaries (ignored by Git)
│
├── Makefile                    # Unified developer interface
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Dev tools (future)
├── README.md                   # Project overview
└── LICENSE                     # MIT
```

---

# 💻 **3. Local Development Setup**

PyStatsV1 supports **Python 3.10**.

### **Step 1: Clone**

```bash
git clone https://github.com/<your-username>/PyStatsV1
cd PyStatsV1
```

### **Step 2: Create & Activate Virtual Environment**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: source .venv/Scripts/activate
```

### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 4: Verify Your Environment**

```bash
make lint
make test
make ch15-ci
```

---

# 🧪 **4. Running the Project**

## **4.1 Full Case Studies**

```bash
make ch13
make ch14
make ch15
```

## **4.2 CI-sized Versions**

```bash
make ch13-ci
make ch14-ci
make ch15-ci
```

## **4.3 Test Suite**

```bash
make test
```

## **4.4 Linting**

```bash
make lint
```

---

# 🧩 **5. Contribution Types**

- New chapters (Ch16+)  
- Fixes / improvements  
- Explain-mode for existing chapters  
- Documentation  
- New plots or statistical metrics  
- Tests  

---

# 🧱 **6. Adding a New Chapter**

Each new case study must include:

### **6.1 Simulator Requirements**

- Use `base_parser()` & `apply_seed()`
- Accept `--seed` and `--outdir`
- Save only to `data/synthetic/`
- Include a JSON metadata file
- Deterministic under fixed seed

### **6.2 Analyzer Requirements**

- Accept `--datadir` or `--data`
- Save outputs (JSON, plots) to `outputs/chXX/`
- Use:

```python
import matplotlib
matplotlib.use("Agg")
```

### **6.3 Makefile Update**

Add:

- `chXX` target  
- `chXX-ci` target  
- Help entry  

### **6.4 Update CLI Smoke Tests**

Add the module names to `tests/test_cli_smoke.py`.

---

# 🔍 **7. Coding Standards**

- Lint with `make lint`
- Use type hints
- Clear docstrings explaining the *purpose*
- Prefer simple, educational code
- Avoid unnecessary dependencies

---

# 🧪 **8. Tests**

All new chapters **must**:

- Be added to CLI smoke tests  
- Produce deterministic outputs under `--seed`  
- Pass `pytest` on Windows/Python 3.10  

Run:

```bash
make test
```

---

# 🔀 **9. Branching & Pull Requests**

### **Branch Naming**

```
feat/ch15-explain-mode
fix/ch14-ci-warning
docs/improve-readme
```

### **Open PRs against `main`**

```bash
git push -u origin <branch>
gh pr create --fill --base main --head <branch>
```

### **PR Requirements**

- Lint passes  
- Tests pass  
- CI green  
- No synthetic data committed  

---

# 📦 **10. Filing Issues**

Use GitHub Issues for:

- Bugs  
- Enhancements  
- New chapter proposals  
- Documentation improvements  

Use labels:

- `bug`
- `enhancement`
- `good first issue`
- `help wanted`

---

# 🤝 **11. Code of Conduct**

Please be respectful.  
See `CODE_OF_CONDUCT.md`.

---

# 🚀 **12. Roadmap**

Planned additions:

- Explain-mode for Ch14 & Ch15  
- Epidemiology: Risk Ratio w/ strata  
- Reliability extensions  
- Additional inference & regression chapters  

---

# 🎉 **Thank You**

PyStatsV1 is an open educational project — your contributions make it meaningful and useful for learners everywhere.  
We’re excited to build new “chapter cars” with you! 🚗📊✨
