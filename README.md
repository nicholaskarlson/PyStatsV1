# PyStatsV1 — Applied Statistics (R ↔ Python)

[![ci](https://img.shields.io/github/actions/workflow/status/nicholaskarlson/PyStatsV1/ci.yml?branch=main)](https://github.com/nicholaskarlson/PyStatsV1/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/tag/nicholaskarlson/PyStatsV1?label=release)](https://github.com/nicholaskarlson/PyStatsV1/tags)

Plain Python scripts that mirror R recipes so non-specialists can run analyses from the command line, save figures/tables, and compare results across R/Python.

---

## Quick start

### macOS / Linux
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### Windows (Git Bash or PowerShell)
```bash
# Try Git Bash first; if that fails, PowerShell will activate the venv
python -m venv .venv; source .venv/Scripts/activate 2>/dev/null || .venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

---

## Example
Run the Chapter 1 example:
```bash
python scripts/ch01_introduction.py
```

### Chapter 13 quick smoke (fast)
```bash
make ch13-ci
```
This generates tiny synthetic datasets and saves a couple of plots to `outputs/`.

Full chapter run:
```bash
make ch13
```

See **[docs/README.md](docs/README.md)** for chapter notes, commands, and links.

---

## License

MIT © 2025 Nicholas Elliott Karlson