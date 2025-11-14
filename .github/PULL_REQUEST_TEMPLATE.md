## Summary

Briefly describe what this PR does in one or two sentences.

Example:  
> Adds a new Chapter 15 reliability case study (Cronbach’s alpha & ICC) with simulator, analyzer, and CI smoke test.

---

## Changes

Check all that apply:

- [ ] New chapter / case study
- [ ] Changes to an existing simulator
- [ ] Changes to an existing analyzer
- [ ] New or updated plots / figures
- [ ] Tests added or updated
- [ ] Documentation updates (README / CONTRIBUTING / CHAPTERS / ROADMAP)

Describe the main changes:

- …
- …
- …

---

## How to Test

List commands a reviewer can run to verify the changes.

```bash
make lint
make test

# If applicable:
make ch13-ci
make ch14-ci
make ch15-ci
# or:
python -m scripts.sim_chXX_example --outdir data/synthetic --seed 123
python -m scripts.chXX_example_analysis --datadir data/synthetic --outdir outputs/chXX --seed 123
```

Mention expected outputs (plots, JSON files) and where they appear.

---

## Checklist

- [ ] I have read and followed `CONTRIBUTING.md`.
- [ ] `make lint` passes locally.
- [ ] `make test` passes locally.
- [ ] I did **not** commit generated data in `data/synthetic/` or plots in `outputs/`.
- [ ] Documentation has been updated if needed (README / CHAPTERS / ROADMAP).

---

## Related Issues

Link any issues this PR addresses:

- Closes #…
- Related to #…

---

## Additional Notes

Anything else that would help reviewers? (Design decisions, trade-offs, future follow-ups.)
