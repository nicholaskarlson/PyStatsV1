# Support

Welcome! 👋

PyStatsV1 is a teaching-oriented statistics codebase. This document explains how to get help, ask questions, and report problems.

---

## 1. I have a question about how to run something

First, please check:

- `README.md` – overview of the project and basic usage
- `CONTRIBUTING.md` – detailed setup, `make` targets, and development workflow
- `CHAPTERS.md` – summary of the existing chapter case studies

If your question is still not answered, please open a **GitHub Issue** with:

- A clear title, e.g. “How do I run the Ch15 reliability example on Windows?”
- The exact command(s) you ran
- Your OS and Python version
- Any error messages, copied as text

---

## 2. I think I found a bug

Please open a **Bug Report issue** using the “Bug report” template, and include:

- Steps to reproduce (commands, parameters)
- Expected behavior
- Actual behavior (with full error text)
- Output from `make lint` and/or `make test`, if relevant

This helps us quickly reproduce and fix the problem.

---

## 3. I want to request a new feature or chapter

Great! 🎓

Use the **Feature request** issue template, and describe:

- The statistical idea or method (e.g., “Epidemiology RR with strata,” “Power analysis for t-test”)
- Why it’s useful from a teaching perspective
- Any references (textbook chapter, paper, blog post) if you have them

We try to keep new examples small, well-scoped, and educational.

---

## 4. I want to contribute code or documentation

Please read:

- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`

Then:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run:

   ```bash
   make lint
   make test
   ```

5. Open a Pull Request using the PR template

We especially welcome:

- New chapters that follow the existing simulator/analyzer pattern
- “Explain mode” enhancements that make scripts more educational
- Documentation improvements and small bug fixes

---

## 5. Response Expectations

PyStatsV1 is maintained on a best-effort basis. Response times will vary, but we aim to:

- Acknowledge new issues and PRs in a reasonable time
- Provide clear feedback on proposed changes
- Keep the main branch stable and CI-green

If your question is time-sensitive, please mention that in the issue description.

---

Thank you for using and contributing to PyStatsV1! 🚗📊✨
