# Security Policy

## Supported Versions

PyStatsV1 is a small, teaching-focused open source project. We do not maintain multiple long-term support branches.

At any given time, **only the `main` branch** (and the latest tagged release) is considered supported for security-related fixes.

## Reporting a Vulnerability

If you believe you have found a security issue in PyStatsV1 (for example, something that could:

- execute arbitrary code unexpectedly,
- leak sensitive information from local files or environment variables, or
- be exploited when running example scripts on untrusted data),

please **do not** open a public GitHub issue immediately.

Instead, contact the maintainer privately at:

**nicholaskarlson@gmail.com**

Please include:

- A short description of the issue
- Steps to reproduce, if possible
- Any relevant environment information (OS, Python version)

You can optionally include a minimal example or patch suggestion.

We will:

1. Acknowledge receipt of your report.
2. Investigate the issue.
3. Decide whether a fix is required.
4. Prepare a patch and coordinate disclosure if appropriate.

Because PyStatsV1 is not a networked service and does not process untrusted input by default, many issues will be low risk; however, we still appreciate careful reports.

## Public Disclosure

Once a fix is available (if required), we may:

- Tag a new release,
- Note the fix in the release notes / changelog, and
- Optionally open a public issue describing the impact at a high level.

If you would like credit for your discovery, please indicate how you would like to be acknowledged (or if you prefer to remain anonymous).
