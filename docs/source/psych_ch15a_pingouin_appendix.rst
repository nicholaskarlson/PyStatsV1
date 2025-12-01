Chapter 15 Appendix – Pingouin for Correlation and Partial Correlation
=====================================================================

In :doc:`psych_ch15_correlation`, you learned how to:

* define and interpret Pearson's correlation coefficient :math:`r`,
* visualize relationships with scatterplots and heatmaps,
* compute correlations using both NumPy and :mod:`pingouin`, and
* run a single partial correlation (exam score ~ study hours | stress).

In that main chapter, the focus was on the *concepts* of correlation and
partial correlation.  This appendix shifts the emphasis to the
:mod:`pingouin` library itself.  We will treat Pingouin as a compact
“stats workbench” that can:

* compute all pairwise correlations (and effect sizes) for a whole set
  of variables at once,
* correct p-values for multiple comparisons,
* and estimate partial correlations while adjusting for one or more
  covariates.

As before, all examples are reproducible with PyStatsV1 and use synthetic
datasets so that you can freely experiment without privacy concerns.

Why this appendix?
------------------

In introductory courses it is common to present correlation as a single
number between two variables.  In real research, we rarely look at just
one pair.  A typical psychology study might collect ten or more measures
(stress, sleep, anxiety, mood, study hours, exam score, etc.).  The
interesting questions are then:

* which variables are most strongly related,
* whether those relationships survive correction for multiple testing,
* and whether an association remains after we statistically control for
  one or more third variables.

Doing all of this by hand, or with low-level functions, is tedious and
error-prone.  :mod:`pingouin` provides higher-level helpers that match how
researchers actually work.  In this appendix we highlight two of them:

* :func:`pingouin.pairwise_corr` – compute all pairwise correlations in
  one shot (with effect sizes, confidence intervals, p-values, and optional
  p-value correction); and
* :func:`pingouin.partial_corr` – compute partial correlations while
  adjusting for one or more covariates.

All examples below assume that Pingouin is installed and that you have
completed the Chapter 15 lab at least once.

A quick reminder: installing Pingouin
-------------------------------------

If you are working on your own machine (rather than the course server),
you can install or update Pingouin as follows::

    pip install --upgrade pingouin

or, if you use Conda::

    conda install -c conda-forge pingouin

For details, see the official documentation at https://pingouin-stats.org.

Pairwise correlations with :func:`pingouin.pairwise_corr`
---------------------------------------------------------

In the Chapter 15 lab (:mod:`scripts.psych_ch15_correlation`) we created a
synthetic dataset with several variables:

* ``stress``
* ``sleep_hours``
* ``anxiety``
* ``study_hours``
* ``exam_score``

To compute *all* pairwise Pearson correlations among these variables using
Pingouin, we can write:

.. code-block:: python

    import pingouin as pg

    from scripts.psych_ch15_correlation import simulate_psych_correlation_dataset

    df = simulate_psych_correlation_dataset(n=200, random_state=456)

    pairwise = pg.pairwise_corr(
        data=df,
        columns=df.columns,
        method="pearson",
        padjust="none",   # or "fdr_bh", "bonf", ...
    )

    print(pairwise.head())

The resulting :class:`pandas.DataFrame` has one row per *unique* variable
pair and includes:

* ``X`` and ``Y`` – the variable names,
* ``r`` – Pearson's correlation coefficient,
* ``CI95%`` – a 95% confidence interval for :math:`r`,
* ``p-unc`` – the uncorrected p-value,
* ``BF10`` – an optional Bayes Factor,
* and several other useful columns.

Because each pair appears only once (e.g., ``stress``–``exam_score`` but
not also ``exam_score``–``stress``), the number of rows is:

.. math::

    \text{n_pairs} = \frac{k(k-1)}{2},

where :math:`k` is the number of variables.

Correcting for multiple comparisons
-----------------------------------

When you compute many correlations at once, some may look “significant”
purely by chance.  Pingouin helps you control the family-wise error rate
or the false discovery rate by adjusting p-values.

For example, to apply the Benjamini–Hochberg false discovery rate (FDR)
correction, use ``padjust="fdr_bh"``:

.. code-block:: python

    pairwise_fdr = pg.pairwise_corr(
        data=df,
        columns=df.columns,
        method="pearson",
        padjust="fdr_bh",
    )

The output now includes a ``p-adjust`` column with corrected p-values.
In this course we mostly treat the corrected p-values as “advanced tools”
for research projects, but it is important for students to see that the
option exists and is easy to use.

Spearman correlations
---------------------

Sometimes you may not want to assume a strictly linear relationship or you
might worry about outliers.  In those cases, Spearman's rank correlation
can be more robust.  Switching methods is as simple as:

.. code-block:: python

    pairwise_spearman = pg.pairwise_corr(
        data=df,
        columns=df.columns,
        method="spearman",
        padjust="fdr_bh",
    )

You can then compare Pearson and Spearman estimates for the same pair of
variables to see whether outliers or non-linearity are having a large
impact.

Partial correlations with :func:`pingouin.partial_corr`
-------------------------------------------------------

In the main chapter we computed a single partial correlation between
``study_hours`` and ``exam_score`` while controlling for ``stress``.
Pingouin makes it easy to extend this idea to multiple covariates.

The basic usage is:

.. code-block:: python

    import pingouin as pg

    df = simulate_psych_correlation_dataset(n=200, random_state=456)

    partial = pg.partial_corr(
        data=df,
        x="study_hours",
        y="exam_score",
        covar=["stress"],     # one or more covariates
        method="pearson",
    )

    print(partial)

The result again is a one-row :class:`pandas.DataFrame` with columns:

* ``r`` – the partial correlation,
* ``CI95%`` – a confidence interval for :math:`r`,
* ``p-val`` – the p-value,
* plus the sample size ``n``.

Controlling for *multiple* covariates is just as easy:

.. code-block:: python

    partial_two = pg.partial_corr(
        data=df,
        x="study_hours",
        y="exam_score",
        covar=["stress", "anxiety"],
        method="pearson",
    )

In a research context, partial correlations are especially useful when
trying to decide whether a relationship is likely to be “direct” or
whether it can be explained away by a third (or fourth) variable.

PyStatsV1 demo scripts for Chapter 15a
--------------------------------------

To keep the main Chapter 15 lab focused, the PyStatsV1 repository includes
two small helper scripts that live in this appendix:

* :mod:`scripts.psych_ch15a_pingouin_pairwise_demo`

  Shows how to:

  * generate the Chapter 15 synthetic dataset,
  * compute all pairwise Pearson and Spearman correlations with
    :func:`pingouin.pairwise_corr`,
  * apply FDR correction to the p-values,
  * and save the resulting tables to ``outputs/track_b``.

* :mod:`scripts.psych_ch15a_pingouin_partial_demo`

  Shows how to:

  * compare a zero-order correlation with one or more partial correlations,
  * control for multiple covariates at once,
  * and summarize the results in a compact table.

You can run these scripts from the command line (inside your PyStatsV1
virtual environment) using:

.. code-block:: bash

    python -m scripts.psych_ch15a_pingouin_pairwise_demo
    python -m scripts.psych_ch15a_pingouin_partial_demo

or, if you prefer the Makefile shortcuts (once they have been added):

.. code-block:: bash

    make psych-ch15a

Unit tests for Chapter 15a
--------------------------

To make sure that the demos behave as expected, we include two small test
files:

* :mod:`tests.test_psych_ch15a_pingouin_pairwise_demo`
* :mod:`tests.test_psych_ch15a_pingouin_partial_demo`

The tests do not check every value.  Instead, they verify structural and
conceptual properties such as:

* :func:`pingouin.pairwise_corr` returns the expected number of pairs,
* the sign and approximate strength of the correlation between ``stress``
  and ``exam_score`` match the design of the synthetic dataset,
* partial correlations shrink (but do not reverse) the positive
  association between ``study_hours`` and ``exam_score`` when we control
  for ``stress``.

You can run just these tests with:

.. code-block:: bash

    pytest tests/test_psych_ch15a_pingouin_pairwise_demo.py
    pytest tests/test_psych_ch15a_pingouin_partial_demo.py

or run the full Track B test suite with:

.. code-block:: bash

    pytest

Suggested student exercises
---------------------------

1. Add a new variable to the Chapter 15 synthetic dataset (for example,
   ``social_support``) that is negatively related to ``stress`` and
   positively related to ``sleep_hours`` and ``exam_score``.  Re-run the
   pairwise correlation demo and interpret the changes in the correlation
   matrix.

2. Use :func:`pingouin.pairwise_corr` with ``method="spearman"`` and
   compare the results to the Pearson correlations.  Are any pairs
   sensitive to outliers or non-linearity?

3. Choose a pair of variables where you suspect a third variable might
   explain part of the relationship (for example, ``stress`` and
   ``exam_score`` with ``sleep_hours`` as a covariate).  Compute zero-order
   and partial correlations and compare the results.

4. For your own research project, design a small correlation study with at
   least five variables.  Use PyStatsV1 and Pingouin to:

   * compute all pairwise correlations,
   * adjust for multiple comparisons,
   * and report at least one partial correlation in APA style.

This appendix is meant as a bridge between the introductory correlation
chapter and more advanced courses in multivariate statistics.  The goal is
not to memorize every option of Pingouin, but to develop a habit of using
well-tested tools to explore relationships among multiple psychological
variables in a principled way.
