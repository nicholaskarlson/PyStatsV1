.. _applied_stats_with_python_ch13_model_diagnostics:

Applied Statistics with Python – Chapter 13
===========================================

Model diagnostics for regression
--------------------------------

This chapter parallels the *Model Diagnostics* chapter from the R notes, but
uses a Python–first workflow. The statistical ideas are the same: we fit a
regression model, then carefully check whether its assumptions are reasonable
before trusting p-values, confidence intervals, or predictions. :contentReference[oaicite:0]{index=0}

By the end of this chapter (R + Python versions), you should be able to:

* state the core assumptions of a linear regression model,
* diagnose violations of these assumptions using residual plots and tests,
* understand leverage, outliers, and influential points,
* compute and interpret standardized residuals and Cook’s distance,
* and know what to do next when diagnostics look “off”.

13.1 Regression model assumptions (recap)
----------------------------------------

We work with the multiple linear regression model

.. math::

   Y_i = \beta_0 + \beta_1 x_{i1} + \dots + \beta_{p-1} x_{i,p-1} + \varepsilon_i,
   \qquad i = 1,\dots,n,

or in matrix form

.. math::

   \mathbf{Y} = X \beta + \varepsilon.

The least–squares estimator is

.. math::

   \hat\beta = (X^\top X)^{-1} X^\top y.

The *assumptions* live in the error term :math:`\varepsilon_i`:

* **Linearity** – the mean of :math:`Y` is a linear function of the predictors.
* **Independence** – errors :math:`\varepsilon_i` are independent.
* **Normality** – errors are Normally distributed.
* **Equal variance** – errors have constant variance :math:`\sigma^2`
  for all combinations of predictors (homoscedasticity). :contentReference[oaicite:1]{index=1}

If these assumptions hold, our familiar t–tests, F–tests, and confidence
intervals are valid. If they fail badly, we can still *compute* them, but the
results are not trustworthy.

In Python, these assumptions underlie models such as
:class:`statsmodels.api.OLS` and :func:`statsmodels.formula.api.ols`.

13.2 Checking assumptions in Python
-----------------------------------

In this section we mirror the R diagnostic tools using NumPy, pandas,
Matplotlib, SciPy, and statsmodels.

Throughout, imagine we have already fit a regression model

.. code-block:: python

   import statsmodels.formula.api as smf

   model = smf.ols("y ~ x1 + x2", data=df).fit()

   fitted = model.fittedvalues
   resid  = model.resid

13.2.1 Fitted versus residuals plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A **fitted vs residuals** plot (sometimes called *residuals vs fitted*) is our
workhorse diagnostic.

* x-axis: fitted values :math:`\hat y_i`
* y-axis: residuals :math:`e_i = y_i - \hat y_i`

In Python:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots()
   ax.scatter(fitted, resid, alpha=0.6)
   ax.axhline(0, color="black", linewidth=1)
   ax.set_xlabel("Fitted values")
   ax.set_ylabel("Residuals")
   ax.set_title("Fitted vs residuals")

What to look for:

* **Linearity**

  * At each fitted value, residuals should be centered around 0.
  * A clear curve (e.g. U-shape) suggests the *form* of the model is wrong
    (missing polynomial terms, interactions, or other nonlinear structure).

* **Equal variance**

  * Spread of residuals should be roughly constant along the x-axis.
  * “Funnel” shapes (narrow then wide) suggest heteroscedasticity
    (non-constant variance). :contentReference[oaicite:2]{index=2}

For simple linear regression you might see problems directly in the
:math:`(x, y)` scatterplot, but for multiple regression this fitted–vs–residuals
plot is essential.

13.2.2 Breusch–Pagan test for constant variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Breusch–Pagan test provides a formal check of homoscedasticity:

* :math:`H_0`: error variance is constant (homoscedastic).
* :math:`H_1`: error variance depends on the predictors (heteroscedastic). :contentReference[oaicite:3]{index=3}

In Python (statsmodels):

.. code-block:: python

   from statsmodels.stats.diagnostic import het_breuschpagan

   bp_stat, bp_pvalue, _, _ = het_breuschpagan(
       model.resid,
       model.model.exog,   # design matrix X
   )

   print(f"BP statistic = {bp_stat:.3f}, p-value = {bp_pvalue:.3g}")

Interpretation (typical conventions):

* **Large p-value** (say > 0.05): no strong evidence against constant variance.
* **Small p-value**: evidence that variance changes with the predictors;
  consider transformations, adding missing structure, or using robust
  standard errors.

13.2.3 Histograms of residuals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To check the normality assumption, a simple starting point is a histogram of
residuals:

.. code-block:: python

   fig, ax = plt.subplots()
   ax.hist(resid, bins=20, edgecolor="black")
   ax.set_xlabel("Residuals")
   ax.set_ylabel("Frequency")
   ax.set_title("Histogram of residuals")

You are looking for a roughly symmetric, bell-shaped distribution. However,
histograms can be ambiguous (especially with small samples), so we usually
follow up with Q–Q plots and formal tests. :contentReference[oaicite:4]{index=4}

13.2.4 Normal Q–Q plots
~~~~~~~~~~~~~~~~~~~~~~~

A **Normal Q–Q plot** (quantile–quantile plot) compares the sorted residuals
to what we would expect if they were sampled from a Normal distribution.

In Python, we can use SciPy or statsmodels:

.. code-block:: python

   import statsmodels.api as sm

   fig = sm.qqplot(resid, line="45")
   plt.title("Normal Q–Q plot of residuals")

Guidelines:

* Points close to the line → residuals are plausibly Normal.
* Systematic curvature (e.g. S-shape) or heavy tails (points far from the
  line at extremes) → Normality assumption is questionable.
* With small *n*, random noise in the plot is expected; with large *n*, even
  small deviations become visible.

13.2.5 Shapiro–Wilk normality test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Shapiro–Wilk test** is a widely used test of Normality. :contentReference[oaicite:5]{index=5}

In Python (SciPy):

.. code-block:: python

   from scipy.stats import shapiro

   W_stat, pvalue = shapiro(resid)
   print(f"Shapiro–Wilk W = {W_stat:.3f}, p-value = {pvalue:.3g}")

Interpretation:

* :math:`H_0`: the data are sampled from a Normal distribution.
* Small p-value → residuals are unlikely to be Normal.
* Large p-value → no strong evidence against Normality.

As always, combine this with visual tools (histogram, Q–Q plot); a test alone
does not tell the whole story.

13.3 Unusual observations: leverage, outliers, influence
--------------------------------------------------------

Diagnostics are not only about assumptions; we also care about **unusual data
points** that can distort a regression:

* **High leverage** points – unusual predictor values (extreme in :math:`X`).
* **Outliers** – points with large residuals (poorly fit by the model).
* **Influential** points – observations that substantially change the fitted
  model when removed. :contentReference[oaicite:6]{index=6}

13.3.1 Leverage and the hat matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recall the fitted values

.. math::

   \hat y = X \hat\beta
          = X (X^\top X)^{-1} X^\top y
          = H y,

where

.. math::

   H = X (X^\top X)^{-1} X^\top

is the **hat matrix**. Its diagonal entries :math:`h_i` are the leverage values:

.. math::

   h_i = H_{ii}, \qquad i = 1,\dots,n.

Properties:

* :math:`0 \le h_i \le 1`,
* :math:`\sum_i h_i = p`, the number of regression parameters.

Heuristic:

* Average leverage :math:`\bar h = p / n`.
* Points with :math:`h_i > 2\bar h` are often flagged as **high leverage**.

In Python, statsmodels makes this easy:

.. code-block:: python

   influence = model.get_influence()
   leverage  = influence.hat_matrix_diag

   avg_h = leverage.mean()
   high_lev = leverage > 2 * avg_h
   df.loc[high_lev, :]

13.3.2 Standardized residuals and outliers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raw residuals have different variances for different :math:`h_i`. Under the
model assumptions we have

.. math::

   \operatorname{Var}(e_i) = (1 - h_i)\,\sigma^2.

We therefore look at **standardized (studentized) residuals**

.. math::

   r_i = \frac{e_i}{s_e \sqrt{1 - h_i}},

which are approximately :math:`N(0,1)` when the model is correct. :contentReference[oaicite:7]{index=7}

Rule of thumb:

* |r\_i| > 2 → potentially an outlier in the regression sense.
* |r\_i| > 3 → very suspicious.

In Python:

.. code-block:: python

   std_resid = influence.resid_studentized_internal
   outliers = abs(std_resid) > 2
   df.loc[outliers, :]

Remember: an outlier is defined *relative to the model*. A point may be
perfectly reasonable in the original data scale but still be an outlier for a
particular regression.

13.3.3 Cook’s distance: measuring influence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cook’s distance** combines leverage and residual size into a single measure
of how much each point influences the fitted model. :contentReference[oaicite:8]{index=8}

Heuristic rule:

.. math::

   D_i > \frac{4}{n} \quad \text{→ observation } i \text{ is influential}.

In Python:

.. code-block:: python

   cooks_d, _ = influence.cooks_distance
   influential = cooks_d > 4 / len(cooks_d)

   df.loc[influential, :]

   # Optionally, sort by Cook's distance
   df.assign(cooks_d=cooks_d).sort_values("cooks_d", ascending=False).head()

Influential points are not automatically “bad”, but they deserve extra scrutiny:
they may be data entry errors, unusual cases that require a different model, or
scientifically interesting exceptions.

13.4 Examples in Python
-----------------------

Here we sketch how the textbook examples translate into Python. The exact
datasets and scripts live in the PyStatsV1 repository.

13.4.1 Example: `mtcars` additive model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the model

.. math::

   \text{mpg} = \beta_0 + \beta_1 \text{hp} + \beta_2 \text{am} + \varepsilon,

where ``hp`` is horsepower and ``am`` is a 0/1 indicator for manual
transmission.

In Python:

.. code-block:: python

   import pandas as pd
   import statsmodels.formula.api as smf
   import statsmodels.api as sm

   mtcars = pd.read_csv("data/mtcars.csv")  # or similar helper in PyStatsV1
   mpg_hp_add = smf.ols("mpg ~ hp + am", data=mtcars).fit()

   print(mpg_hp_add.summary())

Diagnostics:

.. code-block:: python

   influence = mpg_hp_add.get_influence()
   leverage  = influence.hat_matrix_diag
   std_resid = influence.resid_studentized_internal
   cooks_d, _ = influence.cooks_distance

   # How many high leverage points?
   high_lev = leverage > 2 * leverage.mean()
   print("High leverage:", high_lev.sum())

   # How many large residuals?
   large_resid = abs(std_resid) > 2
   print("Large standardized residuals:", large_resid.sum())

   # Influential points by Cook's distance
   influential = cooks_d > 4 / len(cooks_d)
   print("Influential points:", influential.sum())

The R version labels specific cars; in Python you can inspect those rows by
index and refit the model excluding them to see how much the coefficients
change.

13.4.2 Example: a large interaction model for Auto MPG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The R notes end with a “big” interaction model for an Auto MPG dataset and
show that diagnostics can look poor when the model is over-complex or the
data contain many influential points. :contentReference[oaicite:9]{index=9}

In Python the pattern is the same:

.. code-block:: python

   autompg = pd.read_csv("data/autompg.csv")

   big_model = smf.ols(
       "mpg ~ disp * hp * domestic",
       data=autompg,
   ).fit()

   sm.qqplot(big_model.resid, line="45")
   plt.title("Q–Q plot: big_model")
   plt.show()

   # Many influential points?
   infl = big_model.get_influence()
   cooks_d, _ = infl.cooks_distance
   influential = cooks_d > 4 / len(cooks_d)
   print("Number of influential points:", influential.sum())

You can then refit on a subset (for example, excluding the most influential
points) or, better, reconsider the model structure (transformations, simpler
interaction structure, different predictors).

13.5 What you should take away
------------------------------

By the end of this chapter you should be comfortable with:

* stating the assumptions of a linear regression model and where they live
  in :math:`Y = X\beta + \varepsilon`,
* using fitted vs residuals plots to check linearity and constant variance,
* using histograms, Q–Q plots, and the Shapiro–Wilk test to assess Normality,
* running and interpreting the Breusch–Pagan test for heteroscedasticity,
* computing leverage, standardized residuals, and Cook’s distance to detect
  high-leverage, outlying, and influential observations,
* and combining these diagnostics to decide whether your model is reasonable
  or needs to be revised.

In later PyStatsV1 chapters and case studies, these tools will underpin more
advanced modeling (logistic regression, generalized linear models, mixed
effects) and will be part of a standard “checklist” whenever we fit a model to
real data.

If any of the diagnostics here feel abstract, try the following in a Python
shell or notebook:

* simulate data where you *know* a model is correct, and verify that the
  diagnostics look good;
* then deliberately break assumptions (non-constant variance, nonlinear
  relationship, heavy-tailed noise) and see how the plots and tests react;
* finally, apply the same tools to real datasets in PyStatsV1 and compare.

That hands-on experimentation will make the ideas in this chapter stick.

