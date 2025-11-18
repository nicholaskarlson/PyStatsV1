.. _applied-stats-python-ch8:

Applied Statistics with Python – Chapter 8
==========================================

Inference for simple linear regression
--------------------------------------

In Chapter 7 you met the **simple linear regression (SLR)** model and learned how
to fit a line to data:

.. math::

   Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad
   \varepsilon_i \sim \mathcal{N}(0, \sigma^2), \quad i = 1,\dots,n.

We focused on *estimating* :math:`\beta_0` and :math:`\beta_1` with least squares and
interpreting the fitted line.

In this chapter we ask questions like:

* *How variable are* :math:`\hat\beta_0` *and* :math:`\hat\beta_1` *from sample to sample?*
* *How sure are we about the true slope?*
* *Is there statistically significant evidence that the slope is non-zero?*
* *How do we build confidence intervals and prediction intervals?*

The R notes use ``lm()`` and its printed output. Here we work mostly with
Python’s :mod:`statsmodels` and :mod:`scipy.stats`, but the statistical ideas are
exactly the same.

Throughout, keep the ``cars`` example in mind:

* :math:`Y` – stopping distance (feet),
* :math:`X` – speed (mph).

We’ll use the same fitted model from Chapter 7 to make the ideas concrete.

.. code-block:: python

   import pandas as pd
   import statsmodels.formula.api as smf

   cars = pd.read_csv("data/cars.csv")
   model = smf.ols("dist ~ speed", data=cars).fit()

   print(model.summary())

8.1 Recap: least squares and notation
-------------------------------------

From Chapter 7:

* The **least squares estimates** :math:`\hat\beta_0` and :math:`\hat\beta_1` are the
  intercept and slope that minimize the sum of squared residuals

  .. math::

     \mathrm{RSS}(\beta_0, \beta_1)
     = \sum_{i=1}^n \big(y_i - (\beta_0 + \beta_1 x_i)\big)^2.

* We can write them in terms of centered sums

  .. math::

     S_{xx} = \sum_{i=1}^n (x_i - \bar x)^2, \qquad
     S_{xy} = \sum_{i=1}^n (x_i - \bar x)(y_i - \bar y).

  Then

  .. math::

     \hat\beta_1 = \frac{S_{xy}}{S_{xx}}, \qquad
     \hat\beta_0 = \bar y - \hat\beta_1 \bar x.

* The fitted values and residuals are

  .. math::

     \hat y_i = \hat\beta_0 + \hat\beta_1 x_i, \qquad
     e_i = y_i - \hat y_i.

* The residual standard error (RSE) estimates :math:`\sigma`:

  .. math::

     s_e = \sqrt{\frac{1}{n-2} \sum_{i=1}^n e_i^2}.

In Python, ``model.params`` stores :math:`\hat\beta_0` and :math:`\hat\beta_1`, and
``model.mse_resid`` is :math:`s_e^2`.

.. code-block:: python

   beta0_hat, beta1_hat = model.params
   se = model.mse_resid**0.5
   n = model.nobs

   print(beta0_hat, beta1_hat, se, n)

8.2 Gauss–Markov in plain language (why least squares is “good”)
----------------------------------------------------------------

The **Gauss–Markov theorem** is one of the main theoretical results behind
ordinary least squares.

Under the SLR assumptions (in particular, linear mean and constant variance),
the least squares estimates :math:`\hat\beta_0` and :math:`\hat\beta_1` are:

* **Linear**: they are linear combinations of the responses :math:`Y_i`.
* **Unbiased**: their expectations equal the true parameters,

  .. math::

     \mathbb{E}[\hat\beta_0] = \beta_0, \qquad
     \mathbb{E}[\hat\beta_1] = \beta_1.

* **Best** among all linear unbiased estimators: they have the **smallest
  variance** in that class.

This is often summarized as:

.. centered:: **OLS estimates are BLUE – Best Linear Unbiased Estimators.**

You do *not* need to prove Gauss–Markov to use regression, but you should
remember the practical message:

*If the linear model assumptions are reasonable, least squares is a very
sensible default – you are not throwing away precision by using it.*

8.3 Sampling distributions of :math:`\hat\beta_0` and :math:`\hat\beta_1`
-------------------------------------------------------------------------

Because the errors :math:`\varepsilon_i` are normal and independent, the least
squares estimates themselves are random variables with normal distributions.

Their exact variances (in terms of the unknown :math:`\sigma^2`) are:

.. math::

   \mathrm{Var}(\hat\beta_1) = \frac{\sigma^2}{S_{xx}}, \qquad
   \mathrm{Var}(\hat\beta_0)
   = \sigma^2\left(\frac{1}{n} + \frac{\bar x^2}{S_{xx}}\right).

So, under the model,

.. math::

   \hat\beta_1 \sim \mathcal{N}\!\left(\beta_1,\, \frac{\sigma^2}{S_{xx}}\right), \qquad
   \hat\beta_0 \sim \mathcal{N}\!\left(\beta_0,\, \sigma^2\left(\frac{1}{n} + \frac{\bar x^2}{S_{xx}}\right)\right).

These distributions describe how the slope and intercept *would vary* if you
could repeatedly collect new samples of size :math:`n` from the same population.

A quick simulation check in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can verify this by simulation, just as the R notes do.

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(42)

   n = 100
   x = np.linspace(-1, 1, n)
   Sxx = np.sum((x - x.mean())**2)

   beta0_true = 3.0
   beta1_true = 6.0
   sigma_true = 2.0

   num_samples = 10_000
   beta0_hats = np.empty(num_samples)
   beta1_hats = np.empty(num_samples)

   for i in range(num_samples):
       eps = rng.normal(loc=0.0, scale=sigma_true, size=n)
       y = beta0_true + beta1_true * x + eps

       sim_model = smf.ols("y ~ x", data={"y": y, "x": x}).fit()
       beta0_hats[i] = sim_model.params["Intercept"]
       beta1_hats[i] = sim_model.params["x"]

   print(beta1_hats.mean(), beta1_true)
   print(beta1_hats.var(), sigma_true**2 / Sxx)

The empirical mean and variance of the simulated slopes should be very close to
the theoretical values.

8.4 Standard errors and :math:`t` statistics
--------------------------------------------

In practice we do not know :math:`\sigma^2`, so we replace it with :math:`s_e^2`. This
gives **standard errors** for the estimates:

.. math::

   \mathrm{SE}(\hat\beta_1) = \frac{s_e}{\sqrt{S_{xx}}}, \qquad
   \mathrm{SE}(\hat\beta_0)
   = s_e \sqrt{\frac{1}{n} + \frac{\bar x^2}{S_{xx}}}.

If you standardize with these estimated standard deviations, you get
:math:`t`-distributed statistics:

.. math::

   T_{\beta_1}
   = \frac{\hat\beta_1 - \beta_1}{\mathrm{SE}(\hat\beta_1)}
   \sim t_{n-2}, \qquad
   T_{\beta_0}
   = \frac{\hat\beta_0 - \beta_0}{\mathrm{SE}(\hat\beta_0)}
   \sim t_{n-2}.

This is why regression output uses the :math:`t` distribution with :math:`n-2`
degrees of freedom for tests and intervals.

In Python, ``model.bse`` stores the standard errors:

.. code-block:: python

   se_beta0, se_beta1 = model.bse
   print(se_beta0, se_beta1)

8.5 Confidence intervals for slope and intercept
------------------------------------------------

The generic shape of a confidence interval is

.. math::

   \text{estimate} \;\pm\; (\text{critical value}) \times \text{standard error}.

For the regression coefficients we get

.. math::

   \hat\beta_1 \pm t_{\alpha/2,\, n-2}\,\mathrm{SE}(\hat\beta_1), \qquad
   \hat\beta_0 \pm t_{\alpha/2,\, n-2}\,\mathrm{SE}(\hat\beta_0),

where :math:`t_{\alpha/2,\, n-2}` is a critical value from the :math:`t_{n-2}` distribution.

In Python you can compute these either by hand or via helper methods:

.. code-block:: python

   from scipy import stats

   alpha = 0.01   # 99% CI
   df = int(model.df_resid)
   tcrit = stats.t.ppf(1 - alpha/2, df=df)

   ci_beta1 = (
       beta1_hat - tcrit * se_beta1,
       beta1_hat + tcrit * se_beta1,
   )
   print("99% CI for beta1:", ci_beta1)

Statsmodels also offers a convenience method:

.. code-block:: python

   print(model.conf_int(alpha=0.01))

Interpretation in the ``cars`` example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the slope :math:`\beta_1` (change in mean stopping distance per 1 mph
increase in speed), a 99% CI might look like

.. math::

   [2.82,\ 5.05] \text{ feet per mph}.

We read this as:

*“We are 99% confident that each 1 mph increase in speed is associated with
between about 2.8 and 5.0 additional feet of average stopping distance.”*

Notice that this interval is **entirely above 0**, which is closely tied to the
significance tests in the next section.

8.6 Hypothesis tests for slope and intercept
--------------------------------------------

We often want to test hypotheses such as:

* Is the slope zero? (Is there any linear relationship?)
* Is the intercept equal to some value?

The generic :math:`t` statistic has the form

.. math::

   t
   = \frac{\text{estimate} - \text{hypothesized value}}
          {\text{standard error}}.

Testing whether the slope is zero
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most common regression test is

.. math::

   H_0 : \beta_1 = 0 \quad \text{vs} \quad H_1 : \beta_1 \neq 0.

Under :math:`H_0`, the model reduces to :math:`Y_i = \beta_0 + \varepsilon_i`; the
response does *not* depend linearly on :math:`x`.

The test statistic is

.. math::

   t
   = \frac{\hat\beta_1 - 0}{\mathrm{SE}(\hat\beta_1)}
   \sim t_{n-2}\ \text{ under } H_0.

In Python:

.. code-block:: python

   t_beta1 = beta1_hat / se_beta1
   p_beta1 = 2 * stats.t.sf(abs(t_beta1), df=df)
   print(t_beta1, p_beta1)

The same values appear in ``model.summary()`` in the row for ``speed``:
``coef``, ``std err``, ``t``, and ``P>|t|``.

If the p-value is very small (for the ``cars`` data it is tiny), we reject
:math:`H_0` and conclude there is a statistically significant linear relationship
between speed and stopping distance.

8.7 The ``cars`` example in Python
----------------------------------

Here is a compact version of the “coefficients table” extraction that parallels
the R code in the original notes:

.. code-block:: python

   coefs = model.summary2().tables[1]
   print(coefs)

   beta0_hat = coefs.loc["Intercept", "Coef."]
   se_beta0  = coefs.loc["Intercept", "Std.Err."]
   t_beta0   = coefs.loc["Intercept", "t"]
   p_beta0   = coefs.loc["Intercept", "P>|t|"]

   beta1_hat = coefs.loc["speed", "Coef."]
   se_beta1  = coefs.loc["speed", "Std.Err."]
   t_beta1   = coefs.loc["speed", "t"]
   p_beta1   = coefs.loc["speed", "P>|t|"]

   print(beta1_hat, se_beta1, t_beta1, p_beta1)

This mirrors the R output:

* ``Estimate``  → ``Coef.``
* ``Std. Error`` → ``Std.Err.``
* ``t value`` → ``t``
* ``Pr(>|t|)`` → ``P>|t|``

8.8 Confidence intervals for mean response
------------------------------------------

Sometimes we want an interval for the **mean response** at a given predictor
value :math:`x_0`:

.. math::

   \mu(x_0) = \mathbb{E}[Y \mid X = x_0] = \beta_0 + \beta_1 x_0.

Our point estimate is :math:`\hat y(x_0) = \hat\beta_0 + \hat\beta_1 x_0`. Its variance
(and therefore its standard error) accounts for uncertainty in the fitted line:

.. math::

   \mathrm{SE}\big(\hat y(x_0)\big)
   = s_e \sqrt{ \frac{1}{n} + \frac{(x_0 - \bar x)^2}{S_{xx}} }.

A :math:`(1-\alpha)\times 100\%` confidence interval for the mean response at
:math:`x_0` is

.. math::

   \hat y(x_0) \pm t_{\alpha/2,\,n-2}\,
   \mathrm{SE}\big(\hat y(x_0)\big).

In statsmodels you can obtain these with ``get_prediction(..., obs=False)`` or
by specifying ``"confidence"`` in the original R workflow:

.. code-block:: python

   new_speeds = pd.DataFrame({"speed": [5, 21]})

   mean_pred = model.get_prediction(new_speeds)
   print(mean_pred.summary_frame(alpha=0.01))  # 99% CI

The output includes columns like ``mean``, ``mean_ci_lower``, and
``mean_ci_upper``.

8.9 Prediction intervals for new observations
---------------------------------------------

A **prediction interval** describes the likely range of a *new individual*
observation :math:`Y_\text{new}` at :math:`x_0`.

There are two sources of variability:

1. Uncertainty in the fitted line (same as the mean response case).
2. The random noise :math:`\varepsilon` around the line.

This adds an extra :math:`\sigma^2` term inside the square root:

.. math::

   \mathrm{SE}_\text{pred}(x_0)
   = s_e \sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar x)^2}{S_{xx}}}.

The prediction interval has the same basic form but is *wider*:

.. math::

   \hat y(x_0) \pm t_{\alpha/2,\,n-2}\,
   \mathrm{SE}_\text{pred}(x_0).

In statsmodels you can request prediction intervals via
``get_prediction(...).summary_frame()`` and use the ``obs_ci_*`` columns:

.. code-block:: python

   pred = model.get_prediction(new_speeds)
   frame = pred.summary_frame(alpha=0.01)

   print(frame[["mean", "mean_ci_lower", "mean_ci_upper",
                "obs_ci_lower", "obs_ci_upper"]])

Compare the **mean** intervals with the **observation** intervals: the latter are
always wider.

8.10 Confidence and prediction bands
------------------------------------

Instead of intervals at a single :math:`x_0`, we can compute intervals across a
grid of :math:`x` values to form **bands**.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   speed_grid = np.linspace(cars["speed"].min(),
                            cars["speed"].max(), 200)
   grid_df = pd.DataFrame({"speed": speed_grid})

   pred_grid = model.get_prediction(grid_df).summary_frame(alpha=0.01)

   mean_lwr = pred_grid["mean_ci_lower"]
   mean_upr = pred_grid["mean_ci_upper"]
   obs_lwr  = pred_grid["obs_ci_lower"]
   obs_upr  = pred_grid["obs_ci_upper"]

   plt.scatter(cars["speed"], cars["dist"], alpha=0.5, label="Data")
   plt.plot(speed_grid, pred_grid["mean"], label="Fitted line")
   plt.plot(speed_grid, mean_lwr, "--", label="99% mean CI")
   plt.plot(speed_grid, mean_upr, "--")
   plt.plot(speed_grid, obs_lwr, ":", label="99% prediction band")
   plt.plot(speed_grid, obs_upr, ":")
   plt.xlabel("Speed (mph)")
   plt.ylabel("Stopping distance (ft)")
   plt.legend()
   plt.show()

Things to notice (mirroring the R version):

* The bands are narrowest near :math:`\bar x` (the mean speed).
* Prediction bands are wider than confidence bands.
* Both bands flare out toward the extremes of the :math:`x` range.

8.11 F-test and ANOVA: another view of “significance of regression”
-------------------------------------------------------------------

In simple linear regression there are two mathematically equivalent ways to
test the significance of the slope:

* A **:math:`t`-test** on :math:`\beta_1 = 0`.
* An **:math:`F`-test** for the overall regression.

The :math:`F`-test uses the decomposition of total variability:

.. math::

   \underbrace{\sum_{i=1}^n (y_i - \bar y)^2}_{\text{SST}}
   =
   \underbrace{\sum_{i=1}^n (y_i - \hat y_i)^2}_{\text{SSE}}
   +
   \underbrace{\sum_{i=1}^n (\hat y_i - \bar y)^2}_{\text{SSReg}}.

The **:math:`F` statistic** is

.. math::

   F
   = \frac{\text{SSReg}/1}{\text{SSE}/(n-2)}
   \sim F_{1,\,n-2} \quad \text{under } H_0 : \beta_1 = 0.

In simple linear regression,

.. math::

   F = t^2,

where :math:`t` is the :math:`t` statistic for the slope test. So the :math:`p`-values
from the :math:`t`-test and the :math:`F`-test always agree.

In statsmodels you can see this in two ways:

1. In ``model.summary()``, the bottom of the table shows the :math:`F`-statistic
   and its :math:`p`-value.

2. You can construct an ANOVA table:

.. code-block:: python

   import statsmodels.api as sm

   anova_table = sm.stats.anova_lm(model, typ=1)
   print(anova_table)

You will see rows for ``speed`` and ``Residual``, with the familiar “sum of
squares”, “mean square”, ``F`` and ``PR(>F)`` columns.

8.12 What you should take away
------------------------------

By the end of this chapter you should be comfortable with:

* How least squares estimates behave as random variables
  (their **sampling distributions**).
* How to compute **standard errors** for the slope and intercept.
* How to build **confidence intervals** for :math:`\beta_0` and :math:`\beta_1`.
* How to run and interpret:
  
  * :math:`t`-tests for individual coefficients,
  * the “significance of regression” test (:math:`H_0 : \beta_1 = 0`),
  * and the equivalent :math:`F`-test / ANOVA view.

* The difference between:

  * **Confidence intervals for a mean response** at a given :math:`x`,
  * **Prediction intervals for a new observation** at that :math:`x`.

* How to construct **bands** over a grid of :math:`x` values to visualize
  uncertainty around the fitted line.

In later PyStatsV1 chapters, these tools will appear repeatedly:

* to compare different models on the same data,
* to quantify uncertainty in estimated effects,
* and to connect simulation results back to theoretical distributions.

If any of the formulas feel abstract, revisit the ``cars`` example in a Python
shell:

* compute the statistics by hand from ``model``’s attributes,
* verify that the printed summary matches your calculations,
* and try changing the model (e.g., adding a new predictor) to see how the
  tests and intervals change.
