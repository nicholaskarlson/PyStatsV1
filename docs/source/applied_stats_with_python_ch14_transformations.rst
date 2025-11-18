.. _applied_stats_with_python_ch14_transformations:

Applied Statistics with Python – Chapter 14
===========================================

Transformations
---------------

In Chapter 13 we focused on *diagnosing* regression models: checking
assumptions, looking for non-constant variance, and finding unusual
observations.

This chapter asks a natural follow-up question:

.. rst-class:: spaced

*What can we do when the diagnostics say our model is not OK?*

A major tool is to transform variables:

* transform the **response** to stabilize variance or make residuals
  more nearly Normal;
* transform **predictors** to capture non-linear relationships while
  keeping a linear model;
* use **polynomial terms** to fit smooth curves.

By the end of this chapter you will be able to:

* explain what a variance-stabilizing transformation is and why it
  matters;
* fit and interpret regression models with a log-transformed response;
* use the Box–Cox family to choose a transformation automatically;
* transform predictors (for example, log–log relationships);
* fit polynomial regression models using ``statsmodels`` (and understand
  how Patsy’s ``I()`` works);
* recognize over-fitting and the dangers of extrapolating high-degree
  polynomials.


14.1 Response transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We start from a simple but common problem: the residual variance grows
with the mean.

Example: salaries at Initech
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose we have data on salary versus years of experience at a fictional
company Initech, stored in ``data/initech.csv``:

.. code-block:: python

   import pandas as pd
   import statsmodels.formula.api as smf

   initech = pd.read_csv("data/initech.csv")
   initech.head()

We first fit a simple linear regression,

.. math::

   Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i,

where :math:`Y_i` is salary and :math:`x_i` is years of experience.

.. code-block:: python

   lin_mod = smf.ols("salary ~ years", data=initech).fit()
   print(lin_mod.summary())

If you reuse the residual plots from Chapter 13 you will typically see:

* the *mean* relationship is roughly linear;
* the *variance* of the residuals increases with the fitted value.

This violates the constant variance assumption
(:math:`\mathsf{Var}[\varepsilon_i] = \sigma^2` for all :math:`i`).

Variance-stabilizing transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In symbols, our assumption is

.. math::

   \varepsilon \sim N(0, \sigma^2), \qquad
   \text{so } \operatorname{Var}[Y\mid X=x] = \sigma^2.

But in the Initech plot the variance looks like it depends on the mean:

.. math::

   \operatorname{Var}[Y\mid X=x] = h\big(\mathbb{E}[Y\mid X=x]\big),

for some increasing function :math:`h`.

A **variance-stabilizing transformation** is a function :math:`g` of the
response such that

.. math::

   \operatorname{Var}[g(Y)\mid X=x] \approx c,

a constant that no longer changes with the mean.

In practice we often try simple monotone transforms:

* log: :math:`g(y) = \log y`;
* square root: :math:`g(y) = \sqrt{y}`;
* reciprocal: :math:`g(y) = 1 / y`.

A good rule of thumb:

* If the response is strictly positive and spans multiple orders of
  magnitude, trying a log transform is almost always reasonable.

Log-transforming salary
^^^^^^^^^^^^^^^^^^^^^^^

For Initech we try

.. math::

   \log Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i.

In Python we can express this directly in the model formula:

.. code-block:: python

   import numpy as np

   log_mod = smf.ols("np.log(salary) ~ years", data=initech).fit()
   print(log_mod.summary())

Note a few things:

* The *response* is now ``np.log(salary)``. We did **not** create a new
  column; Patsy (the formula library used by ``statsmodels``) evaluates
  the NumPy call on the fly.
* The model is still *linear* in the parameters :math:`\beta_0` and
  :math:`\beta_1`.

To visualize:

* scatterplot of ``years`` versus ``np.log(salary)`` with the fitted
  straight line;
* residual plots for ``log_mod``.

You should see:

* the residual spread is much more constant;
* the Normal Q–Q plot looks closer to a straight line.

Interpreting coefficients on the original scale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the transformed scale we have

.. math::

   \log \hat{y}(x) = \hat{\beta}_0 + \hat{\beta}_1 x.

Exponentiating both sides,

.. math::

   \hat{y}(x)
   = \exp(\hat{\beta}_0)\,\exp(\hat{\beta}_1 x).

Each **additional year of experience** multiplies the *median* salary by

.. math::

   \exp(\hat{\beta}_1).

For example, if :math:`\hat{\beta}_1 = 0.079`, then

.. math::

   \exp(0.079) \approx 1.08,

meaning “about an 8% increase in salary per year of experience”.

Comparing fit on the original scale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

“Smaller residual standard error” is not comparable across models that
use different scales. Instead, compare the root mean squared error on
the original salary scale:

.. code-block:: python

   # original model
   rmse_lin = np.sqrt(np.mean((initech["salary"] - lin_mod.fittedvalues) ** 2))

   # log model, transformed back to dollars
   rmse_log = np.sqrt(
       np.mean((initech["salary"] - np.exp(log_mod.fittedvalues)) ** 2)
   )

   rmse_lin, rmse_log

Typically ``rmse_log`` is smaller, supporting the transformed model.


14.1.1 The Box–Cox family
^^^^^^^^^^^^^^^^^^^^^^^^^^

The log transform works well here, but we can also *let the data* suggest
a transformation.

The **Box–Cox family** of transforms for a strictly positive response
:math:`y` is

.. math::

   g_\lambda(y) =
   \begin{cases}
     \dfrac{y^\lambda - 1}{\lambda}, & \lambda \neq 0, \\
     \log y, & \lambda = 0.
   \end{cases}

The idea:

* For each candidate :math:`\lambda`, transform the response with
  :math:`g_\lambda`, fit a linear model, and compute the log-likelihood.
* Choose the :math:`\lambda` that maximizes this likelihood (or a nearby
  “nice” value, like 0, 0.5, 1, -0.5, etc.).
* Optionally, build a confidence interval for :math:`\lambda` to decide
  whether a simple transformation such as log is adequate.

In Python you can use ``scipy.stats.boxcox`` to obtain the MLE of
:math:`\lambda`:

.. code-block:: python

   from scipy import stats

   y = initech["salary"].to_numpy()
   # returns transformed y and the MLE lambda_
   y_bc, lambda_ = stats.boxcox(y)
   lambda_

You can then fit a model to ``y_bc`` instead of ``salary`` and compare
diagnostics as before.

For the Initech data the Box–Cox profile (not shown here) typically
puts :math:`\lambda=0` (log) very near the maximum and inside its
confidence interval, justifying our simpler choice.


14.2 Transforming predictors and using polynomials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far we transformed the *response*. We can also transform
**predictors**:

* to make non-linear relationships look linear;
* to build more flexible models while staying within the linear
  regression framework.

Log–log relationships
^^^^^^^^^^^^^^^^^^^^^

Recall the Auto MPG data used in earlier chapters. Suppose we want to
model fuel economy ``mpg`` as a function of horsepower ``hp``:

.. code-block:: python

   auto = pd.read_csv("data/autompg.csv")
   auto.head()

A simple linear model often shows curvature and non-constant variance:

.. code-block:: python

   mpg_hp_lin = smf.ols("mpg ~ hp", data=auto).fit()
   mpg_hp_lin.summary()

Diagnostic plots usually reveal:

* mpg decreases as hp increases;
* the relationship is not quite linear;
* residual variance grows for large hp.

A common fix is to log-transform one or both variables.

.. code-block:: python

   mpg_hp_logy = smf.ols("np.log(mpg) ~ hp", data=auto).fit()
   mpg_hp_loglog = smf.ols("np.log(mpg) ~ np.log(hp)", data=auto).fit()

The log–log model

.. math::

   \log(\text{mpg}) = \beta_0 + \beta_1 \log(\text{hp}) + \varepsilon

has a convenient interpretation:

.. math::

   \beta_1 \approx \text{elasticity of mpg with respect to hp},

the percent change in mpg for a 1% change in horsepower (holding other
variables fixed).

In practice, residual plots for the log–log model are often much cleaner
than for the raw variables.


14.2.1 Polynomial regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another powerful tool is to include **polynomial terms** of a predictor.

Example: marketing and diminishing returns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose ``data/marketing.csv`` contains monthly sales of a product
(``sales``) and the advertising budget (``advert``), both measured in
tens of thousands of dollars.

Plotting the data suggests that

* sales increase with advertising,
* but with *diminishing returns*.

We first fit a straight line:

.. code-block:: python

   marketing = pd.read_csv("data/marketing.csv")

   mod_lin = smf.ols("sales ~ advert", data=marketing).fit()
   mod_lin.summary()

A linear model ignores curvature. To capture diminishing returns we add a
quadratic term:

.. math::

   Y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \varepsilon_i.

In Patsy / ``statsmodels`` we can write

.. code-block:: python

   mod_quad = smf.ols("sales ~ advert + I(advert**2)", data=marketing).fit()
   print(mod_quad.summary())

Key points:

* ``I(advert**2)`` tells Patsy to treat ``advert**2`` as a new predictor,
  not as part of formula syntax—``I()`` stands for “inhibit”.
* With the linear term already in the model, the ``I(advert**2)`` term
  is highly significant.
* Residual plots are much cleaner; the fitted curve bends the right way.

You can continue this pattern with higher-order terms, but beware of
over-fitting and strange behavior outside the data range.

Over-fitting and extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In principle we can fit a polynomial of any degree:

.. math::

   Y_i = \beta_0 + \beta_1 x_i + \dots + \beta_{p-1} x_i^{p-1}
         + \varepsilon_i.

With degree :math:`p-1 = n-1` (one less than the number of points) a
polynomial can *perfectly interpolate* the data: residuals are exactly
zero.

That is rarely useful:

* the fit becomes extremely wiggly;
* predictions outside the observed range (extrapolation) can be absurd;
* standard errors explode because the design matrix is nearly singular.

The moral:

* Use low-degree polynomials (quadratic, maybe cubic, occasionally
  quartic).
* Always check residuals and, importantly, **plots of the fitted curve
  versus data**, not just :math:`R^2`.

Example: fuel economy versus speed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider experimental data on a car’s fuel efficiency at different
speeds, stored in ``data/fuel_econ.csv`` with columns ``mph`` and
``mpg``.

We expect:

* mpg increases with speed up to some optimal point,
* then decreases again—roughly a smooth hump-shaped curve.

We can start with

.. code-block:: python

   econ = pd.read_csv("data/fuel_econ.csv")

   fit1 = smf.ols("mpg ~ mph", data=econ).fit()
   fit2 = smf.ols("mpg ~ mph + I(mph**2)", data=econ).fit()
   fit4 = smf.ols("mpg ~ mph + I(mph**2) + I(mph**3) + I(mph**4)", data=econ).fit()
   fit6 = smf.ols(
       "mpg ~ mph + I(mph**2) + I(mph**3) + I(mph**4) + I(mph**5) + I(mph**6)",
       data=econ,
   ).fit()

Use residual plots and F-tests for nested models to decide how far to
go:

.. code-block:: python

   from statsmodels.stats.anova import anova_lm

   anova_lm(fit4, fit6)

If the :math:`p`-value is moderate but the residuals look clearly better
for ``fit6``, you might keep the degree-6 model *but* still be cautious
about extrapolation beyond the range of observed speeds.

Orthogonal polynomials (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

High-degree raw polynomials (``x``, ``x**2``, ``x**3``, …) are often
highly correlated, which can lead to numerical instability and large
standard errors.

In R, ``poly(x, degree)`` constructs **orthogonal polynomials**. The
Patsy library used by ``statsmodels`` has similar tools but they are
less commonly used in basic workflows.

If you need high-degree polynomials in Python, a pragmatic strategy is:

* use ``numpy.vander`` or ``sklearn.preprocessing.PolynomialFeatures``
  to generate columns;
* standardize predictors (center and scale) before forming high powers;
* keep degrees fairly small unless you have a strong reason and plenty
  of data.


14.3 Examples in Python
~~~~~~~~~~~~~~~~~~~~~~~

This section sketches complete Python workflows that combine the ideas
above. (Feel free to open a notebook and run them step by step.)

Example 1: fixing non-constant variance via log
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Load Initech salary data and fit the straight-line model
   ``salary ~ years``.
2. Reuse the diagnostic helper from Chapter 13 to make residual plots.
3. Fit the log-response model ``np.log(salary) ~ years``.
4. Compare residual plots and RMSE on the original dollar scale.
5. Translate :math:`\hat{\beta}_1` into a “percent change per year”
   interpretation.

Example 2: log–log relationship between mpg and horsepower
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Load Auto MPG data from ``data/autompg.csv``.
2. Fit three models:
   ``mpg ~ hp``, ``np.log(mpg) ~ hp``, and
   ``np.log(mpg) ~ np.log(hp)``.
3. Compare:

   * :math:`R^2` and residual standard error;
   * residual plots;
   * interpret the slope in the log–log model.

Example 3: polynomial fuel-economy curve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Load ``data/fuel_econ.csv``.
2. Fit models with degrees 1, 2, 4, and 6 in ``mph``.
3. For each model:

   * draw the fitted curve overlaid on the scatterplot;
   * inspect residual plots.

4. Use ``anova_lm`` or an information criterion (AIC) to compare nested
   models.
5. Choose a final degree (for example 4 or 6) and interpret its
   implications for “best speed for fuel economy”.


14.4 How this connects to PyStatsV1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transformations and polynomial terms are building blocks that reappear
throughout PyStatsV1:

* In later material on **model selection** we will compare many models
  that differ only in which transformed variables they include.
* In **logistic regression** and generalized linear models, link
  functions play a role very similar to response transformations here.
* In chapters on **experimental design**, we often build models that
  include polynomial terms in quantitative factors (for example,
  quadratic response-surface models) and interactions between factors.
* In applied work, much of the “art” of modeling is about choosing
  sensible transformations that make diagnostics (Chapter 13) look good
  without sacrificing interpretability.

PyStatsV1 examples and exercises will encourage you to:

* try simple transformations when diagnostics suggest problems;
* check that you can still explain the model in context (e.g., percent
  changes instead of additive changes);
* avoid blindly adding very high-degree polynomials just because they
  increase :math:`R^2`.


14.5 What you should take away
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By the end of this chapter you should be comfortable with:

* **When and why** to transform the response:

  * to stabilize variance;
  * to make residuals more nearly Normal;
  * to convert additive effects into multiplicative (percentage) effects.

* Using the **log transform** for strictly positive variables and
  interpreting coefficients back on the original scale.

* The **Box–Cox family** as a way to choose a transformation:

  * the special role of :math:`\lambda = 0` (log);
  * reading the profile of log-likelihood vs :math:`\lambda`.

* Transforming **predictors**:

  * log and other monotone transforms;
  * building **polynomial regression** models with ``I(x**2)``,
    ``I(x**3)``, etc.;
  * using nested model comparisons (or AIC) to decide how much
    complexity is warranted.

* Recognizing **over-fitting** and the dangers of extrapolating high-degree
  polynomials.

* The distinction between:

  * the *statistical model* (what assumptions we make about
    :math:`Y\mid X`);
  * the *formula* language we use to tell Python which transformed
    variables to include.

Most importantly:

.. rst-class:: spaced

*Diagnostics come first; transformations are tools you apply in response
to what the diagnostics show.*
