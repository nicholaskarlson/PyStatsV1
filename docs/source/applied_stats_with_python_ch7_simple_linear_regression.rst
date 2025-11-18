Applied Statistics with Python – Chapter 7
==========================================

Simple linear regression in Python and R
----------------------------------------

This chapter parallels the *Simple Linear Regression* chapter from the R
notes. The statistical ideas are the same:

* We have a **numeric predictor** (``x``) and a **numeric response** (``y``).
* We want to describe how the *mean* of ``y`` changes as ``x`` changes.
* We quantify that relationship with a **line** plus **random noise**.

In the R notes, the running example is the classic ``cars`` dataset:

* ``speed`` – car speed in miles per hour,
* ``dist`` – stopping distance in feet.

In PyStatsV1 we reuse the same data, but we work in **Python-first** terms:
NumPy, pandas, Matplotlib, and the ``statsmodels`` regression API.

By the end of this chapter you should be able to:

* write down the **simple linear regression model** and its assumptions,
* compute least–squares estimates by hand (with NumPy),
* fit the same model using a high-level tool (``statsmodels.ols``),
* interpret the slope, intercept, residuals, and :math:`R^2`,
* use the fitted model to make **predictions** (and know when not to trust them),
* understand how the R function ``lm()`` corresponds to the Python tools we use.

.. note::

   Throughout this chapter we assume you have the PyStatsV1 repository
   checked out locally and that you can run the chapter script

   ``scripts/ch07_simple_linear_regression.py``

   which contains a full, executable version of the examples.

7.1 From scatterplots to models
-------------------------------

We start with a scatterplot: *speed vs. stopping distance*.

In Python, with a DataFrame ``cars`` that has ``speed`` and ``dist`` columns:

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt

   cars = pd.read_csv("data/cars.csv")  # see PyStatsV1 data folder

   fig, ax = plt.subplots()
   ax.scatter(cars["speed"], cars["dist"], alpha=0.7)
   ax.set_xlabel("Speed (mph)")
   ax.set_ylabel("Stopping distance (ft)")
   ax.set_title("Stopping distance vs. speed")
   plt.show()

The plot tells us:

* Faster cars tend to have **longer** stopping distances.
* The points do not fall exactly on a line—there is **random variation**.

We can express this informally as

*response = pattern + noise*.

In math notation we write

.. math::

   Y = f(X) + \varepsilon,

where

* :math:`X` – predictor (speed),
* :math:`Y` – response (stopping distance),
* :math:`f(\cdot)` – unknown systematic pattern,
* :math:`\varepsilon` – random error.

In this chapter we **restrict** :math:`f(\cdot)` to be a *line*.

7.2 The simple linear regression model
--------------------------------------

The **simple linear regression (SLR)** model uses a straight line to describe
how the mean of ``Y`` changes with ``X``:

.. math::

   Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \qquad i = 1,\dots,n.

Here

* :math:`x_i` – observed predictor value (fixed, not random),
* :math:`Y_i` – random response,
* :math:`\beta_0` – intercept,
* :math:`\beta_1` – slope,
* :math:`\varepsilon_i` – random error term.

We assume the errors satisfy the usual **LINE** conditions:

* **L – Linear**: the mean of :math:`Y` is a straight line in :math:`x`,

  .. math:: \mathbb{E}[Y_i \mid X_i = x_i] = \beta_0 + \beta_1 x_i.

* **I – Independent**: errors :math:`\varepsilon_i` are independent.
* **N – Normal**: errors are normally distributed,

  .. math:: \varepsilon_i \sim \mathcal{N}(0, \sigma^2).

* **E – Equal variance**: all errors share the same variance :math:`\sigma^2`.

Under these assumptions, the conditional distribution of :math:`Y_i` is

.. math::

   Y_i \mid X_i = x_i \sim \mathcal{N}(\beta_0 + \beta_1 x_i,\; \sigma^2).

The three unknown parameters are :math:`\beta_0`, :math:`\beta_1`, and
:math:`\sigma^2`. Our task is to estimate them from data.

7.3 Least squares: estimating the line
--------------------------------------

Given observed pairs :math:`(x_i, y_i)`, the **least–squares** idea is:

*Choose the line that makes the squared vertical errors as small as possible.*

Vertical errors (residuals) are

.. math:: e_i = y_i - (\beta_0 + \beta_1 x_i).

We want :math:`\beta_0` and :math:`\beta_1` that minimize

.. math::

   \text{SSE}(\beta_0,\beta_1)
   = \sum_{i=1}^n \big(y_i - (\beta_0 + \beta_1 x_i)\big)^2.

Solving the resulting equations gives the familiar closed forms

.. math::

   \hat\beta_1 =
   \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}
        {\sum_{i=1}^n (x_i - \bar{x})^2},
   \qquad
   \hat\beta_0 = \bar{y} - \hat\beta_1 \bar{x},

where :math:`\bar{x}` and :math:`\bar{y}` are the sample means.

7.3.1 Computing :math:`\hat\beta_0` and :math:`\hat\beta_1` in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using NumPy with the ``cars`` dataset:

.. code-block:: python

   import numpy as np
   import pandas as pd

   cars = pd.read_csv("data/cars.csv")

   x = cars["speed"].to_numpy()
   y = cars["dist"].to_numpy()

   x_bar = x.mean()
   y_bar = y.mean()

   Sxx = np.sum((x - x_bar) ** 2)
   Sxy = np.sum((x - x_bar) * (y - y_bar))

   beta1_hat = Sxy / Sxx
   beta0_hat = y_bar - beta1_hat * x_bar

   print(beta0_hat, beta1_hat)

Interpretation (for this dataset):

* :math:`\hat\beta_1 \approx 3.93` – for each +1 mph of speed, the **mean**
  stopping distance increases by about **3.9 feet**.
* :math:`\hat\beta_0 \approx -17.6` – the predicted mean distance at 0 mph
  (an extrapolation; not physically meaningful, but needed for the line).

7.3.2 Predictions and interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once we have :math:`\hat\beta_0` and :math:`\hat\beta_1`, the fitted line is

.. math::

   \hat{y} = \hat\beta_0 + \hat\beta_1 x.

Predictions in Python are straightforward:

.. code-block:: python

   def predict_stopping_distance(speed_mph: float) -> float:
       return beta0_hat + beta1_hat * speed_mph

   predict_stopping_distance(8)   # within data range (interpolation)
   predict_stopping_distance(21)  # within range but not observed
   predict_stopping_distance(50)  # outside range (extrapolation – be careful!)

Key idea:

* **Interpolation** inside the observed ``x`` range is usually reasonable.
* **Extrapolation** far beyond the data range is risky, even with a good line.

7.4 Residuals, variance, and :math:`R^2`
----------------------------------------

Once the line is fitted, we inspect how far each point is from that line.

Residuals
^^^^^^^^^

Residuals are observed minus fitted values:

.. math::

   e_i = y_i - \hat{y}_i
       = y_i - (\hat\beta_0 + \hat\beta_1 x_i).

In NumPy:

.. code-block:: python

   y_hat = beta0_hat + beta1_hat * x
   residuals = y - y_hat

   residuals[:5]

A good first diagnostic is a **residual vs. fitted** plot; you should see:

* no strong curve (supports linearity),
* roughly constant spread (supports equal variance),
* no obvious pattern or clustering (supports independence).

Residual variance and residual standard error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sum of squared residuals is

.. math:: \text{SSE} = \sum_{i=1}^n e_i^2.

We estimate :math:`\sigma^2` by

.. math::

   s_e^2 = \frac{\text{SSE}}{n - 2},

where ``n − 2`` reflects the two parameters we estimated.

In Python:

.. code-block:: python

   n = len(y)
   sse = np.sum(residuals ** 2)
   s2_e = sse / (n - 2)
   s_e = np.sqrt(s2_e)   # residual standard error

The residual standard error :math:`s_e` is in the same units as ``y``
(feet here). You can read it as:

   “Our fitted mean stopping distances are typically off by about
   :math:`s_e` feet.”

Decomposition of variation and :math:`R^2`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can decompose total variation in ``y`` into explained and unexplained parts:

.. math::

   \underbrace{\sum (y_i - \bar{y})^2}_{\text{SST}}
   =
   \underbrace{\sum (\hat{y}_i - \bar{y})^2}_{\text{SSReg}}
   +
   \underbrace{\sum (y_i - \hat{y}_i)^2}_{\text{SSE}}.

Then the **coefficient of determination** is

.. math::

   R^2 = \frac{\text{SSReg}}{\text{SST}}
       = 1 - \frac{\text{SSE}}{\text{SST}}.

Interpretation:

* :math:`R^2` is the **proportion of variation in ``y``** explained by the
  regression on ``x``.
* Values near 1 mean the line explains most of the variability;
  values near 0 mean the line explains little.

In Python:

.. code-block:: python

   sst = np.sum((y - y_bar) ** 2)
   ss_reg = np.sum((y_hat - y_bar) ** 2)
   r2 = ss_reg / sst   # or 1 - sse / sst

For the ``cars`` example, :math:`R^2` is around 0.65 – about 65% of the
variation in stopping distance is explained by speed alone.

7.5 Using ``statsmodels``: Python’s version of ``lm()``
-------------------------------------------------------

In R, we would write

.. code-block:: r

   cars <- datasets::cars
   fit  <- lm(dist ~ speed, data = cars)

   summary(fit)
   predict(fit, newdata = data.frame(speed = 8))

In Python, the closest equivalent is ``statsmodels``’ formula API:

.. code-block:: python

   import pandas as pd
   import statsmodels.formula.api as smf

   cars = pd.read_csv("data/cars.csv")

   model = smf.ols("dist ~ speed", data=cars).fit()

   print(model.params)      # beta_0_hat and beta_1_hat
   print(model.rsquared)    # R^2
   print(model.summary())   # detailed regression table

   model.predict({"speed": [8, 21, 50]})

You should recognize many familiar quantities in the summary:

* coefficient estimates (:math:`\hat\beta_0`, :math:`\hat\beta_1`),
* their standard errors and t-statistics,
* :math:`R^2` and adjusted :math:`R^2`,
* residual standard error (called ``scale`` or ``sigma``).

Mapping R → Python
^^^^^^^^^^^^^^^^^^^

* ``lm(dist ~ speed, data=cars)`` → ``smf.ols("dist ~ speed", data=cars).fit()``
* ``coef(fit)`` → ``model.params``
* ``resid(fit)`` → ``model.resid``
* ``fitted(fit)`` → ``model.fittedvalues``
* ``summary(fit)`` → ``model.summary()``
* ``predict(fit, newdata=...)`` → ``model.predict(new_dataframe)``

7.6 Simulation: seeing SLR in action
------------------------------------

Simulation is a powerful way to *see* what a model means.

Suppose the **true** relationship is

.. math::

   Y = 5 - 2x + \varepsilon,
   \qquad \varepsilon \sim \mathcal{N}(0, 3^2).

We can simulate a dataset, fit a line, and compare estimates to truth:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import statsmodels.formula.api as smf

   rng = np.random.default_rng(seed=1)

   n = 21
   beta0_true = 5.0
   beta1_true = -2.0
   sigma_true = 3.0

   x = np.linspace(0, 10, n)
   eps = rng.normal(loc=0.0, scale=sigma_true, size=n)
   y = beta0_true + beta1_true * x + eps

   sim = pd.DataFrame({"x": x, "y": y})

   fit = smf.ols("y ~ x", data=sim).fit()
   print(fit.params)

If you plot the data and overlay both lines (true and fitted), you will see that
the estimated line is close, but not perfect—that’s sampling variability.

This kind of simulation becomes even more useful in later chapters (e.g.,
to study confidence intervals, hypothesis tests, and power).

7.7 What you should take away
------------------------------

By the end of this chapter you should be comfortable with:

* thinking in terms of **models**:

  .. math:: Y = \beta_0 + \beta_1 X + \varepsilon,

  and “response = prediction + error”,

* interpreting the **slope** and **intercept** in context,
* computing least–squares estimates :math:`\hat\beta_0` and :math:`\hat\beta_1`,
* computing and interpreting **residuals**, **SSE**, **residual standard error**,
* interpreting :math:`R^2` as “fraction of variance explained,”
* using ``statsmodels`` in Python as the counterpart of R’s ``lm()``:

  .. code-block:: python

     model = smf.ols("dist ~ speed", data=cars).fit()
     model.summary()
     model.predict({"speed": [10, 20]})

In later PyStatsV1 chapters, we will:

* extend SLR to **multiple** regression,
* build models with **transformed** predictors (e.g., quadratic terms),
* and connect regression more formally to **inference** (tests and confidence
  intervals).

If any of the algebra feels fuzzy, focus first on the **pictures** and the
Python code; the formulas will slowly become familiar as you keep applying them
to real datasets.

