Applied Statistics with Python – Chapter 4
==========================================

Summarizing data
----------------

This chapter parallels the “Summarizing Data” chapter from the R notes.
The statistical ideas are the same:

* For **numeric variables**, we summarize the distribution using measures
  of **center** and **spread**.
* For **categorical variables**, we summarize using **counts** and
  **proportions**.
* We then use **plots** to visualize those summaries.

In the R version you see functions like ``mean()``, ``median()``, ``sd()``,
``IQR()``, ``hist()``, ``boxplot()``, and ``plot()`` for scatterplots.  Here
we’ll use Python, NumPy, pandas, and Matplotlib to achieve the same goals.

Throughout, imagine we have a DataFrame ``mpg`` that mirrors the R
``ggplot2::mpg`` dataset, with columns like:

* ``cty`` – city miles per gallon
* ``hwy`` – highway miles per gallon
* ``drv`` – drivetrain (``"f"``, ``"r"``, ``"4"``)
* ``displ`` – engine displacement in liters

You could obtain this DataFrame in several ways, for example:

.. code-block:: python

   import pandas as pd
   import seaborn as sns  # only needed if you want to load the example

   # Option 1: seaborn’s built-in mpg dataset
   mpg = sns.load_dataset("mpg")

   # Option 2: read from a CSV bundled with your project
   # mpg = pd.read_csv("data/mpg.csv")


4.1 Summary statistics
----------------------

We start with **summary statistics**: numbers that describe center, spread,
and distribution shape for a variable.

4.1.1 Numeric variables: center and spread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In R you saw a table of summaries like:

* mean
* median
* variance
* standard deviation
* interquartile range (IQR)
* minimum, maximum, range

We can compute the same quantities with pandas:

.. code-block:: python

   import numpy as np
   import pandas as pd

   # City miles per gallon
   cty = mpg["cty"]

   # Center
   cty_mean = cty.mean()      # average
   cty_median = cty.median()  # median

   # Spread
   cty_var = cty.var(ddof=1)  # sample variance (n-1)
   cty_sd = cty.std(ddof=1)   # sample standard deviation (n-1)
   cty_iqr = cty.quantile(0.75) - cty.quantile(0.25)

   cty_min = cty.min()
   cty_max = cty.max()
   cty_range = cty_max - cty_min

   summary = {
       "mean": cty_mean,
       "median": cty_median,
       "variance": cty_var,
       "sd": cty_sd,
       "IQR": cty_iqr,
       "min": cty_min,
       "max": cty_max,
       "range": cty_range,
   }

   summary

A quick shortcut for many of these is ``describe``:

.. code-block:: python

   mpg["cty"].describe()

which returns count, mean, standard deviation, quartiles, min, and max.

**Conceptual recap**

* Mean: arithmetic average; sensitive to outliers.
* Median: middle value; robust to outliers.
* Variance/SD: average squared (or square-rooted) distance from the mean.
* IQR: distance between the 25th and 75th percentiles (middle 50% of data).
* Min/Max/Range: show the extremes of the distribution.

Python vs R differences:

* R’s ``var()`` and ``sd()`` use ``n-1`` by default (unbiased estimators).
* pandas uses ``ddof=1`` for ``DataFrame.var`` and ``DataFrame.std`` by default.
* NumPy’s ``np.var`` and ``np.std`` default to ``ddof=0`` (divide by ``n``).
  Use ``ddof=1`` to match the R textbook.


4.1.2 Categorical variables: counts and proportions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For categorical variables, we care about **how often** each level appears.

In R, you saw ``table(mpg$drv)`` and relative frequencies with
``table(mpg$drv) / nrow(mpg)``.

In pandas:

.. code-block:: python

   drv_counts = mpg["drv"].value_counts()
   drv_props = mpg["drv"].value_counts(normalize=True)

   drv_counts
   drv_props

This gives frequency and proportion for each drivetrain category.

Key ideas:

* ``value_counts()`` is the pandas analogue of ``table()`` in R.
* ``normalize=True`` turns counts into proportions.
* These summaries are the numerical counterpart of a bar chart.


4.2 Plotting
------------

Numeric tables are useful, but most of the time we learn more from good
visualization.

We’ll mirror the same four plot types as the R chapter:

* Histograms
* Bar charts
* Boxplots
* Scatterplots

We will use Matplotlib and pandas plotting helpers. These examples assume:

.. code-block:: python

   import matplotlib.pyplot as plt


4.2.1 Histograms
~~~~~~~~~~~~~~~~

When you have **one numeric variable**, a histogram is the workhorse plot.

In R: ``hist(mpg$cty)`` and a more polished version with axis labels,
title, breaks, colors.

In Python/pandas:

.. code-block:: python

   fig, ax = plt.subplots()

   mpg["cty"].hist(
       bins=12,                # similar idea to breaks =
       color="dodgerblue",
       edgecolor="darkorange",
       ax=ax,
   )

   ax.set_xlabel("Miles per gallon (city)")
   ax.set_ylabel("Frequency")
   ax.set_title("Histogram of MPG (city)")

   plt.tight_layout()

Notes:

* ``bins`` is analogous to R’s ``breaks`` argument.
* Always label axes and add a clear title.
* ``hist`` gives the familiar histogram shape: bars whose area
  corresponds to counts (or densities).


4.2.2 Bar charts
~~~~~~~~~~~~~~~~

Bar charts summarize **categorical** variables (or numeric variables
with a small number of distinct values).

R example: ``barplot(table(mpg$drv))``.

Python:

.. code-block:: python

   drv_counts = mpg["drv"].value_counts().sort_index()

   fig, ax = plt.subplots()

   drv_counts.plot(
       kind="bar",
       color="dodgerblue",
       edgecolor="darkorange",
       ax=ax,
   )

   ax.set_xlabel("Drivetrain (f = FWD, r = RWD, 4 = 4WD)")
   ax.set_ylabel("Frequency")
   ax.set_title("Drivetrains")

   plt.tight_layout()

If you want **proportions** instead of counts, apply ``value_counts(normalize=True)``:

.. code-block:: python

   drv_props = mpg["drv"].value_counts(normalize=True).sort_index()
   drv_props.plot(kind="bar", ax=ax)   # same idea; y-axis now in [0, 1]


4.2.3 Boxplots
~~~~~~~~~~~~~~

Boxplots are ideal when you want to summarize the distribution of a numeric
variable, especially **across groups** defined by a categorical variable.

Single boxplot
^^^^^^^^^^^^^^

R: ``boxplot(mpg$hwy)``

Python/pandas:

.. code-block:: python

   fig, ax = plt.subplots()

   mpg["hwy"].plot(kind="box", vert=True, ax=ax)

   ax.set_ylabel("Miles per gallon (highway)")
   ax.set_title("Highway MPG – overall distribution")

   plt.tight_layout()

Grouped boxplots
^^^^^^^^^^^^^^^^

R syntax: ``boxplot(hwy ~ drv, data = mpg)`` – highway MPG by drivetrain.

In pandas, we group then call ``boxplot``:

.. code-block:: python

   fig, ax = plt.subplots()

   mpg.boxplot(
       column="hwy",
       by="drv",
       ax=ax,
       grid=False,
   )

   ax.set_xlabel("Drivetrain (f = FWD, r = RWD, 4 = 4WD)")
   ax.set_ylabel("Miles per gallon (highway)")
   ax.set_title("MPG (highway) vs drivetrain")
   # pandas adds its own super-title; remove if you like:
   fig.suptitle("")

   plt.tight_layout()

Interpretation reminders:

* The box shows the interquartile range (IQR): 25th to 75th percentile.
* The line inside the box is the median.
* Whiskers extend to typical minimum/maximum values.
* Points beyond the whiskers are potential outliers.

In Chapter 4 of the R notes, there is also emphasis on the formula syntax
``y ~ x``. The conceptual equivalent here is:

* “Take numeric column ``hwy``”
* “Group by drivetrain ``drv``”
* “Draw separate boxplots for each group”


4.2.4 Scatterplots
~~~~~~~~~~~~~~~~~~

Scatterplots show the relationship between **two numeric variables**.

The R chapter uses

.. code-block:: r

   plot(hwy ~ displ, data = mpg)

We can mirror this with pandas:

.. code-block:: python

   fig, ax = plt.subplots()

   ax.scatter(
       mpg["displ"],
       mpg["hwy"],
       s=30,
       color="dodgerblue",
   )

   ax.set_xlabel("Engine displacement (liters)")
   ax.set_ylabel("Miles per gallon (highway)")
   ax.set_title("MPG (highway) vs engine displacement")

   plt.tight_layout()

Typical interpretation for the ``mpg`` data:

* As engine displacement increases, highway MPG tends to decrease.
* The scatterplot shows not only the trend but also variability and
  potential clusters (e.g., different vehicle types).

A tiny bit of code to add a fitted line (optional, for later chapters):

.. code-block:: python

   import numpy as np

   X = mpg["displ"].to_numpy()
   y = mpg["hwy"].to_numpy()

   # simple least-squares line via NumPy polyfit
   m, b = np.polyfit(X, y, deg=1)

   x_grid = np.linspace(X.min(), X.max(), 100)
   y_hat = m * x_grid + b

   fig, ax = plt.subplots()
   ax.scatter(X, y, s=30, color="dodgerblue", alpha=0.7)
   ax.plot(x_grid, y_hat, color="darkorange", linewidth=2)

   ax.set_xlabel("Engine displacement (liters)")
   ax.set_ylabel("Miles per gallon (highway)")
   ax.set_title("MPG (highway) vs engine displacement with fitted line")

   plt.tight_layout()

You do *not* need to understand regression yet; here the line is just a visual
summary of the overall trend. Later chapters will unpack the model behind it.


4.3 What you should take away
-----------------------------

By the end of this chapter (R + Python versions), you should be comfortable with:

* Computing basic **summary statistics** for numeric data:

  - ``mean``, ``median``, ``variance``, ``sd``, ``IQR``, ``min``, ``max``, ``range``.

* Computing **frequency tables** and **proportions** for categorical variables
  using ``value_counts`` (and ``normalize=True`` for proportions).

* Matching each summary to an appropriate **plot type**:

  - histogram for one numeric variable,
  - bar chart for one categorical variable,
  - boxplot for numeric vs categorical,
  - scatterplot for two numeric variables.

* Translating R’s functions and syntax to Python/pandas/NumPy:

  - ``mean`` ↔ ``Series.mean()``,
  - ``sd`` ↔ ``Series.std(ddof=1)``,
  - ``IQR`` ↔ quantiles / ``Series.quantile``,
  - ``table`` ↔ ``value_counts``,
  - ``hist`` / ``barplot`` / ``boxplot`` / ``plot`` ↔ Matplotlib / pandas plotting.

Most importantly:

*You can now look at a variable, decide whether it is numeric or categorical,
and quickly choose a summary and a plot that make sense.*

These skills will be used constantly in later PyStatsV1 chapters—before
we fit any models, we will always:

1. **Summarize the data numerically** (center, spread, and counts), and
2. **Visualize** the data with one or more of the plots from this chapter.

