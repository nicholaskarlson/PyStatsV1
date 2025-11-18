Applied Statistics with Python – Chapter 11
===========================================

Categorical predictors and interactions
---------------------------------------

So far our regression chapters have mostly used **numeric predictors**:
continuous variables like horsepower, weight, or displacement.

In practice we also care about:

* **Categorical predictors** – e.g., transmission type (automatic vs manual),
  number of cylinders, country of origin.
* **Interactions** – situations where the effect of one predictor depends on
  the level of another.

The original R chapter for this material uses datasets like ``mtcars`` and
``autompg``, and leans heavily on R’s treatment of *factors* and the formula
syntax. In this Python-first version we will:

* mirror the main ideas in **NumPy, pandas, and statsmodels**,  
* show how **dummy variables** and **interactions** appear in formulas, and  
* connect back to the matrix notation from Chapter 9.

Throughout, you can think in three parallel languages:

* the **statistical model** in symbols (e.g. :math:`Y = \beta_0 + \beta_1 x_1 + \dots`),
* the **R formula** (e.g. ``mpg ~ hp + am``),
* the **Python formula** for ``statsmodels`` (e.g. ``"mpg ~ hp + am_manual"``).

The goal is not to memorize syntax, but to see how the *same ideas* travel
between R and Python.

.. contents::
   :local:
   :depth: 2


11.1 Dummy variables (indicator variables)
-----------------------------------------

A **dummy variable** is a numerical 0/1 variable that encodes a *binary*
category. For example, in the classic ``mtcars`` data:

* ``mpg`` – fuel efficiency (miles per gallon, our response),
* ``hp`` – horsepower (numeric predictor),
* ``am`` – transmission (0 = automatic, 1 = manual).

In R, you might fit

.. code-block:: r

   lm(mpg ~ hp + am, data = mtcars)

In Python, the same idea with ``statsmodels`` looks like:

.. code-block:: python

   import pandas as pd
   import statsmodels.formula.api as smf
   import statsmodels.api as sm

   # mtcars from the R datasets shipped with statsmodels
   mtcars = sm.datasets.get_rdataset("mtcars", "datasets").data

   # am is already coded 0/1, but let's make the meaning explicit
   mtcars["am_manual"] = (mtcars["am"] == 1).astype(int)

   model = smf.ols("mpg ~ hp + am_manual", data=mtcars).fit()
   print(model.params)

Statistically the model is

.. math::

   Y = \beta_0 + \beta_1 \,\text{hp} + \beta_2 \,\text{am\_manual} + \varepsilon,

where ``am_manual`` is 1 for manual and 0 for automatic.

Interpretation:

* :math:`\beta_1` – change in mean mpg for a one-unit increase in horsepower,
  holding transmission type fixed.
* :math:`\beta_2` – *difference* in mean mpg between manual and automatic
  transmissions at the same horsepower.
* :math:`\beta_0` – mean mpg for an automatic car with hp = 0
  (not realistic, but the algebraic intercept).

Notice how the dummy variable lets us write **two lines** with a shared slope:

* automatic: :math:`Y = \beta_0 + \beta_1 x_1 + \varepsilon`,
* manual: :math:`Y = (\beta_0 + \beta_2) + \beta_1 x_1 + \varepsilon`.

Same slope, different intercepts.

In PyStatsV1-style code, you’ll usually see:

* a ``pandas`` column containing 0/1 indicators, and
* a ``statsmodels`` formula that includes that column as an extra predictor.


11.2 Interactions: when slopes depend on context
------------------------------------------------

Dummy variables give us **two parallel lines**. Often we want something more
flexible: different *slopes* for different groups.

A simple example
~~~~~~~~~~~~~~~~

Suppose we have a cleaned ``autompg`` DataFrame with columns:

* ``mpg`` – response,
* ``disp`` – engine displacement,
* ``domestic`` – 1 if the car is built in the US, 0 if foreign.

An *additive* model with a dummy variable is

.. math::

   Y = \beta_0 + \beta_1 \,\text{disp} + \beta_2 \,\text{domestic} + \varepsilon.

Both domestic and foreign cars share the same slope :math:`\beta_1`.

An *interaction* model allows different slopes:

.. math::

   Y = \beta_0 + \beta_1 \,\text{disp}
       + \beta_2 \,\text{domestic}
       + \beta_3 \,\text{disp} \cdot \text{domestic}
       + \varepsilon.

In Python’s formula syntax:

* ``disp + domestic`` – additive model (no interaction),
* ``disp * domestic`` – additive terms **plus** ``disp:domestic`` interaction.

.. code-block:: python

   mpg_disp_add = smf.ols("mpg ~ disp + domestic", data=autompg).fit()
   mpg_disp_int = smf.ols("mpg ~ disp * domestic", data=autompg).fit()

   # compare nested models (F-test)
   f_stat, p_value, _ = mpg_disp_int.compare_f_test(mpg_disp_add)
   print(p_value)

If the p-value is very small, the interaction term meaningfully improves fit, and
we prefer the more flexible model.

Interpreting the interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write the interaction model separately for foreign (domestic = 0) and domestic
(domestic = 1) cars:

* foreign: :math:`Y = \beta_0 + \beta_1\,\text{disp} + \varepsilon`,
* domestic: :math:`Y = (\beta_0 + \beta_2) + (\beta_1 + \beta_3)\,\text{disp} + \varepsilon`.

So:

* **Intercepts** differ by :math:`\beta_2`.
* **Slopes** differ by :math:`\beta_3`.

Graphically you now have **two lines that can cross**, not just parallel lines.
This is the key idea: *interactions let the effect of one variable depend on the
value of another*.

Numeric–numeric interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactions are not just “dummy × numeric”. You can also interact two numeric
predictors, for example:

.. code-block:: python

   model = smf.ols("mpg ~ disp * hp", data=autompg).fit()
   print(model.summary().tables[1])  # coefficient table

The term ``disp:hp`` corresponds to :math:`\beta_3 x_1 x_2` in

.. math::

   Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2 + \varepsilon.

A quick algebra trick:

.. math::

   Y = \beta_0
     + (\beta_1 + \beta_3 x_2) x_1
     + \beta_2 x_2
     + \varepsilon.

The “slope in disp” is now :math:`\beta_1 + \beta_3 x_2`, which **depends on hp**.
That is exactly what “interaction” means.


11.3 Factor variables and automatic dummies
-------------------------------------------

R has a special data type for categorical variables: **factors**. When you use
a factor in a formula, R silently creates dummy variables for all but one
reference level.

Python has the same idea split in two pieces:

* ``pandas.Categorical`` (and ``dtype="category"``) to mark categorical data,
* ``statsmodels`` formulas that call ``C(name)`` to apply categorical coding.

Example: number of cylinders (4, 6, 8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose ``autompg["cyl"]`` has values 4, 6, and 8.

In R you might write:

.. code-block:: r

   lm(mpg ~ disp * cyl, data = autompg)

In Python, the equivalent is:

.. code-block:: python

   autompg["cyl"] = autompg["cyl"].astype("category")

   model_add = smf.ols("mpg ~ disp + C(cyl)", data=autompg).fit()
   model_int = smf.ols("mpg ~ disp * C(cyl)", data=autompg).fit()

Behind the scenes ``statsmodels`` creates dummy columns such as ``C(cyl)[T.6]``
and ``C(cyl)[T.8]``; 4 cylinders is the reference level by default (because "4"
is first in the sorted order).

Interpretation:

* ``Intercept`` – mean mpg for 4-cylinder cars when ``disp = 0``.
* ``C(cyl)[T.6]`` – difference in intercept between 6- and 4-cylinder cars.
* ``C(cyl)[T.8]`` – difference in intercept between 8- and 4-cylinder cars.
* In the interaction model, additional terms like ``disp:C(cyl)[T.6]`` modify
  the slope for 6-cylinder cars relative to 4-cylinder cars.

This matches the R output conceptually, even though the labels look slightly
different.

Changing the reference level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just as in R you can relevel a factor, in Python you can change which category
is treated as the baseline. One simple pattern is to reorder categories:

.. code-block:: python

   autompg["cyl"] = autompg["cyl"].cat.reorder_categories(["6", "4", "8"])
   model = smf.ols("mpg ~ disp * C(cyl)", data=autompg).fit()

Now 6 cylinders is the reference group, so all coefficient interpretations shift
accordingly—but the **fitted values and residuals are identical**.


11.4 Different parameterizations, same model
--------------------------------------------

The same regression surface can be written in many algebraically equivalent
ways.

For example, with 4/6/8 cylinders you could:

* use **two** dummy variables (6 vs not 6, 8 vs not 8) plus an intercept, or
* use **three** dummy variables and **no intercept**, or
* let ``statsmodels`` construct dummies via ``C(cyl)`` with your chosen
  reference level.

All of these parameterizations represent the same family of three lines, one for
each cylinder count.

A check in Python:

.. code-block:: python

   # model 1: intercept + C(cyl) parameterization
   m1 = smf.ols("mpg ~ disp * C(cyl)", data=autompg).fit()

   # model 2: explicit dummies, no intercept
   dummies = pd.get_dummies(autompg["cyl"], prefix="cyl", drop_first=False)
   df2 = pd.concat([autompg[["mpg", "disp"]], dummies], axis=1)

   m2 = smf.ols("mpg ~ 0 + disp:cyl_4 + disp:cyl_6 + disp:cyl_8 "
                "+ cyl_4 + cyl_6 + cyl_8", data=df2).fit()

   import numpy as np
   np.allclose(m1.fittedvalues, m2.fittedvalues)  # should be True

Takeaway: **coefficients depend on parameterization; fitted values do not.**
When comparing models, always compare their predictions or residuals, not just
raw coefficient values.


11.5 Building larger models with interactions
---------------------------------------------

Once you are comfortable with dummy variables and the ``*`` / ``:`` syntax,
you can specify fairly rich models compactly.

A “big” three-way interaction model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the same variables as above:

* ``mpg`` – response,
* ``disp`` – displacement (numeric),
* ``hp`` – horsepower (numeric),
* ``domestic`` – 1 if US-built, 0 otherwise,

we might start with the full three-way interaction:

.. code-block:: python

   big = smf.ols("mpg ~ disp * hp * domestic", data=autompg).fit()
   print(big.summary().tables[1])

The formula

.. math::

   \text{mpg} ~ \text{disp} * \text{hp} * \text{domestic}

expands to:

.. math::

   \text{disp} + \text{hp} + \text{domestic}
   + \text{disp:hp} + \text{disp:domestic} + \text{hp:domestic}
   + \text{disp:hp:domestic}.

This matches the **hierarchy principle**: if you include a higher-order
interaction, you also include all lower-order pieces that lead up to it.

Model simplification with nested F-tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just like in the R notes, we typically *remove* high-order interactions unless
they truly help.

In Python, you can compare nested models with ``compare_f_test``:

.. code-block:: python

   # full three-way interaction
   big = smf.ols("mpg ~ disp * hp * domestic", data=autompg).fit()

   # all two-way interactions, no three-way
   two_way = smf.ols(
       "mpg ~ disp * hp + disp * domestic + hp * domestic",
       data=autompg
   ).fit()

   # additive model (no interactions)
   additive = smf.ols("mpg ~ disp + hp + domestic", data=autompg).fit()

   # (1) do we need the three-way interaction?
   print(big.compare_f_test(two_way))   # (F, p-value, df_diff)

   # (2) do we need any interactions at all?
   print(two_way.compare_f_test(additive))

Typical workflow:

1. Start with a rich, but sensible model (respecting hierarchy).
2. Compare against a simpler nested model with an F-test.
3. If the p-value is large, prefer the simpler model (fewer parameters, easier
   to interpret).
4. Stop when further simplification clearly harms fit or removes important
   structure.

You can also inspect:

* residual standard error / RMSE,
* :math:`R^2` and adjusted :math:`R^2`,
* domain knowledge (does the simpler model still make scientific sense?).


11.6 How this connects to PyStatsV1
-----------------------------------

In PyStatsV1, these ideas show up repeatedly:

* **Dummy variables** are how we bring categorical information (treatment
  vs control, male vs female, exposed vs not exposed) into linear models.
* **Interactions** are how we let effects vary across groups, or across levels
  of another variable.
* **Factor / categorical coding** is how we bridge from the R ecosystem
  (where factors are everywhere) to Python’s ``pandas`` + ``statsmodels``
  world.

When you read or write PyStatsV1 code for regression chapters, watch for:

* columns that are 0/1 (hand-made dummies),
* formulas that contain ``C(variable)`` (automatic categorical coding),
* ``*`` and ``:`` in model formulas (interactions).

These tools are the building blocks for:

* logistic regression with categorical predictors,
* ANOVA-style comparisons between treatments,
* models with policy indicators and time trends,
* and many of the case studies we’ll add on top of this mini-textbook.


11.7 What you should take away
------------------------------

By the end of this chapter (R + Python versions), you should be able to:

* Construct **dummy variables** and interpret their coefficients as
  *differences* between groups.
* Distinguish between:

  * additive models (parallel lines, shared slopes) and
  * interaction models (group-specific slopes and intercepts).

* Use Python’s formula interface to:

  * include categorical predictors via ``C(...)``,
  * add interactions with ``*`` and ``:``,
  * and change the reference level when needed.

* Read regression output and translate coefficients back to:

  * intercepts and slopes for specific groups, and
  * differences between those groups.

* Recognize that different **parameterizations** (different dummy codings)
  can represent the *same* model, even though the coefficient labels differ.

* Use nested model F-tests (or ``compare_f_test`` in ``statsmodels``) to decide
  whether higher-order interactions are worth keeping.

In later PyStatsV1 chapters, these ideas will underpin:

* regression models with categorical predictors and interactions,
* comparisons of several models on the same data,
* and case studies where interpretation of group differences really matters.
