Applied Statistics with Python – Chapter 10
==========================================

Model building: explanation and prediction
------------------------------------------

In earlier chapters we focused on **fitting a single model**:

* simple linear regression (Chapters 7–8),
* multiple linear regression (Chapter 9).

Here we step back and ask a bigger question:

.. rubric:: How do we choose *which* model to use?

We will:

* separate the ideas of **family**, **form**, and **fit**,
* distinguish between models aimed at **explanation** vs **prediction**,
* see how **overfitting** and **train–test splits** enter the picture.

Throughout, you can imagine the familiar Auto MPG example:

* response :math:`y` = miles per gallon (``mpg``),
* predictors :math:`x_1, x_2, \dots` = car attributes (weight, horsepower, …).


10.1 Family, form, and fit
--------------------------

When we say "build a model", there are really *three* choices hiding inside:

1. **Family** – the broad class of models we are willing to consider.
2. **Form** – the specific predictors and transformations included.
3. **Fit** – the numerical values of the parameters, estimated from data.

We will mostly stay inside one family:

.. rubric:: Family: linear models

.. math::

   y = \beta_0 + \beta_1 x_1 + \cdots + \beta_{p-1} x_{p-1} + \varepsilon,

with :math:`\varepsilon` capturing noise or unexplained variation.

Other families exist (trees, smoothers, neural nets), but linear models are:

* the **standard starting point**,
* easy to fit and interpret,
* an excellent gateway to more advanced methods.


10.1.1 Fit
~~~~~~~~~~

Suppose we choose a simple form with one predictor:

.. math::

   y = \beta_0 + \beta_1 x_1 + \varepsilon.

To **fit** this model in Python we choose a loss function and minimize it.
In this course we almost always use **least squares**:

.. math::

   \min_{\beta_0, \beta_1}
   \sum_{i=1}^n \left(y_i - (\beta_0 + \beta_1 x_{1i})\right)^2.

In practice:

* with :mod:`statsmodels`, this is done by ``smf.ols(...).fit()``;
* with :mod:`sklearn`, by ``LinearRegression().fit(X, y)``.

The result is a **fitted model**:

.. math::

   \hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1,

which we can use for **interpretation** or **prediction**.


10.1.2 Form
~~~~~~~~~~~

The **form** of a linear model is determined by:

* which predictors are included,
* which transformations and interactions we use.

Examples, using ``mpg`` as the response:

* Simple linear regression:

  .. math::

     \text{mpg} = \beta_0 + \beta_1 \,\text{weight} + \varepsilon.

* Multiple linear regression:

  .. math::

     \text{mpg} =
       \beta_0 + \beta_1 \,\text{weight}
             + \beta_2 \,\text{horsepower}
             + \beta_3 \,\text{year}
             + \varepsilon.

* Model with a transformation and an interaction:

  .. math::

     \text{mpg} =
       \beta_0 + \beta_1 \,\text{weight}
             + \beta_2 \,\text{weight}^2
             + \beta_3 \,\text{year}
             + \beta_4 \,\text{weight} \times \text{year}
             + \varepsilon.

All of these are still **linear models**: linear in the parameters
:math:`\beta_j`.  The form controls *flexibility*:

* more predictors and terms → more flexibility,
* but also more risk of **overfitting** and harder interpretation.


10.1.3 Family
~~~~~~~~~~~~~

The **family** is the broad modeling approach.  Some examples:

* linear regression,
* generalized linear models (logistic, Poisson, …),
* non-parametric smoothers,
* trees and ensembles (random forests, boosting).

In this mini-textbook we focus on the **linear regression family** because:

* it is the standard tool for many applied problems,
* it has a rich theory of **inference** (standard errors, t-tests, F-tests),
* many ideas (design matrices, loss functions, regularization) carry directly
  into more advanced models.

You should keep a mental picture:

* **family** = which toolbox?
* **form**   = which tools from that box?
* **fit**    = how we use data to tune the tools (estimate parameters).


10.1.4 Assumed model vs fitted model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When we write a formula like

.. math::

   \text{mpg} =
     \beta_0 + \beta_1 \,\text{weight}
           + \beta_2 \,\text{horsepower}
           + \varepsilon,

we are specifying the **assumed model**:

* linear family,
* particular form (which variables and interactions),
* often with additional assumptions about :math:`\varepsilon`
  (e.g. Normal errors with constant variance).

After fitting, we obtain a **fitted model** such as:

.. math::

   \widehat{\text{mpg}} =
     46.2 - 3.1 \,\text{weight}
         - 0.02 \,\text{horsepower}.

Important:

* Fitting only gives the **best model within the chosen form**.
* If the family or form is poorly chosen, even a perfectly fitted model
  can be misleading.


10.2 Explanation versus prediction
----------------------------------

Why are we building a model?

* To **explain** how predictors relate to the response?
* Or to **predict** future responses as accurately as possible?

The distinction matters.  The modeling steps can look similar, but:

* For **explanation**, we prioritize *interpretability* and valid inference.
* For **prediction**, we prioritize *accuracy on new data* and resistance
  to overfitting.


10.2.1 Explanation
~~~~~~~~~~~~~~~~~~~

For explanation we want models that are:

* **small** – using as few predictors as reasonably possible,
* **interpretable** – each coefficient has a clear story,
* **well-behaved** – assumptions are at least approximately satisfied.

In linear regression, we often:

* start from a **full model** with many predictors,
* use:
  * t-tests for individual coefficients,
  * F-tests / ANOVA for comparing nested models,
  * residual plots to check model assumptions,
* gradually simplify to a **parsimonious model** that still fits well.

Example goals for the Auto MPG data:

* quantify how **weight** and **year** relate to fuel efficiency,
* understand which car attributes matter *most*,
* communicate results to non-statisticians.

Here, even if a larger model slightly improves prediction, we may prefer
a **simpler model** that tells a clearer story.


10.2.1.1 Correlation and causation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A crucial warning for explanatory models:

.. rubric:: Correlation does not imply causation.

Linear models detect **associations** between variables.  They do *not* prove
that one variable *causes* another.

* Observational data (like Auto MPG) can show that higher horsepower is
  associated with lower fuel efficiency.
* But this does not prove that "increasing horsepower by 10 automatically
  reduces mpg by 3" in a causal sense.

To argue for causation we usually need:

* a carefully designed **experiment**,
* or strong subject-matter reasoning and supporting evidence.

In PyStatsV1, we will often treat our models as tools for **description** and
**exploration**, with appropriate caution about causal claims.


10.2.2 Prediction
~~~~~~~~~~~~~~~~~

For prediction, the priority is different:

* We care about how well the model predicts **new, unseen data**.
* We are less concerned with:
  * whether each coefficient is statistically significant,
  * whether the model is easy to explain in words.

We need a **numerical measure of prediction error**.  A common choice is
root mean squared error (RMSE):

.. math::

   \text{RMSE} =
   \sqrt{\frac{1}{n} \sum_{i=1}^n \left(y_i - \hat{y}_i\right)^2}.

In Python, given arrays ``y_true`` and ``y_pred``:

.. code-block:: python

   import numpy as np

   def rmse(y_true, y_pred):
       return np.sqrt(np.mean((y_true - y_pred) ** 2))

Lower RMSE means better predictive performance on the data we are evaluating.


10.2.2.1 Train–test split and overfitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A key problem in predictive modeling is **overfitting**:

* A very flexible model can track the noise in the training data.
* It will have **low error on the data it saw**, but **high error on new data**.

To detect overfitting we mimic the "magic extra data" thought experiment by
splitting our data:

* **training set** – used to fit the model,
* **test set** – held out and only used to evaluate predictions.

In code, using scikit-learn:

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   import numpy as np

   X = mpg_df[["weight", "horsepower", "year"]].to_numpy()
   y = mpg_df["mpg"].to_numpy()

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42
   )

   model = LinearRegression().fit(X_train, y_train)

   y_train_pred = model.predict(X_train)
   y_test_pred = model.predict(X_test)

   train_rmse = rmse(y_train, y_train_pred)
   test_rmse = rmse(y_test, y_test_pred)

   print(train_rmse, test_rmse)

Typical pattern:

* As we add predictors and complexity:
  * **train RMSE** almost always decreases,
  * **test RMSE** may first decrease, then increase once we overfit.
* The best **predictive** model is often the one with the **lowest test RMSE**,
  even if it is not the largest or most complex.


10.3 What you should take away
------------------------------

By the end of this chapter (and its R + Python versions), you should be able to:

* distinguish clearly between:
  * **family** of models,
  * **form** of a model,
  * **fit** of a model;
* explain the difference between models aimed at:
  * **explanation** – small, interpretable, inference-friendly,
  * **prediction** – chosen to minimize error on new data;
* understand why:
  * linear models are often the **first choice**,
  * more complex models can **overfit**;
* compute and interpret:
  * **RMSE**, and
  * **train vs test** prediction error;
* describe why a train–test split is essential for honest assessment;
* articulate the warning:
  * "Correlation does not imply causation" in the context of regression.


10.4 How this connects to PyStatsV1
-----------------------------------

In PyStatsV1 you will see these ideas used repeatedly:

* **Explanatory models**

  * ``statsmodels`` regressions with detailed summaries,
  * ANOVA tables and F-tests for comparing nested models,
  * clean, compact models that are easy to discuss in class.

* **Predictive checks**

  * simple train–test splits for case studies,
  * side-by-side train vs test RMSE,
  * examples where a smaller model outperforms a more complex one on
    held-out data.

As you work through the code in later chapters, keep asking:

* "Am I trying to **explain** or **predict** here?"
* "Have I thought about **family**, **form**, and **fit** separately?"

That habit will pay off in any future modeling you do, whether with linear
models, machine learning methods, or more advanced tools.
