.. _psych_ch8_hypothesis_testing:

Psychological Science & Statistics – Chapter 8
==============================================

Hypothesis Testing and the One-Sample t-Test
--------------------------------------------

In Chapters 6 and 7 you learned:

* how to model a variable with the **normal distribution** and interpret
  **z-scores** (Chapter 6), and
* how repeated sampling leads to a **sampling distribution of the mean**
  (Chapter 7).

In this chapter we put those pieces together to introduce **null hypothesis
significance testing (NHST)** for a **single mean** using the
**one-sample t-test**.

Our goals are to help you:

* understand the logic of NHST as a decision procedure,
* interpret the **one-sample t-statistic**,
* see how sampling variability drives the **p-value**, and
* connect the theoretical t-test to a **simulation-based view**.

The Logic of NHST for a Single Mean
-----------------------------------

Suppose we have a quantitative variable such as ``stress_score`` and we want
to test whether the mean in a population is equal to some reference value
:math:`\mu_0` (for example, a published norm or a policy target).

The one-sample t-test follows these steps:

1. **State the hypotheses.**

   .. math::

      H_0 : \mu = \mu_0
      \qquad\text{(null hypothesis)}

      H_1 : \mu \ne \mu_0
      \qquad\text{(two-sided alternative)}

2. **Collect a sample** of size :math:`n` from the population and compute:

   * the sample mean :math:`\bar{x}`,
   * the sample standard deviation :math:`s`.

3. **Compute the test statistic.**

   Because the population standard deviation :math:`\sigma` is unknown, we
   use the **t-statistic**:

   .. math::

      t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}

   This tells us how many **standard errors** the sample mean is from the
   null value :math:`\mu_0`.

4. **Ask: how surprising is this t if :math:`H_0` were true?**

   Under :math:`H_0` and certain assumptions (independent observations,
   approximately normal population), the t-statistic follows a **t
   distribution** with :math:`n - 1` degrees of freedom.

   The **p-value** is the probability of obtaining a t-statistic as extreme
   as (or more extreme than) the one we observed if :math:`H_0` is true.

5. **Make a decision.**

   * If the p-value is small (for example, :math:`p < 0.05`), we say our
     observed t is **unlikely under the null**, and we **reject** :math:`H_0`.
   * If the p-value is not small, we **fail to reject** :math:`H_0`. This
     does *not* prove that :math:`H_0` is true; it only says the data are not
     very inconsistent with it.

Connecting to Chapter 7: Sampling Distributions
-----------------------------------------------

In Chapter 7 you simulated the **sampling distribution of the mean** for a
stress scale. You saw that:

* the distribution of sample means is centered near the true population mean,
* its spread shrinks as :math:`n` increases (standard error idea), and
* most sample means lie close to the population mean, with a few in the tails.

The one-sample t-statistic builds directly on that idea, but with an extra
wrinkle: we do not know the population standard deviation :math:`\sigma`, so
we estimate it with the sample standard deviation :math:`s`. Because :math:`s`
varies from sample to sample, the distribution of t has **heavier tails**
than the normal distribution.

Simulation-Based View of a One-Sample Test
------------------------------------------

In Chapter 7, you simulated the sampling distribution of the mean. You
tracked where sample means ended up when repeatedly sampling from a fixed
population.

In this chapter's lab, we use a similar idea to approximate a p-value, but
with a crucial twist: instead of just tracking means, we track
**t-statistics**.

1. Generate a large synthetic population of **stress scores**.

2. Specify a null hypothesis about the population mean:

   .. math::

      H_0 : \mu = \mu_0

3. Draw a single random sample from this population and compute:

   * the sample mean :math:`\bar{x}`,
   * the sample standard deviation :math:`s`,
   * the observed t-statistic :math:`t_\text{obs}`.

4. Construct a world where :math:`H_0` is exactly true by **recentering** the
   population so its mean is :math:`\mu_0`. The shape and spread stay the
   same; only the center changes.

5. Draw many random samples from this recentered population. For **each**
   simulated sample, compute its own mean, its own standard deviation, and
   its own t-statistic :math:`t_\text{sim}`.

6. Approximate the two-sided p-value as:

   .. math::

      \hat{p} =
      \frac{\text{number of simulations with }
      |t_\text{sim}| \ge |t_\text{obs}|}
      {\text{number of simulations}}

   This is a simulation-based analogue of the theoretical p-value. By checking
   how extreme our *t-statistic* is compared to the distribution of *simulated
   t-statistics*, we correctly account for the uncertainty in estimating the
   standard deviation.

PyStatsV1 Lab: A One-Sample Test on Stress Scores
-------------------------------------------------

In this lab, you will:

1. Generate a large synthetic population of stress scores.
2. State a null hypothesis about the population mean.
3. Draw one random sample of size :math:`n` and compute the observed
   :math:`t`-statistic.
4. Use simulation to generate a **null distribution of t-values**.
5. Approximate the two-sided p-value by locating your observed :math:`t` in
   that distribution.
6. Make a decision about whether to reject the null hypothesis at
   :math:`\alpha = 0.05`.

All code for this lab lives in:

* ``scripts/psych_ch8_one_sample_test.py``

and it will write outputs to:

* ``data/synthetic/psych_ch8_population_stress.csv`` (population),
* ``data/synthetic/psych_ch8_null_t_values.csv`` (simulated t-values),
* optionally ``outputs/track_b/ch08_null_t_distribution.png`` (plot).

Running the Lab Script
~~~~~~~~~~~~~~~~~~~~~~

From the project root, run:

.. code-block:: bash

   python -m scripts.psych_ch8_one_sample_test

If your Makefile defines a convenience target, you can instead run:

.. code-block:: bash

   make psych-ch08

This will:

* Generate a synthetic ``stress_score`` population.
* Specify a null value :math:`\mu_0` (for example, 20).
* Draw a sample of size :math:`n` (for example, 25).
* Compute the observed t-statistic for :math:`H_0 : \mu = \mu_0`.
* Simulate a null distribution by recentring the population, resampling, and
  computing :math:`t` for every resample.
* Estimate a two-sided p-value as a long-run relative frequency.
* Print a verbal conclusion (reject vs fail to reject at
  :math:`\alpha = 0.05`).
* Optionally, save a plot of the simulated t-distribution with the observed
  t-statistic marked.

Expected Console Output
~~~~~~~~~~~~~~~~~~~~~~~

Your exact numbers will vary, but the output will look similar to:

::

   Generated population with 50000 individuals
   Population mean stress_score = 19.98
   Population SD   stress_score = 9.95

   Null hypothesis: mu = 20.00
   Observed sample size n = 25
   Observed sample mean   = 22.13
   Observed sample SD     = 10.31
   t statistic            = 1.03

   Using 4000 simulations under H0...
   Approximate two-sided p-value = 0.31
   Decision at alpha = 0.05: fail to reject H0

Interpreting the Output
~~~~~~~~~~~~~~~~~~~~~~~

Focus on the following pieces:

* The **observed t-statistic**: how many standard errors the sample mean is
  from the null value.
* The **p-value**: the probability of obtaining a t-statistic this extreme
  (or more) if the null hypothesis were true.
* The **decision**: the binary result based on your alpha threshold.
  Remember that "fail to reject" is not the same as "prove the null true."

Your Turn: Practice with Different Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Change the null hypothesis**

   Modify the null value :math:`\mu_0` in the script. How does this change
   the t-statistic and the resulting p-value?

2. **Change the sample size**

   Increase the sample size :math:`n` (for example, from 25 to 100).
   Notice how the t-statistic changes. The :math:`\sqrt{n}` in the
   denominator makes the test more sensitive to small departures as the
   sample size grows.

3. **Replicate the experiment**

   Run the script multiple times. Do you always reach the same decision?
   If the true mean is close to :math:`\mu_0`, you may see the decision flip
   back and forth — this is the nature of sampling variability.

Optional Plot: Null t-Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If enabled in the script, a plot file is saved to:

``outputs/track_b/ch08_null_t_distribution.png``

The figure shows:

* a histogram of simulated **t-statistics** under :math:`H_0`, and
* a vertical line marking your **observed t-statistic**.

Questions to consider:

* Does the histogram look roughly bell-shaped and centered at 0?
* Is your observed line in the main bulk of the distribution (a common
  result) or out in the thin tails (a rare result)?

Summary
-------

In this chapter you learned how to:

* frame a research question as a **null hypothesis** about a mean,
* compute and interpret the **one-sample t-statistic**,
* approximate a **p-value** using a simulated null distribution of
  t-statistics, and
* make a decision to reject or fail to reject :math:`H_0`.

These ideas form the backbone of classical inference and set you up for the
next steps:

* confidence intervals for a mean,
* comparisons of two means (independent and paired-samples t-tests), and
* more complex models such as ANOVA and regression.
