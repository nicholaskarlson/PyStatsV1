Psychological Science & Statistics – Chapter 10
===============================================

The Independent-Samples t-Test
------------------------------

In Chapter 8, you learned the **logic of hypothesis testing** using a
one-sample t-test and a simulated null distribution of t-statistics.

In Chapter 9, you computed an **analytic one-sample t-test and confidence
interval** for a single mean using the theoretical :math:`t` distribution.

In this chapter, we extend those ideas to **comparing two independent groups**.
This is the standard "between-subjects" design in experimental psychology:
participants are randomly assigned to one of two conditions, and we compare the
means.

Typical examples include:

* Control vs. Treatment
* Placebo vs. Drug
* No-training vs. Training

Our running example will again use a **stress_score** variable.

When to Use the Independent-Samples t-Test
------------------------------------------

Use an independent-samples t-test when:

* You have **two groups** that are **independent** of each other.
  (No person appears in both groups.)
* Your dependent variable (DV) is **approximately continuous** and
  **approximately Normal** within each group.
* You are interested in whether the **population means differ**:

  .. math::

     H_0: \mu_1 = \mu_2
     \\
     H_1: \mu_1 \ne \mu_2

Here, :math:`\mu_1` is the population mean for group 1 and
:math:`\mu_2` is the population mean for group 2.

The Logic of the Independent-Samples t-Test
-------------------------------------------

The basic logic mirrors the one-sample case:

1. **State the hypotheses**

   .. math::

      H_0: \mu_1 = \mu_2
      \\
      H_1: \mu_1 \ne \mu_2

2. **Compute the observed difference in sample means**

   .. math::

      \bar{x}_1 - \bar{x}_2

3. **Estimate the standard error of the difference (assuming equal variances)**

   When we assume the two populations have **equal variances**, we first
   compute a **pooled standard deviation**:

   .. math::

      s_p^2
      = \frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}

   where

   * :math:`n_1, n_2` are the group sample sizes
   * :math:`s_1, s_2` are the sample standard deviations

   Then the **standard error of the difference in means** is

   .. math::

      \mathrm{SE}_{\bar{x}_1 - \bar{x}_2}
      = s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}.

.. admonition:: Modern Best Practice: Welch’s t-test vs. Pooled t-test

   The pooled-variance t-test above is the **classical Student’s t-test**
   taught in most introductory textbooks. It assumes that the two populations
   have **equal variances**.

   In real research, however, variances are often **not** equal, especially
   when group sizes differ. In those situations, the pooled test can have an
   inflated Type I error rate (too many false positives).

   Modern statistical software therefore defaults to **Welch’s t-test**, which
   does **not** assume equal variances and adjusts the degrees of freedom
   using the Welch–Satterthwaite equation.

   .. list-table:: Classical Pooled t-test vs. Welch’s t-test
      :header-rows: 1

      * - Method
        - Variance assumption
        - Degrees of freedom
        - Typical use
      * - Pooled t-test
        - Assumes :math:`\sigma_1^2 = \sigma_2^2`
        - :math:`n_1 + n_2 - 2`
        - Teaching; balanced designs with similar spreads
      * - Welch’s t-test
        - Does **not** assume equal variances
        - Approximate df (Welch–Satterthwaite)
        - Modern default; safer when variances or :math:`n` differ

   In this chapter we focus on the **pooled** test to explain the logic of
   comparing two means. In applied work, however, it is good practice to
   **report Welch’s t-test as well**, especially when the variances or sample
   sizes are noticeably different.

   The PyStatsV1 Chapter 10 script prints **both** pooled and Welch results so
   you can see how they compare on the same data.

4. **Compute the t-statistic (pooled version)**

   .. math::

      t_{\text{pooled}} =
      \frac{\bar{x}_1 - \bar{x}_2}{\mathrm{SE}_{\bar{x}_1 - \bar{x}_2}}

   with degrees of freedom

   .. math::

      \mathrm{df}_{\text{pooled}} = n_1 + n_2 - 2.

5. **Find the p-value and make a decision (pooled version)**

   Under :math:`H_0`, the statistic :math:`t_{\text{pooled}}` follows a
   :math:`t` distribution with :math:`\mathrm{df}_{\text{pooled}}` degrees of
   freedom. For a two-sided test, the p-value is

   .. math::

      p_{\text{pooled}}
      = 2 \cdot P\bigl(T_{\mathrm{df}_{\text{pooled}}}
      \ge |t_{\text{obs}}|\bigr).

   If :math:`p_{\text{pooled}} < \alpha` (typically :math:`0.05`), we **reject**
   :math:`H_0` and conclude that the group means differ.

Effect Size: Cohen's d
----------------------

Statistical significance does not tell us **how large** the effect is.
For independent groups with a pooled standard deviation, a common effect size is
**Cohen's :math:`d`**:

.. math::

   d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}.

Rough guidelines (Cohen, 1988):

* :math:`d \approx 0.2` – small effect
* :math:`d \approx 0.5` – medium effect
* :math:`d \approx 0.8` – large effect

Confidence Interval for the Mean Difference (Pooled Version)
------------------------------------------------------------

We can also construct a **confidence interval** for the difference in means:

.. math::

   (\bar{x}_1 - \bar{x}_2)
   \pm t_{\mathrm{crit}} \cdot \mathrm{SE}_{\bar{x}_1 - \bar{x}_2},

where :math:`t_{\mathrm{crit}}` is the critical :math:`t` value from the
:math:`t` distribution with :math:`\mathrm{df}_{\text{pooled}} = n_1 + n_2 - 2`
at your chosen :math:`\alpha` level (e.g., :math:`\alpha = 0.05` for a 95% CI).

PyStatsV1 Lab: Independent-Samples t-Test on Stress Scores
----------------------------------------------------------

In this lab you will:

1. Generate a synthetic dataset of **stress scores** for two independent groups:

   * **control**
   * **treatment**

2. Compute sample means, standard deviations, and group sizes.
3. Compute the **pooled standard deviation** and **standard error**.
4. Compute the **independent-samples t-statistic (pooled version)** and its
   **two-sided p-value**.
5. Compute **Cohen's :math:`d`** as an effect size.
6. Construct a **95% confidence interval** for the difference in means.
7. Compute **Welch’s t-test** as a modern, variance-robust comparison.
8. Optionally, visualize the group means with error bars.

All code for this lab lives in:

* ``scripts/psych_ch10_independent_t.py``

and the script will optionally write outputs to:

* ``data/synthetic/psych_ch10_independent_groups.csv``
* ``outputs/track_b/ch10_group_means_with_ci.png``

Running the Lab Script
~~~~~~~~~~~~~~~~~~~~~~

From the project root, run:

.. code-block:: bash

   python -m scripts.psych_ch10_independent_t

If your Makefile defines a convenience target, you can instead run:

.. code-block:: bash

   make psych-ch10

This will:

* Generate a synthetic dataset with two groups
  (e.g., 25 participants per group).
* Compute the independent-samples t-test comparing **control** vs.
  **treatment** using the **pooled** version.
* Compute **Welch’s t-test** on the same data as a **safety check**.
* Compute Cohen's :math:`d` and a 95% confidence interval for the mean
  difference.
* Print a short APA-style summary line.
* Optionally, save a bar plot of the group means with error bars.

Expected Console Output
~~~~~~~~~~~~~~~~~~~~~~~

Your exact numbers will vary, but the output will look similar to:

::

   Generated independent groups with n = 25 per condition
   Group: control    mean = 18.48  SD =  8.74  n = 25
   Group: treatment  mean = 16.82  SD =  9.87  n = 25

   --- Pooled-variance independent-samples t-test (classic Student's t) ---
   Mean difference (control - treatment) = 1.66
   Pooled SD = 9.32
   SE of difference = 2.64
   df (pooled) = 48
   t (pooled) = 0.63
   Two-sided p-value (pooled) = 0.53
   95% CI (pooled) for mean difference: [-3.65, 6.97]
   Cohen's d (pooled) = 0.18

   --- Welch's t-test (modern default, equal_var = False) ---
   df (Welch) ≈ 45.2
   t (Welch) = 0.61
   Two-sided p-value (Welch) = 0.54

   Wrote data to: data/synthetic/psych_ch10_independent_groups.csv
   Wrote plot to: outputs/track_b/ch10_group_means_with_ci.png

Interpreting the Output
~~~~~~~~~~~~~~~~~~~~~~~

Focus on the following pieces:

* **Mean difference**: How far apart are the sample means?
* **t statistic and p-value (pooled vs. Welch)**: Do the methods agree about
  whether the difference is statistically significant?
* **Confidence interval**: Does the 95% CI for
  :math:`\mu_1 - \mu_2` include zero?
* **Cohen's :math:`d`**: How large is the effect in standardized units?

Your Turn: Practice Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Change the group means**

   In ``psych_ch10_independent_t.py``, try changing the assumed population
   means for the two groups. How does this affect the mean difference, t, and
   Cohen's :math:`d`?

2. **Change the sample size**

   Increase :math:`n` per group (e.g., from 25 to 100). Notice how the standard
   error shrinks and the test becomes more sensitive to small differences.

3. **Make the variances very different**

   Use very different standard deviations for the two groups. Compare the
   pooled and Welch results. How do the degrees of freedom and p-values differ?

4. **Practice APA-style reporting**

   Using the script output, practice writing a short APA-style sentence, e.g.:

   *"Participants in the treatment condition did not differ significantly from
   those in the control condition on stress scores,
   :math:`t(48) = 0.63`, :math:`p = .53`, :math:`d = 0.18` (pooled)."*
