
Chapter 18 – Analysis of Covariance (ANCOVA)
===========================================

.. contents:: Chapter overview
   :local:
   :depth: 2

In Chapter 17 you learned how to analyse *mixed-model* designs with both
between-subjects and within-subjects factors. In this chapter we introduce
a closely related idea: using a *covariate* to statistically control for
pre-existing differences between participants.

Analysis of covariance (ANCOVA) combines the logic of regression and ANOVA.
It answers questions like:

* Do two treatment groups differ on a post-test **after controlling for**
  baseline differences?
* Does a new therapy reduce anxiety **over and above** what we can predict
  from initial symptom severity?

Throughout the chapter we will work with a simple psychology example and
use :mod:`pystatsv1` and :mod:`pingouin` to fit and interpret an ANCOVA model.

Learning goals
--------------

After working through this chapter you should be able to:

* explain what a covariate is and why researchers include covariates
  in experimental designs,
* distinguish between **raw** (unadjusted) group means and **adjusted**
  means from an ANCOVA,
* describe the assumptions of ANCOVA (linearity, homogeneity of regression
  slopes, reliability of the covariate),
* run a basic one-way ANCOVA in Python using :mod:`pingouin`,
* interpret the output (F statistic, p-value, effect size, adjusted means),
* understand how ANCOVA is related to multiple regression.

18.1 Statistical control and covariates
---------------------------------------

Suppose a researcher is evaluating a new study-skills workshop designed
to improve exam performance. Students volunteer and are randomly assigned
to either a **control** group (no workshop) or a **treatment** group
(workshop). Everyone completes a pre-test measuring current study skills
and a final exam at the end of term.

In an ideal randomized experiment, random assignment ensures that the two
groups are similar *on average* before the intervention. In practice,
however, there will always be some pre-existing differences. In our example,
some students may start with better study skills or higher motivation.

A **covariate** is a continuous variable that is:

* measured prior to the manipulation (e.g., pre-test score),
* related to the outcome (e.g., final exam score), and
* **not** directly affected by the experimental treatment.

ANCOVA uses the covariate to statistically control for pre-existing
differences. Conceptually, we are asking:

    *“If all students had started with the **same** pre-test score,
    would the treatment and control groups still differ on the exam?”*

18.2 The logic of ANCOVA
------------------------

ANCOVA can be viewed in two equivalent ways:

* as an ANOVA that has been extended to include a continuous predictor, or
* as a multiple regression in which group membership is coded as a
  categorical predictor and the covariate is a continuous predictor.

The key idea is to partition the variance in the outcome into:

* variance explained by the covariate,
* variance explained by the group factor *after controlling for the
  covariate*, and
* residual (error) variance.

If the group factor explains a non-trivial amount of variance **over and
above** the covariate, the ANCOVA will yield a significant F statistic for
the group effect. The adjusted means provide a way to visualize that effect.

18.3 Adjusted means and interpretation
--------------------------------------

Because the covariate is continuous, each participant has a unique
combination of covariate value and outcome value. ANCOVA uses the
regression of the outcome on the covariate to compute **adjusted means**
for each group at a common reference value of the covariate
(often the overall mean).

In our example, imagine that we adjust all students to have the same
pre-test score. The adjusted means then tell us what the average exam
score *would have been* for each group **if** they had started at the
same baseline.

When reporting ANCOVA results in APA style, researchers typically:

* report the F statistic, degrees of freedom, p-value, and effect size
  for the group effect,
* describe the direction and magnitude of the adjusted group difference,
* mention the covariate and whether it was a significant predictor of
  the outcome.

For example:

    *“Controlling for pre-test study skills, students in the workshop
    condition scored higher on the final exam than those in the control
    condition, :math:`F(1, 77) = 8.42`, :math:`p = .005`, partial
    :math:`\eta^2 = .10`.”*

18.4 Assumptions of ANCOVA
--------------------------

ANCOVA shares many assumptions with regression and ANOVA:

* **Linearity** – the relationship between the covariate and outcome
  is approximately linear within each group.
* **Homogeneity of regression slopes** – the slope relating the covariate
  to the outcome is similar for each group. If the slopes differ
  substantially, a model with an interaction between group and covariate
  may be more appropriate.
* **Independence of observations** – the usual assumption for between-
  subjects designs.
* **Normality and homogeneity of variance** – residuals are approximately
  normal and have similar variance across groups.
* **Reliable covariate** – the covariate should be measured with reasonable
  reliability; noisy covariates provide little benefit and can even reduce
  power.

In practice, researchers check these assumptions using plots (e.g.,
scatterplots and residual plots) and model diagnostics.

18.5 PyStatsV1 Lab – One-way ANCOVA with a pre-test covariate
-------------------------------------------------------------

The Chapter 18 lab shows how to run a simple one-way ANCOVA using a
synthetic psychology dataset.

The script :mod:`scripts.psych_ch18_ancova`:

* simulates data for a control and treatment group,
* includes a **pre-test** covariate that is correlated with the **post-test**
  exam score,
* compares an ordinary one-way ANOVA on the post-test scores to a one-way
  ANCOVA that controls for the pre-test,
* uses :func:`pingouin.ancova` to fit the ANCOVA and report the F statistic,
  p-value, partial :math:`\eta^2`, and adjusted means,
* saves the synthetic dataset and ANCOVA table to the usual
  ``data/synthetic`` and ``outputs/track_b`` folders, and
* produces a simple plot that visualizes the group effect before and after
  adjusting for the covariate.

To run the lab from the command line, use the Makefile target:

.. code-block:: bash

   make psych-ch18

or, equivalently:

.. code-block:: bash

   python -m scripts.psych_ch18_ancova

To run the tests for this chapter only:

.. code-block:: bash

   make test-psych-ch18

As in earlier chapters, the tests provide a lightweight “contract” for the
simulation:

* the covariate must be positively correlated with the outcome,
* the ANCOVA model must show a significant treatment effect when it is
  present in the data-generating process, and
* the adjusted mean for the treatment group should exceed that of the
  control group.

Concept check
-------------

* Why might an experimenter include a pre-test covariate instead of simply
  comparing post-test scores with a t-test or one-way ANOVA?
* What does it mean to say that ANCOVA “controls for” a covariate?
* How are adjusted means different from raw means?
* What does the assumption of homogeneity of regression slopes require?
* How is ANCOVA related to multiple regression?

In the next chapter we will turn to non-parametric statistics – tools that
relax some of the assumptions we have relied on so far and allow us to
analyse ordinal and highly non-normal data.
