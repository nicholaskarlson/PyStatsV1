.. _psych_ch1_thinking_like_a_scientist:

Psychological Science & Statistics – Chapter 1
==============================================

Thinking like a psychological scientist
---------------------------------------

This chapter sets the stage for the rest of the psychology track. Our goals are to
connect:

- the *questions* psychologists ask,
- the *studies* they design, and
- the *analyses* they run,

so that statistics feels like part of the scientific process, not a separate math class.


1.1 Why we cannot just "trust our gut"
--------------------------------------

Everyday life encourages us to rely on intuition: "It feels true, so it must be true."
In psychological science, that is not enough.

Three classic biases:

- **Confirmation bias**

  We notice and remember information that supports what we already believe, and
  ignore disconfirming evidence.

  *Example*: if you believe “screens before bed ruin sleep,” you may pay attention
  to the nights you slept badly after using your phone and forget about good nights.

- **Hindsight bias ("I knew it all along")**

  After we know the outcome, we feel as if it was obvious.

  *Example*: once you hear the result of a study, the conclusion feels inevitable,
  even if you would never have predicted it in advance.

- **Availability heuristic**

  We judge how common something is based on how easily examples come to mind.

  *Example*: you may think exam anxiety is “universal” because anxious students are
  more vivid in memory or more likely to talk about it, even if many students are
  relatively calm.

These biases are **not character flaws**; they are how human cognition works. The
point of scientific methods and statistics is to *protect us from our own minds*.


1.2 The scientific method: systematizing curiosity
--------------------------------------------------

A cartoon version of the scientific method is:

.. math::

   \text{Theory} \;\Rightarrow\; \text{Hypothesis} \;\Rightarrow\; \text{Observation} \;\Rightarrow\; \text{Revision}.

In more detail:

1. **Theory**

   A framework or story about how the world works.

   *Example*: using social media late at night increases arousal and interferes with
   sleep, which increases next-day anxiety.

2. **Hypothesis**

   A specific, testable prediction derived from the theory.

   *Example*: students who restrict screen time in the hour before bed will report
   lower test anxiety than students who do not.

3. **Observation / Study**

   You design a study, collect data, and analyze the results.

4. **Revision**

   You compare the results to the theory:

   - If the data support the hypothesis, your confidence in the theory may increase.
   - If the data do **not** support it, you refine or replace the theory.

Psychological science is not a straight line from theory to proof. It is a looping
process of **asking, testing, and revising**.


1.3 Claims, variables, and hypotheses
-------------------------------------

Research methods courses often distinguish three broad kinds of claims:

- **Frequency claims** – how often something happens.
  - Example: "About 30% of students report test anxiety before exams."

- **Association claims** – how two measured variables move together.
  - Example: "Students with higher sleep quality tend to report lower anxiety."

- **Causal claims** – whether one variable *causes* a change in another.
  - Example: "Mindfulness training reduces test anxiety."

To study any of these, we need **variables**:

- A *variable* is something that can take different values (score, condition, gender,
  reaction time, etc.).
- A *construct* is the underlying idea (test anxiety, motivation, working memory).
- An *operational definition* is how we turn the construct into data.
  - Example: "Test anxiety" → total score on a 10-item questionnaire.

A **hypothesis** links constructs in a testable way, using variables we can measure:

- Construct level: "Higher sleep quality leads to lower anxiety."
- Operational level: "Students with higher scores on the sleep quality scale will
  have lower scores on the anxiety questionnaire."

Throughout this track we will move between:

- *words* (psychological story),
- *variables* (how we measure it), and
- *numbers* (what we analyze in Python).


1.4 The four big validities
---------------------------

When psychologists evaluate a study, they often ask about four types of *validity*:
how well does the study support the claim being made?

- **Construct validity**

  Are we really measuring what we think we are?

  - Does our test anxiety questionnaire actually capture "anxiety about tests"
    rather than general stress or depression?

- **External validity**

  Do the findings generalize beyond this specific sample and setting?

  - Would the results hold at a different university, age group, or culture?

- **Statistical validity**

  Do the numbers support the claim?

  - Are the analyses appropriate for the design and variables?
  - Is the sample big enough to detect realistic effects?
  - Are we interpreting p-values, confidence intervals, and effect sizes correctly?

- **Internal validity**

  Did A really cause B?

  - Are there alternative explanations (confounds)?
  - Was there random assignment, or could pre-existing differences explain the
    results?

No single study is perfect on all four dimensions. The real skill is to:

- match the **claim** to the **design** and **analysis**, and
- be honest about which kinds of validity are strongest (and weakest).


1.5 Where Python and PyStatsV1 fit in
-------------------------------------

So far this chapter has been mostly conceptual. That is intentional: you need a
strong sense of **questions, claims, and validities** before statistics and code
really make sense.

In this mini-book, Python is the environment where we will:

- store and clean psychological data,
- compute descriptive statistics and effect sizes,
- fit models (e.g. t tests, regression, logistic regression),
- create plots that tell clear stories, and
- keep an executable record of what we did.

PyStatsV1 provides:

- example datasets that look like real psychology studies,
- reusable analysis scripts and "labs," and
- documentation (like this mini-book) that connects theory to code.

For now, we just want a **gentle "hello world"**.

.. code-block:: python

   import pandas as pd

   # Replace this path with the actual location in PyStatsV1 once the labs are set up
   data = pd.read_csv("data/study1_sleep_anxiety.csv")

   print(data.head())

   print(data.dtypes)

This tiny script already brings several ideas together:

- a dataset with rows = participants and columns = variables,
- variable *types* (numeric vs categorical) in ``dtypes``,
- a repeatable analysis (you can rerun the same code on the same file).

In later chapters we will build full **PyStatsV1 labs** where:

- each chapter has a corresponding dataset and notebook/script,
- you run analyses that mirror the textbook examples, and
- you can adapt the code to your own projects.


1.6 What you should take away
-----------------------------

By the end of this chapter you should be able to:

- explain why psychological science cannot rely on intuition alone,
- distinguish between frequency, association, and causal claims,
- describe the basic scientific method loop (theory → hypothesis → observation → revision),
- name and briefly define the four big validities (construct, external, statistical, internal),
- see where Python and PyStatsV1 will fit into the research cycle.

When later chapters become more technical, come back here and ask:

- *What is the claim?*  
- *What is the design?*  
- *Which kind(s) of validity are strongest?*  
- *What exactly are the numbers telling us about the psychological story?*
