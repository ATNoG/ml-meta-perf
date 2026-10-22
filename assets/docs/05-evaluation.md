# 5. Evaluation

> The measurements quoted in this chapter were made on the MCC corpus of the paper (release `v1.0.0` of the
> repository). They are not regenerated: the numbers of a study are in the `report.md` and the CSV files that
> `ml-meta-perf fit` writes next to its equations.

*Implemented in `ml_meta_perf.validate` and `ml_meta_perf.stats`.*

The metric abbreviations used in this chapter are coefficient of determination (**R²**),
Matthews Correlation Coefficient (**MCC**), mean absolute error (**MAE**), root mean squared
error (**RMSE**), symmetric mean absolute percentage error (**SMAPE**), average precision
(**AP**), mean reciprocal rank (**MRR**), hit at rank one (**Hit@1**), and F1 score (**F1**).
The equation labels are **E1** (dataset features only), **E2** (model features only),
**E3-Valid** (the plateau-selected equation using both feature groups), and **E3-MAX**
(the maximum-capability equation using both feature groups).
The evaluation protocols are in-sample (**IS**), leave-one-dataset-out (**LODO**),
leave-one-model-out (**LOMO**), and doubly held out (**DHO**).

## Four protocols

| protocol | fitted on | scored on | question answered |
|---|---|---|---|
| **IS** | all observed rows | the same observed rows | how well does the equation *describe* the data |
| **LODO** | 19 datasets | the held-out 20th, rotated | what will these models score on a dataset nobody has run |
| **LOMO** | 24 models | the held-out 25th, rotated | what will a new model score on datasets we know |
| **DHO** | every row sharing neither the dataset nor the model | one observed (dataset, model) cell, rotated over all observed cells | what will *this* model score on *this* dataset, when neither has been run |

**The fourth is the question the study is actually for**, and it is the strictest of the
four: LODO still shows a predictor the held-out learner on nineteen other
problems, and LOMO still shows it the held-out dataset. Only DHO
denies both. `ml_meta_perf.validate.cross_validate_doubly_held_out` implements it, one refit
per observed cell rather than per group.

It is also where the study's decisions are tested. Every equation's length is read off
`selection.floor_curve`, the **minimum over all four** protocols — so in practice off DHO —
and the shared configuration was chosen by how well E3-Valid's DHO predictions rank the models
([chapter 3](03-term-selection.md#how-the-configuration-itself-was-chosen)). The ranking and
threshold tables report E3-Valid under DHO.

IS describes fit to the observed corpus. Restricted terms, bounded arity, and ridge fitting
limit capacity without guaranteeing absence of overfitting. The differences between IS and
coefficient-refit protocols are useful diagnostics, conditional on global form selection.

## One equation, refit — not one search per fold

The equation's form is selected once; each exclusion refits its term statistics,
intercept, and ridge coefficients. `cross_validate_fixed_form` therefore assesses this
selected form under restricted fitting histories. Because discovery used all rows, these
scores are conditional and can be optimistic for the complete discovery procedure.
Repeating term selection within LODO folds provides a separate sensitivity diagnostic.

### What licenses fixing the form

Term reselection frequencies describe how often the selected expressions return after
a dataset is excluded. A separate diagnostic also scores predictions from those
within-fold searches. It still retains global configuration and requested length, and is
not a fully nested assessment of discovery. Stability of some terms cannot by itself
remove selection optimism from the fixed-form scores.

### What is still shared across folds

The vocabulary, numerical admissibility, initial screening, length selection, and
configuration selection use the full corpus. Only coefficient recalibration and its
standardisation are isolated in the primary refits. This shared information is a limitation
of the fixed-form experiment; a complete discovery assessment needs outer grouped folds.

## Random k-fold is a leak, not a protocol

Dataset meta-features are **constant across a dataset's observed rows**. A random split
therefore puts the same dataset on both sides of the fold and does not test transfer to an
unseen dataset.

On the MCC corpus of the paper (`v1.0.0` tag) the same equation scored far higher under a random split than under
leave-one-dataset-out. The gap changes with the corpus and the form, and a small gap would not make random
splitting a test of transfer to unseen datasets.

This is the same concern as subject-wise splitting in clinical machine learning, and the
same recommendation made by community reporting standards (Walsh et al., 2021).

## Metrics

| metric | units | why |
|---|---|---|
| **R²** | — | variance explained; the comparable headline |
| **MAE** | MCC | the honest error headline — what a practitioner actually asks |
| RMSE | MCC | outlier-sensitive companion to MAE |
| SMAPE | % | scale-free, but see the caveat below |
| Spearman | — | rank quality over all observed rows, insensitive to the MCC ceiling |

Spearman is reported for the regression task and **not** used to compare model rankings.
Computed over all observed rows, it mixes between-dataset and within-dataset order and can assign
correlation even to a predictor that is constant within each held-out group. Average
precision, reciprocal rank, hit@1, and regret instead evaluate each dataset's model list and
weight its head, which is where a recommendation is actually read.

### Where the numbers are

Every equation on every metric under every protocol is in the `scores.csv` and the `report.md` of a
study. This chapter keeps no copy of those numbers: a hand-written table beside a generated one is a
second source of truth for numbers that have exactly one.

Two things about it are worth saying in prose, because they are not visible in the table.

**The hardest trivial predictor is a different one for R² than for MAE and SMAPE.** MAE is
minimised by the median and R² by the mean, so a comparison that reports only mean baselines
is not reading each metric against the predictor that is hardest to beat on it.

### A caveat on R²

R² uses the mean of the evaluated targets in its denominator. Scores over aggregated
dataset means and scores over observed pairs therefore measure different quantities.
All three primary equations here are fitted and scored on the same observed pairs.

### A caveat on SMAPE

SMAPE divides by $|y| + |\hat{y}|$, and the revised corpus contains valid zero-MCC observations. Each
contributes the full 200% unless the prediction is also exactly 0, so the metric is
dominated by the rows the equation is already known to handle worst rather than by its
typical error. It is reported because it is scale-free and was requested, but MAE is the
honest headline for a target that legitimately passes through zero.

## Two questions the R² does not answer

Regression accuracy, threshold qualification, and classifier ranking are different
questions. The equation is evaluated on all three.

### Will this model clear a bar?

Given a dataset and a threshold, does the equation put each model on the right side of it?
`ml_meta_perf.validate.decision_report` scores that at five thresholds, against the accuracy
of always answering with the larger class — the bar any such rule must clear.

The decision metrics are reported at several thresholds, with precision, recall and MCC beside the
three columns named here (`ml_meta_perf.validate.decision_report`).


F1 and binary-decision MCC accompany accuracy because the qualifying class proportions
change with the threshold. The majority-class reference and precision/recall reveal the
trade-off between accepting an inadequate model and rejecting a suitable one.

### Which model should I run?

Ranking relevance means observed MCC within 0.01 of the dataset's best, accommodating
near-best ties. This band is an operational tolerance, not statistical equivalence.
AP, MRR, Hit@1, and regret are averaged equally over datasets. Every result names its
protocol, and historical baselines are compared under matched LODO information.


The per-dataset values expose variation in top-choice regret that an overall mean hides.

## Baselines

An equation earns its place only by beating the obvious alternatives, and there are more of
them than one. Every trivial predictor is reported at **two centres**, because the metrics
disagree about which is the honest opponent: the mean minimises squared error, the median
minimises absolute error. An MAE quoted against a mean baseline is quoted against a predictor
that is not minimising the metric it is being judged on.

Each metric must be compared with the relevant baseline under the same protocol, with both centres
considered and no assumption that one wins on every measure. These centres are computed from the
fitting groups only; the evaluated observation is excluded.

Every per-model baseline also has a limit no metric shows: it is **empty under
LOMO**. A held-out model has no training row, so "how well does this model
usually do" does not exist. The per-dataset baselines remain defined under that protocol;
the per-model baselines are competitors only when model history is available.

### On ranking, compare matched historical baselines

Historical per-model means and medians require observations of that learner on other
datasets, so their matched comparison uses LODO, with paired dataset-level AP comparisons. Intervals containing zero do not establish a
ranking advantage. Under DHO the target learner's history is absent and these baselines
are unavailable.

### Numerical estimation and threshold decisions

The regression tables report numerical estimation; the decision tables report threshold
qualification. They answer different questions from near-best ranking, so an advantage on
one should not be inferred from another. Each comparison retains its stated protocol and
information availability.

## Term stability

Exact-term frequencies across LODO reselection searches (`validate.term_stability`) accompany the
fitted equation. Terms that never reappear receive frequency zero. This measures expression
recurrence, rather than coefficient-sign stability (`experiment._sign_stability`) or independent
discovery performance.

Read regression, ranking, and threshold outcomes separately: a higher MAP need not imply
higher Hit@1 or lower regret. These conditional comparisons do not establish independent
superiority of the full symbolic-discovery procedure.
