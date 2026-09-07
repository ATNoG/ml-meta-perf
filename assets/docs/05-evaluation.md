# 5. Evaluation

*Implemented in `ml_meta_perf.validate` and `ml_meta_perf.stats`.*

## Protocols

| protocol | fitted on | scored on | question answered |
|---|---|---|---|
| **in-sample** | all 476 rows | the same 476 rows | how well does the equation *describe* the data |
| **leave-one-dataset-out** | 19 datasets | the held-out 20th, rotated | what will these models score on a dataset nobody has run |
| **leave-one-model-out** | 24 models | the held-out 25th, rotated | what will a new model score on datasets we know |

In-sample is reported as a first-class result rather than dismissed. Term count is capped
and terms are drawn from a screened pool, so this is **equation fitting, not model
fitting**: the capacity to memorise 476 rows with 20 terms is limited, and the gap between
in-sample and cross-validated columns is itself the diagnostic. For contrast, a
RandomForest reaches 0.910 in-sample and 0.067 leave-one-dataset-out on the same features.

Leave-one-dataset-out is the harder and more useful number, and is the one the study
leads with for transfer claims.

## One equation, refit — not one search per fold

**The equation's form is chosen once. In every fold, only its weights are refit.** That is
`ml_meta_perf.validate.cross_validate_fixed_form`, and it is a position about what the
artefact is rather than a statistical convenience.

The form of an equation is its conceptual claim — a statement about which quantities govern
how well a learner does on a dataset. For a simpler problem one would write that form down
from domain expertise and never search for it at all. What cross-validation then asks is
whether the claim survives data it has not seen, with its constants recalibrated. It does not
ask whether a search would have rediscovered the same form, because that is a question about
the search.

Re-running selection inside every fold answers that other question, and answering it *as
though it were the first* has a cost: twenty folds fit twenty different equations, so the
pooled R² is an average over twenty models. Measured here, that made leave-one-dataset-out
move by as much as 0.3 when the requested length changed by two, while the fixed form varies
by 0.014 over the same range. The reported curve is smooth because it is one equation being
asked the same question at different lengths.

### What licenses fixing the form

A form arrived at by searching all 476 rows is only a claim about the phenomenon if a
different sample would have produced it. That is measured rather than assumed.
`ml_meta_perf.validate.fold_selections` re-runs selection in each fold and returns **term
names and no predictions** — so the re-selecting protocol cannot produce a reported score
even by accident — and `term_stability` counts how often each of the published equation's
terms comes back when a fifth of the data is removed. A form whose terms churn fold to fold
has not earned this protocol. The table is in [chapter 10](10-report.md).

### What is still shared across folds

The term *vocabulary* — which terms exist at all — is built from the feature columns, not the
target, and is shared. That is required rather than conceded: rebuilding it per fold would
make a term *admissible* on the training rows but undefined on the held-out ones, and the
fold would score `NaN` rather than score badly. Admissibility is a property of the feature
columns and has to be decided once, globally.

## Random k-fold is a leak, not a protocol

Dataset meta-features are **constant across a dataset's 25 rows**. A random split
therefore puts the same dataset on both sides of the fold, and the equation recognises the
dataset rather than generalising to it.

The **same equation**, three protocols:

| protocol | R² | MAE |
|---|---|---|
| random 10-fold | **0.651** | 0.138 |
| leave-one-dataset-out | **0.627** | 0.144 |
| leave-one-model-out | **0.622** | 0.142 |

**The three now agree, and that is itself the finding.** A random split used to score 0.083
above leave-one-dataset-out, because putting rows from the same dataset on both sides of the
split let the equation recognise the dataset rather than generalise to it. The gap has closed
to 0.001. Two changes did it, and neither was a better search: the equation's *form* is now
fixed and only its weights are refit per fold (below), so there is far less for a leaky split
to leak into; and the model features are now constant within a learner rather than partly
functions of the dataset.

A gap of that kind is a diagnostic worth keeping even when it reads zero.
`ml_meta_perf.validate.random_kfold_groups` exists **only** to produce this comparison; it is
never used to score a result.

This is the same concern as subject-wise splitting in clinical machine learning, and the
same recommendation made by community reporting standards (Walsh et al., 2020).

## Metrics

| metric | units | why |
|---|---|---|
| **R²** | — | variance explained; the comparable headline |
| **MAE** | MCC | the honest error headline — what a practitioner actually asks |
| RMSE | MCC | outlier-sensitive companion to MAE |
| SMAPE | % | scale-free, but see the caveat below |
| Spearman | — | rank quality, insensitive to the MCC ceiling |

### A caveat on R²

R² is computed against the mean of the **evaluated** rows, so its denominator is the
variance of whatever is being scored. Two R² values computed over different row sets share
no denominator and **cannot be compared**.

This used to bite here. E1 was fitted and scored on the 20 aggregated per-dataset means,
which put its 0.506 on a twenty-point denominator beside E3's on a 476-row one —
inviting exactly the comparison the caveat forbids, and in the direction that flatters the
control. All three equations are now fitted and scored on the same 476 rows
([chapter 4](04-equation.md)), so the caveat is a general warning rather than a live hazard
in this study's own tables. E1's transfer figure on the common scale is 0.341.

### A caveat on SMAPE

SMAPE divides by $|y| + |\hat{y}|$, and **15 of the 476 rows have MCC exactly 0**. Each
contributes the full 200% unless the prediction is also exactly 0, so the metric is
dominated by the rows the equation is already known to handle worst rather than by its
typical error. It is reported because it is scale-free and was requested, but MAE is the
honest headline for a target that legitimately passes through zero.

![Mean absolute error against equation length](../figures/error_curve_mae.png)

## Two questions the R² does not answer

An R² of 0.65 is a statement about how precisely the equation states an MCC. Nobody asks it
that. The two questions people actually ask are coarser, and the equation is graded on both.

### Will this model clear a bar?

Given a dataset and a threshold, does the equation put each model on the right side of it?
`ml_meta_perf.validate.decision_report` scores that at five thresholds, against the accuracy
of always answering with the larger class — the bar any such rule must clear.

| threshold | accuracy | F1 | MAP | majority-class baseline |
|---|---|---|---|---|
| 0.5 | 0.899 | 0.935 | 0.976 | 0.767 |
| 0.6 | 0.882 | 0.919 | 0.978 | 0.733 |
| 0.7 | 0.861 | 0.891 | 0.948 | 0.668 |
| 0.8 | 0.807 | 0.825 | 0.948 | 0.626 |
| 0.9 | 0.773 | 0.724 | 0.923 | 0.508 |

![Decision quality against threshold](../figures/decision_quality.png)

F1 is reported beside accuracy because the classes are unbalanced at the outer thresholds and
accuracy alone hides it: the majority rule reaches 0.51 accuracy at a threshold of 0.9 at an
F1 of exactly zero. **A regression too imprecise to state an MCC is accurate on the decision**,
because thresholding discards exactly the precision it lacks.

### Which model should I run?

Only the head of the list is ever used — a practitioner tries the top few and never sees the
tail — so the ranking is graded the way a search engine is. A model counts as a right answer
if its MCC is within 0.01 of the best on its dataset, rather than by a fixed top-k cut: 80 of
the 476 rows sit at exactly MCC 1.0, so many datasets have several genuinely tied best models
and a top-3 rule would score a correct answer as a miss.

| metric | E3 |
|---|---|
| average precision | 0.822 |
| mean reciprocal rank | 0.867 |
| hit@1 — best model ranked first | 0.800 |
| top-1 regret | 0.006 |
| Spearman | 0.706 |

![Per-dataset ranking quality](../figures/ranking_quality.png)

**Read the first four and treat Spearman as weak evidence.** Measured on this corpus it sits
between 0.63 and 0.73 for every predictor *and* every baseline, including a constant, so it
cannot separate the things this study compares. The regret figure is the one in the target's
own units: picking the model E3 ranks first costs 0.006 MCC against the best available.

## Baselines

An equation earns its place only by beating the obvious alternatives, and there are more of
them than one. Every trivial predictor is reported at **two centres**, because the metrics
disagree about which is the honest opponent: the mean minimises squared error, the median
minimises absolute error. An MAE quoted against a mean baseline is quoted against a predictor
that is not minimising the metric it is being judged on.

| baseline | R² | MAE | SMAPE |
|---|---|---|---|
| global mean (loo-dataset) | -0.040 | 0.293 | 50.9 |
| per-model mean (loo-dataset) | 0.201 | 0.238 | 47.3 |
| global mean (loo-model) | -0.024 | 0.290 | 50.7 |
| per-dataset mean (loo-model) | **0.296** | 0.213 | 43.8 |
| global median (loo-dataset) | -0.303 | 0.258 | 43.6 |
| per-model median (loo-dataset) | 0.090 | 0.226 | 45.2 |
| global median (loo-model) | -0.294 | 0.256 | 43.4 |
| per-dataset median (loo-model) | 0.129 | **0.192** | **40.3** |

The two halves of that table disagree, and predictably: the strongest R² baseline is a mean
(0.296 against 0.129) and the strongest MAE and SMAPE baselines are medians (0.192 against
0.213, 40.3 against 43.8). Each metric has to be read against whichever is harder on it.
E3 clears both — 0.627 R², 0.144 MAE, 35.2 SMAPE under leave-one-dataset-out.

All of these are computed leave-one-group-out. With 17–25 rows per group, letting the row
being predicted into its own group's centre inflates the per-model mean's R² by 0.082;
`validate.baseline_group_centre` does it correctly and a naive group centre does not.

Every group-conditioned baseline also has a limit no metric shows: it is **empty under
leave-one-model-out**. A held-out model has no training row, so "how well does this model
usually do" does not exist. It is a competitor on one protocol and undefined on the other.

### On ranking, the equation does not beat "use whatever usually works"

This is the honest result, and an earlier draft of this chapter got it wrong in a way worth
recording, because the mistake is the one this chapter exists to warn about.

| | AP | MRR | hit@1 | top-1 regret |
|---|---|---|---|---|
| per-model mean (leave-one-dataset-out) | 0.798 | 0.835 | 0.750 | 0.011 |
| **E3** | 0.822 | 0.867 | 0.800 | 0.008 |
| per-model **median** (leave-one-dataset-out) | **0.837** | **0.885** | **0.850** | 0.009 |

Read as means, E3 beats the mean baseline and loses to the median one. **Neither reading
survives a paired test.** Compared dataset by dataset with `validate.paired_comparison` —
exact sign test plus a bootstrap over the twenty groups:

| comparison | mean difference in AP | equation better on | p | 95% CI |
|---|---|---|---|---|
| E3 vs per-model mean | +0.024 | 7 of the 17 that differ | 0.63 | [-0.068, +0.133] |
| E3 vs per-model median | -0.016 | 6 of the 17 that differ | 0.33 | [-0.064, +0.040] |

Every interval spans zero. **On ranking, the equation is indistinguishable from ordering the
models by how well they usually do.** An earlier draft announced that this caveat "has
closed", on the strength of the +0.024 in the first row alone — a difference of two means
over twenty folds, which is precisely what this study has three times mistaken for a result.
Note also that `hit@1` can only move in steps of 0.05 on twenty datasets, so the 0.800
against 0.750 is two datasets flipping their top pick.

That is not a failure of the equation so much as a statement about the problem: knowing which
models are generally good is most of what *ranking* needs, which is why the trivial baseline
is so hard to beat there. The generated report states this verdict from the paired test
rather than from the table, so it cannot drift back into the optimistic reading.

### On predicting the value, and on the go/no-go decision, it wins clearly

| | R² (loo-dataset) | threshold MCC @ 0.7 |
|---|---|---|
| per-model mean | 0.201 | 0.439 |
| per-model median | 0.090 | 0.304 |
| **E3** | **0.627** | **0.683** |

Knowing *how well* a particular model will do on a particular dataset is what needs the
meta-features, and no group centre has anything to say about it. The same holds once the
prediction is thresholded into the go/no-go rule a practitioner actually asks for: against a
majority-class floor of 0.668 accuracy, E3 reaches MCC 0.683 where the trivial predictors
reach 0.439 and 0.304.

So the summary across the three questions is: **clearly better at predicting the value,
clearly better at the threshold decision, and no better at ranking.**

## Term stability

A fitted equation presents a term chosen in 19 of 20 folds and one chosen in 3
identically. `CrossValidation.stability()` counts selection frequency across folds, and no
extracted practice ([chapter 6](06-practices.md)) is published from a term below 50%.

## Flexible models do worse, not better

Standard regressors on the same raw features, under the same protocols:

| model | in-sample R² | LOO-dataset R² | LOO-model R² |
|---|---|---|---|
| RidgeCV (linear, 18 features) | 0.418 | **-2.002** | 0.328 |
| RandomForest (300 trees) | **0.910** | **0.067** | 0.465 |
| GradientBoosting | 0.820 | 0.049 | 0.354 |
| **ml-meta-perf E3 (15 terms)** | 0.658 | **0.638** | 0.622 |

Read the RandomForest row across. With 20 dataset groups a forest memorises dataset
identity almost perfectly and then transfers worse than a 15-term additive equation. This is also
the likely provenance of the R² ≈ 0.9 figures reported for opaque meta-models: an
in-sample or randomly-split forest reproduces them exactly, and the same forest is
near-useless on an unseen dataset.

> Measured with scikit-learn during exploration. It is not a dependency of the package.
