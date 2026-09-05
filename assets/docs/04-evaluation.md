# 4. Evaluation methodology

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
| leave-one-dataset-out | **0.652** | 0.138 |
| leave-one-model-out | **0.633** | 0.142 |

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
([chapter 6](06-results.md)), so the caveat is a general warning rather than a live hazard
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
| average precision | 0.829 |
| mean reciprocal rank | 0.882 |
| hit@1 — best model ranked first | 0.800 |
| top-1 regret | 0.006 |
| Spearman | 0.703 |

![Per-dataset ranking quality](../figures/ranking_quality.png)

**Read the first four and treat Spearman as weak evidence.** Measured on this corpus it sits
between 0.63 and 0.73 for every predictor *and* every baseline, including a constant, so it
cannot separate the things this study compares. The regret figure is the one in the target's
own units: picking the model E3 ranks first costs 0.006 MCC against the best available.

## Baselines

An equation earns its place only by beating the obvious alternatives:

| baseline | R² | MAE |
|---|---|---|
| global mean (loo-dataset) | -0.040 | 0.293 |
| per-model mean (loo-dataset) | 0.201 | 0.238 |
| global mean (loo-model) | -0.024 | 0.290 |
| per-dataset mean (loo-model) | 0.296 | 0.213 |

E3 beats all four on its respective protocol.

**The ranking caveat that stood through earlier drafts has closed.** For *ranking* models
on a new dataset, the trivial "average MCC of this model elsewhere" baseline used to beat E3.
It no longer does, on any head-weighted measure:

| | AP | MRR | hit@1 | top-1 regret | Spearman |
|---|---|---|---|---|---|
| per-model mean (leave-one-dataset-out) | 0.798 | 0.835 | 0.750 | 0.011 | 0.703 |
| **E3** | **0.829** | **0.882** | **0.800** | **0.006** | 0.703 |

**Spearman is exactly tied at 0.703, and every metric that weights the head of the list is
not.** That is the clearest single demonstration of why this chapter demotes rank correlation:
the two predictors order the whole list about equally well, and only one of them reliably puts
a best model first. E3 halves the regret.

The baseline also has a limit no metric shows: it is **empty under leave-one-model-out**. A
held-out model has no training row, so "average MCC of this model elsewhere" does not exist.
It is a competitor on one protocol and undefined on the other.

E3 also wins clearly on predicting the MCC *value* — 0.652 against the baseline's 0.201,
and that was always the larger gap. Knowing which models are generally good is most of what
*ranking* needs, which is why the baseline was so hard to beat there; knowing *how well* a
particular model will do on a particular dataset is what needs the meta-features, and the
baseline has nothing to say about it. The practical reading is unchanged for anyone asking
"will this reach MCC 0.8 on my data" and improved for anyone asking "which of these 25
models should I run".

## Term stability

A fitted equation presents a term chosen in 19 of 20 folds and one chosen in 3
identically. `CrossValidation.stability()` counts selection frequency across folds, and no
extracted practice ([chapter 7](07-practices.md)) is published from a term below 50%.
