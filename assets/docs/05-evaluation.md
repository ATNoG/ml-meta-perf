# 5. Evaluation

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
| **IS** | all 476 rows | the same 476 rows | how well does the equation *describe* the data |
| **LODO** | 19 datasets | the held-out 20th, rotated | what will these models score on a dataset nobody has run |
| **LOMO** | 24 models | the held-out 25th, rotated | what will a new model score on datasets we know |
| **DHO** | every row sharing neither the dataset nor the model | one observed (dataset, model) cell, rotated over all 476 | what will *this* model score on *this* dataset, when neither has been run |

**The fourth is the question the study is actually for**, and it is the strictest of the
four: LODO still shows a predictor the held-out learner on nineteen other
problems, and LOMO still shows it the held-out dataset. Only DHO
denies both, which makes it the one place an equation and an opaque regressor are denied the
same things. `ml_meta_perf.validate.cross_validate_doubly_held_out` implements it, one refit
per observed cell rather than per group.

It is also where the study's decisions are tested. Every equation's length is read off
`selection.floor_curve`, the **minimum over all four** protocols — so in practice off DHO —
and the shared configuration was chosen by how well E3-Valid's DHO predictions rank the models
([chapter 3](03-term-selection.md#how-the-configuration-itself-was-chosen)). The ranking and
threshold tables report E3-Valid under DHO.

IS is reported as a first-class result rather than dismissed. Term count is capped
and terms are drawn from a screened pool, so this is **equation fitting, not model
fitting**: the capacity to memorise 476 rows with a couple of dozen terms at most is limited,
and the gap between IS and the other three is itself the diagnostic. For contrast, a random
forest on all eighteen raw corpus columns fits almost perfectly under IS and reaches little
under LODO and DHO — the numbers are in the generated
[opaque section](#what-an-opaque-model-reaches-and-does-not) below, and that ordering is the
whole of the argument this study makes for a readable form.

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
pooled R² is an average over twenty models. Measured here, that made LODO
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
has not earned this protocol. The table is in the generated section below.

### What is still shared across folds

The term *vocabulary* — which terms exist at all — is built from the feature columns, not the
target, and is shared. That is required rather than conceded: rebuilding it per fold would
make a term *admissible* on the training rows but undefined on the held-out ones, and the
fold would score `NaN` rather than score badly. Admissibility is a property of the feature
columns and has to be decided once, globally.

## Random k-fold is a leak, not a protocol

Dataset meta-features are **constant across a dataset's observed rows**. A random split
therefore puts the same dataset on both sides of the fold and does not test transfer to an
unseen dataset.

The same equation under all three splits is the generated
[Why a random split is not a protocol](#why-a-random-split-is-not-a-protocol) table below —
this section does not keep a second copy, for the reason given
[above](#where-the-numbers-are).

**The three now nearly agree, and that is itself the finding.** A random split used to score
0.083 above LODO, because putting rows from the same dataset on both sides of the split let the
equation recognise the dataset rather than generalise to it; the current gap is in the
generated table below. Two changes closed it, and neither was a better search: the equation's *form* is now
fixed and only its weights are refit per fold (below), so there is far less for a leaky split
to leak into; and only `Processing Units Number` now varies within a learner, rather than
several cost-related model columns varying with the dataset.

A gap of that kind is a diagnostic worth keeping even when it reads zero.
`ml_meta_perf.validate.random_kfold_groups` exists **only** to produce this comparison; it is
never used to score a result.

This is the same concern as subject-wise splitting in clinical machine learning, and the
same recommendation made by community reporting standards (Walsh et al., 2021).

## Metrics

| metric | units | why |
|---|---|---|
| **R²** | — | variance explained; the comparable headline |
| **MAE** | MCC | the honest error headline — what a practitioner actually asks |
| RMSE | MCC | outlier-sensitive companion to MAE |
| SMAPE | % | scale-free, but see the caveat below |
| Spearman | — | rank quality over all 476 rows, insensitive to the MCC ceiling |

Spearman is reported for the regression task and **not** used to compare model rankings.
Computed over all 476 rows, it mixes between-dataset and within-dataset order and can assign
correlation even to a predictor that is constant within each held-out group. Average
precision, reciprocal rank, hit@1, and regret instead evaluate each dataset's model list and
weight its head, which is where a recommendation is actually read.

### Where the numbers are

Every equation on every metric under every protocol, and every trivial predictor beside them,
is in the generated [How well it does](#how-well-it-does) section below. **This chapter used
to carry its own copy of that table and it went stale twice** — once when the fourth protocol
was added and the copy still showed three, and once when the three equations were put on one
configuration and E1 and E2 both moved. A hand-written table beside a generated one is a
second source of truth for numbers that have exactly one.

Two things about it are worth saying in prose, because they are not visible in the table.

**The hardest trivial predictor is a different one for R² than for MAE and SMAPE.** MAE is
minimised by the median and R² by the mean, so a comparison that reports only mean baselines
is not reading each metric against the predictor that is hardest to beat on it. Both centres
are reported, which is what the `at both centres` sub-table below is for.

**E3's margin is widest on R² and narrowest on SMAPE**, and that is the SMAPE caveat below
doing its work rather than a weakness: the metric is dominated by the 15 rows at exactly
MCC = 0, where a median baseline predicting near zero scores well.

### A caveat on R²

R² is computed against the mean of the **evaluated** rows, so its denominator is the
variance of whatever is being scored. Two R² values computed over different row sets share
no denominator and **cannot be compared**.

This used to bite here. E1 was fitted and scored on the 20 aggregated per-dataset means,
which put its 0.506 on a twenty-point denominator beside E3's on a 476-row one —
inviting exactly the comparison the caveat forbids, and in the direction that flatters the
control. All three equations are now fitted and scored on the same 476 rows
([chapter 4](04-equation.md)), so the caveat is a general warning rather than a live hazard
in this study's own tables. E1's transfer figure on the common scale is in the table below.

### A caveat on SMAPE

SMAPE divides by $|y| + |\hat{y}|$, and **15 of the 476 rows have MCC exactly 0**. Each
contributes the full 200% unless the prediction is also exactly 0, so the metric is
dominated by the rows the equation is already known to handle worst rather than by its
typical error. It is reported because it is scale-free and was requested, but MAE is the
honest headline for a target that legitimately passes through zero.

## Two questions the R² does not answer

An R² of 0.65 is a statement about how precisely the equation states an MCC. Nobody asks it
that. The two questions people actually ask are coarser, and the equation is graded on both.

### Will this model clear a bar?

Given a dataset and a threshold, does the equation put each model on the right side of it?
`ml_meta_perf.validate.decision_report` scores that at five thresholds, against the accuracy
of always answering with the larger class — the bar any such rule must clear.

The table is in the generated section below, at all five thresholds and with precision,
recall and MCC beside the three columns named here. It is not repeated at this point in the
chapter: a hand-copied version stood here until 2026-09-07 and had drifted from the generated
one 240 lines further down the same file.

![Decision quality against threshold](../figures/07_decision_quality.png)

The legend uses the four protocol abbreviations: **IS**, **LODO**, **LOMO**, and **DHO**.

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

The equation and every competitor are in the generated [Acting on it](#acting-on-it) tables
below, **each row naming the protocol it was scored under** — which matters more here than
anywhere else in the chapter, because a ranking scored under IS and a ranking scored with
both the dataset and the model held out are not the same claim. The four E3 rows differ by
0.011 of average precision across the four protocols, and the strictest one, DHO, is
what this section's conclusions are read from.

![Per-dataset ranking quality](../figures/06_ranking_quality.png)

In the figure, a star marks a **top choice within 0.01 MCC of the dataset's best** observed
model, matching the relevance tolerance used by AP, MRR, and hit@1.

On six of the twenty held-out datasets the top pick is the dataset's best model to within the
relevance tolerance, so the regret is exactly zero. The largest misses are IoTID20 at 0.062,
KPI-KQI at 0.061, IoT-APD at 0.058, and Social Network Ads at 0.057 — variation a mean over
twenty folds hides, and the reason this is plotted per dataset rather than summarised.

## Baselines

An equation earns its place only by beating the obvious alternatives, and there are more of
them than one. Every trivial predictor is reported at **two centres**, because the metrics
disagree about which is the honest opponent: the mean minimises squared error, the median
minimises absolute error. An MAE quoted against a mean baseline is quoted against a predictor
that is not minimising the metric it is being judged on.

The table is generated: [The trivial predictors, at both
centres](#the-trivial-predictors-at-both-centres), with the strongest of each kind picked out
per metric beneath it.

The two halves of it disagree, and predictably: the strongest R² baseline is a mean and the
strongest MAE and SMAPE baselines are medians. Each metric has to be read against whichever
is harder on it, and E3 clears both on all three.

All of these are computed leave-one-group-out. With 17–25 rows per group, letting the row
being predicted into its own group's centre inflates the per-model mean's R² by 0.082;
`validate.baseline_group_centre` does it correctly and a naive group centre does not.

Every per-model baseline also has a limit no metric shows: it is **empty under
LOMO**. A held-out model has no training row, so "how well does this model
usually do" does not exist. The per-dataset baselines remain defined under that protocol;
the per-model baselines are competitors only when model history is available.

### On ranking, the equation does not beat "use whatever usually works"

Ranking is the one of the three questions where the equation does not clear the trivial
predictors, and reading it correctly needs the paired test rather than the means.

The comparison is in the generated section below, and it carries **four rows for the equation
and two for the baselines**. The four are the four protocols — IS, dataset held out,
model held out, and both held out — because a ranking comparison is only a comparison if
every side is scored under the same one, and this went wrong twice: the ranking was once
scored under IS against leave-one-out baselines, and the leave-one-out baselines then turned
out to be handed the model identity the equation is denied. Every row names its protocol.

Read as means, both per-model centres lead the DHO equation on AP, MRR,
Hit@1 and regret. Compared dataset by dataset with `validate.paired_comparison`, the AP
differences against both centres are significant. This is a qualified comparison because a
per-model centre uses observations of that model on other datasets and is undefined when the
model itself is held out.

**On ranking, ordering models by how well they usually do remains stronger whenever model
history is available.** Note also that `hit@1` moves only in steps of 0.05 on twenty datasets, so a
0.05 gap is one dataset changing its top pick — which is why a difference of two means over
twenty folds is not a measurement here.

That is a statement about the problem rather than a failure of the equation: knowing which
models are generally good is most of what *ranking* needs, which is why a trivial baseline is
hard to beat there.

**The figures show the strictest protocol; the tables show all four.** A figure has room for
one line per dataset and the honest one to draw is DHO, where neither
the dataset nor the model was in the fit. The tables keep every protocol beside it, because
the spread between them is itself the measurement — it is the cost of generalisation on this
task, and it cannot be read off a single line.

### On predicting the value, and on the go/no-go decision, it wins clearly

Knowing *how well* a particular model will do on a particular dataset is what needs the
meta-features, and no group centre has anything to say about it: the trivial predictors sit
far below the equation on R², and the generated tables below give both, each row naming its
protocol. The same holds once the prediction is thresholded into the go/no-go rule a
practitioner actually asks for — the decision table reports the equation and both trivial
centres at every threshold, against the majority-class floor any such rule has to clear.

So the summary across the three questions is: **clearly better at predicting the value,
clearly better at the threshold decision, and no better at ranking.**

## Term stability

A fitted equation presents terms with very different support: the most stable appears in 18
of 20 folds, while two appear in only one. `CrossValidation.stability()` counts selection frequency across folds, and no
extracted practice ([chapter 6](06-practices.md)) is published from a term below 50%.

## Flexible models do worse, not better

This is the other half of the trade the study is making, and it has to be priced rather than
asserted. Three standard regressors — a cross-validated ridge, a random forest and a gradient
boosting ensemble — are fitted on all eighteen corpus features and scored under the same
protocols, with the same clip to the training fold's range. **The table is generated on every
run**, in the section below; it was measured once by hand until 2026-09-07, which for a study
whose argument is that its analysis is generated was the wrong way round, and the hand-copied
numbers had drifted.

This gives the opaque models two descriptors that the retained E3 equation omits, so the
comparison does not withhold available information from them. The shape of the result is the point:

**A forest fits this meta-data almost perfectly and cannot generalise across datasets.** With
twenty dataset groups and dataset features constant within a group, it identifies the dataset
and looks the answer up. Identification is worth nothing on a dataset nobody has run, which
is why the IS and LODO columns of that row have to be read together.

**Every protocol is reported, and the ordering of the four columns is the finding.** Removing
a whole model costs the forest little; removing a whole dataset costs it almost everything;
removing both leaves it approximately at the corpus-mean R² reference of zero. That is the
signature of a model that learned which dataset a row came from rather than a relationship.

**The DHO column is the like-for-like comparison, and the only one.** Under
LODO a forest still has the held-out learner on nineteen other problems;
under LOMO it still has the held-out dataset. Only with both removed is it
denied what the equation is denied — and it is also the protocol on which the trivial
per-model baselines cannot be computed at all, since a model held out of every fold has no
rows to average. A feature-based predictor still predicts.

**And it loses on the two decisions as well, to the equation *and* to the trivial
predictors.** This is the part that was never measured before. On the go/no-go decision at a
0.7 threshold the opaque models reach an MCC around 0.27, against the equation's 0.66 under
the same DHO protocol and 0.44 for a per-model mean scored under an easier one —
so a forest is worse at deciding whether a model will clear a bar than "how well does this
model usually do". On ranking they do not reach the equation either, and again fall short of
the per-model median. Both comparisons are in the generated ranking and decision tables, with
**every row naming its protocol** and every opaque estimator appearing under all four.

The honest summary is not that opaque models are bad at this. It is that **the accuracy this
study traded away was not there to be had** under a protocol where the dataset is genuinely
unseen, so the trade cost less than the R² gap under IS suggests.

<!-- generated: do not edit below -->

## How well it does

| protocol | R² | MAE | RMSE | n |
|---|---|---|---|---|
| IS | 0.6815 | 0.1302 | 0.1937 | 476 |
| LODO | 0.6513 | 0.1368 | 0.2026 | 476 |
| LOMO | 0.6261 | 0.1408 | 0.2098 | 476 |
| DHO | 0.6151 | 0.1438 | 0.2129 | 476 |

Cross-validated rows hold out a whole dataset or a whole model, so the equation is scored on a group it has never seen. That is the number that matters, and it is well below the IS value at this sample size.

Against the baselines and the ceiling that bounds any additive equation:

| equation | n_terms | r2 | mae | rmse | smape | spearman | n |
|---|---|---|---|---|---|---|---|
| E1, dataset only (5 terms) | 5 | 0.3394 | 0.2056 | 0.2789 | 42.5459 | 0.6322 | 476 |
| E1 reference: true dataset means |  | 0.3539 | 0.2042 | 0.2758 | 42.7254 | 0.6533 | 476 |
| E2, model only (5 terms) | 5 | 0.2537 | 0.2331 | 0.2965 | 46.4330 | 0.4460 | 476 |
| E2 reference: true model means |  | 0.2821 | 0.2257 | 0.2908 | 45.8786 | 0.4870 | 476 |
| E3-Valid, dataset + model (17 terms) | 17 | 0.6815 | 0.1302 | 0.1937 | 34.0670 | 0.8413 | 476 |
| E3-MAX, arity 3 (29 terms) | 29 | 0.7271 | 0.1187 | 0.1793 | 34.1240 | 0.8392 | 476 |
| reference: additive mean-based reference |  | 0.6605 | 0.1447 | 0.2000 | 35.2889 | 0.8100 | 476 |

#### The trivial predictors, at both centres

| baseline | r2 | mae | rmse | smape | spearman | n |
|---|---|---|---|---|---|---|
| global mean (LODO) | -0.0396 | 0.2928 | 0.3499 | 50.9160 | -0.6527 | 476 |
| per-model mean (LODO) | 0.2006 | 0.2382 | 0.3068 | 47.3028 | 0.3885 | 476 |
| global mean (LOMO) | -0.0239 | 0.2905 | 0.3473 | 50.6571 | -0.4787 | 476 |
| per-dataset mean (LOMO) | 0.2964 | 0.2132 | 0.2879 | 43.7772 | 0.5808 | 476 |
| global median (LODO) | -0.3032 | 0.2580 | 0.3918 | 43.6423 | -0.6749 | 476 |
| per-model median (LODO) | 0.0899 | 0.2256 | 0.3274 | 45.1719 | 0.3968 | 476 |
| global median (LOMO) | -0.2938 | 0.2556 | 0.3904 | 43.3930 | -0.4798 | 476 |
| per-dataset median (LOMO) | 0.1291 | 0.1920 | 0.3203 | 40.3032 | 0.6069 | 476 |
| additive mean-based reference (IS) | 0.6605 | 0.1447 | 0.2000 | 35.2889 | 0.8100 | 476 |

Every trivial predictor appears at its mean and at its median, because the metrics disagree about which is the honest opponent: the mean minimises squared error and the median minimises absolute error, so an MAE quoted against a mean baseline is quoted against a predictor that is not minimising the metric it is judged on. Reading the strongest baseline of each kind, per metric:

| metric | strongest mean baseline | strongest median baseline | harder |
|---|---|---|---|
| R2 | per-dataset mean (LOMO) (0.2964) | per-dataset median (LOMO) (0.1291) | **mean** |
| MAE | per-dataset mean (LOMO) (0.2132) | per-dataset median (LOMO) (0.1920) | **median** |
| SMAPE | per-dataset mean (LOMO) (43.7772) | per-dataset median (LOMO) (40.3032) | **median** |

## Acting on it

R² is the wrong question for a practitioner, who asks whether a model will work on some data rather than what its MCC will be to three decimals. Thresholding both the prediction and the truth turns the equation into a go/no-go rule, scored here on held-out datasets:

| threshold | accuracy | majority | precision | recall | mcc | f1 | map | n_positive |
|---|---|---|---|---|---|---|---|---|
| 0.5000 | 0.8739 | 0.7668 | 0.9111 | 0.9260 | 0.6412 | 0.9185 | 0.9659 | 365 |
| 0.6000 | 0.8697 | 0.7332 | 0.9309 | 0.8883 | 0.6822 | 0.9091 | 0.9756 | 349 |
| 0.7000 | 0.8298 | 0.6681 | 0.9217 | 0.8145 | 0.6466 | 0.8648 | 0.9489 | 318 |
| 0.8000 | 0.8130 | 0.6261 | 0.9563 | 0.7349 | 0.6573 | 0.8311 | 0.9391 | 298 |
| 0.9000 | 0.7899 | 0.5084 | 0.9551 | 0.6157 | 0.6239 | 0.7487 | 0.9235 | 242 |

`majority` is the floor any such rule has to clear. The harder comparison is a predictor that answers "how well does this model usually do", thresholded the same way — at both centres, for the reason the error metrics report both:

| predictor | threshold | accuracy | majority | precision | recall | mcc | f1 | map | n_positive |
|---|---|---|---|---|---|---|---|---|---|
| equation (IS) | 0.5000 | 0.8950 | 0.7668 | 0.9245 | 0.9397 | 0.7011 | 0.9321 | 0.9772 | 365 |
| equation (IS) | 0.6000 | 0.8845 | 0.7332 | 0.9375 | 0.9026 | 0.7156 | 0.9197 | 0.9812 | 349 |
| equation (IS) | 0.7000 | 0.8466 | 0.6681 | 0.9359 | 0.8270 | 0.6829 | 0.8781 | 0.9536 | 318 |
| equation (IS) | 0.8000 | 0.8298 | 0.6261 | 0.9502 | 0.7685 | 0.6785 | 0.8497 | 0.9302 | 298 |
| equation (IS) | 0.9000 | 0.8025 | 0.5084 | 0.9568 | 0.6405 | 0.6442 | 0.7673 | 0.9237 | 242 |
| equation (LODO) | 0.5000 | 0.8866 | 0.7668 | 0.9169 | 0.9370 | 0.6754 | 0.9268 | 0.9723 | 365 |
| equation (LODO) | 0.6000 | 0.8782 | 0.7332 | 0.9343 | 0.8968 | 0.7010 | 0.9152 | 0.9785 | 349 |
| equation (LODO) | 0.7000 | 0.8424 | 0.6681 | 0.9355 | 0.8208 | 0.6758 | 0.8744 | 0.9506 | 318 |
| equation (LODO) | 0.8000 | 0.8235 | 0.6261 | 0.9534 | 0.7550 | 0.6709 | 0.8427 | 0.9280 | 298 |
| equation (LODO) | 0.9000 | 0.8067 | 0.5084 | 0.9573 | 0.6488 | 0.6510 | 0.7734 | 0.9200 | 242 |
| equation (LOMO) | 0.5000 | 0.8739 | 0.7668 | 0.9067 | 0.9315 | 0.6373 | 0.9189 | 0.9646 | 365 |
| equation (LOMO) | 0.6000 | 0.8824 | 0.7332 | 0.9373 | 0.8997 | 0.7114 | 0.9181 | 0.9735 | 349 |
| equation (LOMO) | 0.7000 | 0.8445 | 0.6681 | 0.9296 | 0.8302 | 0.6754 | 0.8771 | 0.9468 | 318 |
| equation (LOMO) | 0.8000 | 0.8067 | 0.6261 | 0.9518 | 0.7282 | 0.6454 | 0.8251 | 0.9227 | 298 |
| equation (LOMO) | 0.9000 | 0.7920 | 0.5084 | 0.9554 | 0.6198 | 0.6273 | 0.7519 | 0.9249 | 242 |
| equation (DHO) | 0.5000 | 0.8739 | 0.7668 | 0.9111 | 0.9260 | 0.6412 | 0.9185 | 0.9659 | 365 |
| equation (DHO) | 0.6000 | 0.8697 | 0.7332 | 0.9309 | 0.8883 | 0.6822 | 0.9091 | 0.9756 | 349 |
| equation (DHO) | 0.7000 | 0.8298 | 0.6681 | 0.9217 | 0.8145 | 0.6466 | 0.8648 | 0.9489 | 318 |
| equation (DHO) | 0.8000 | 0.8130 | 0.6261 | 0.9563 | 0.7349 | 0.6573 | 0.8311 | 0.9391 | 298 |
| equation (DHO) | 0.9000 | 0.7899 | 0.5084 | 0.9551 | 0.6157 | 0.6239 | 0.7487 | 0.9235 | 242 |
| per-model mean (LODO) | 0.5000 | 0.7395 | 0.7668 | 0.8005 | 0.8795 | 0.1842 | 0.8381 | 0.9788 | 365 |
| per-model mean (LODO) | 0.6000 | 0.6828 | 0.7332 | 0.8113 | 0.7393 | 0.2506 | 0.7736 | 0.9567 | 349 |
| per-model mean (LODO) | 0.7000 | 0.7227 | 0.6681 | 0.8550 | 0.7044 | 0.4391 | 0.7724 | 0.9645 | 318 |
| per-model mean (LODO) | 0.8000 | 0.7122 | 0.6261 | 0.8368 | 0.6711 | 0.4374 | 0.7449 | 0.9332 | 298 |
| per-model mean (LODO) | 0.9000 | 0.6450 | 0.5084 | 0.7744 | 0.4256 | 0.3314 | 0.5493 | 0.9408 | 242 |
| per-model median (LODO) | 0.5000 | 0.7206 | 0.7668 | 0.7829 | 0.8795 | 0.0950 | 0.8284 | 0.9779 | 365 |
| per-model median (LODO) | 0.6000 | 0.7416 | 0.7332 | 0.8038 | 0.8567 | 0.3018 | 0.8294 | 0.9432 | 349 |
| per-model median (LODO) | 0.7000 | 0.6933 | 0.6681 | 0.7671 | 0.7767 | 0.3040 | 0.7719 | 0.9636 | 318 |
| per-model median (LODO) | 0.8000 | 0.7437 | 0.6261 | 0.8121 | 0.7685 | 0.4635 | 0.7897 | 0.9301 | 298 |
| per-model median (LODO) | 0.9000 | 0.6933 | 0.5084 | 0.6967 | 0.7025 | 0.3863 | 0.6996 | 0.9368 | 242 |
| RidgeCV (linear) (IS) | 0.5000 | 0.8382 | 0.7668 | 0.8636 | 0.9370 | 0.5095 | 0.8988 | 0.9558 | 365 |
| RidgeCV (linear) (IS) | 0.6000 | 0.8235 | 0.7332 | 0.8886 | 0.8682 | 0.5583 | 0.8783 | 0.9308 | 349 |
| RidgeCV (linear) (IS) | 0.7000 | 0.7836 | 0.6681 | 0.8799 | 0.7830 | 0.5446 | 0.8286 | 0.9395 | 318 |
| RidgeCV (linear) (IS) | 0.8000 | 0.7332 | 0.6261 | 0.9091 | 0.6376 | 0.5176 | 0.7495 | 0.9212 | 298 |
| RidgeCV (linear) (IS) | 0.9000 | 0.6786 | 0.5084 | 0.8938 | 0.4174 | 0.4301 | 0.5690 | 0.9254 | 242 |
| RidgeCV (linear) (LODO) | 0.5000 | 0.7311 | 0.7668 | 0.8319 | 0.8137 | 0.2668 | 0.8227 | 0.9196 | 365 |
| RidgeCV (linear) (LODO) | 0.6000 | 0.6933 | 0.7332 | 0.8162 | 0.7507 | 0.2701 | 0.7821 | 0.8882 | 349 |
| RidgeCV (linear) (LODO) | 0.7000 | 0.6660 | 0.6681 | 0.7809 | 0.6950 | 0.2902 | 0.7354 | 0.9064 | 318 |
| RidgeCV (linear) (LODO) | 0.8000 | 0.6534 | 0.6261 | 0.8009 | 0.5940 | 0.3364 | 0.6821 | 0.8704 | 298 |
| RidgeCV (linear) (LODO) | 0.9000 | 0.6345 | 0.5084 | 0.7429 | 0.4298 | 0.3027 | 0.5445 | 0.8970 | 242 |
| RidgeCV (linear) (LOMO) | 0.5000 | 0.8235 | 0.7668 | 0.8575 | 0.9233 | 0.4667 | 0.8892 | 0.9224 | 365 |
| RidgeCV (linear) (LOMO) | 0.6000 | 0.8025 | 0.7332 | 0.8806 | 0.8453 | 0.5137 | 0.8626 | 0.9041 | 349 |
| RidgeCV (linear) (LOMO) | 0.7000 | 0.7374 | 0.6681 | 0.8535 | 0.7327 | 0.4566 | 0.7885 | 0.8982 | 318 |
| RidgeCV (linear) (LOMO) | 0.8000 | 0.6891 | 0.6261 | 0.8947 | 0.5705 | 0.4526 | 0.6967 | 0.9038 | 298 |
| RidgeCV (linear) (LOMO) | 0.9000 | 0.6534 | 0.5084 | 0.8738 | 0.3719 | 0.3841 | 0.5217 | 0.8765 | 242 |
| RidgeCV (linear) (DHO) | 0.5000 | 0.7122 | 0.7668 | 0.8257 | 0.7918 | 0.2322 | 0.8084 | 0.8764 | 365 |
| RidgeCV (linear) (DHO) | 0.6000 | 0.6933 | 0.7332 | 0.8182 | 0.7479 | 0.2739 | 0.7814 | 0.8521 | 349 |
| RidgeCV (linear) (DHO) | 0.7000 | 0.6324 | 0.6681 | 0.7678 | 0.6447 | 0.2394 | 0.7009 | 0.8613 | 318 |
| RidgeCV (linear) (DHO) | 0.8000 | 0.6218 | 0.6261 | 0.7921 | 0.5369 | 0.2946 | 0.6400 | 0.8289 | 298 |
| RidgeCV (linear) (DHO) | 0.9000 | 0.5987 | 0.5084 | 0.6889 | 0.3843 | 0.2272 | 0.4934 | 0.8445 | 242 |
| RandomForest (100 trees) (IS) | 0.5000 | 0.9790 | 0.7668 | 0.9810 | 0.9918 | 0.9408 | 0.9864 | 0.9996 | 365 |
| RandomForest (100 trees) (IS) | 0.6000 | 0.9790 | 0.7332 | 0.9829 | 0.9885 | 0.9461 | 0.9857 | 0.9993 | 349 |
| RandomForest (100 trees) (IS) | 0.7000 | 0.9580 | 0.6681 | 0.9686 | 0.9686 | 0.9053 | 0.9686 | 0.9959 | 318 |
| RandomForest (100 trees) (IS) | 0.8000 | 0.9538 | 0.6261 | 0.9894 | 0.9362 | 0.9053 | 0.9621 | 0.9962 | 298 |
| RandomForest (100 trees) (IS) | 0.9000 | 0.9286 | 0.5084 | 0.9952 | 0.8636 | 0.8653 | 0.9248 | 0.9850 | 242 |
| RandomForest (100 trees) (LODO) | 0.5000 | 0.7500 | 0.7668 | 0.8000 | 0.8986 | 0.1956 | 0.8465 | 0.9768 | 365 |
| RandomForest (100 trees) (LODO) | 0.6000 | 0.7248 | 0.7332 | 0.8114 | 0.8138 | 0.2948 | 0.8126 | 0.9484 | 349 |
| RandomForest (100 trees) (LODO) | 0.7000 | 0.7143 | 0.6681 | 0.7993 | 0.7642 | 0.3706 | 0.7814 | 0.9592 | 318 |
| RandomForest (100 trees) (LODO) | 0.8000 | 0.6975 | 0.6261 | 0.8565 | 0.6208 | 0.4341 | 0.7198 | 0.9306 | 298 |
| RandomForest (100 trees) (LODO) | 0.9000 | 0.6639 | 0.5084 | 0.8254 | 0.4298 | 0.3804 | 0.5652 | 0.9390 | 242 |
| RandomForest (100 trees) (LOMO) | 0.5000 | 0.8803 | 0.7668 | 0.9162 | 0.9288 | 0.6601 | 0.9224 | 0.9481 | 365 |
| RandomForest (100 trees) (LOMO) | 0.6000 | 0.8613 | 0.7332 | 0.9174 | 0.8911 | 0.6552 | 0.9041 | 0.9549 | 349 |
| RandomForest (100 trees) (LOMO) | 0.7000 | 0.8529 | 0.6681 | 0.9079 | 0.8679 | 0.6771 | 0.8875 | 0.9213 | 318 |
| RandomForest (100 trees) (LOMO) | 0.8000 | 0.8382 | 0.6261 | 0.9368 | 0.7953 | 0.6840 | 0.8603 | 0.9175 | 298 |
| RandomForest (100 trees) (LOMO) | 0.9000 | 0.8445 | 0.5084 | 0.9516 | 0.7314 | 0.7100 | 0.8271 | 0.8891 | 242 |
| RandomForest (100 trees) (DHO) | 0.5000 | 0.7395 | 0.7668 | 0.7904 | 0.8986 | 0.1453 | 0.8410 | 0.9517 | 365 |
| RandomForest (100 trees) (DHO) | 0.6000 | 0.7101 | 0.7332 | 0.7890 | 0.8252 | 0.2290 | 0.8067 | 0.9300 | 349 |
| RandomForest (100 trees) (DHO) | 0.7000 | 0.6660 | 0.6681 | 0.7751 | 0.7044 | 0.2825 | 0.7381 | 0.9261 | 318 |
| RandomForest (100 trees) (DHO) | 0.8000 | 0.6513 | 0.6261 | 0.8204 | 0.5671 | 0.3508 | 0.6706 | 0.8988 | 298 |
| RandomForest (100 trees) (DHO) | 0.9000 | 0.6134 | 0.5084 | 0.7900 | 0.3264 | 0.2905 | 0.4620 | 0.9082 | 242 |
| GradientBoosting (50 stages) (IS) | 0.5000 | 0.9370 | 0.7668 | 0.9420 | 0.9781 | 0.8187 | 0.9597 | 0.9887 | 365 |
| GradientBoosting (50 stages) (IS) | 0.6000 | 0.9286 | 0.7332 | 0.9462 | 0.9570 | 0.8158 | 0.9516 | 0.9709 | 349 |
| GradientBoosting (50 stages) (IS) | 0.7000 | 0.8803 | 0.6681 | 0.9394 | 0.8774 | 0.7422 | 0.9073 | 0.9722 | 318 |
| GradientBoosting (50 stages) (IS) | 0.8000 | 0.8782 | 0.6261 | 0.9688 | 0.8322 | 0.7640 | 0.8953 | 0.9517 | 298 |
| GradientBoosting (50 stages) (IS) | 0.9000 | 0.8697 | 0.5084 | 0.9839 | 0.7562 | 0.7617 | 0.8551 | 0.9470 | 242 |
| GradientBoosting (50 stages) (LODO) | 0.5000 | 0.7563 | 0.7668 | 0.8059 | 0.8986 | 0.2245 | 0.8497 | 0.9626 | 365 |
| GradientBoosting (50 stages) (LODO) | 0.6000 | 0.7374 | 0.7332 | 0.8164 | 0.8281 | 0.3204 | 0.8222 | 0.9365 | 349 |
| GradientBoosting (50 stages) (LODO) | 0.7000 | 0.7017 | 0.6681 | 0.7953 | 0.7453 | 0.3496 | 0.7695 | 0.9332 | 318 |
| GradientBoosting (50 stages) (LODO) | 0.8000 | 0.6975 | 0.6261 | 0.8156 | 0.6678 | 0.4017 | 0.7343 | 0.9173 | 298 |
| GradientBoosting (50 stages) (LODO) | 0.9000 | 0.6408 | 0.5084 | 0.7483 | 0.4421 | 0.3144 | 0.5558 | 0.9280 | 242 |
| GradientBoosting (50 stages) (LOMO) | 0.5000 | 0.8887 | 0.7668 | 0.9239 | 0.9315 | 0.6858 | 0.9277 | 0.9440 | 365 |
| GradientBoosting (50 stages) (LOMO) | 0.6000 | 0.8466 | 0.7332 | 0.9182 | 0.8682 | 0.6288 | 0.8925 | 0.9332 | 349 |
| GradientBoosting (50 stages) (LOMO) | 0.7000 | 0.8214 | 0.6681 | 0.9146 | 0.8082 | 0.6284 | 0.8581 | 0.9212 | 318 |
| GradientBoosting (50 stages) (LOMO) | 0.8000 | 0.7794 | 0.6261 | 0.9251 | 0.7047 | 0.5901 | 0.8000 | 0.9004 | 298 |
| GradientBoosting (50 stages) (LOMO) | 0.9000 | 0.7920 | 0.5084 | 0.9613 | 0.6157 | 0.6295 | 0.7506 | 0.8983 | 242 |
| GradientBoosting (50 stages) (DHO) | 0.5000 | 0.7374 | 0.7668 | 0.8015 | 0.8740 | 0.1854 | 0.8362 | 0.9431 | 365 |
| GradientBoosting (50 stages) (DHO) | 0.6000 | 0.7101 | 0.7332 | 0.8023 | 0.8023 | 0.2590 | 0.8023 | 0.9339 | 349 |
| GradientBoosting (50 stages) (DHO) | 0.7000 | 0.6597 | 0.6681 | 0.7766 | 0.6887 | 0.2779 | 0.7300 | 0.9248 | 318 |
| GradientBoosting (50 stages) (DHO) | 0.8000 | 0.6492 | 0.6261 | 0.7964 | 0.5906 | 0.3277 | 0.6782 | 0.8930 | 298 |
| GradientBoosting (50 stages) (DHO) | 0.9000 | 0.6218 | 0.5084 | 0.7583 | 0.3760 | 0.2903 | 0.5028 | 0.9050 | 242 |

The ranking and the go/no-go decision are reported with **both the dataset and the model of every cell held out of the fit**. Neither single-group protocol answers the question those tasks pose: LODO has seen the learner on the other nineteen problems, and LOMO has seen the dataset. A recommendation is asked about a pair that has not been run.

| Evaluation protocol | AP | MRR | hit@1 | regret | F1 @ 0.7 | MCC @ 0.7 |
|---|---|---|---|---|---|---|
| IS — nothing held out | 0.816 | 0.875 | 0.80 | 0.011 | 0.878 | 0.683 |
| LODO — the dataset unseen, the model known | 0.808 | 0.850 | 0.75 | 0.012 | 0.874 | 0.676 |
| LOMO — the model unseen, the dataset known | 0.817 | 0.883 | 0.80 | 0.008 | 0.877 | 0.675 |
| **DHO** — **both unseen** | 0.846 | 0.908 | 0.85 | 0.005 | 0.865 | 0.647 |

The per-model mean and median predictors are reported below under LODO, the only held-out protocol in which the test model still has training rows. **Under the strictest protocol they cannot be computed at all**: a model held out of every fold has no rows to average, so "how well does this model usually do" has no value. The best of them reaches AP 0.837 and F1 0.772 while being shown the model identity the strictest row of the equation is denied.

Ranking models within a held-out dataset:

- mean top-1 regret **0.005** MCC — what you give up by taking the model the equation ranks first

| predictor | ap | mrr | hit_at_1 | regret | datasets | ap_vs_e3_p | ap_vs_e3_significant |
|---|---|---|---|---|---|---|---|
| equation (IS) | 0.8156 | 0.8750 | 0.8000 | 0.0106 | 20 | 1.0000 | no |
| equation (LODO) | 0.8084 | 0.8500 | 0.7500 | 0.0119 | 20 |  |  |
| equation (LOMO) | 0.8170 | 0.8833 | 0.8000 | 0.0078 | 20 | 0.2668 | no |
| equation (DHO) | 0.8464 | 0.9083 | 0.8500 | 0.0049 | 20 | 0.5488 | no |
| per-model mean (LODO) | 0.7980 | 0.8350 | 0.7500 | 0.0111 | 20 | 0.6476 | no |
| per-model median (LODO) | 0.8375 | 0.8850 | 0.8500 | 0.0088 | 20 | 0.6476 | no |
| RidgeCV (linear) (IS) | 0.7441 | 0.8125 | 0.7000 | 0.0135 | 20 | 0.8036 | no |
| RidgeCV (linear) (LODO) | 0.6703 | 0.7287 | 0.6000 | 0.0379 | 20 | 0.0309 | yes |
| RidgeCV (linear) (LOMO) | 0.7175 | 0.8100 | 0.7000 | 0.0331 | 20 | 0.0309 | no |
| RidgeCV (linear) (DHO) | 0.6043 | 0.6807 | 0.5000 | 0.0763 | 20 | 0.0001 | yes |
| RandomForest (100 trees) (IS) | 0.8851 | 0.9250 | 0.8500 | 0.0020 | 20 | 0.0127 | no |
| RandomForest (100 trees) (LODO) | 0.7693 | 0.7917 | 0.7000 | 0.0133 | 20 | 0.8238 | no |
| RandomForest (100 trees) (LOMO) | 0.6998 | 0.7063 | 0.5500 | 0.0515 | 20 | 0.0127 | yes |
| RandomForest (100 trees) (DHO) | 0.7307 | 0.7833 | 0.6500 | 0.0349 | 20 | 0.2632 | yes |
| GradientBoosting (50 stages) (IS) | 0.8189 | 0.8542 | 0.7500 | 0.0107 | 20 | 0.1435 | no |
| GradientBoosting (50 stages) (LODO) | 0.7575 | 0.8142 | 0.7500 | 0.0179 | 20 | 0.5034 | no |
| GradientBoosting (50 stages) (LOMO) | 0.7149 | 0.7655 | 0.6000 | 0.0361 | 20 | 0.0118 | yes |
| GradientBoosting (50 stages) (DHO) | 0.7150 | 0.7855 | 0.7000 | 0.0163 | 20 | 0.0414 | yes |

Paired over the datasets, the equation differs significantly from: RidgeCV (linear) (LODO), RidgeCV (linear) (DHO), RandomForest (100 trees) (LOMO), RandomForest (100 trees) (DHO), GradientBoosting (50 stages) (LOMO), GradientBoosting (50 stages) (DHO). The remaining comparisons are ties.

## What an opaque model reaches, and does not

The other side of the trade, priced. Three standard regressors on all eighteen raw corpus columns, under the same protocols, with the same clip to the training fold's range that every reported number uses.

| model | features | r2_IS | mae_IS | r2_LODO | mae_LODO | r2_LOMO | mae_LOMO | r2_DHO | mae_DHO |
|---|---|---|---|---|---|---|---|---|---|
| RidgeCV (linear) | 18 | 0.4439 | 0.1848 | -0.5682 | 0.2908 | 0.3679 | 0.2016 | -0.5935 | 0.3001 |
| RandomForest (100 trees) | 18 | 0.9548 | 0.0421 | 0.1731 | 0.2326 | 0.6329 | 0.1249 | 0.0901 | 0.2501 |
| GradientBoosting (50 stages) | 18 | 0.7992 | 0.0978 | 0.1631 | 0.2293 | 0.5817 | 0.1474 | 0.0593 | 0.2502 |

**Read the RandomForest (100 trees) row across.** It fits this meta-data at R² 0.9548; holding out a whole model leaves it at 0.6329; holding out a whole dataset drops it to 0.1731; and with **both** held out it reaches 0.0901. The published equation is at 0.6815 and 0.6513 in the corresponding IS and LODO settings.

The ordering of those four columns is the whole finding. A flexible model on twenty dataset groups, with dataset features constant inside a group, does not learn a relationship -- it learns which dataset a row came from and looks the answer up. Every column that removes an identity removes some of that, and the column that removes both leaves almost nothing.

**Under full leakage prevention the best opaque estimator reaches 0.0901** (RandomForest (100 trees)), approximately the corpus-mean R² reference of zero. This is the like-for-like comparison in the study: LODO still hands a forest the held-out learner on nineteen other problems, and LOMO still hands it the held-out dataset. Only here is it denied what the equation is denied -- and it is also the protocol on which the trivial per-model baselines cannot be computed at all, since a model held out of every fold has no rows to average. A feature-based predictor still predicts.

This is the likely provenance of the R² near 0.9 figures reported for opaque meta-models: a forest evaluated under IS or a random split reproduces them exactly. The ridge penalty is selected internally by RidgeCV; the tree ensembles use fixed, documented settings rather than a hyperparameter search. Further tuning would answer a different objection -- the failure is that the sample has twenty dataset groups, which no amount of tuning changes. What the table licenses is that the accuracy this study traded away was not there to be had under a protocol where the dataset is genuinely unseen.

## Why a random split is not a protocol

The same equation under three splits. A random k-fold puts rows of one dataset on both sides of the fold, so it does not test transfer to an unseen dataset. The difference between that row and the grouped splits measures how much this potential leakage changes the score; here random k-fold and LODO are essentially identical:

| protocol | r2 | mae | rmse | smape | spearman | n |
|---|---|---|---|---|---|---|
| random 10-fold (leaky) | 0.6545 | 0.1356 | 0.2017 | 34.8318 | 0.8300 | 476 |
| LODO | 0.6513 | 0.1368 | 0.2026 | 35.6741 | 0.8337 | 476 |
| LOMO | 0.6261 | 0.1408 | 0.2098 | 35.5131 | 0.8213 | 476 |

<!-- end generated -->

## Limitations of the evaluation

### Ranking: the trivial baselines remain stronger

The current standing is above and in the generated tables; what belongs here is **how it got
there**, because two of the three steps were failures and the successful one is easy to
oversell.

Before `Model Capability` joined the model side, the trivial per-model-mean baseline
out-ranked E3 outright while E3 won on predicting the MCC *value*. Two responses were tried.

A learning-to-rank objective was the obvious one, and it **failed**. Squared error over
within-dataset pairs is ordinary least squares after centring both the design and the target
inside each dataset, so it costs one extra step and stays closed-form; it ranked *worse* than
the objective it was meant to beat. The loss function was never the limitation.

Adding `Model Capability` improved the model side, but it did not close the ranking gap.
Both per-model centres lead E3 on all four reported ranking summaries, and their AP
advantages survive pairing over the twenty held-out datasets. Reporting the baseline beside
the equation prevents the descriptor improvement from being mistaken for a ranking win.

The result also exposes the baseline's information advantage: a per-model centre knows how
that learner performed on other datasets. It cannot score a genuinely unseen learner, while
the descriptor equation can. The defensible conclusion is therefore conditional: **when
model history exists, the trivial ranking remains stronger; when it does not, the baseline
is unavailable.**

### What would change the conclusions

| if | then |
|---|---|
| more datasets (OpenML-scale) | would settle whether the 0.6605 additive ceiling is a property of this sample or of the approach |
| richer measured model descriptors | would test whether the remaining model-side transfer gap is reachable without asserted ordinals |
| an interaction-aware but interpretable term family | would test whether the +0.122 rank-1 gap can be closed without abandoning readability |
| a ranking-specific interpretable objective beyond centred least squares | would test whether the ranking gap against the per-model baselines closes |
| an independent meta-dataset | would test whether the retained configuration and plateau-selected equation transfer beyond this corpus |
| recording failed training runs | would let the study speak about whether to try a model at all, not only about how good a trained one will be |
| training without the 100k sampling cap, or recording the sampled size | would make training-set size a variable the study can reason about at all |
