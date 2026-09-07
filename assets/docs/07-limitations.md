# 7. Limitations and threats to validity

Stated plainly, because a study whose contribution is honesty about ceilings should be
honest about its own.

## Sample size

**Twenty datasets is the binding constraint on every cross-validated number here.**

- Leave-one-dataset-out R² varies by ±0.07 between adjacent term counts from fold noise
  alone. The curve should be read, never a single cell.
- E1's cross-validated numbers rest on 20 folds and are correspondingly unstable — its
  leave-one-dataset-out R² is negative at 1–4 terms and 0.217 at 7.
- The knee detector, the Pareto front and the "best cross-validated" rule all operate on a
  curve whose points carry that much noise. That they agree on 12–14 terms is reassuring,
  not conclusive.

Twenty-five models is more comfortable but still small for the leave-one-model-out
protocol.

## Domain

All twenty datasets are networking, IoT and security tabular benchmarks. Nothing here
should be assumed to transfer to images, text, or tabular data from other domains. The
extracted practices in particular are statements about this corpus.

## The additive form

The equation is additive in its terms. [Chapter 4](04-equation.md) quantifies what that
costs: a rank-1 interaction component is worth +0.122 R² and the equation captures
essentially none of it. This is a limitation of the model family, not of the fitting.

## Training-set size is not a variable here

**Every model was trained on a stratified sample capped at 100,000 rows.** Ten of the
twenty datasets exceed that cap, so above it every training set is the same size and
`nr_inst` records the *source* dataset rather than what the model saw.

This is a hard boundary on what the study can be asked. Anything of the form "does X change
as the training set grows" is untestable here, and a split of the corpus by `nr_inst` is a
split by source size, which is not the same variable. An earlier draft of
[chapter 10](10-report.md) used exactly such a split to weigh the practice that *neural
architectures catch up on larger datasets*; the practice has been withdrawn from the
catalogue because this corpus cannot speak to it, not because the answer came out one way
or the other.

`nr_inst` and `inst_to_attr` remain legitimate meta-features — they are knowable before
training and they describe the problem a practitioner is facing — but no term over them
should be read as a statement about sample size.

## Every prediction is conditional on training succeeding

**Runs that failed to train were discarded when the meta-dataset was built.** That was a
deliberate choice — a crashed run has no MCC to record — but it means the corpus is a
sample of *completed* runs, and the equation is fitted on and can only speak about that
population.

Twenty-four of the 500 (dataset, model) cells are absent, and not at random: eight models
are missing from the same three datasets. So the exclusions are concentrated exactly where
one would expect a model to have struggled, which is the classic shape of selection bias.

Two consequences, and they point in different directions:

- **For the model comparison, the bias is measurable and small.** Restricting to the 17
  datasets where every model ran moves the model ranking by Spearman 0.975 and no mean by
  more than 0.07. Every family-level figure in [chapter 10](10-report.md) is computed on
  that complete subset for this reason rather than on all rows.
- **For the go/no-go rule, the bias is not correctable.** A rule trained only on runs that
  completed answers "will this trained model be any good", not "should I try this at all".
  It has never seen a failure and cannot warn about one.

The 15 rows at exactly MCC = 0 are *not* the failures. They are classifiers that converged
and learned nothing useful — the majority-class predictor and its relatives. Predictions
never fall below 0.17, so those rows sit above the diagonal, but that is shrinkage toward
the middle of the observed range rather than an inability to recognise a failure mode the
data does not contain.

## Ranking: the equation caught up with the trivial baseline, and no further

Through earlier drafts the trivial per-model-mean baseline out-ranked E3 on both measures —
mean Spearman 0.703 against 0.648, top-1 regret 0.011 against 0.019 — while E3 won on
predicting the MCC *value*. Two responses were tried and one worked, but "worked" needs
stating carefully, and an earlier version of this section did not state it carefully enough.

A learning-to-rank objective was the obvious response and it **failed**. Squared error over
within-dataset pairs is ordinary least squares after centring both the design and the target
inside each dataset, so it costs one extra step and stays closed-form; it ranked *worse* than
the objective it was meant to beat, 0.532 against 0.625 (below). The loss function was never
the limitation.

Adding `Model Capability` to the model side is what moved it. On the means E3 now leads the
per-model **mean** baseline on every head-weighted metric — average precision 0.850 against
0.798, reciprocal rank 0.882 against 0.835, hit@1 0.800 against 0.750 — and **none of those
margins survives pairing over the twenty held-out datasets** (average precision p = 0.21,
interval spanning zero).

**And the mean is not the hardest baseline.** The per-model **median** is a different ordering
and a better one on two of the four measures: average precision 0.837, reciprocal rank 0.885,
hit@1 0.850 against E3's 0.800, top-1 regret 0.009 against 0.015. That comparison does not
survive pairing either (p = 0.14). Reporting only the mean baseline made the equation look
like it had won a contest it had drawn.

| | AP | MRR | hit@1 | top-1 regret |
|---|---|---|---|---|
| per-model mean | 0.798 | 0.835 | 0.750 | 0.011 |
| **E3** | **0.850** | 0.882 | 0.800 | 0.015 |
| per-model median | 0.837 | **0.885** | **0.850** | **0.009** |

So the honest statement is that **the equation caught up with the trivial baselines and did
not pass them**. That is still the diagnosis confirming itself — a per-model centre
out-ranked the equation because it knew something the equation did not, roughly which models
are good, and the fix was to tell the equation rather than to change how it was fitted. But a
study that stops at "E3 wins on all four" is reading four means over twenty folds, which is
what [`validate.paired_comparison`](../../src/ml_meta_perf/validate.py) exists to prevent,
and hit@1 moves only in steps of 0.05 on twenty datasets.

Where the equation does clear the trivial predictors, and clearly, is the two questions that
are not ranking: predicting the MCC value (0.638 leave-one-dataset-out against 0.201) and the
go/no-go threshold decision (MCC 0.683 against 0.439 and 0.304 at a threshold of 0.7).
[Chapter 5](05-evaluation.md) reports all three together.

Spearman does not enter any of this. It sits between 0.63 and 0.73 for every predictor and
every baseline on this corpus, including a constant.

## Model descriptors are thin, and one of them is asserted

The five model features the corpus originally shipped captured 63% of what model identity
explains. That gap is what the asserted ordinals were added to close, and they close most of
it: E2 goes from 63% to **88%** of its ceiling.

**Five of the six model features are asserted, not measured, and that is the sharpest
limitation in this chapter.** `Processing Units Number` is computed from a trained instance.
`Model Capability` ranks the ten learner families on a ladder taken from the tabular-ML
literature (Grinsztajn et al. 2022;
Shwartz-Ziv & Armon 2022; McElfresh et al. 2023; Hollmann et al. 2023). Three consequences
follow and none should be glossed:

- **It encodes prior knowledge, so it cannot be evidence for that knowledge.** Any result
  that amounts to "capable families do better" is partly built in. What the column can
  support is the *conditional* claim — how family capability interacts with dataset
  properties — which is not something the ladder asserts.
- **The corpus contradicts it in places.** Ordering families by observed mean MCC agrees
  with the ladder at only 6 of 9 steps, and `generic NN` sits at rung 6 of 10 while holding
  the *lowest* family mean here, 0.454. The overall association is nonetheless strong
  (Spearman 0.733 conditional, 0.391 marginal).
- **It does not extend to an unclassified model.** A new learner needs a human to place it
  on the ladder. So does each of `Solution Stochasticity`, `Loss Margin Behaviour`,
  `Input Distribution Modelling` and `Fitting Regime` — all four are read off a published
  description of the algorithm, not off a training run. Five of six model features therefore
  require an act of classification before the equation can be applied to a genuinely new
  method.

The four mechanism ordinals share the first and third caveats and are **weaker than
`Model Capability` on the second**, because they were not fitted against anything: each grades
one mechanism on a stated scale, with `Loss Margin Behaviour` the least defensible of them —
placing the perceptron criterion above hinge is a choice rather than a consensus. Any term
over one of these columns is evidence about the ordering claimed here, not about a quantity
anyone observed.

Richer *measured* descriptors — inductive bias, hypothesis-space characteristics, optimiser
behaviour — would carry none of these caveats and remain the better answer.

How much more is now measured rather than guessed. Tabulating that missing third per model
instead of describing it is worth **+0.106** of leave-one-dataset-out R²
(this chapter) — which is at most what a *perfect* set of extra
descriptors would be worth, since free per-model numbers are the best any descriptor set
could do at telling these 25 models apart. It is also why a table is not a substitute for
descriptors: it transfers to a new dataset and not to a new model, while a descriptor would
do both.

## A per-feature association can flip sign between equations

`Training Operations` is the worked example. It is the reason the study treats per-feature
associations as evidence rather than as advice — and, in the end, the reason the column was
removed from the corpus altogether. The measurement below is retained because the retirement
does not make the lesson go away; it is what the lesson cost.

| equation | association | effect | confidence |
|---|---|---|---|
| earlier default (14 terms, arity 2, λ=20) | higher training cost → **higher** MCC | 0.16 | moderate |
| a later default (24 terms, arity 3, λ=5) | higher training cost → **lower** MCC | 0.38 | moderate |

Same data, same extraction procedure, same confidence rating, opposite sign — and the later
equation is the better one on every metric, so this is not a case of a bad equation being
corrected by a good one.

**Neither reading is an error.** Each correctly describes the equation it was extracted
from. The problem is the feature: `Training Operations` moves with two things at once. It
rises with model capacity, which raises MCC, and it rises with dataset size, which is where
the hard datasets are. Which of the two an equation ends up expressing depends on what
*other* terms it has available to absorb the other half — and that depends on the grammar,
the penalty and the length, none of which the practitioner reading the practice can see.

**The feature was eventually retired for exactly this.** A column that moves with model
capacity and dataset size at once is not a model descriptor, and two of the other three
retired columns had the same defect. That is a resolution of this particular case, not of the
general problem: nothing guarantees the six current features are free of it, and the mitigations
below are still what stands between a fitted weight and a stated practice.

This generalises past this one feature, and it is the reason the study does not present
per-feature associations as advice. A measurement like "higher training cost went with
lower MCC" is **evidence**, and evidence whose sign depends on the equation it came from is
weak evidence. It is not, on its own, a best practice: a practice is a general
recommendation that a body of evidence can support, and no single fitted equation is that
body.

That is why [chapter 10](10-report.md) states its practices at the level of received
guidance from the literature and uses this study to weigh each one, rather than reading new
guidance out of the weights. A recommendation that survives being weighed against several
independent measurements is worth something; one extracted from a single equation inherits
that equation'"'"'s instabilities, of which this is a worked example.

Three mitigations are in place and none of them is sufficient:

- terms selected in fewer than half the folds produce no practice, which catches
  instability *within* one configuration but says nothing about instability *across*
  configurations — `Training Operations` sat at 0.66 fold stability in the equation
  published at the time, which was not low enough to suppress it;
- the marginal correlation is printed beside the conditional direction
  ([chapter 10](10-report.md)), so a reader can at least see when the two disagree, as they
  do here;
- features whose rank direction and decile effect disagree in sign are dropped outright.

What would actually settle it is refitting across a grid of configurations and reporting
only the practices whose sign is stable across all of them. That is a sign-stability
analogue of the fold-stability filter, it is affordable, and it has not been done.

## Hyperparameter selection is not nested

The stability cap, penalty and equation length were tuned by inspecting
leave-one-dataset-out scores. Those scores are therefore **mildly optimistic** as estimates
of performance on genuinely new data. A fully nested protocol would cost another factor of
20 in compute and, at this sample size, would mostly measure noise; the honest reading is
that the reported transfer numbers are an upper estimate rather than an unbiased one.

The **in-sample** numbers are unaffected by this.

## The single negative row

One row (NSL-KDD / SGD, MCC = -0.29) is a legitimate anti-correlated result and is
retained. It is an outlier in a target that is otherwise non-negative, and it falls outside
the scatter's axes. Removing it was considered and rejected: dropping the worst observed
result biases the target upward.

## The bound on what better model descriptors could buy

*Implemented in `ml_meta_perf.identity`. This was chapter 7 of an earlier draft. It is a
limitation rather than a result -- it measures the headroom the study cannot reach with
the descriptors the corpus records -- so it belongs here.*

*Implemented in `ml_meta_perf.identity`. **Not part of the reported study** — see the last
section for why it was measured and then withdrawn.*

[Chapter 4](04-equation.md) measures a gap and stops there. Dataset identity explains 0.354
of MCC variance and the twelve dataset meta-features recover **98%** of it; model identity
explains 0.282 and the six model meta-features recover **88%**. What is left is written down
nowhere in this corpus — the features record what a model *is*, its capacity and five
asserted facts about how it is built, but nothing about what it is good at.

**That gap was a third when this chapter was written, and it is now an eighth.** The four
asserted ordinals merged on 2026-09-05 closed most of it, which is why the ceiling measured
below is much smaller than the text originally described.

[Chapter 4](04-equation.md) measures the same gap from the other side: a rank-1 interaction
component is worth **+0.122 R²** over the additive oracle, and the equation reaches about a
third of that pattern in-sample and a quarter out of fold — measured, not inferred from an
R² comparison.

Both are statements that something is missing. Neither says **how much** a better set of
model descriptors would be worth, and that is a number a reader will want before deciding
whether to go and collect them. This chapter measures it.

### The measurement

Under leave-one-dataset-out **every one of the 25 models appears in every training fold**.
So the equation's residual can be averaged per model on the training rows and applied to
the held-out dataset without any leak. Doing that replaces the model descriptors with the
best possible substitute — the model's *identity*, fitted freely — and the improvement is
therefore an upper bound on what any descriptor set could achieve at telling these 25
classifiers apart.

Two numbers per model: a level, and a slope on one dataset feature chosen inside the fold.

$$\text{correction}_m = b_m + c_m \cdot \log(\mathrm{gravity})$$

| | LOO-dataset R² | MAE |
|---|---|---|
| E3, 15 terms | 0.6381 | 0.1425 |
| + levels only | 0.6693 | 0.1351 |
| **+ levels and slope** | **0.6693** | **0.1351** |
| *per-model mean baseline* | *0.2006* | *0.2382* |

**+0.017 of leave-one-dataset-out R² is what perfect model descriptors would still be
worth.** This chapter's whole reason for existing was that the number used to be **+0.106**,
and before `Model Capability` was added, **+0.121** against an E3 scoring 0.4658. It is now
small enough that the question it was asked to answer is closed.

That is the strongest evidence in the study that the model side is adequately described.
Free per-model numbers are the best any descriptor set could do at telling these 25 models
apart — they are model identity itself, fitted out of fold — so what they add on top of the
equation is exactly the information the descriptors are missing. Six columns now leave 0.017
of it on the table, where five columns left 0.106.

**The slope adds nothing at all**, where it used to add +0.065 on top of the levels. A
per-model slope on a dataset feature is an interaction the equation could not express; the
equation now expresses it directly, in nine mixed terms out of sixteen. The row is kept at
its unchanged value rather than deleted, because a measurement going to zero is the result.

The rank columns of this table are dropped rather than refreshed, because the ranking
comparison they fed now lives in [chapter 5](05-evaluation.md), where E3 beats the baseline
on every head-weighted measure.

### Why the model side, specifically

A purely predictive version of chapter 4's rank-1 oracle would estimate *both* latents from
meta-features. Replacing each latent with a ridge fit on its own features, scored
leave-one-out over the latent's own rows:

| latent | rows | best 1 feature | best 2 | best 3 |
|---|---|---|---|---|
| dataset, $u_d$ | 20 | **+0.597** (`gravity`) | +0.629 | +0.662 |
| model, $v_m$ | 25 | +0.227 (`Processing Units Number`) | +0.321 | +0.286 |

And the interaction those fitted latents reconstruct, scored on all 476 rows:

| interaction estimated as | R² | gain over additive |
|---|---|---|
| additive oracle, no interaction | 0.6605 | — |
| $\hat{u}$ from covariates, **true** $v$ | 0.7294 | +0.069 |
| **$\hat{u}$ and $\hat{v}$ both from covariates** | **0.6780** | **+0.018** |
| true $u$, true $v$ (the oracle) | 0.7828 | +0.122 |

**Of chapter 4's +0.122, about +0.018 is reachable from meta-features on both sides, and
the bottleneck is entirely the model side.** The dataset latent is largely predictable —
one feature gets 0.597 of it — and the model latent is not, at 0.32 from its best two.

This is the same conclusion as the +0.106 above, reached independently: the dataset half of
this meta-data is close to exhausted and the model half is not. Chapter 4 says it from the
ceilings, chapter 4 from the interaction ladder, and this chapter from both directions at
once. **Richer model descriptors — inductive bias, hypothesis-space characteristics,
optimiser behaviour — are the only open direction in this study with room left in it.**

### What the correction is, mechanically

Two stages, in this order. The level is the shrunk mean residual of the model's own rows;
the slope is then fitted to what the levels leave behind. Both are shrunk toward zero,
which is the empirical-Bayes estimator for a group effect under a common prior (Efron &
Morris, 1975), with the prior-to-noise ratio fixed rather than estimated — fixed because it
barely matters:

| intercept shrinkage | levels only | with slope |
|---|---|---|
| 0 | 0.5219 | 0.5882 |
| 2 | 0.5217 | 0.5886 |
| **5 (default)** | 0.5198 | **0.5874** |
| 10 | 0.5155 | 0.5836 |
| 20 | 0.5073 | 0.5756 |

R² moves by 0.015 across a range of 20; the slope's shrinkage moves it by 0.005 across the
same range. Nothing here was tuned against the reported figure.

The construction is a **factorial regression** in the sense used for genotype-by-environment
trials: a two-way table modelled with covariates on one margin and free coefficients on the
other (Denis, 1988; van Eeuwijk, Denis & Kang, 1996). In machine-learning terms it is the
collaborative half of a hybrid recommender, and algorithm selection has been posed as
collaborative filtering before (Mısır & Sebag, 2017; Fusi et al., arXiv:1705.05355; Yang et
al., arXiv:1808.03233).

### Two-stage, not backfitting

Fitting the two blocks jointly — backfitting, the standard scheme for an additive model
with a tabulated block (Hastie & Tibshirani, 1990) — is much worse, measured with levels
only:

| rounds | LOO-dataset R² | mixed terms in the equation |
|---|---|---|
| **1 (two-stage)** | **0.5198** | 11.4 / 24 |
| 2 | 0.4068 | 12.2 / 24 |
| 3 | 0.3751 | 12.2 / 24 |
| 5 | 0.3316 | 12.3 / 24 |

It fails in an informative direction. Once the table absorbs model capability the equation
stops needing to explain it and spends its terms on within-dataset detail instead — the
mixed-term count rises every round — and detail like that does not transfer. **The equation
has to be fitted against MCC to remain a statement about MCC.**

### A dense dataset-side score is worse than one feature

The general form of the dataset side is a score over several features, fitted with the
model side by alternating least squares:

| dataset score | LOO-dataset R² | MAE | mean regret |
|---|---|---|---|
| all 12 features | 0.5195 | 0.1627 | **0.016** |
| 3 features | 0.4760 | 0.1702 | 0.017 |
| 2 features | 0.5546 | 0.1610 | 0.030 |
| **1 feature (`gravity`), closed form** | **0.5874** | **0.1494** | 0.032 |

A twelve-loading score is worth no more than the levels alone. Twenty datasets do not
support twelve free loadings, and with a single feature the alternation is unnecessary —
each model's slope is a closed-form ridge fit of its own residuals.

### Negative result: a ranking objective ranks worse

[Chapter 5](05-evaluation.md) reports that the equation is indistinguishable from the
trivial "how well does this model usually do" baselines at ranking, and the natural response is
to fit for ranking rather than for squared error. Squared error over
all within-dataset *pairs* expands to ordinary least squares on a design and target both
centred inside each dataset — the within, or fixed-effects, transform (Mundlak, 1978) — so
a pairwise ranking objective costs one extra centring step and stays a closed-form ridge
solve. It was implemented and cross-validated:

| | mean ρ | mean regret |
|---|---|---|
| E3, squared error, 24 terms | 0.625 | 0.064 |
| **pairwise ranking objective, 24 terms** | **0.532** | **0.095** |
| per-model mean baseline | 0.703 | 0.011 |

<sub>Measured against the 24-term E3, before `Model Capability` joined the model side. The
comparison is between two *objectives* on one feature set, which is what makes it evidence
about the loss function; the published equation now reaches 0.706 with the extra descriptor
and squared error unchanged ([chapter 5](05-evaluation.md)), which is the same conclusion
from the other end.</sub>

**It ranks worse than the objective it was meant to beat.** Centring inside each dataset
removes the dataset main effect, which is most of what the terms explain; what remains is
the within-dataset ordering, for which the equation has only five model meta-features. The
loss function was never the limitation — the model descriptors are, which is this chapter's
conclusion arrived at by a third route.

### Status: measured, and deliberately not part of the study

An earlier draft published this as a fourth equation, "E4". **It has been withdrawn**, for
four reasons that are worth recording because the numbers are good enough to be tempting:

- **It is not a unified equation.** Fifty fitted numbers in a table cannot be read the way a
  term can, cannot be evaluated by hand, and support no term or term-group analysis. The
  additive form exists so that a reader can decompose the prediction; a lookup table
  refuses that at exactly the point where the study's contribution lives.
- **It yields per-model advice, not practices.** The study's unit of output is a statement
  about a *meta-feature* that generalises ([chapter 6](06-practices.md)). A table says "add
  0.24 for logistic regression on this corpus", which is a fact about these 25 runs and this
  equation's bias, not transferable guidance.
- **It cannot extrapolate to a new model.** Under leave-one-model-out the correction is
  empty and its predictions are bitwise E3's. Every number comes from rows where that model
  was already run.
- **It is still below what an opaque regressor reaches** on this kind of meta-data. Trading
  the readable equation for a table does not even win the accuracy argument outright, so it
  wins nothing worth the trade.

What survives is the **measurement**: +0.106 as the upper bound on better model descriptors,
and +0.018 as the covariate-reachable part of the interaction ladder. Those belong to
this chapter's argument about thin model descriptors, and they are the
reason that argument now carries a number.

`ml_meta_perf.identity` ships tested and is wired into no pipeline. It stays in the tree where
the agglomerative-construction experiment of [chapter 3](03-term-selection.md) did not,
because this one produces a number the study quotes — the +0.106 ceiling — rather than only
a conclusion. `ml_meta_perf.identity.correct_out_of_fold`, applied to a finished `cross_validate_fixed_form`, reproduces the table above.

## What was tried against the additive form, and failed

> **These were measured against the previous grammar**, whose default reached 0.5582 —
> before `sum_ratio` was made symmetric and the term budget re-tuned. They are reported
> unchanged rather than silently rebased, because none of them was re-run afterwards. What
> did beat 0.5582 was the grammar fix, not any of the six.

That earlier default was attacked from six directions. All keep the equation form
`MCC = Σ wᵢtᵢ` intact, and none beats it:

| attempt | in-sample R² |
|---|---|
| **baseline (ordinary least squares, uniform weights)** | **0.5582** |
| downweight the rows at MCC ∈ {0, 1} by 0.5 | 0.5380 |
| downweight them by 0.25 | 0.4147 |
| Huber IRLS, 8 iterations | 0.5526 |
| equal weight per dataset | 0.5509 |
| two-stage: 7 dataset terms, then 7 on the residual | 0.5532 |
| adding `f^3`, `1/sqrt(f)`, `f^0.25` to the vocabulary | 0.5582 (unchanged) |

Widening the operator set deserves its own note, since `f^2` and `sqrt(f)` are **already**
in the vocabulary. Adding `f^3`, `1/sqrt(f)` and `f^0.25` grows the library from 172 terms
to 182 — most of the 51 new candidates fail admissibility — and the beam then selects
**none of the ten that survive**. In-sample R² is identical to four decimal places at both
8, 14 and 20 terms. Under a looser arity-4 grammar the same extension is actively harmful,
dropping 20-term R² from 0.647 to 0.632. Higher powers are
near-duplicates of the ones already present, and the collinearity guard treats them as
such.

Reweighting was the most promising idea and is the clearest failure: R² is reported on all
476 rows with uniform weight, so any reweighting optimises a *different* objective and
necessarily scores worse on the one being reported. Downweighting the saturated rows in
particular removes 118 of 476 observations' worth of influence — the pile-ups at 0 and 1
are a third of the data, not outliers to be discounted.

Together with the earlier negatives — search strength ([chapter 3](03-term-selection.md)),
transform vocabulary and feature scaling ([chapter 2](02-additive-model.md)), agglomerative
construction, and marginal-impact filtering — the additive form at this configuration is
exhausted.

**One thing does work and neither keeps the equation form intact.** The interaction the
equation cannot reach is worth **+0.122** on its own ([chapter 4](04-equation.md)), and the
part of model capability the model features still miss is worth **+0.106** of
leave-one-dataset-out R² if it is tabulated per model rather than described
(this chapter). The first is unreachable from these features; the
second replaces terms with a lookup table and is reported as a ceiling rather than as a
result. Both are measurements about where the headroom is, not attempts on it. Loosening the library and the
shrinkage also raises in-sample R², but only by giving up transfer at roughly six to one,
which is why that configuration is no longer reported as a result
([chapter 2](02-additive-model.md) measures the trade).

For that last gap the literature points at **GA2M / Explainable Boosting Machines** —
generalised additive models with explicit pairwise interaction terms (Lou et al., 2013;
GAMI-Net, arXiv:2003.07132) — which are exactly the model class that adds interaction while
staying inspectable. Billa et al. (arXiv:2601.00428) find EBMs and symbolic regression
dominate interpretable tabular regression. The trade is real: an EBM is a set of shape
functions rather than a closed-form equation, so it can be plotted but not written down.

## What would change the conclusions

| if | then |
|---|---|
| more datasets (OpenML-scale) | would settle whether the 0.6605 additive ceiling is a property of this sample or of the approach |
| richer model descriptors | would test whether the 42% unexplained model capability is reachable |
| an interaction-aware but interpretable term family | would test whether the +0.122 rank-1 gap can be closed without abandoning readability |
| a learning-to-rank objective | would test whether the ranking gap against the per-model-mean baseline closes |
| refitting across a configuration grid | would separate evidence that describes the data from evidence that describes one equation |
| recording failed training runs | would let the study speak about whether to try a model at all, not only about how good a trained one will be |
| training without the 100k sampling cap, or recording the sampled size | would make training-set size a variable the study can reason about at all |
