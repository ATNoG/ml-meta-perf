# 9. Model identity: the bound on what better descriptors could buy

*Implemented in `metafit.identity`. **Not part of the reported study** — see the last
section for why it was measured and then withdrawn.*

[Chapter 6](06-results.md) measures a gap and stops there. Dataset identity explains 0.354
of MCC variance and twelve dataset meta-features recover **99%** of it; model identity
explains 0.282 and five model meta-features recover only **63%**. The missing third of
model capability is real and is written down nowhere in this corpus — the features record
what a model *costs* (`Processing Units Number`, `Training Operations`,
`Prediction Operations`) and two facts about its construction, but nothing about what it
is good at.

[Chapter 5](05-oracles.md) measures the same gap from the other side: a rank-1 interaction
component is worth **+0.122 R²** over the additive oracle, and the equation captures
essentially none of it.

Both are statements that something is missing. Neither says **how much** a better set of
model descriptors would be worth, and that is a number a reader will want before deciding
whether to go and collect them. This chapter measures it.

## The measurement

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
| E3, 20 terms | 0.4779 | 0.1788 |
| + levels only | 0.5194 | 0.1702 |
| **+ levels and slope** | **0.5839** | **0.1493** |
| *per-model mean baseline* | *0.2006* | *0.2382* |

**+0.106 of leave-one-dataset-out R² is what perfect model descriptors would still be
worth**, and it remains the largest number this study has attached to any single change:
no modification to the equation itself — the grammar fix, the arity, the length, the
penalty — moved transfer by more than 0.03.

**It was +0.121 before `Model Capability` was added**, against an E3 that scored 0.4658.
The column took roughly an eighth of the available headroom and left the rest, which is the
cleanest statement of both what it is worth and what it is not: knowing which of ten
families a learner belongs to is a real part of model identity, and a small part of it.
The rank columns of this table are dropped rather than refreshed, because the ranking
comparison they fed now lives in [chapter 4](04-evaluation.md), where E3 no longer trails
the baseline.

The slope's carrier is `log(gravity)`, chosen by training residual sum of squares over all
twelve dataset features, in **20 folds of 20**. `gravity` is the separation between class
centres, and it enters log-compressed under the library's own rule
([chapter 2](02-equation-form.md)).

## Why the model side, specifically

A purely predictive version of chapter 5's rank-1 oracle would estimate *both* latents from
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

**Of chapter 5's +0.122, about +0.018 is reachable from meta-features on both sides, and
the bottleneck is entirely the model side.** The dataset latent is largely predictable —
one feature gets 0.597 of it — and the model latent is not, at 0.32 from its best two.

This is the same conclusion as the +0.106 above, reached independently: the dataset half of
this meta-data is close to exhausted and the model half is not. Chapter 6 says it from the
ceilings, chapter 5 from the interaction ladder, and this chapter from both directions at
once. **Richer model descriptors — inductive bias, hypothesis-space characteristics,
optimiser behaviour — are the only open direction in this study with room left in it.**

## What the correction is, mechanically

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

## Negative result: a ranking objective ranks worse

[Chapter 4](04-evaluation.md) reports that the per-model-mean baseline out-ranks E3, and
the natural response is to fit for ranking rather than for squared error. Squared error over
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
and squared error unchanged ([chapter 4](04-evaluation.md)), which is the same conclusion
from the other end.</sub>

**It ranks worse than the objective it was meant to beat.** Centring inside each dataset
removes the dataset main effect, which is most of what the terms explain; what remains is
the within-dataset ordering, for which the equation has only five model meta-features. The
loss function was never the limitation — the model descriptors are, which is this chapter's
conclusion arrived at by a third route.

## Status: measured, and deliberately not part of the study

An earlier draft published this as a fourth equation, "E4". **It has been withdrawn**, for
four reasons that are worth recording because the numbers are good enough to be tempting:

- **It is not a unified equation.** Fifty fitted numbers in a table cannot be read the way a
  term can, cannot be evaluated by hand, and support no term or term-group analysis. The
  additive form exists so that a reader can decompose the prediction; a lookup table
  refuses that at exactly the point where the study's contribution lives.
- **It yields per-model advice, not practices.** The study's unit of output is a statement
  about a *meta-feature* that generalises ([chapter 7](07-practices.md)). A table says "add
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
[chapter 8](08-limitations.md)'s argument about thin model descriptors, and they are the
reason that argument now carries a number.

`metafit.identity` ships tested and is wired into no pipeline. It stays in the tree where
the agglomerative-construction experiment of [chapter 3](03-search-and-fitting.md) did not,
because this one produces a number the study quotes — the +0.106 ceiling — rather than only
a conclusion. `examples/model_identity_ceiling.py` reproduces the table above.
