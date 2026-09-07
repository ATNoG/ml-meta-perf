# 5. Oracles and ceilings

*Implemented in `ml_meta_perf.validate.additive_oracle`, `interaction_oracle` and
`oracle_ladder`.*

An oracle here is **not a model**. It is fitted with the true target values, including
those of the rows it is scored on, uses no features, and can predict nothing about a
dataset it has not seen. Its purpose is to answer a structural question: *if a model of
this form knew everything it could possibly know, how well would it do?* The gap between
an equation and its oracle is the part of the error that better term engineering could in
principle remove; the gap between the oracle and 1.0 is the part it could not.

## The additive oracle

$$\hat{y}_{dm} = \bar{y} + \underbrace{(\bar{y}_{d\cdot} - \bar{y})}_{\text{dataset effect}} + \underbrace{(\bar{y}_{\cdot m} - \bar{y})}_{\text{model effect}}$$

Take the true per-dataset and per-model mean MCC and add them. On this data:

$$R^2_{\text{additive}} = 0.6605$$

Interpretation: **even knowing perfectly how hard every dataset is and how good every
model is, pure addition explains only 66% of MCC.** The remaining 34% is dataset×model
interaction — cases where a specific model is unusually well or badly suited to a specific
dataset.

Two narrower ceilings come from the same construction with one group at a time:

| knowing only | variance of MCC explained |
|---|---|
| which **dataset** it is | 0.354 |
| which **model** it is | 0.282 |

These bound E1 and E2 respectively ([chapter 6](06-results.md)).

## What the additive oracle does *not* bound

It bounds a **two-way additive** form. It does **not** bound E3.

Half of E3's terms are *mixed* — products and ratios pairing a dataset feature with a
model feature — and those express exactly the interaction the two-way form cannot. Given
enough terms, E3 crosses it:

| terms | in-sample R² | LOO-dataset R² |
|---|---|---|
| 24 | 0.653 | 0.280 |
| 28 | **0.6608** | **-0.283** |
| 32 | **0.6632** | **-0.591** |

But read the second column. E3 passes the oracle at exactly the point its cross-validated
score collapses. The crossing is not interaction being *captured*, it is interaction being
*invented* — and the oracle turns out to mark, quite closely, the level past which
apparent gains stop transferring at all. That makes it a useful line to draw even though
it is not a hard ceiling, and the claim must be stated carefully in any write-up.

## A better oracle: the AMMI ladder

The additive oracle's residual is not noise. It is a (dataset × model) matrix of
interactions, and interaction matrices are typically dominated by a few components. So
decompose it by SVD and add the leading components back:

$$\hat{y}^{(r)}_{dm} = \bar{y} + \alpha_d + \beta_m + \sum_{j=1}^{r} \sigma_j u_{dj} v_{mj}$$

This is the **AMMI model** — additive main effects, multiplicative interaction — long used
for genotype-by-environment trials in agronomy, which is structurally the same problem: a
grid of subjects crossed with conditions, where particular pairings suit each other.
Unobserved cells (24 of 500 here) contribute zero residual, so they neither distort the
decomposition nor enter any score.

| interaction rank | R² | gain |
|---|---|---|
| 0 (additive oracle) | 0.6605 | — |
| **1** | **0.7828** | **+0.122** |
| 2 | 0.8537 | +0.071 |
| 3 | 0.8968 | +0.043 |
| 4 | 0.9278 | +0.031 |
| 8 | 0.9842 | +0.019 |
| 20 (full) | 1.0000 | — |

At full rank it reproduces every observed cell exactly, so the interesting question is not
where the ladder ends but **how fast it climbs**.

## The central finding

**The first interaction component alone is worth +0.122 R².**

That is more than the entire difference between a 2-term and a 14-term equation. The
question is then whether the fitted equation reaches any of it, and comparing two R² values
cannot answer that — a number below rank 0 is equally consistent with an equation that
misses the pattern and one that finds it but is inaccurate elsewhere.

Comparing the two interaction *structures* does answer it. Lay both the truth and the
equation's predictions on the (dataset × model) grid, strip each of its own additive part,
and correlate what is left. `validate.interaction_capture` does this and
[chapter 10](10-report.md) regenerates it with the equation:

| protocol | alignment with rank 1 | with ranks 1–2 |
|---|---|---|
| in-sample | 0.31 | 0.28 |
| leave-one-dataset-out | 0.27 | 0.21 |

So the equation reaches **about a third** of the leading interaction pattern in-sample and
a little over a quarter out of fold — not most of it, and not none. An earlier version of
this chapter said "essentially none", read off an R² comparison against a 14-term equation
that no longer exists; that claim was wrong in method as well as in number.

Interaction is 36% of MCC's variance once the additive part is removed, and the leading
component carries 38% of *that*. The remaining headroom is therefore real but smaller than
the +0.122 makes it look, and it is not obviously reachable: the ceiling is set by an
oracle handed 20 free dataset numbers and 25 free model numbers.

A rank-1 interaction is a product of a dataset-side latent and a model-side latent. The
equation's mixed terms are products of *single raw features*, which recover that structure
only where it happens to align with one feature pair — and the alignment above is a
measurement of how often that happens. Closing the rest without abandoning interpretability
is the clearest direction for future work, and doing it with principal components would
close it at the cost of the very thing the project exists to provide.

## How much of the ladder is reachable

The latents above are 20 free dataset numbers and 25 free model numbers, so the ladder is
an upper bound and not a target. [Chapter 9](09-model-effects.md) measures what is left
once each latent has to be *predicted*: replacing both with ridge fits on their own
meta-features leaves **+0.018 of the +0.122**, and the shortfall is almost entirely on the
model side — the dataset latent is 0.597 predictable from `gravity` alone, the model latent
only 0.32 from its best two features.

Keeping the model side free instead — legitimate under leave-one-dataset-out, where every
model appears in every training fold — recovers **+0.106** of leave-one-dataset-out R²
rather than +0.018. That is not an equation and is not reported as a result; it is the
ceiling on what better *model descriptors* could be worth, which is what
[chapter 9](09-model-effects.md) uses it for.
