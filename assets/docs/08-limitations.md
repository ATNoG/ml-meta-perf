# 8. Limitations and threats to validity

Stated plainly, because a study whose contribution is honesty about ceilings should be
honest about its own.

## Sample size

**Twenty datasets is the binding constraint on every cross-validated number here.**

- Leave-one-dataset-out R² varies by ±0.07 between adjacent term counts from fold noise
  alone. The curve should be read, never a single cell.
- E1's cross-validated numbers rest on 20 points and are correspondingly unstable —
  negative at 1–2 terms, 0.506 at 5.
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

The equation is additive in its terms. [Chapter 5](05-oracles.md) quantifies what that
costs: a rank-1 interaction component is worth +0.122 R² and the equation captures
essentially none of it. This is a limitation of the model family, not of the fitting.

## The equation cannot detect failure

Predictions never fall below 0.17 while 38 rows sit at exactly 0. E3 is a usable estimator
in the range where models work and a **poor detector of the range where they do not**.
Anyone using it to screen candidate models should know that it will not warn them about a
model that is about to fail outright.

## Ranking is not dominated

For selecting a model on a new dataset, the trivial per-model-mean baseline out-ranks E3 on
both measures — mean Spearman 0.703 against 0.648, mean top-1 regret 0.011 against 0.019.
E3 wins on predicting the MCC *value* (0.466 against 0.201), which is a different question:
knowing which models are generally good is enough to order them, while knowing how well one
will do on *this* data is what needs the meta-features. A learning-to-rank objective rather
than squared error is the natural next step and has not been tried.

## Model descriptors are thin

Five model features, one of them constant per model, capture 58% of what model identity
explains. The missing 42% is real and unwritten. Richer descriptors — inductive bias,
hypothesis-space characteristics, optimiser behaviour — would probably help more than any
change to the fitting.

## A practice can flip sign between equations

`Training Operations` is the worked example, and it is the most serious limitation on the
guidance the study extracts.

| equation | practice | effect | confidence |
|---|---|---|---|
| earlier default (14 terms, arity 2, λ=20) | higher training cost → **higher** MCC | 0.16 | moderate |
| published default (24 terms, arity 3, λ=5) | higher training cost → **lower** MCC | 0.38 | moderate |

Same data, same extraction procedure, same confidence rating, opposite sign — and the later
equation is the better one on every metric, so this is not a case of a bad equation being
corrected by a good one.

**Neither reading is an error.** Each correctly describes the equation it was extracted
from. The problem is the feature: `Training Operations` moves with two things at once. It
rises with model capacity, which raises MCC, and it rises with dataset size, which is where
the hard datasets are. Which of the two an equation ends up expressing depends on what
*other* terms it has available to absorb the other half — and that depends on the grammar,
the penalty and the length, none of which the practitioner reading the practice can see.

This generalises past this one feature. **A practice is a property of an equation, not of
the data**, and it only transfers to the data when the feature it names is not confounded
inside the meta-data. The study has no test that separates the two cases, so the practices
in [chapter 7](07-practices.md) should be read with the confidence column *and* this
caveat, not the confidence column alone.

Three mitigations are in place and none of them is sufficient:

- terms selected in fewer than half the folds produce no practice, which catches
  instability *within* one configuration but says nothing about instability *across*
  configurations — `Training Operations` sits at 0.66 fold stability in the published
  equation;
- the marginal correlation is printed beside the conditional direction
  ([chapter 9](09-report.md)), so a reader can at least see when the two disagree, as they
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

## What would change the conclusions

| if | then |
|---|---|
| more datasets (OpenML-scale) | would settle whether the 0.6605 additive ceiling is a property of this sample or of the approach |
| richer model descriptors | would test whether the 42% unexplained model capability is reachable |
| an interaction-aware but interpretable term family | would test whether the +0.122 rank-1 gap can be closed without abandoning readability |
| a learning-to-rank objective | would test whether the ranking gap against the per-model-mean baseline closes |
| refitting across a configuration grid | would separate practices that describe the data from practices that describe one equation |
