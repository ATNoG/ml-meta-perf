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

Predictions never fall below 0.17 while 38 rows sit at exactly 0. E2 is a usable estimator
in the range where models work and a **poor detector of the range where they do not**.
Anyone using it to screen candidate models should know that it will not warn them about a
model that is about to fail outright.

## Ranking is not dominated

For selecting a model on a new dataset, the trivial per-model-mean baseline out-ranks E2
(mean Spearman 0.70 vs 0.61) at comparable top-1 regret. E2 wins on predicting the MCC
value. A learning-to-rank objective rather than squared error is the natural next step and
has not been tried.

## Model descriptors are thin

Five model features, one of them constant per model, capture 58% of what model identity
explains. The missing 42% is real and unwritten. Richer descriptors — inductive bias,
hypothesis-space characteristics, optimiser behaviour — would probably help more than any
change to the fitting.

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
