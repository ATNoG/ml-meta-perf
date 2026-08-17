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

## Training-set size is not a variable here

**Every model was trained on a stratified sample capped at 100,000 rows.** Ten of the
twenty datasets exceed that cap, so above it every training set is the same size and
`nr_inst` records the *source* dataset rather than what the model saw.

This is a hard boundary on what the study can be asked. Anything of the form "does X change
as the training set grows" is untestable here, and a split of the corpus by `nr_inst` is a
split by source size, which is not the same variable. An earlier draft of
[chapter 9](09-report.md) used exactly such a split to weigh the practice that *neural
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
  more than 0.07. Every family-level figure in [chapter 9](09-report.md) is computed on
  that complete subset for this reason rather than on all rows.
- **For the go/no-go rule, the bias is not correctable.** A rule trained only on runs that
  completed answers "will this trained model be any good", not "should I try this at all".
  It has never seen a failure and cannot warn about one.

The 15 rows at exactly MCC = 0 are *not* the failures. They are classifiers that converged
and learned nothing useful — the majority-class predictor and its relatives. Predictions
never fall below 0.17, so those rows sit above the diagonal, but that is shrinkage toward
the middle of the observed range rather than an inability to recognise a failure mode the
data does not contain.

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

## A per-feature association can flip sign between equations

`Training Operations` is the worked example, and it is the reason the study treats
per-feature associations as evidence rather than as advice.

| equation | association | effect | confidence |
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

This generalises past this one feature, and it is the reason the study does not present
per-feature associations as advice. A measurement like "higher training cost went with
lower MCC" is **evidence**, and evidence whose sign depends on the equation it came from is
weak evidence. It is not, on its own, a best practice: a practice is a general
recommendation that a body of evidence can support, and no single fitted equation is that
body.

That is why [chapter 9](09-report.md) states its practices at the level of received
guidance from the literature and uses this study to weigh each one, rather than reading new
guidance out of the weights. A recommendation that survives being weighed against several
independent measurements is worth something; one extracted from a single equation inherits
that equation'"'"'s instabilities, of which this is a worked example.

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
| refitting across a configuration grid | would separate evidence that describes the data from evidence that describes one equation |
| recording failed training runs | would let the study speak about whether to try a model at all, not only about how good a trained one will be |
| training without the 100k sampling cap, or recording the sampled size | would make training-set size a variable the study can reason about at all |
