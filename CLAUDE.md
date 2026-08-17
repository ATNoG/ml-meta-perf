# Working notes for `metafit`

Orientation for a session picking this up cold. The study itself is documented in
[`assets/docs/`](assets/docs/index.md); this file is the working context that is *not* in
the chapters — where things stand, what has already been tried and failed, and which
facts about the data are easy to get wrong.

## What this is

A research prototype for an academic paper. It fits short, readable equations

```
MCC = w0 + w1*t1 + ... + wk*tk
```

predicting the Matthews Correlation Coefficient a classifier reaches on a dataset, from
meta-features of both. It trades accuracy for explainability **on purpose**: opaque
regressors reach R² ≈ 0.9 on this kind of meta-data and nothing can be read off them.

Three equations, differing only in which features they may use:

| | features | terms | R² (all 476 rows) | LOO-dataset |
|---|---|---|---|---|
| **E1** | dataset only | 5 | 0.337 | 0.506 (on 20 means) |
| **E2** | model only | 9 | 0.164 | 0.055 |
| **E3** | both — *the published one* | 24 | **0.600** | **0.466** |

`python -m metafit` runs everything in ~38s and writes `assets/docs/09-report.md`,
`results/*.{json,csv}` and `assets/figures/*.{png,pdf}`.

## Facts about the data that are easy to get wrong

- **476 rows**, 20 datasets × 25 models. **24 cells are missing**, not at random: eight
  models absent from the three *smallest* datasets (165, 389, 400 instances). Failed
  training runs were discarded when the corpus was built.
- **Every model was trained on a stratified sample capped at 100,000 rows.** Ten datasets
  exceed that. `nr_inst` is therefore the **source** dataset size, not the training-set
  size, and **no question about the effect of more training data is testable here.**
- **15 rows sit at exactly MCC = 0**, 80 at exactly 1.0, one is negative (−0.29, NSL-KDD /
  SGD, retained deliberately). An earlier draft said 38 zeros everywhere; it was wrong and
  came from a threshold rather than an equality.
- The zeros are **not** training failures — those were excluded. They are classifiers that
  converged and learned nothing useful.

## Do not re-attempt these

All measured, all worse, all documented in the chapter named:

| tried | outcome | where |
|---|---|---|
| Divide-and-Learn (Gong & Chen) | halves LOO-dataset R²; 20 datasets split 3 ways is 6 per division | removed from docs; user scrapped it |
| Agglomerative / dendrogram term construction (`construct.py`) | worse than enumeration; guided merge cheaper but worse still | ch. 3 — kept in the tree, **not part of the reported study** |
| Concordance between discovery methods | scrapped with the guided method | — |
| Extra transforms `f^3`, `1/sqrt(f)`, `f^0.25` | in-sample identical to 4dp, worse under the looser grammar | ch. 2 |
| `1/f` and `f^2` (already in the vocabulary) | offered every run, **selected zero times** | ch. 2 |
| Scaling raw features up front | fails three separate ways | ch. 2 |
| Marginal / Pareto filtering of terms before search | destructive — terms are not independent (top-8 marginal 0.200 vs beam 0.533) | ch. 3 |
| Row weighting, Huber loss, two-stage fitting | all worse; R² is reported unweighted, so reweighting optimises a different objective | ch. 6 |
| `max_arity = 4` | +0.035 fit for −0.203 transfer | ch. 2 |
| A second "accuracy-leaning" configuration | dropped — two headline equations invite quoting whichever suits | ch. 6 |
| More search (8× compute) | converges to the 4th decimal | ch. 3 |

**Withdrawn rather than reported:** the best practice *"neural architectures catch up on
larger datasets"*. It looked testable and is not — see the 100k cap above. The split
produced a clean-looking number that meant nothing.

## Standing constraints

- **Dependencies:** polars, numpy, matplotlib, kneeliverse. **No scipy** — it is too large
  for this. Ask before adding anything else.
- **`venv/`, not `.venv/`** — explicit user preference.
- **Figures carry no titles or annotations.** LaTeX captions do that work;
  `figures.captions()` supplies them. Written as PNG **and** PDF, transparent background.
- **The published equation must be simplified** (`terms.simplify` via `fit.prune`) and free
  of duplicate terms. Tested.
- **The report is generated, never narrated.** `metafit.report` and `metafit.guidance`
  derive every sentence arithmetically. If an analysis needs an agent to write it, it
  belongs in code instead — this is the project's central claim, not a style preference.
- `ci.sh` is the gate and pre-commit runs it: unittest + coverage, ruff, basedpyright,
  vulture. All must pass before committing.

## The distinction that took a while to get right

A **measurement** is what the equation does as a feature moves — "higher `ns_ratio` went
with lower MCC here". A **best practice** is general, transferable advice that already
circulates in the field, which measurements can *support*, *qualify* or *challenge*.

Do not present the first as the second. Chapter 8 has the worked example: `Training
Operations` flips sign between two equations fitted on the same data, so it is weak
evidence and not advice at all. `metafit.guidance` holds ten literature practices with
computed verdicts — currently 9 supported, 1 not tested.

## Where the remaining headroom is

- The additive form cannot express dataset×model interaction beyond its mixed terms. **A
  rank-1 interaction component is worth +0.122 R²** and the equation captures ~none of it
  (ch. 5). This is the single largest known gap.
- Model meta-features capture 58% of what model identity explains, against 95% for the
  dataset side. **Richer model descriptors** would likely help more than any change to
  the fitting (ch. 8).
- The per-model-mean baseline **out-ranks E3** (Spearman 0.703 vs 0.648). E3 wins on
  predicting the value, not on ordering candidates. A learning-to-rank objective is untried.

## Layout

```
src/metafit/    data, terms, fit, model, validate, selection, attribution,
                practices (measurements), guidance (literature practices),
                report (generated analysis), plots, figures, experiment, cli
assets/docs/    chapters 1-8 written; chapter 9 regenerated every run
results/        generated, gitignored
```
