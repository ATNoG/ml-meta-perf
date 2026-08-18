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

Three equations, differing **only** in which features they may use. All three are fitted on
all 476 rows by one function (`experiment.run_equation`) and scored on all 476 rows under
both protocols — that uniformity is load-bearing, since the gaps between them are the
study's evidence about what each half of the meta-data is worth:

| | features | terms | in-sample R² | LOO-dataset | LOO-model |
|---|---|---|---|---|---|
| **E1** | dataset only | 7 | 0.349 | 0.217 | 0.294 |
| **E2** | model only | 12 | 0.281 | 0.151 | 0.188 |
| **E3** | both — *the published one* | 20 | **0.614** | **0.478** | 0.428 |

Configuration: `max_arity=3`, `max_abs_zscore=3.0`, **λ=20**, 20 terms, 316-term library.

**Do not compare their R² values as achievements.** They share a scale (476 rows) but not
a ceiling: E1 can only predict a per-dataset constant, so 0.354 is its structural maximum
and 0.349 means it is *done*. The comparable quantity is the fraction of its own ceiling —
98% / 99% / 93%.

**The 63% that used to sit in the middle of that row is the study's central measurement,
and `Model Capability` is what closed it.** On the five descriptors the corpus ships, E2
reaches 63% of its ceiling; adding one ten-level family ordinal asserted from the
tabular-ML literature takes it to 99%. The conclusion is therefore *the corpus lacks model
features, and most of what it lacks is recoverable from knowing which of ten families a
learner belongs to* — not that the gap was never there. Three costs, all in ch. 8: the
column is asserted rather than measured; the corpus agrees with its ordering at only 6 of 9
steps (`generic NN` sits at rung 6 with the lowest family mean, 0.454); and it does not
extend to an unclassified model, which is why LOO-model fell from 0.489 to 0.428.

**E1 used to be fitted on 20 aggregated dataset means** and reported 0.953 / 0.506. Those
numbers were on a 20-point denominator and invited the reading that the control transfers
better than E3. Fitting on all rows costs nothing (+0.012 in-sample) and puts it at 0.217.
Aggregating E2 to 25 model means — the mirror move — was measured and is *worse*: three of
the five model features are functions of the dataset too, so averaging discards real
variation. Do not reintroduce either aggregation.

`python -m metafit` runs everything in ~27s and writes `assets/docs/10-report.md`,
`results/*.{json,csv}` and `assets/figures/*.{png,pdf}`. It was ~38s until two changes that
altered no result: the beam search's candidate solves are batched into one stacked
`np.linalg.solve` per step (1.16M solve calls → 87k), and `guided_screen` ranks the target
once per fold instead of once per candidate. The profile is now the linear algebra itself.

## Facts about the data that are easy to get wrong

- **476 rows**, 20 datasets × 25 models. **24 cells are missing**, not at random: eight
  models absent from the three *smallest* datasets (165, 389, 400 instances). Failed
  training runs were discarded when the corpus was built.
- **Every model was trained on a stratified sample capped at 100,000 rows.** Ten datasets
  exceed that. `nr_inst` is therefore the **source** dataset size, not the training-set
  size, and **no question about the effect of more training data is testable here.**
- **Each row is the best of three seeds, not their mean.** The upstream corpus trains every
  (dataset, model) pair under three seeds and `idxmax`es on MCC. The target is therefore
  optimistic by ~0.02 MCC on average, seeds spread by 0.05 on average (0.10 at the 90th
  percentile), and that spread is the noise floor for any prediction. Verified against
  `meta2perf-symbolic-regression/src/exp_stage_create_meta_dataset.py`.
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
| Agglomerative / dendrogram term construction | worse than enumeration; guided merge cheaper but worse still. Code removed from the tree; `git log -- src/metafit/construct.py` recovers it | ch. 3 |
| Concordance between discovery methods | scrapped with the guided method | — |
| Extra transforms `f^3`, `1/sqrt(f)`, `f^0.25` | in-sample identical to 4dp, worse under the looser grammar | ch. 2 |
| `1/f` and `f^2` (already in the vocabulary) | offered every run, **selected zero times** | ch. 2 |
| Scaling raw features up front | fails three separate ways | ch. 2 |
| Marginal / Pareto filtering of terms before search | destructive — terms are not independent (top-8 marginal 0.200 vs beam 0.533) | ch. 3 |
| Row weighting, Huber loss, two-stage fitting | all worse; R² is reported unweighted, so reweighting optimises a different objective | ch. 6 |
| `max_arity = 4` | +0.035 fit for −0.203 transfer | ch. 2 |
| A second "accuracy-leaning" configuration | dropped — two headline equations invite quoting whichever suits | ch. 6 |
| More search (8× compute) | converges to the 4th decimal | ch. 3 |
| Learning-to-rank objective (within-dataset pairwise least squares) | ranks *worse* than squared error, 0.532 vs 0.625 Spearman; per-model mean still leads at 0.703 | ch. 9 |
| Backfitting the equation against a per-model table | 0.520 → 0.407 over two rounds and falling; the equation stops explaining model capability and spends its terms on non-transferring detail | ch. 9 |
| A dense (12-feature) dataset-side score in the rank-1 term | 0.520, no better than levels alone; one feature gets 0.587 | ch. 9 |
| Estimating both AMMI latents from covariates | +0.018 of the +0.122, bottlenecked on the model side | ch. 5, 9 |
| **Publishing the per-model table as a fourth equation ("E4")** | withdrawn: not a unified equation, no term or term-group analysis, per-model advice rather than practices, empty under leave-one-model-out, and still under an opaque regressor. **Kept only as the ceiling measurement** | ch. 9 |
| Aggregating E1 to 20 dataset means | R² on a 20-point denominator, not comparable with E3's; 0.506 read as beating 0.466 when the common-scale figure is 0.217 | ch. 6 |
| Aggregating E2 to 25 model means | worse: optimum collapses to 1 term, in-sample 0.125 vs 0.177 | ch. 6 |
| An ordinal "model complexity" / family column | LOO-dataset 0.466 → 0.297, below baseline at *every* equation length; capped at +0.032 before it is tried, and the per-family residuals are not monotone in complexity. A **capability**-ordered rung is a different proposal and measures much better (0.478) but is rejected too — the asserted order matches this corpus at 6 of 9 steps | `todo.md` §2 |
| One-hot family dummies | 7 of the 10 families fall below the 10%-of-rows floor the z-score cap imposes on a binary column; loosening the cap to admit them degrades the baseline too | `todo.md` §2.5 |
| Judging any library expansion at the incumbent penalty | the ridge was tuned for the smaller library; at λ=5 the mechanistic axes look like a failure, at λ=20 they beat the published equation on both protocols. Always sweep λ **and** run a matched-penalty baseline | `todo.md` §2.5 |

**Fixed (`todo.md` §3 stage 0):** `Library` used to de-duplicate on `Term.name` only, so
one column could enter twice under two names. It now also drops a term whose *centred*
column correlates above `1 - 1e-9` with one already kept. This mattered before any binary
feature existed: `inst_to_attr` is `nr_inst / nr_attr`, so
`log(inst_to_attr) + log(nr_attr)` **is** `log(nr_inst)` exactly and the grammar emitted
both — three collinear pairs in the published 281-term library, now 278. The beam search
would never have selected both (it refuses candidates correlating above
`fit.COLLINEARITY_LIMIT` = 0.95), so the cost was pool slots and duplicated entries in the
term rankings, not a singular solve. E1, E2 and E3 came out byte-identical. **Any feature defined as a ratio of
two others will do this again.**
| Log-differences of the three cost columns | LOO-dataset 0.466 → 0.353; fit up, transfer down | `todo.md` §1.1 |

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

[`todo.md`](todo.md) holds the open directions with their measurements, and **`todo.md` §3
is the staged plan** — read it before starting any of them.

**Standing decision (2026-08-17): no binary indicator features.** They do not suit an
additive model — inside a product an indicator is identically zero on the rows where it is
off, which is a per-group slope in product notation rather than a relationship, and inside
this grammar only `f` and `f^2` are even defined for a 0/1 column. Five binary mechanistic
axes *were* measured at 0.494 LOO-dataset / 0.535 LOO-model against the published
0.466 / 0.489; that number stands as a measurement and **is not a merge candidate**. Same
for the complexity ordinal and one-hot family dummies. See `todo.md` §2.7 and §3 stage 1.

**Standing decision (2026-08-17): no re-running any model or experiment from the
meta-dataset.** That closes behavioural probing as well as the corpus rebuild, so every
model descriptor this study can still add is *asserted* taxonomy rather than measurement.

**Explainability is the constraint, not a cost to trade against R².** A feature that
improves transfer while making a term unreadable has failed, not traded. `todo.md` opens
with the five-rule gate every proposal must pass; apply it *before* measuring anything.
Sessions keep drifting the same way — find a gain, then quote the interpretability loss as a
price. Do not.

Rejected against that gate rather than against their numbers, and not to be resurrected by
citing R²: the complexity ordinal, one-hot family dummies, five binary mechanistic axes
(0.494/0.535), axes graded to {0.30, 0.95}, and **principal-coordinate embeddings of the
axis matrix (0.457/0.547)** — the best-transferring variant measured anywhere in this study,
inadmissible because two of its five columns have no plain-language reading.

**No new model descriptor is currently admissible.** Everything derivable from the five
existing columns has been measured and fails; everything else needs a measurement that is
ruled out. That is not a dead end for the paper — see the headroom section below.

A **capability**-ordered rung is not the same proposal as the complexity ladder and measures
much better (0.478 LOO-dataset at λ=20 k=20, against 0.466 published and 0.386 matched
baseline; LOO-model falls to 0.428). It is still rejected, and on a subtler ground worth
remembering: the asserted order matches this corpus at **6 of 9 steps**, barely above chance
— `generic NN` has the *lowest* mean MCC of any family (0.454) at rung 6 of 10. A
transparently **mis**named feature is worse than an opaque one, because chapter 7 will turn
it into advice. `todo.md` has the surviving version of the idea (external published mean
rank) and the full spec for what to record when the meta-dataset can be recomputed.

- **Richer model descriptors are now the only open direction with room in it.** Three
  separate measurements point at the same wall: model meta-features capture 63% of what
  model identity explains (99% on the dataset side, ch. 6); the AMMI model latent is only
  0.32 predictable from those features against 0.597 for the dataset latent (ch. 9); and a
  ranking objective failed for the same reason (ch. 9). **+0.106 is the upper bound on what
  perfect descriptors could still buy**, measured by replacing them with model identity
  itself. It was +0.121 before `Model Capability`, so that column took about an eighth of
  the available headroom.
- **A per-*family* effect table is the only model-side description that survives
  leave-one-model-out.** Per-model identity buys exactly 0.000 there — a held-out model has
  no training row — while free levels over the 10 families of `MODEL_FAMILY` buy **+0.075**,
  because the rest of the family stays in the training fold (+0.079 LOO-dataset). It is
  still a table rather than an equation, so it re-opens three of E4's four objections; the
  fourth is gone. Undecided, see `todo.md` §2.
- The rank-1 gap of ch. 5 is **mostly not reachable**: +0.018 of the +0.122 with covariates
  on both margins. It is no longer the largest *actionable* gap.
- The per-model-mean baseline **no longer out-ranks E3**: 0.703 vs 0.706 mean Spearman,
  0.011 vs 0.008 top-1 regret. Read the rank-correlation margin as a tie and the regret as
  the real gain. A ranking *objective* was tried and is worse (ch. 9); what closed the gap
  was the descriptor.

## Layout

```
src/metafit/    data, terms, fit, model, validate, selection, attribution,
                practices (measurements), guidance (literature practices),
                report (generated analysis), plots, figures, experiment, cli
                identity  -- measured, wired into nothing, backs the ch. 9 ceiling
assets/docs/    chapters 1-9 written; chapter 10 regenerated every run
results/        generated, gitignored
```
