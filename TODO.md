# Working notes

> **You are on `wip/beam-search-op`, and it is the only working branch.**
> `fix/ci-docs-plots` and `feature/descriptor-selection` were deleted on 2026-09-08, local and
> remote, after checking that every commit on them was reachable from here or from `main`.
> **Do not create branches without being asked.** `origin/main` is one commit ahead
> (`811253c`, a collaborator's) and does **not** merge cleanly; see "Documentation" below.

## Start here — 2026-09-08

The beam-search line is **closed as a negative** and its machinery is **removed**. The session
that closed it also removed the configuration grid search, folded everything into one CLI
command, and started rebuilding how the study picks its equations. Four commits:

```
2ef4005  Choose the equation with one number instead of a chain of conditions
8a6182e  Remove the configuration grid search: three local fits give what it bought
d92d8d3  Stop stamping the PDF figures with the wall clock
5a78c04  Remove the beam and grid-search experiments, and fold the search into one command
f7b4a7c  Score beam policies on four protocols, and use ESS as a sampler not a design
```

**The working tree is clean and the whole suite is green (499 tests).** Nothing is half-done in
the tree; what is unfinished is unstarted, and it is listed under "The plan" below.

### Read this first

**`selection.best_configuration` gives the right answer on this corpus for the wrong reason.**
It is committed, tested and unused. Before wiring it into the pipeline (C1), decide whether to
replace it -- see "The selection rule" below, which measures how it fails. Wiring a rule that
will flip its verdict as the corpus grows is worse than leaving the length written down, because
a constant at least does not change silently.

### The one thing to pick up: C1

`selection.best_configuration` now derives the equation, and **nothing uses it yet.**
`experiment.py:356` still reads `size = min(config.headline_terms, available)`. Wiring that up
is C1 and it is the next commit. Everything needed is in place and measured:

```python
from ml_meta_perf.selection import best_configuration, most_capable
best_configuration(curves, rows=frame.height)   # -> (2, 15)   E3-Valid, the study's equation
most_capable(curves)                            # -> (3, 23)   E3-MAX, the capability bound
```

where `curves` is `{arity: EquationReport.curve}` for arities 2, 3 and 4.

## The plan — C0 to C7

**C0. Done.** `results/cluster/` deleted (79 MB of artifacts from cluster jobs that no longer
exist and nothing can regenerate; their findings are recorded below with their numbers).
CI stays push-and-PR only and gets its first runner pass whenever this branch opens a PR --
deliberately not now.

**C1. Derive the length instead of asserting it. NEXT.** `experiment.py:356` is the only line
that *decides*, and it throws away work already done: `fit()` returns an equation at every
length in one pass and `run_equation` cross-validates all of them, so the curve exists at that
point. Replace the assertion, carry the chosen size on `EquationReport`, and have the six read
sites take it from there -- 566/568 (protocol scores), 757/759 (decision report), 1038/1040
(baselines). `length_comparison` already derives and keeps `headline_terms` only as a fallback;
that fallback becomes `max(path)`. `Configuration.headline_terms` then has no readers and goes.

**C2. Search the arity too.** `--arity` becomes a list, default `(2, 3, 4)`: one `build_library`
and one `fit` per arity, then `best_configuration` across the combined curves. Measured cost
about 25 s, nearly all of it the arity-4 library build, against a 282 s run:

| arity | library | build | fit |
|---:|---:|---:|---:|
| 2 | 220 terms | 0.03 s | 0.36 s |
| 3 | 838 terms | 0.16 s | 0.59 s |
| 4 | 4,223 terms | 22.76 s | 1.60 s |

**C3. Four equations, and only one of them is evaluated.**

- **E1** dataset features, **E2** model features -- both stay, both with lengths derived.
- **E3-Valid** -- the study's equation, `best_configuration`. **Every evaluation lands here**:
  the protocols, the decision and ranking tasks, the practice comparison, the baselines.
- **E3-MAX** -- `most_capable`. Bounds how far the additive form reaches, is reported in the
  docs, and is **not** evaluated as a predictor.

**C4. Drop `--terms`, keep `--max-terms`.** Different things: the horizon is a knob and a cost
control, the length is what C1 derives.

**C5. Drop `--quick`.** A preset of flags that already exist (`--max-terms 3 --pool 40
--penalty 20`) carrying two Configuration objects whose only job is a third copy of a length.
Three tests use it -- `test_plots` x2, `test_experiment` x3 -- and pass the flags explicitly
instead, which makes what "quick" meant visible at the call site.

**C6. One set of named constants feeding argparse**, with a comment recording that they are
where the 2026-09 sweep landed: `PENALTY`, `MAX_ABS_ZSCORE`, `POOL_SIZE`, `BEAM_WIDTH`,
`MAX_TERMS`, `ARITIES`. **Shared by all three equations** -- E1 and E2 carried their own
(E1 at z-cap 3.0/penalty 20/pool 200, E2 at 3.0/5/100, against E3's 4.25/20/600) and no longer
do. This *will* move E1's and E2's numbers, and that is the point: three hyperparameter sets
meant the comparison between the equations was never quite like-for-like. Gone:
`DEFAULT_E1`, `DEFAULT_E2`, `DEFAULT_E3`, `DEFAULT_E3_CAPABILITY`, `QUICK_E1`, `QUICK_E3`.

**C7. Rename so the code says what the study claims** -- after C1-C6, as its own commit with no
behaviour change. `fit.fit()` performs a *search* (a beam over term subsets) and uses a ridge
fit only as its scoring function; calling the whole thing `fit` blurs the form-versus-weights
line the study is built on, which `validate.cross_validate_fixed_form` names correctly and
chapter 3 calls "term selection". Split into `search.py` (`guided_screen`, `Subset`,
`Selector`, `search()`, `SearchResult`) and `fit.py` (`Standardizer`, `ridge_solve`,
`to_equation`, `prune`). `search` imports from `fit`; that direction is correct and the
docstring should say so rather than implying a clean layering.

**How each step is checked.** C1-C3 are *meant* to move numbers, so the snapshot check does not
apply. C4-C7 change nothing and are verified by snapshot-and-compare over `results/`,
`assets/docs/` and all fourteen figures -- which the PDF determinism fix now makes possible.

## The selection rule -- WORKS HERE, WRONG IN PRINCIPLE, MUST BE REPLACED

```
consensus(a, k) = median of the three protocol R2 at arity a, k terms   (already computed)
p               = a * k        feature slots the equation spends
n               = 476          corpus rows

E3-Valid = argmax over (a, k) of  1 - (1 - consensus) * (n - 1) / (n - p - 1)
E3-MAX   = argmax over (a, k) of  consensus
```

One number, one max, no branches, no tuned constant. On this corpus: **(2, 15)** and **(3, 23)**
-- the equations the study already publishes, now derived.

The degrees-of-freedom factor exceeds one and grows with `p`, inflating the *unexplained* share,
so a term pays for itself only if the consensus it buys outweighs the residual degrees of
freedom it costs. At 476 rows that exchange rate is **about 0.0008 of consensus per slot**, set
by the corpus size rather than by a threshold. Arity 3 buys +0.0058 of consensus for 39 more
slots and is charged 0.0303 -- five times the price of the goods.

**`p = arity * terms` carries the decision.** With `p = terms` the margin between (2, 15) and
(3, 23) is **0.0006**, a coin flip; with the arity penalty it is 0.0086 and the top eight
configurations are all arity 2. So the justification -- a term of arity `a` names `a` raw
features, which is what tells fifteen terms at arity 2 apart from fifteen at arity 3 when a
coefficient count cannot -- is load-bearing and has to be argued, not asserted in passing.

**Three departures from the textbook, all in the docstring:** adjusted R2 corrects *in-sample*
optimism and the consensus is already cross-validated, so this is a second penalty on top of
one that is there; `p` is a complexity budget, not a parameter count; and ridge shrinkage puts
the effective degrees of freedom below even the coefficient count. The honest description is
"the consensus discounted by adjusted R2's functional form against a complexity budget".

**IT IMPLEMENTS THE LETTER OF THE RULE, NOT THE INTENTION.** Recorded 2026-09-08 by the
owner, and then measured. The intention was a statement about the *curve*: the shortest
equation whose combined R2 does not improve significantly with more terms -- saturation, a
plateau. What this implements is a *cost trade-off*: an argmax of consensus discounted by a
price per slot. Those coincide on this corpus and diverge elsewhere, because the price is a
function of the corpus size:

| n | E3-Valid | E3-MAX | picks | slot price |
|---:|---:|---:|---|---:|
| 476 | 0.6137 | 0.5834 | (2, 15) | 0.00081 |
| 1500 | 0.6307 | 0.6267 | (2, 15) | 0.00024 |
| 5000 | 0.6359 | 0.6389 | **(3, 23)** | 0.00007 |

**Same curve, same plateau, same 0.0058 gap -- only `n` changes, and the answer flips.** As the
corpus grows, slots get cheaper, the penalty vanishes, and the rule converges on the raw argmax,
which is E3-MAX. It will stop shortening at all. **This corpus is explicitly going to grow**, so
this is not hypothetical: the rule will change its verdict without anything about the equations
changing.

**What a correct rule would look like.** It has to read the *shape* of the consensus curve --
where the increments stop being distinguishable from noise -- rather than charge a
size-dependent price. The fold-to-fold spread is the natural scale for "distinguishable", and it
does not depend on `n` the way a degrees-of-freedom correction does. Whatever replaces this must
be checked against the flip above: **run it at several hypothetical corpus sizes with the curve
held fixed, and require the answer not to move.** That test is cheap and would have caught this.

**THIS RULE NEEDS REVISING.** It was arrived at knowing the answer it had to reproduce -- the
15-term arity-2 equation the study already defends. That is the shape of reasoning this project
rejected once before, in the capability-ordered rung. What makes it admissible rather than
fitted: it has no free parameter to have tuned, and its complexity measure is justified
independently of the answer. What it has **not** had is a test against a corpus where the right
answer is not already known. Revisit when there is one, and disclose the provenance in the
chapter.

**What it replaced**, so it is not re-attempted: "the simplest grammar whose own best length is
not significantly worse than the best overall". Same answer, three conditions and a paired test
-- and that test *rejected nothing*: all three arities came out "not significantly worse", with
arity 4 at 26 terms winning 13 of 20 folds. The discrimination came entirely from preferring
the simplest grammar, so the statistics were decoration. Earlier attempts and why each failed:

| rule | picked | LOO-dataset | |
|---|---|---:|---|
| shortest whose paired CI vs E3-MAX spans zero | arity 4, 9 terms | 0.5813 | 0.057 below the floor |
| + sign test not against it | arity 4, 14 terms | 0.6148 | 0.023 below |
| + mean not negative | arity 3, 23 terms | 0.6439 | degenerates to E3-MAX |
| Pareto knee / gRDP knee | 4 to 8 terms | -- | every one significantly worse; already rejected 2026-09-07 |

**The floor that rules these out:** E3-Valid may not land below the 15-term result (0.6381).
`selection.pareto_knee` decides nothing and stays a reported diagnostic.

## Documentation -- replaced, not merged

`origin/main`'s chapters are **superseded**; this branch's six-chapter structure is correct.
Main's `811253c` is the pipeline re-run on a collaborator's machine (the report header still
reads `C:\Datos\IT projects\...`) plus prose updated to match, against the eleven-chapter
pre-audit layout. **Nothing from it should come over**: its docstring edits claim a
configuration sweep is "pending" that job 15335 completed, and every chapter link it adds
(`06-results.md`) does not exist here.

Audited topic by topic on 2026-09-08: the consolidation lost almost nothing. Main's standalone
`08-limitations` was *distributed* into a "Limitations of ..." section inside each of the six
chapters, which puts each threat beside the thing it threatens; `09-model-effects` is absorbed
into `04-equation` with current numbers (per-model slope +0.0162, CI [+0.0049, +0.0290], 14
folds, against the +0.017 the old chapter recorded).

**One real gap, missing from both branches: the multiple-comparisons disclosure.** The ordinal,
the dummies, the axes, five embeddings, `Model Capability`, and every beam and sampler variant
recorded below were all tried against the same 476 rows and the same protocols, so the
surviving result comes out of a search and **its p-values are unadjusted**. That belongs in the
text and is in neither branch. Add the same disclosure for the selection rule above.

**One editorial call left open:** seven distributed "Limitations of ..." sections read better in
place, but a reviewer looking for threats to validity has no single section to open. A short
index in `index.md` linking the seven would keep both.

## The beam and sampler negatives, in full

**Job 15337** (the full beam sweep, 728,640 points, 15h03m on 62 cores) printed "5 of 14
policies beat 'vanilla' on a paired test", and **every row of that table is at nine terms** --
`OBJECTIVE_WEIGHTS` sends `_best_configuration` short, so the incumbent it paired against scored
0.5751, which is 0.063 below the published equation. At the published operating point:

| policy | terms shared with the published equation | LOO-dataset | delta |
|---|---:|---:|---:|
| `prune-relative-0.02` / `-0.05` / `cap-3` | **15 of 15** | 0.6381 | 0.0000 |
| `cap-2` / `cap-2+prune-0.02` | 9 of 15 | **0.4355** | **-0.2026** |

`cap-2`'s leave-one-*model* R2 barely moves (-0.003) while leave-one-dataset drops 0.203 and
leave-one-cell drops 0.213 -- the terms it loses are dataset-side, so the damage only shows when
the dataset is withheld. Its protocol spread widens from 0.0416 to 0.2479. **And the paired test
calls that collapse a "tie"** (p = 0.115, 6/20): on this comparison the paired test can
under-call a catastrophic R2 loss, so the R2 delta must be read beside it, never instead of it.

**R2 and ranking can disagree:** `cap-2` loses 0.213 of cell R2 while its cell ranking MAP goes
*up*, 0.854 against 0.831. A ranking-only check would have promoted it.

**Speed differences vanish under repeated timing.** Best-of-five puts every policy at 0.957 to
1.008x, against the 1.12-1.28x single-run numbers. Three policies returning the *identical*
equation still spread 0.978 to 1.002, which is the machine's noise floor.

**ESS, three ways, all negative.** As a one-shot seeder (`seed_terms`): worse and 1.5-2.8x
slower. As a continuous sampler inside the beam (`ProbeField`, anchors accumulating with
measured objectives, attractiveness = -objective, cauchy attraction, refitted every step): the
**identical equation** at 6 and 15 terms, at 3-7x the cost. Over the configuration grid, replayed
against the exhaustive 48,576-point table as a lookup oracle, 100 seeds, budget 2,000:

| | mean | vs random | 95% CI | wins |
|---|---:|---|---|---:|
| random | 0.6766 | -- | -- | -- |
| Sobol | 0.6778 | tie +0.0011 | [-0.0003, +0.0025] | 53/100 |
| ESS w=1.0 | 0.6770 | tie +0.0004 | [-0.0009, +0.0018] | 53/100 |
| LHS | 0.6759 | tie -0.0007 | [-0.0022, +0.0008] | 43/100 |

**Nothing beats uniform random.** The only significant effect anywhere is Sobol over LHS. Why,
measured: the grid is 37.4% categorical by variance (arity 29.4%, feature subset 6.8%) against
5.9% for the two continuous axes with `penalty` at 0.3%; adjacent penalties differ by 69% as
much as random pairs; every optimum sits on a grid boundary; and one point in 48,576 is within
0.005 of the best. Sobol samples with 3.7x lower discrepancy than random and the **correlation
between a sample's discrepancy and the best score it finds is +0.02** -- coverage is a guarantee
about integration error, and this is a hitting problem on an isolated corner.

### Three harness failures that each produced a confident wrong answer

Worth more than the results. All three were mine, in one session.

1. **A sampler that was never called.** Blocks of 8 over 32 strata meant a budget of 200 spent
   every draw on the random cold start, so `ess.esa` ran zero times and the run reported "ESS
   does not help". The tell was in the output and went unchased: two policies printed
   byte-identical means *and* standard deviations.
2. **Discretisation that halved the boundary sampling rate.** Rounding a continuous draw to the
   *nearest* lattice level gives the two extreme levels a half-width cell and every interior
   level a full one -- 0.200 against 0.333 on a six-level axis. Sobol and LHS hit an axis
   extreme 21.5 times per 62 draws against random's 36.8, and every optimum on that grid is on
   an axis extreme. That artifact alone produced four significant "worse than random" verdicts.
   Equal-probability binning fixed it exactly.
3. **Twelve seeds against a 0.001-0.005 effect.** Random's own mean moved 0.0038 between 12 and
   100 seeds, so a 12-seed run reported ESS, Sobol *and* LHS all beating random, and an earlier
   one reported ESS beating random by +0.0045. Neither survived. **Match the seed count to the
   effect size, not to the slowest sampler** -- the shared baseline is where power is free.

**The grid is truncating its own optimum**, independent of any sampler: both R2 argmaxes sit at
`k = 28` and `z = 5.00`, the top of both axes. That is worth more than any search method and is
a one-line change if the grid is ever rebuilt.

**Where a smarter search would pay**, none of it ESS: successive halving over the protocol
ladder (rank cheaply on in-sample, promote survivors to leave-one-dataset-out, only finalists to
leave-one-cell's 476 solves) is the strongest fit and assumes nothing about geometry; Optuna's
TPE handles the mixed categorical/integer/float shape this grid actually has; constraint or
branch-and-bound over the feature-subset lattice is the only method addressing the axis that
explodes (2^k in optional columns). A* is rejected on the same ground as OBL: it needs an
admissible heuristic on partial configurations, and inventing one adds a knob rather than
testing an idea.

## Repository state

- **One CLI command.** `ml-meta-perf --help` is the only interface again; `ml-meta-perf-search`
  and `ml-meta-perf-beam` are gone. `scripts/` is empty of jobs.
- **No optional dependencies.** The `search` and `diversity` extras are gone. **joblib is a base
  dependency now, and that was a fix**: `opaque` imports it at module level for the
  leave-one-cell protocol's 476 refits, `experiment` imports `opaque`, and `run()` calls it
  unconditionally -- so `pip install ml-meta-perf && ml-meta-perf` raised ImportError while
  `pyproject.toml` claimed a base install reproduced the whole study. CI missed it because CI
  installed `.[search]`, for a reason that was about basedpyright rather than about the study
  running. CI now installs `.` and would catch it.
- **PDF figures are deterministic.** matplotlib stamped `CreationDate` from the wall clock, so
  every run rewrote seven tracked files with byte-different, visually identical content --
  making `git status` dirty after any run and making the compare-the-outputs check impossible
  for the vector figures. `metadata={"CreationDate": None}` fixes it; PNGs were already stable.
- **The removals were verified as no-ops** the way this repo verifies changes: snapshot
  `results/`, `assets/docs/` and the figures, remove, re-run, compare. Identical throughout,
  across all fourteen figures.


## Where the study stands

| | grammar | terms | features used | in-sample | LOO-dataset | LOO-model |
|---|---|---:|---:|---:|---:|---:|
| E1 — dataset features | arity 3 | 7 | | 0.349 | 0.341 | 0.306 |
| E2 — model features | arity 2 | 6 | | 0.248 | 0.185 | 0.228 |
| **E3 — both (published)** | **arity 2** | **15** | **12 of 16** | **0.658** | **0.638** | **0.622** |
| E3 — capability | arity 3 | 23 | | 0.707 | 0.678 | 0.651 |

`DEFAULT_E3`: arity 2, z-cap 4.25, penalty 20, 15 terms, four-feature model pool.
`DEFAULT_E3_CAPABILITY`: arity 3, z-cap 4.25, penalty 3, 23 terms, same pool.

**The equation's model-feature pool is four; the corpus keeps six.** `MODEL_FEATURES` is the
corpus schema and identification is a corpus property; `EQUATION_MODEL_FEATURES` is what the
equation may build terms from, because fitting is judged on *compression*. Dropping
`Solution Stochasticity` and `Loss Margin Behaviour` from the term pool removes nothing from
the corpus. Do not re-run the identification argument against the equation's pool — that
mistake was made and corrected this session.

**The length comes from a rule, not a constant.** `selection.best_length`: fit at every
length, take the median of the three protocol R² per length, publish the argmax. No
threshold, no smoothing, no sensitivity parameter. It selects 15 under arity 2 and 23 under
arity 3; neither number appears anywhere in the code.

**Four protocols, and the ranking and threshold decisions are reported under the strictest.**

| what the equation was shown | R² | AP | hit@1 | F1 @ 0.7 | MCC @ 0.7 |
|---|---:|---:|---:|---:|---:|
| in-sample | 0.658 | 0.850 | 0.80 | 0.900 | 0.730 |
| leave-one-dataset-out | 0.638 | 0.847 | 0.80 | 0.897 | 0.719 |
| leave-one-model-out | 0.622 | 0.838 | 0.85 | 0.886 | 0.695 |
| **leave-one-cell-out (both unseen)** | **0.616** | **0.831** | **0.80** | **0.886** | **0.695** |

`validate.cross_validate_doubly_held_out` refits with the dataset row-block *and* the model
column-block removed and predicts the cell. **It is used for the ranking and the threshold
decision only** — the R² curve and the length rule stay on the single-group protocols, which
is the right scope for them. 476 solves in 0.6s.

**The headline this licenses:** with neither the dataset nor the model seen, the equation
still reaches R² 0.616 and AP 0.831 — and the per-model mean and median **cannot be computed
at all** under that protocol, because a model held out of every fold has no rows to average.
Every earlier comparison handed them the model identity the equation was denied.

The corpus is one file, `src/ml_meta_perf/meta_dataset.csv`: 2 identifiers, 12 dataset
features, 6 model features, `MCC`.

## 3. The LOO-dataset craters — diagnosed and disclosed

Two different things, previously conflated as "E3's worst fold is -10.62".

**The negative per-fold R² was a metric artefact, not a fit failure.** `5G_Slicing` has a
within-dataset MCC standard deviation of **0.0516** — all 25 of its models score about 0.986.
R² divides by that, so an ordinary error returns a large negative number; its predictions are
all within 0.05 of the truth. `NSR` (0.0551) and `DeepSlice` (0.0639) are the same, and those
three are the only strongly negative folds. `CrossValidation.dispersion` now reports per-fold
**MAE** alongside, plus `low_variance_folds`. The worst fold by MAE is 0.256 and is a
different dataset — which is the one actually worth looking at.

**The pooled craters are real, and they are extrapolation.** At k=15 the held-out
`ASNM-CDX-2009` fold is predicted at **-2.41** — raw pooled R² **-2.06** — and
`_clip_to_training` pins it to the training floor of -0.29, recovering 0.393. The design is
*well* conditioned there (cond 10, max |w| 0.14): a held-out dataset can sit outside the
convex hull of the other 19 in term space, and a linear equation extrapolates without limit.

**The clip is load-bearing everywhere**, now measured and reported: it moves **54 of 476**
predictions under leave-one-dataset-out and **70** under leave-one-model-out, at every length.
`CrossValidation.clipped` counts it. Still open: whether to bound extrapolation some other
way, or to accept the clip and justify it prominently in chapter 5.

## Open, in the order it is worth picking up

1. **C1 through C7 above** -- the CLI and selection rebuild. C1 is the next commit.
2. **The model-descriptor question, which is the study's live problem.**
   The identity ceiling is +0.050, not the +0.017 the chapter recorded, and it was settled by
   the study's own paired test rather than by judgement: a per-model **level** is a tie, and a
   per-model **slope** is not — bootstrap interval [+0.0049, +0.0290], entirely above zero, on
   14 winning folds of 20. (Its sign test is p = 0.115, so the gain is carried by magnitude
   rather than by consistency; the generated section reports both.) The surviving half is
   exactly the one that lets a learner's advantage depend on the data, which is what a
   *capability* descriptor would do and what nothing in this corpus records. **Behavioural
   probing therefore stops being future work and becomes the next step** — see "When the
   meta-dataset can be recomputed", Group A, which is costed and specific.

3. **The documentation replacement** -- six chapters, everything regenerated here, plus the
   multiple-comparisons disclosure missing from both branches. See "Documentation" above.
4. **Open a PR and let CI run.** Nothing has ever run on a runner, and this branch has changed
   the dependency set twice: scikit-learn became a base dependency on 2026-09-07, joblib on
   2026-09-08. Deliberately deferred, not forgotten.
5. **Chapter 6's practice catalogue.** The verdicts are regenerated and each practice is paired
   with the terms carrying it, but only 3 of the 10 make a claim any term can answer -- the rest
   are about a protocol, a metric, or a family of learners. The ten were chosen before that
   pairing existed. A catalogue chosen *for* it would raise coverage and make the term-level
   check the chapter's contribution rather than a footnote. Practices about class imbalance,
   feature count, class entropy and the instances-per-attribute ratio are all expressible as
   feature claims with terms to check against.
6. **The per-family effect table.** The one measured model-side description that transfers to an
   *unseen* model: +0.075 LOO-model where per-model identity gives exactly 0.000. A table rather
   than an equation, so it fails the single-equation gate. Publish as a second component or keep
   as a ceiling? Numbers are pre-2026-09-05 and need re-measuring.
7. **The opaque comparison costs four minutes of the ~4.7-minute run**, nearly all of it the 522
   estimator refits the leave-one-cell protocol needs. `opaque._doubly_held_out` already threads
   the cell loop. If runtime becomes a problem the honest lever is fewer trees -- the conclusion
   does not depend on 300 -- not dropping a protocol.
8. **Correcting `Training Operations` for the 100k cap -- blocked, not rejected.** Needs each
   trained instance's tuned hyperparameters; the upstream `results_stage_ml_eval.csv` covers 348
   of 476 rows and the 128 gaps are exactly the 8 GPU-trained models x 16 datasets.
9. **Re-read chapters 0 and 2 end to end.** Chapters 1 and 3-6 had a continuous read on
   2026-09-07; 0 and 2 did not, beyond labelling chapter 2's historical tables.

## Practices are paired with *terms* now, not with features

`guidance.equation_evidence` was rewritten on 2026-09-08 and the unit changed. It first paired
each practice with a raw **feature** and reported one direction per feature, which was the
wrong shape twice over: the equation is a sum of *terms*, and a feature enters several of them
in different positions. Collapsing that into one direction threw away the only reading a
readable equation can offer.

One row per (practice, term) pair now, with the term's standardised weight, its 10-90 effect,
its fold stability, and a **measured** direction — measured because reading a sign off the
weight is wrong the moment the feature sits in a denominator, and several do.

**What it found, and it is the best single illustration of why the study fits equations at
all.** `Processing Units Number` carries five of the fifteen terms, all with negative weights.
Where it is a *denominator* (`log(eq_num_attr)/log(PUN)`, `log(gravity)/log(PUN)`,
`nr_cor_attr/log(PUN)`) the term rises with capacity; where it is a *numerator*
(`log(PUN)/log(nr_class)`) it falls; and in the one product (`log(IDM) * log(PUN)`) the
association is 0.05, which is no direction at all.

That is arithmetic, not contradiction — a negatively-weighted ratio contributes more as its
denominator grows — and read with the `position` column it is one coherent statement: **the
equation has no marginal claim about capacity, only claims about capacity relative to
something the dataset demands.** Which is the conditional form of "match capacity to the
problem", the practice's own headline.

**The reported mismatch was therefore in the expectation, not in the equation.** The practice
is a conditional claim and it had been encoded as a marginal one (`capacity lowers MCC`); a
marginal expectation can never match a term that only speaks conditionally. Two fixes on
2026-09-08 made that legible rather than confusing: `guidance.feature_position` reports which
slot the feature occupies, and `MIN_TERM_DIRECTION` (the same 0.15 floor the per-feature
statements use) stops a pairing with |rho| = 0.05 being given a sign it cannot support. The
counts moved from "4 agree, 4 disagree" to **4 agree, 3 disagree, 1 undirected, 1 not
selected**, and all three disagreements are the same feature in a denominator.

**The open question this leaves is whether the expectation language should carry conditional
claims at all** — `capacity relative to dataset difficulty raises MCC` rather than a bare
per-feature direction. It would score the capacity practice as agreeing, which it does. It
would also need a way to say *relative to what*, and inventing one risks fitting the encoding
to the answer. Left as-is deliberately, with the position column doing the work instead.

Two intermediate designs were tried and dropped, both recorded so they are not re-attempted:

- **Feature-level directions.** Too coarse, as above. It also mislabelled `nr_outliers` as
  "no direction" because the *summed* per-feature effect failed the monotonicity filter, while
  the single term carrying it has a perfectly clear one.
- **Family-level row predictions** — asking whether the equation's out-of-fold predictions
  rank tree families above neural ones. It works (all three claims came out `yes`), and it is
  answering a different question: whether the equation *predicts well*, which chapter 5 already
  establishes. It says nothing about which part of the equation encodes the advice.

**Coverage is honest and low, and the chapter says so.** Only 3 of the 10 practices make a
claim about a quantity the equation contains; the other seven are about a protocol, a metric,
or a family of learners and have no term to be checked against. The generated section counts
this explicitly so the verdict tally ("5 supported") is not read as the equation having
validated five practices — those verdicts come from corpus statistics that any study with this
corpus could compute.

**If the low coverage is unsatisfying, the fix is the catalogue, not the check.** Ten practices
were chosen before this pairing existed. A catalogue chosen *for* it — practices that make
claims about quantities the corpus records — would have far higher coverage and would be a
more focused contribution. That is a research decision, and item 5 under "Open" already
flagged that the ten have never been re-checked against the 15-term equation.

# Facts worth not rediscovering

**A comparison is only a comparison if every side is scored under the same protocol.** This
went wrong twice in one session and both times it flattered the equation: the ranking was
scored in-sample against leave-one-out baselines, and then the leave-one-out baselines turned
out to be seeing the model identity the equation is denied. Every row of
`ranking_baselines` and `decision_baselines` now names its protocol. When adding a predictor
to either table, name the protocol in the label or it will drift again.

**The protocol is fixed-form and there is no second one.** `cross_validate_fixed_form` fits
the equation once and refits only its weights per fold. `fold_selections` re-runs selection
inside the folds and returns **term names and no predictions**, so the re-selecting protocol
cannot produce a reported score; `term_stability` over it is the safeguard that licenses
fixing the form. `tests/test_model_features.py::TestReportedProtocol` pins this.

**Numbers from the two protocols differ by up to 0.3 and must never share a table.** The old
published 0.474 / 0.428 were re-selecting; anything current is fixed-form. **Every number
inherited from the deleted `todo.md` is pre-2026-09-05 and therefore re-selecting.** None of
them may be quoted beside a current figure without being re-measured first.

**Re-sweep the configuration whenever the feature set *or the grammar* changes.** This has
bitten four times. `DEFAULT_E3` tuned on the re-selecting metric scored 0.246 under fixed
form. The fourth was the 2026-09-07 constraint, and job 15335 closed it — but the rule stands
for the next change.

**Library construction is order-independent as of 2026-09-05** — `build_library` sorts both
feature groups. Before that, `A * B` and `B * A` entered as two names for one column and the
de-duplication kept different terms depending on argument order, moving leave-one-dataset-out
by up to 0.286. If a future change reintroduces order sensitivity, that is the cause.

**Any feature defined as a ratio of two others creates perfectly collinear library terms.**
`inst_to_attr` is `nr_inst / nr_attr`, so `log(inst_to_attr) + log(nr_attr)` *is*
`log(nr_inst)` exactly, and the grammar generates both sides. `Library` now de-duplicates on
centred column *values* rather than on `Term.name`, which catches it — but the right move is
not to define features that way.

**E3's model-only terms carry almost none of its output variance, and that is expected.**
Currently 0.047 of the share across 2 terms, against 0.42 for the 4 dataset-only terms and
0.53 for the 10 mixed ones. It is not a defect: a model feature on its own can only shift
every row of a dataset by the same amount, so the work it does is *conditional* on the data
and lands in the mixed terms by construction. That is precisely why E3 beats E1 and E2
combined. The shares are a covariance decomposition and sum to 1, so a small model-only share
means those terms are correlated with the mixed block, not that they are idle. Do not
re-open this as an anomaly.

**Never read a difference of two means over twenty folds as a result.** This has produced
three wrong conclusions in one session: that the equation out-ranked the trivial baseline
(paired, it wins on 7 of the 17 datasets that differ, every interval spanning zero); that
ranking quality falls with equation length (it is two datasets flipping their top pick, and
hit@1 can only move in steps of 0.05 on twenty folds); and that a stricter form of the new
constraint was catastrophic (one bad landing at one length). Use
`validate.paired_comparison` — exact sign test plus a bootstrap over groups, no scipy.

**The group-mean baselines need a leave-one-out correction.** With 17-25 rows per group,
including the row being predicted inflates the per-model mean's R² by 0.082.
`validate.baseline_group_mean` does it correctly; a naive group mean does not.

**The configuration is selected on the protocol it is reported on**, and always has been.
Length and penalty are chosen by maximising LOO-dataset R², which is then the headline. That
is optimistic, and nested selection is the fix if a reviewer presses. Related: the ordinal,
the dummies, the axes, five embeddings and `Model Capability` were all tried against the same
476 rows and the same two protocols, so the surviving result comes out of a search and its
p-values are unadjusted. Worth disclosing in the limitations chapter.

---

# Measured and rejected — do not re-attempt

All measured, all worse, **all under the pre-2026-09-05 re-selecting protocol**. The
conclusions stand; the numbers would need re-measuring before publication.

| attempt | outcome, and what its failure rules out |
|---|---|
| Divide-and-Learn (Gong & Chen) | halves LOO-dataset R²; 20 datasets split 3 ways is 6 per division |
| Agglomerative / dendrogram term construction | worse than enumeration; `git log -- src/metafit/construct.py` recovers it |
| Extra transforms `f^3`, `1/sqrt(f)`, `f^0.25` | identical in-sample, worse under the looser grammar |
| Scaling raw features up front | fails three separate ways |
| Marginal / Pareto filtering of terms before search | destructive — terms are not independent |
| `max_arity = 4` | +0.035 fit for −0.203 transfer |
| More search (8× compute, 16× beam) | converges to the 4th decimal; the search was never the binding constraint |
| Learning-to-rank objective | ranks *worse* than squared error |
| Backfitting against a per-model table | falls over two rounds and keeps falling |
| Aggregating E1 to 20 dataset means, or E2 to 25 model means | incomparable denominator; both worse |
| Log-differences of the three cost columns | the five original columns hold no unexploited structure; the grammar was not the limitation |
| An ordinal "model complexity" column | capability is not monotone in any complexity ladder — per-family residuals run *backwards* against it |
| A capability-ordered rung (rather than complexity-ordered) | performs far better, and is rejected because **the label would be false**: against this corpus the asserted order beats chance at only 6 of 9 steps, and `generic NN` has the lowest mean MCC of any family while sitting at rung 6 of 10. A transparently *mis*named feature reads fine and is wrong, which for an explainability-first study is worse than no feature |
| One-hot family dummies | family is too coarse to encode per-family at this sample size — 7 of 10 fail the z-score floor |
| Five binary mechanistic axes | family information *does* help, but only as per-group slopes: an indicator inside a product is identically zero on the rows where it is off, which is a piecewise model, not one equation |
| Principal-coordinate embeddings of the axis matrix | best transfer measured anywhere here, and inadmissible: the information is worth ~5 dimensions and two of the five have no plain-language reading |
| The 61 AI-proposed hyperparameter descriptors | a hyperparameter a learner does not have has no value; encoding absence as zero collapses applicability into magnitude — seven of the best-scoring ten were zero on 72–99% of rows |
| The four retired model columns | removing each *helps* LOO-model; three varied within a model; two were zero-based so no log, root or reciprocal applied |
| Spearman as a ranking metric here | 0.63 to 0.73 for every predictor *and* every baseline, including a constant. Use AP, MRR, hit@1, regret |
| NDCG@3 and regret@3 | saturated: 0.97–0.99 and 0.002–0.009 for everything, because 9.3 of 25 models are tied at the top on average |

### The detail behind three of those rows

Folded in from `assets/docs/08-appendix.md`, deleted 2026-09-07 when the chapters were cut to
the six the study reports. **All of it predates the 2026-09-05 protocol change**, so the
conclusions carry and the magnitudes would need re-measuring.

**Agglomerative construction, instead of enumeration.** Build terms by repeatedly merging the
most-correlated pair rather than enumerating the grammar. It *does* find terms enumeration
misses, and it is cheaper. It is also worse on transfer at every length tried, and the
variants make the reason plain: target-free pairing adds nothing over enumeration; cutting
the dendrogram and using the clusters as the equation is worse again; a guided merge is
cheaper and worse still; and nesting merges deeper does not pay on this data. The pattern
across all four is that **every increase in expressiveness cost transfer**, which is the same
signature the feature-side negatives show. `git log -- src/metafit/construct.py` recovers the
implementation.

**A richer transform vocabulary.** Adding `f^3`, `1/sqrt(f)` and `f^0.25` grows the library
from 172 terms to 182 — most of the 51 new candidates fail admissibility — and the beam
selects **none of the ten that survive**. In-sample R2 is identical to four decimal places at
8, 14 and 20 terms. Under a looser arity-4 grammar the same extension is actively harmful.
Higher powers are near-duplicates of `f^2` and `sqrt(f)`, and the collinearity guard treats
them as such.

**Six attempts on the additive form's accuracy**, against a then-baseline of 0.5582
in-sample: downweighting the rows at MCC in {0, 1} by 0.5 (0.5380) and by 0.25 (0.4147),
Huber IRLS over 8 iterations (0.5526), equal weight per dataset (0.5509), a two-stage fit of
7 dataset terms then 7 on the residual (0.5532), and the transform extension above
(unchanged). Reweighting was the most promising and is the clearest failure: R2 is reported
on all 476 rows with uniform weight, so any reweighting optimises a *different* objective and
necessarily scores worse on the one being reported. Downweighting the saturated rows removes
118 of 476 observations' worth of influence — the pile-ups at 0 and 1 are a third of the
data, not outliers to be discounted.

**Standing decision (2026-08-17): no binary indicator features.** Inside a product an
indicator is identically zero on the rows where it is off, which is a per-group slope rather
than a relationship, and only `f` and `f^2` are defined for a 0/1 column — and for a binary
column those two are numerically identical.

**Standing decision (2026-08-17): no re-running any model or experiment from the
meta-dataset.** That closes behavioural probing and the corpus rebuild, so every model
descriptor this study can still add is *asserted* taxonomy rather than measurement.

## What the negatives are worth

Read as a list of failures this looks like a wasted search. Read as evidence it is the
strongest support the study's central claim has, because every attempt failed *for the same
reason*, and each closes a different escape route a reviewer would otherwise ask about.
Together they say something sharper than "we tried and could not improve it": **the missing
signal is model capability, it is worth +0.121 leave-one-dataset-out, and it is not
recoverable by re-encoding anything the corpus records.**

Every model descriptor the corpus has says what a model **costs** (`Processing Units Number`,
the operation counts) or asserts a static property. **None says what a model is good at.**
That belongs in the limitations chapter, with the five negatives as its evidence — it is a
property of the corpus, not of the search.

**The one direction that survives on the merits is behavioural probing**, closed only by the
decision not to re-run models. Say so in future work: it is a known, costed, specific next
step, not a vague gesture. Landmarking (Pfahringer, Bensusan & Giraud-Carrier, 2000)
characterises a *dataset* by running fast learners on it; inverted, it characterises a *model*
by running it on a battery of tiny synthetic tasks with known structure. ~25 models x ~10
probes of a few hundred rows is minutes, once, and independent of the 20 real datasets.

**If a capability descriptor is wanted without re-running anything**, the one admissible form
is *measured elsewhere, not asserted here*: published mean normalised rank of each algorithm
across external tabular benchmarks — the "average rank" prior of algorithm selection (Brazdil
& Soares, 2000). Continuous, citable, non-circular. It would have to be assembled by hand,
with coverage gaps for `TabICL` and `TabPFN`. Not started.

---

# When the meta-dataset can be recomputed: what to record

The diagnosis dictates the specification: add *behavioural* descriptors — things the model
does, measured — and nothing else.

## Group A — dataset-independent, measured once per model

Run each model on a battery of small synthetic tasks with known structure and record its
score. Never touches a real dataset, so there is no leakage question at all. This is the
group most likely to reach the +0.121, because it measures capability directly rather than
proxying it.

| descriptor | probe | separates |
|---|---|---|
| `Rotation Tolerance` | axis-aligned vs rotated boundary, same task | trees from linear models |
| `Interaction Capture` | XOR / parity | linear from non-linear classes |
| `Label Noise Tolerance` | MCC drop at 10% and 20% flipped labels | `PassiveAggressive` from `SGD`, `Perceptron` |
| `Redundancy Tolerance` | correlated / duplicated features | `BernoulliNB` from `LinearSVC` |
| `Irrelevance Tolerance` | padding with pure-noise features | `DT` from `ExtraTree` |
| `Imbalance Tolerance` | 95:5 class split | generative from discriminative losses |
| `Sample Efficiency` | MCC at 100 rows / MCC at 1000 rows, same generator | in-context and instance models from the rest |

These reach the collisions no family-level encoding can touch — `DT`/`ExtraTree`,
`SGD`/`Perceptron`, `LightGBM_RF`/`LightGBM_ExtraTrees` — which are 121 of the 151 tied rows.

## Group B — per (model, dataset), computed inside the training fold only

| descriptor | how | why it is worth the cost |
|---|---|---|
| `Learning Curve Slope` | MCC at 25 / 50 / 100% of the training sample, slope against `log(n)` | answers the question the 100k cap made untestable |
| `Tuning Sensitivity` | IQR of cross-validated MCC across the tuning trials | already latent in the optuna studies — just record it |
| `Calibration Error` | Brier or ECE on the training folds | describes something MCC cannot see |
| `Train Validation Gap` | train MCC − validation MCC | overfitting propensity, per instance |

These are **landmarks** and legitimate as such, but they are computed on the target dataset,
so the discipline is absolute: training folds only, never the fold being predicted. Group A
has no such hazard, which is why it comes first.

## Group C — fixes to what is already recorded

- **Record the actual sampled training size** and compute `Training Operations` from it.
- **Record per-instance hyperparameters for every row**, GPU models included — the 128 gaps
  block any recomputation from tuned values.
- **Keep `MCC_Fold_Std` out.** It is an outcome of the run being predicted, not a descriptor.

## Rules to bake into the generator

Continuous; strictly positive; narrow dynamic range (max/min ≤ 20, or the column can never be
a denominator); clear of 1.0 (log 1 = 0 blocks the preferred denominator form); per-trained-
instance where possible; named as a physical quantity; never a nominal category as an integer;
never an outcome of the run being predicted; **never defined as a ratio of two other
features**.

A probe score in roughly **[0.3, 0.95]** satisfies the first four. Raw MCC in [−1, 1] is
unusable: at zero or below, `log`, `sqrt` and `1/f` are all undefined and the column can never
divide, which strands it in the corner the binary axes were in. Use balanced accuracy, or map
MCC to (0, 1] with a floor.

---

# Reproducing

```bash
venv/bin/python -m ml_meta_perf        # the whole study, ~16s
venv/bin/pre-commit run --all-files    # the gate
```

The run writes `results/*.{json,csv}`, `assets/figures/*.{png,pdf}`, and the **generated
sections of chapters 4, 5 and 6** in place, between the markers in `report.BEGIN`/`report.END`.
`--docs DIR` moves the last of those; `--no-report` skips it.

`ml-meta-perf-search` (`src/ml_meta_perf/equation_search_cli.py`,
`scripts/equation_search.sbatch`) is the configuration sweep. It ran on 2026-09-07 as Slurm
job 15335 and the shipped configuration survives it — see item 1, and read it before
re-running: the cluster copy goes stale and the memory request is load-bearing.

## Working habits worth keeping

- **Audit prose numbers against `results/` mechanically, not by reading.** Harvest every decimal
  from `results/*.csv|json` and the generated blocks, then flag every decimal in hand-written
  prose that does not appear at the precision it is written.
- **Verify a change by byte-comparing every output, one change at a time.** Snapshot `results/`,
  `assets/docs/` and the figures, make one change, re-run, `cmp` everything. A batch of four
  changes cannot be verified this way; four separate ones can.
- **When a number belongs in a chapter, generate it.** The rule is not "check the numbers", it
  is "give the chapter nowhere to keep a number of its own".
- **Score every predictor under every protocol, including the strictest.** A comparison missing
  its strictest column is not conservative; it is flattering whichever side had more left over.
- **Pick the unit before building the check.** The practice comparison was built three times --
  against features, against family-level row predictions, and finally against terms -- and only
  the third says anything a scatter plot could not.
- **Match the seed count to the effect size, not to the slowest arm** of the comparison.
- **Never read a difference of two means as a result.** Use `validate.paired_comparison`. And
  read the R2 delta beside it: the paired test called a -0.203 collapse a tie.
- **`git checkout -- assets/docs/` twice destroyed uncommitted work.** Commit before any bulk
  restructure; prefer moving files to a scratch directory over reverting.
- **Splicing text with `s.index(a)` / `s.index(b)` silently duplicated a hundred lines** when the
  end marker occurred before the start marker. Assert `end > start`.

