# WIP: a model-feature set, a unified equation, and two downstream tests

## Start here (written 2026-09-05 to hand the branch to a cold session)

**What this branch is doing.** Replacing `data.MODEL_FEATURES` with a set that (a) identifies
every model in the corpus, (b) is admissible under the project's explainability gate, and
(c) holds up on two new practical tests — a binary above/below-threshold decision and a
ranking of models per dataset. Nothing is wired into production yet.

**The candidate: `proposed-6`.** `Model Capability`, `Processing Units Number`, and four
asserted mechanism ordinals — `Solution Stochasticity`, `Loss Margin Behaviour`,
`Input Distribution Modelling`, `Fitting Regime`. Built by
`scripts/build_proposed_frame.py` into `assets/meta_dataset_proposed.csv` (generated, not
tracked — rebuild it before anything else). It is the only set measured anywhere here that
**identifies all 25 models and all 10 families**, and it currently leads the search objective.

**The protocol, settled and load-bearing.** The equation's *form* is the conceptual claim; it
is fitted once and only the **weights** are refit per fold. The re-selecting ("nested")
protocol is deliberately not scored — it tests the search, not the science. The safeguard that
makes this legitimate is `stability`: the fraction of terms selection picks again when a fifth
of the data is removed, weighted 0.15 in the objective. See "The protocol decision".

**Numbers under this protocol are NOT comparable with the published 0.613 / 0.474 / 0.428**,
which are re-selecting. Under the fixed form the *shipped six themselves* reach 0.608 / 0.578,
so most of any apparent jump is the protocol. The study must restate its headline under one
convention and say which. This is the single easiest mistake to make when picking this up.

**What was fixed in `src/` this session** (gate green: 459 tests, ruff, basedpyright, vulture):

1. `terms.build_library` is order-independent — this closed the leave-one-dataset-out ordering
   band completely, spread 0.286 → **0.0000**. It was *not* the beam's tie-breaking.
2. `validate._clip_to_training` bounds each fold by its own training target range.
3. `CrossValidation.dispersion()` reports median and worst per-fold R2.
4. `fit._descending` — a real latent bug (unstable sort, then reversed) that changed no number.

**Where things stand.** Cluster job **15333** finished: 48,576 points in 67 minutes, in
`results/cluster/equation_search.csv`. **It picked four features, not six** —
`Model Capability`, `Processing Units Number`, `Fitting Regime`,
`Input Distribution Modelling` at 16 terms, lambda 25, z 4.5, arity 2, scoring in-sample
0.664 / LODO 0.641 / LOMO 0.627 with stability 0.872. It dropped `Solution Stochasticity` and
`Loss Margin Behaviour`, which are the two columns that make the set identify individual
models. See "Job 15333 result" — the identification-versus-fit choice there is the first thing
to settle.

**Read next, in order:** "The protocol decision", "Applied to `src/`", "The six-feature set",
then "Open work". Everything before those is the history of how the branch got here and can be
skimmed.

## Open work

| # | task | why it is next |
|---|---|---|
| 1 | **Settle family-versus-model resolution** | the winner identifies all 10 families but leaves 134/476 rows ambiguous between models. If the claim is at family level it is complete; if at model level it needs the two ordinals the search dropped, which cost stability and ranking. Re-rank `equation_search.csv` under alternative weights first — that is free |
| 2 | **Run the cleanup plan** | "Cleanup and documentation plan" — one corpus, a validity test, the new feature chapter, the two eval figures, then the deletions. Order is **B → A → D → E → F → C**; C removes ~80 tests and eight modules and goes last |
| 3 | **Audit the four ordinal tables** | 25 rungs each, all asserted; one pass already found three errors. Check each against a published description, and check each ladder against per-rung mean MCC the way ch. 8 checks `Model Capability` (which matches at only 6 of 9 steps) |
| 4 | **Read the chosen equation aloud** | the k=23 form failed this: reciprocal pairs (`A/B` and `B/A` both selected), a -1.77 weight, `log(gravity)` fanned over six terms |
| 5 | **Screen the dataset side** | `DATASET_FEATURES` is 12 inherited columns and no chapter shows a candidate was ever rejected — the same criticism being levelled at the model side. Needs no re-running of models |
| 6 | **Restate the published headline** under the fixed-form protocol | or say plainly that the paper reports the re-selecting one; the two must never share a table |
| 7 | Phase B | wire the winner into `data.py`, re-sweep `DEFAULT_E2`/`DEFAULT_E3`, regenerate `assets/docs/` |

### Known-unfixed, deliberately

- `candidates` (the beam's per-step frontier, fixed at 24) is unreachable from `Configuration`
  and ignored by `cross_validate_path`. **Measured, it does not matter**: in-sample is
  identical to three decimals across a 16x change in it.
- `guided_screen` is a marginal filter that switches on silently once the library exceeds
  `pool_size`. It is a no-op at every configuration the study uses (libraries 175-316 against
  `pool_size` 600) and only cuts at z 5.0.
- `build_library` applies the z-cap over all 476 rows, outside the fold loop, so which terms
  *exist* is decided with held-out rows in view. Under the fixed-form protocol this is no
  longer a leak in the scoring sense, but it is worth a sentence in the methods.
- The k-sweep is still jagged in leave-one-dataset-out (max adjacent jump 0.297). That is
  genuinely different equations at different lengths, not a bug — and it is the reason
  `stability` carries weight.

### Files this branch added

```
src/ml_meta_perf/  equation_search{,_cli}.py   the multi-objective search (current)
                   admissibility.py            the descriptor gate
                   reference_subsets.py        the external run, with canonical_* rescoring
                   joint_search.py             SearchPoint / evaluate_point
                   cluster_search{,_cli}.py    the earlier descriptor search
                   ess_sampler.py, ess_search.py   the ESA sampler (superseded)
scripts/           build_proposed_frame.py     writes assets/meta_dataset_proposed.csv
                   downstream_tests.py         binary + ranking evals, with baselines
                   controls.py                 order-stability and configuration sweeps
                   proposed_ordinals.py        identification harness
                   equation_search.sbatch      the cluster job
```

`results/cluster/` is gitignored and holds every measurement referenced below.

## Goal

`assets/ai_model_descriptors.md` documents ~61 candidate model-hyperparameter descriptors
proposed by three AIs, computed against this project's corpus and shipped as
`assets/meta_dataset_all_descriptors.csv` (476 rows: the 12 dataset features, the 6 current
`MODEL_FEATURES`, ~55 candidate descriptors, `MCC`). Find a small, defensible
replacement/extension for `data.MODEL_FEATURES` — not just "whatever ends up in one fitted
equation", but a set that would survive a search methodology described in an academic
write-up.

## What reproduction showed

The student's inputs arrived as `assets/best_lodo_vs_best_mean_r2.md` and
`assets/subset_search_semantic_10v_24h_atari.zip` (the zip is **untracked**: 1.2 MB against
the `check-added-large-files` 512 KB hook — see "Housekeeping"). Both are transcribed into
`src/ml_meta_perf/reference_subsets.py`.

**The result reproduces exactly.** `ATARI_BEST` gives 0.7058 / 0.5938 / 0.5857 against the
published 0.7058 / 0.5938 / 0.5856. The run's own `search_manifest.json` records
`original_reference` = 0.6141 / 0.4779 / 0.4276, identical to this package's baseline, so it
ran against this corpus and this pipeline. Two transcription traps, both handled:
`claude2__` there is `claude1__` here (`descriptors.canonical`), and the manifest's
`semantic_policy` places the *production* descriptors in their groups where
`ai_model_descriptors.md` §2 left them ungrouped (`descriptors.MANIFEST_SEMANTIC_GROUPS`).

### Problem 1: the configuration is part of the result

`ATARI_BEST` was found at **penalty 10, 23 terms, z-cap 5.0**, not at `DEFAULT_E3`'s
penalty 20 / 20 terms / z-cap 3.0. The same ten features:

| fitted at | in-sample | LODO | LOMO |
|---|---:|---:|---:|
| its own configuration | 0.7058 | **0.5938** | 0.5857 |
| `DEFAULT_E3` | 0.6183 | **0.3439** | 0.5211 |
| baseline (shipped six) at `DEFAULT_E3` | 0.6141 | 0.4779 | 0.4276 |

So the best feature set anyone has found here is *also* a 0.13 regression against the
shipped six, and which of the two you see is decided entirely by knobs this branch's
`evaluate_subset` and `greedy_forward_search` were holding fixed. **The earlier greedy
search would have rejected the winner on its first step.** That is the whole explanation
for "current best found here is nowhere near that" in the previous version of this file —
it was not a search-power problem.

`02-equation-form.md` and `experiment.py:81-87` both already say the penalty must be
re-swept whenever the feature set changes. This is that warning arriving as a measurement.
`src/ml_meta_perf/joint_search.py` is the fix: a `SearchPoint` is features **and** penalty
**and** length **and** z-cap **and** arity, scored as one unit.

### Problem 2: feature declaration order moves LODO by up to 0.286

Handing `run_equation` the *same* columns in a different order changes the score, because
`terms.build_library` emits terms in declaration order and the beam breaks near-ties by
position. Over seven orderings (declared, sorted, reversed, four shuffles):

| set / configuration | LODO min | LODO max | spread |
|---|---:|---:|---:|
| shipped six at `DEFAULT_E3` | 0.4388 | 0.4779 | 0.039 |
| `ATARI_BEST` at `DEFAULT_E3` | 0.3390 | 0.4435 | 0.105 |
| `ATARI_BEST` at its own configuration | 0.3456 | **0.5938** | **0.248** |

In-sample (spread 0.006) and LOMO (0.005) are stable across the same orderings. **Only
LODO is affected**, which is worth chasing separately — it points at the leave-one-dataset-
out path rather than at the library or the solve.

Three consequences:

1. **The spread grows with feature count** — 0.039 at six features, 0.105–0.248 at ten. It
   is exactly the richer sets this branch exists to evaluate that are least stable.
2. **Both published headlines are the maxima of their own orderings.** The shipped six
   reach 0.4779 in shipped order against a 0.4388 floor; `ATARI_BEST` reaches 0.5938 in
   declared order against a 0.3456 floor. Neither is wrong; neither is reproducible from a
   feature *set* alone.
3. **The comparison still holds at a fixed convention.** Sorted order throughout: 0.4743
   baseline against 0.5854 for `ATARI_BEST`, so **+0.111 LODO**, plus **+0.157 LOMO** which
   is stable under every ordering tried. The external result survives; it is the precision
   of the margin, not its direction, that this qualifies.

Recorded as `reference_subsets.ORDER_SENSITIVITY`, with
`joint_search.evaluate_over_orders` to re-measure it for any new candidate. The widest band
measured since is 0.286, on the moderate tier's winner.

**Cause found -- see Control 1.** It is not general beam nondeterminism and not a function of
feature count. It is degenerate near-ties, produced by low-support spike columns and arity-3
products together; a continuous, well-supported pool at arity 2 is order-invariant to four
decimal places.

## What exists in this branch

New this session:

- `reference_subsets.py` — the four externally-found sets with the configuration each was
  found under, their published scores, the manifest's hyperparameter grid, and
  `ORDER_SENSITIVITY`. Doubles as search seeds and as regression fixtures.
- `joint_search.py` — `SearchPoint` / `PointScore` / `evaluate_point`, scoring features and
  configuration together through the real `run_equation`. `OBJECTIVES` names four (`lodo`,
  `lomo`, `mean_r2`, `combined`); every `PointScore` carries all of them, so a finished log
  re-ranks without re-fitting. Plus `sweep_configuration`, `rescore_at`,
  `evaluate_over_orders`.
- `ess_sampler.py` — the ESA/attraction-field sampler (see below).
- `ess_search.py` — `run_search`, the batch-sequential driver, warm-started from
  `REFERENCE_SUBSETS`.
- `descriptors.py` gained `canonical`/`PREFIX_ALIASES` and `MANIFEST_SEMANTIC_GROUPS`.
- 50 new tests across `tests/test_{reference_subsets,joint_search,ess_sampler,ess_search}.py`.
  Full suite: **457 tests, ~3 min** (was 407). ruff / basedpyright / vulture all clean.

Pre-existing, unchanged: `descriptors.py`'s registry machinery, `feature_search.py`'s four
backends, `feature_search_cli.py`. **`feature_search.evaluate_subset` and
`greedy_forward_search` are now known to score at a fixed configuration and therefore to
mis-rank** — they are kept because their stability-selection and greedy machinery is still
useful, but treat any ranking they produce as conditional on `DEFAULT_E3`.

## The ESS sampler

`/home/mantunes/git/Optuna_ESA_Sampler/` is a **skeleton** — `utils.point_to_sample` is
`pass`, `ESSSampler.sample_relative` returns `{}` and calls it with no arguments. Nothing
was reusable directly, but the idea was, and the installed `EmptySpaceSearch` **0.7.1**
(the repo pins 0.1.4) has exactly what was described: `esa(..., attractiveness=...,
attraction_weight=..., attraction_metric=...)`.

`ess_sampler.py` implements it as:

- **Encoding.** A random-key relaxation: one continuous coordinate per candidate, the *k*
  largest select the subset, then four hyperparameter axes (penalty in log10, terms, z-cap,
  arity). Decoding repairs the subset through `Registry.resolve_conflicts`, so no proposal
  can break semantic exclusion however the optimizer moves.
- **Attraction field.** Observed objective values are passed as `attractiveness`, refit on
  every round, so each batch fills empty space *biased toward* the regions that scored —
  the "repeated ESS call into high-productive areas" this was for.
- **Not an `optuna.samplers.BaseSampler`.** Optuna's protocol is per-trial and
  per-parameter; ESA's whole contribution is placing a **batch** jointly so its members
  repel each other. A point placed without seeing the rest of its batch is just a random
  draw, so the loop is driven directly instead.
- **Force laws.** `ess` *raises* if attraction wins at contact and only *warns* if it never
  wins anywhere — and the second failure silently disables the exploitation half. Of the
  pairs in `METRIC_REGISTRY` satisfying both conditions, `gaussian` repulsion + `cauchy`
  attraction has the widest margin (F_rep(0)/F_att(0) = 2.5). Asserted in
  `test_ess_sampler.test_the_force_laws_form_a_genuine_attraction_well` — do not change the
  defaults without re-running it.

**Budget reality.** The external run did 6,608 evaluations on 56 workers in 12 hours. One
`evaluate_point` is 10–40s here, so a single-machine overnight run is ~2,000. That gap is
why placement quality matters and why the search is warm-started rather than cold.

## Where the earlier single-machine run stands (superseded)

`results/ess_search_combined.csv` (gitignored) is the log of a 12-round x 12-batch run,
objective `combined`, 148 evaluations, ~37 minutes. It did not beat the `ATARI_BEST` seed
(0.5039 against 0.5855 combined). Superseded by the cluster run below, but the mechanics it
established still hold: 0 duplicates in 148 proposals, so the random-key decode is not
collapsing, and points spread across the whole penalty and length range rather than clumping.

## What the cluster run showed

Three Slurm jobs, one per admissibility tier, 11 hours each on `cpuPartition`
(`scripts/search.sbatch`; jobs 15327-15329 on atari / sega / case-hpc). Logs, evaluation
tables, Pareto fronts and `status_*.json` are synced to `results/cluster/` (gitignored).

| tier | pool | cores | evals | rounds | dups | front | best LODO | best LOMO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strict | 10 | 14 | 8,470 | 605 | 0 | 29 | 0.5252 | 0.5540 |
| moderate | 24 | 22 | 9,615 | 437 | 0 | 36 | 0.5631 | 0.5345 |
| permissive | 55 | 62 | 13,520 | 218 | 0 | **1** | 0.5938 | 0.5857 |

Three things are visible before any control is applied.

**The permissive front has one point on it, and that point is the `ATARI_BEST` seed.** All
three external reference sets are in the permissive top three; the best *genuinely searched*
permissive point reaches **LODO 0.5145 / LOMO 0.5120**, which both gated tiers beat. 13,520
evaluations over the 55-column pool could not rediscover the seed or anything near it. That
is the signature of an isolated optimum, not a basin -- consistent with Problem 2's finding
that the seed's 0.5938 sits at the top of a 0.231-wide ordering band.

**The strict pool had no warm start at all** (`seeds inside this tier: 0 of 4` -- even the
shipped six fail it, on `Active Regularization Mechanisms` and `Robust to Outliers`) and
still cleared the published baseline from round 1. With 10 candidates choosing 8, it is
closer to an exhaustive feature search wrapped in a configuration search, and it converged
by round ~200 of 605.

**Both gated winners chose `max_arity = 2` and 17 terms** -- a *simpler* grammar and a
shorter equation than the published `max_arity = 3`, 20 terms. That is the opposite of the
usual drift this project guards against: the gain arrives with an explainability gain.

### The two gated winners

| | strict (8 of 10) | moderate (8 of 24) |
|---|---|---|
| features | `Model Capability`, `Processing Units Number`, `chatgpt__component_count`, `chatgpt__representation_width`, `chatgpt__sequential_depth`, `chatgpt__structural_size_proxy`, `chatgpt__train_work_proxy`, `perplexity__reg_l2_log` | `Model Capability`, `Processing Units Number`, `Robust to Outliers`, `chatgpt__representation_width`, `chatgpt__structural_size_proxy`, `perplexity__iters_log`, `perplexity__lr_log`, `perplexity__reg_l2_log` |
| configuration | lambda 25.6, 17 terms, z 4.25, arity 2 | lambda 14.8, 17 terms, z 4.36, arity 2 |
| min / median support | **55.7% / 90.1%** | 14.9% / 90.1% |
| binary members | none | none |
| zero-sentinel members | none | `perplexity__iters_log` (20.4%), `perplexity__lr_log` (14.9%) |

Contrast `ATARI_BEST`: min support **0.6%**, median 20.4%, one binary
(`claude1__no_tunable_hp`), seven zero-sentinel columns. The strict winner is the first set
found anywhere in this work that has no member the "Point 2 answered" audit rejects.

### Stability selection agrees with the strict winner

Over the top 5% of the strict log by LODO (423 of 8,470 points), seven of the eight winning
columns appear in **70-99.5%** of them. It is the consensus set, not one lucky draw. The
moderate log's top 5% is diffuse by comparison (31-88%), which matches its ordering band
below. One near-member worth recording: `Training Operations`, a *shipped* descriptor,
appears in 85.8% of strict's top points without making the winner.

## The controls, and what survived them

Two controls, both of which `CLAUDE.md` requires and neither of which the external run ran.
Scripts are the three subcommands of `scripts/controls.py`; outputs are `results/cluster/{confirmation,config_sweep,orders_gridbest}.csv`.

### Control 1: order stability (7 orderings, sorted first)

| set @ configuration | LODO median | LODO band | LOMO median | LOMO band |
|---|---:|---:|---:|---:|
| shipped six @ `DEFAULT_E3` | 0.4702 | 0.021 | 0.4281 | 0.003 |
| **strict winner @ own** | **0.4995** | **0.026** | **0.5525** | **0.002** |
| moderate winner @ own | 0.5611 | **0.286** | 0.5367 | 0.029 |
| `ATARI_BEST` @ own | 0.5563 | **0.231** | 0.5895 | 0.011 |
| strict winner @ grid-best (25 / 20 / 4.25 / 2) | 0.5273 | **0.001** | 0.5511 | 0.001 |
| strict winner @ grid-comb (25 / 17 / 4.25 / 2) | 0.5227 | **0.000** | 0.5555 | 0.000 |
| shipped six @ grid-best (10 / 20 / 4.25 / 3) | 0.5013 | **0.177** | 0.5279 | 0.027 |

**The moderate winner is disqualified here**: a 0.286 LODO band is wider than
`ATARI_BEST`'s, so its headline 0.5631 is a draw from a distribution whose floor is 0.278.
It is not a result.

**The strict winner is order-invariant to four decimal places** at its grid configuration.
That answers "Next" item 2 from the previous version of this file, and the answer is not
what was expected: the LODO order sensitivity is not general beam nondeterminism, it is a
symptom of degenerate near-ties, and it is produced by the *low-support spike columns and
arity-3 products together*. Remove both -- a continuous, well-supported pool at arity 2 --
and the beam has no near-ties left to break by position. The shipped six at arity 3 still
show a 0.177 band, so it is the grammar and the columns, not the feature count.

### Control 2: matched configuration sweep (280 cells, both sets, sorted order)

lambda in {5, 10, 15, 20, 25, 30, 40} x terms in {13, 17, 20, 23, 27} x z in {3.0, 4.0, 4.25, 5.0} x arity in {2, 3}.
The shipped six get the *same* 280 configurations the strict set does, which is the
best-vs-best comparison `CLAUDE.md` demands and which nothing on this branch had yet run.

| | best LODO in grid | best LOMO in grid | median LOMO over grid | cells LOMO > 0.50 |
|---|---:|---:|---:|---:|
| shipped six | **0.5153** (10 / 20 / 4.25 / 3) | 0.5430 | 0.4833 | 80 / 280 |
| strict winner | 0.5283 (25 / 20 / 4.25 / 2) | 0.5637 | **0.5365** | **276 / 280** |

Paired cell by cell (strict minus baseline, same 280 configurations):

| metric | median delta | mean delta | cells won |
|---|---:|---:|---:|
| LODO | **+0.0001** | +0.0108 | 140 / 280 |
| LOMO | **+0.0472** | +0.0528 | **280 / 280** |

**LODO: the descriptor gain is zero.** Median paired delta +0.0001 and 140 of 280 cells is
exactly chance. The shipped six reach 0.5153 LODO on their own once the configuration is
swept, against the published 0.4779. The whole apparent leave-one-dataset-out improvement --
the strict set's, the moderate set's and `ATARI_BEST`'s +0.111 alike -- is the configuration
moving, not the features. `02-equation-form.md` and `experiment.py:81-87` warned that the
penalty must be re-swept whenever the feature set changes; this is the second time on this
branch that warning has arrived as a measurement, and this time it eliminates the headline.

**LOMO: the descriptor gain is real, and it is uniform.** The strict set wins in **every one
of the 280 cells**, minimum delta +0.0064, median +0.0472. There is no configuration in the
grid at which the shipped six are the better model-side description. This is the first
model-descriptor result in this study that does not depend on which knobs it is read at.

### What that is worth

`CLAUDE.md` records **+0.106** as the upper bound on what perfect model descriptors could
buy, measured by substituting model identity itself. The strict set's controlled **+0.047**
LOMO is **~44% of that remaining headroom**, taken by eight columns of which six are new,
all continuous or many-levelled, none binary, none zero-sentinel, at arity 2 and 17 terms.

The headroom arithmetic is not double-counting `Model Capability`. That column is in *both*
sides of the paired sweep -- it is a shipped `MODEL_FEATURES` member and a strict-winner
member -- so the +0.047 sits on top of the eighth of the headroom `CLAUDE.md` already
credits to it, and the six new columns are what buys it. What the strict set drops from the
shipped six is `Training Operations`, `Prediction Operations`, `Active Regularization
Mechanisms` and `Robust to Outliers`; what it adds is five `chatgpt__` structural
descriptors and `perplexity__reg_l2_log`.

The honest summary of this branch, then:

- **A defensible replacement for `MODEL_FEATURES` exists**, and its claim is a LOMO claim.
  The next section shows it is not the strict winner but a two-column subset of what is
  already shipped -- reading the fitted equation, rather than its score, is what found that.
- **The LODO story is a re-tuning story, not a descriptor story**, and it applies to the
  published equation regardless of what happens to the descriptors: `DEFAULT_E3` is not at
  its own LODO optimum. That is a separate finding and probably a separate paragraph.
- **`ATARI_BEST` is not the result to chase.** Its margin over the gated set is bought with a
  0.6%-support column and a binary indicator, sits in a 0.231-wide ordering band, and its
  LODO half evaporates under a matched sweep. Keep it as a measured upper bound, as
  `CLAUDE.md` already keeps E4.

## The result is subtractive: two columns do all of it

Reading the fitted equation rather than its score changed the conclusion. Written after the
controls above, and it supersedes their recommendation.

### Which features the equation actually uses

The strict-8 set at its chosen cell (lambda 25 / 17 terms / z 4.25 / arity 2) scores
**in-sample 0.6274, LODO 0.5227, LOMO 0.5557**. Its 17 terms are 6 dataset-only, 1
model-only, 10 mixed -- and they draw on **two** of its eight features:

| feature | terms |
|---|---:|
| `Processing Units Number` | 8 |
| `Model Capability` | 2 |
| `perplexity__reg_l2_log` | 1 (weight -0.0048, on a squared term) |
| `chatgpt__component_count`, `representation_width`, `sequential_depth`, `structural_size_proxy`, `train_work_proxy` | **0** |

Five of the six AI-proposed columns that survived the strict gate contribute **nothing**. At
20 terms one of them enters, `chatgpt__sequential_depth`, at weight +0.0016. So the +0.047
LOMO of Control 2 was never bought by the new descriptors: it was bought by *removing* four
of the shipped six from the pool, which frees term slots for `Processing Units Number` and
`Model Capability` to interact with the dataset side.

### Confirmed by direct ablation

Same paired-grid method as Control 2, `results/cluster/{minimal_sets,ablation}.csv`.
Paired against `shipped-6` over 112 cells:

| model features | n | LOMO delta (median) | cells won | LODO delta | in-sample delta |
|---|---:|---:|---:|---:|---:|
| **`MC` + `PUN`** | **2** | **+0.0513** | **111/112** | -0.0137 | -0.0212 |
| `MC` + `PUN` + `reg_l2_log` | 3 | +0.0486 | 112/112 | -0.0133 | -0.0185 |
| strict-8 | 8 | +0.0469 | 112/112 | -0.0128 | -0.0178 |
| `MC` + `PUN` + `RtO` | 3 | +0.0403 | 108/112 | -0.0165 | -0.0147 |
| shipped-6 minus `ARM` | 5 | +0.0127 | 75/112 | +0.0000 | +0.0000 |

**Two features beat eight.** And leave-one-out / add-one over 60 cells says exactly which
of the shipped six is responsible (positive = removing it helps):

| removed from shipped-6 | LOMO delta | | added to `MC`+`PUN` | LOMO delta | LODO delta |
|---|---:|---|---|---:|---:|
| `Prediction Operations` | **+0.0224** | | `Prediction Operations` | **-0.0248** (0/60) | -0.0017 |
| `Robust to Outliers` | +0.0163 | | `Robust to Outliers` | -0.0070 | -0.0100 |
| `Active Regularization Mechanisms` | +0.0154 | | `Active Regularization Mechanisms` | -0.0042 | +0.0000 |
| `Training Operations` | +0.0070 | | `Training Operations` | +0.0008 | **+0.0269** |
| `Model Capability` | -0.0018 | | | | |
| `Processing Units Number` | -0.0131 | | | | |

Only `Model Capability` and `Processing Units Number` are load-bearing; removing either also
costs LODO heavily (-0.077 and -0.065). Every other shipped descriptor is neutral or harmful
under leave-one-model-out.

**The mechanism is already in `CLAUDE.md`:** three of the five original model features are
functions of the dataset too. That is why `Training Operations` and `Prediction Operations`
behave the way the table shows -- they carry dataset information, so they help LODO
(`MC+PUN+TrOp` is +0.027 LODO) and hurt LOMO, because none of that information transfers to
a model that was never trained. A descriptor that is partly a dataset feature is a liability
in exactly the protocol that model descriptors exist to serve.

### The recommendation, at a common cell

| model features | n | in-sample | LODO | LOMO |
|---|---:|---:|---:|---:|
| shipped-6 @ `DEFAULT_E3` (published) | 6 | 0.6134 | 0.4743 | 0.4283 |
| shipped-6 @ its own grid best (10 / 20 / 4.25 / 3) | 6 | 0.6719 | 0.5153 | 0.5309 |
| strict-8 @ 25 / 17 / 4.25 / 2 | 8 | 0.6274 | 0.5227 | 0.5557 |
| **`MC` + `PUN` @ 20 / 17 / 4.25 / 2** | **2** | 0.6264 | **0.5275** | **0.5558** |
| `MC` + `PUN` + `TrOp` @ 25 / 20 / 4.25 / 3 | 3 | 0.6316 | 0.5239 | 0.5535 |
| `ATARI_BEST` @ 10 / 23 / 5.0 / 3 | 10 | 0.7058 | 0.5854 | 0.5857 |

**`MC` + `PUN` weakly dominates strict-8** -- better LODO, equal LOMO, equal in-sample, with
a quarter of the columns. Against the published equation it is **+0.053 LODO, +0.128 LOMO,
+0.013 in-sample**, and Control 2 attributes the LODO half to the configuration rather than
to the features.

Beware one cell: `MC`+`PUN` at 20 / 13 / 4.25 / **arity 3** posts the highest LOMO in the
grid (0.5645) at **LODO -0.0113**. Arity 2 is not a stylistic preference here.

### The AI descriptors: a clean negative result

Of ~61 candidates from three AIs, **none survives**. That is worth reporting as a finding
rather than buried as a failed branch, and the audit says why in terms that do not depend on
this corpus: the proposals are hyperparameter *values*, and a hyperparameter that a learner
does not have has no value, so every one of them collapses applicability into magnitude at
zero. `assets/ai_model_descriptors.md` asked for descriptors and got a configuration dump.

### Validity of every model-side column in play

The question the equation grammar actually asks of a column: is it continuous, or ordered
with even increments, and is it positive (three of the four transforms take a log)?

| column | levels | range | spacing | support | valid? |
|---|---:|---|---|---:|---|
| `Model Capability` | 10 | 1 .. 10 | **even** | 100% | **yes** -- ordered, positive, no zero |
| `Processing Units Number` | 169 | 1.79 .. 20.4 | continuous | 100% | **yes** |
| `Training Operations` | 216 | 6.69 .. 27.7 | continuous | 100% | yes, but partly a dataset feature |
| `Prediction Operations` | 147 | 0.69 .. 18.9 | continuous | 100% | yes, but partly a dataset feature |
| `Active Regularization Mechanisms` | 4 | 0 .. 3 | even | 50% | a real count; zero means "none", not "n/a" |
| `Robust to Outliers` | 3 | 0 .. 2 | even | 47.9% | a real grade; same reading |
| `perplexity__reg_l2_log` | 20 | **-7 .. +7** | uneven | 55.7% | **no** -- negative, and 0 is both "no L2" and log(1) |
| `chatgpt__component_count` | 5 | 0, 1, 50, 100, 300 | uneven | 92.9% | no -- sentinel zero, not incremental |
| `chatgpt__sequential_depth` | 9 | 0 .. 40 | uneven | 90.1% | no -- sentinel zero |
| `chatgpt__representation_width` | 21 | 0 .. **1.1e12** | uneven | 90.1% | no -- sentinel zero, pathological range |
| `chatgpt__structural_size_proxy` | 33 | 0 .. **2.2e12** | uneven | 90.1% | no -- same, and a *proxy* by name |
| `chatgpt__train_work_proxy` | 38 | 0 .. **4.4e13** | uneven | 90.1% | no -- same |
| every remaining `ATARI_BEST` member | 2-9 | 0 .. varies | uneven | 0.6-27.9% | no -- sentinel zero; one is binary |

`perplexity__reg_l2_log` is the worst case in the table and it is the one that reached the
strict pool: zero is a *legitimate* value of a log, so its applicability sentinel is not even
distinguishable from a real reading of 1.0. That it also goes negative rules it out of every
log transform the grammar has. It should never have cleared the gate --
`admissibility._strict` tests support and level count but not sign, and not whether zero is
reachable as a genuine value.

## Four defects the score-based search never saw

Found by asking what the columns *are* rather than what they score. All four are
pre-existing; none was introduced by the reduction, and two of them affect the published
equation today.

### Defect 1: no feature set identifies the 25 models

The set has to be injective on models -- two learners with the same feature vector on the
same dataset must get the same prediction, so any MCC difference between them is
irreducible. Counting models caught in a colliding cell, out of 476:

| model features | n | models in colliding cells | irreducible groups |
|---|---:|---:|---|
| `MC` + `PUN` | 2 | 174 | `DT`/`ExtraTree`; the six linear learners; `LightGBM_RF`/`LightGBM_ExtraTrees` |
| shipped-6 | 6 | 121 | same, minus a partial split of the linear group |
| best greedy over every admissible column | 10 | **16** | `DT`/`ExtraTree`, `LightGBM_RF`/`LightGBM_ExtraTrees` |

**No set reaches zero.** `DT` vs `ExtraTree` (best split versus random split) and
`LightGBM_RF` vs `LightGBM_ExtraTrees` are identical in every descriptor the corpus has or
that anyone proposed. That is a documented limit of the meta-dataset, not a search failure,
and it belongs in the chapter on what the corpus lacks.

**This is the ceiling argument, and it settles the in-sample question.** The finest partition
any equation over a feature set can distinguish is one cell per (dataset, model-feature
vector), so the set's in-sample R2 is capped:

| model features | cells | in-sample ceiling | achieved | % of ceiling |
|---|---:|---:|---:|---:|
| `MC` + `PUN` | 349 | 0.9092 | 0.6264 | 68.9% |
| `MC` + `PUN` + `RtO` | 369 | 0.9377 | 0.6385 | 68.1% |
| `MC` + `PUN` + `TrOp` | 369 | 0.9521 | 0.6316 | 66.3% |
| shipped-6 | 410 | 0.9882 | 0.6719 | 68.0% |
| model identity (oracle) | 476 | 1.0000 | -- | -- |

Every set reaches ~68% of its own ceiling, so the equation grammar -- not identifiability --
is what currently binds. But the two-feature set's ceiling is **0.909 against shipped-6's
0.988**, and that gap is permanent. The in-sample loss of the reduction is real and it is
structural, which is the reason to reject `MC`+`PUN` as the published set even though it wins
on transfer.

**And it is the principled, fit-free argument for a larger set that R2 could not supply.**
The AI candidates *do* split the collision groups the shipped six cannot -- 13 of them
separate `DT` from `ExtraTree`, 11 separate the linear learners. Screening ~61 candidates on
whether they resolve a named ambiguity is a defensible selection methodology in a way that
screening them on LOMO is not, and it is the search this branch should have run.

### Defect 2 (corrected): the *AI* columns pre-transform without a reason; the shipped ones have one

The first reading of this was wrong and is recorded because the distinction is the whole
point. `exp()` of `Processing Units Number` returns integers on 100% of its 169 distinct
values, so the column is a natural log of a count -- but **that log is by definition, and the
definition is part of the study's argument**: capacity has bounded returns, adding nodes
linearly does not buy accuracy linearly, and the log is what encodes that. `Training
Operations` and `Prediction Operations` are logs of cost for the same stated reason. These
belong in the discussion, not in a defect list.

What *is* a defect is the same operation performed without a reason. The AI-proposed columns
arrive pre-transformed -- `reg_l2_log` is `log10(x + eps) + 5`, `lr_log` and `iters_log` the
same shape -- and none of the three source documents states why, or what the `+5` is, or why
a log is the right scale for that quantity. **The shipped transforms are modelling choices
with a published rationale; the proposed ones are formatting.** Worse, they are not found in
the common literature at all: a search for algorithm-side meta-features returns work on
*dataset* characterisation almost exclusively, which is itself the reason this corpus is
short of model descriptors.

Two consequences worth keeping separate:

- **For the paper.** A feature handed to the grammar should be raw *unless* its transform is
  itself a claim. `PUN` passes that test; `reg_l2_log` fails it.
- **For the term strings.** `log(Processing Units Number)` still reads as a first log when it
  is a second one. Measured, un-logging is worse (-0.046 in-sample, 5 of 60 cells), so the
  fix is naming -- call the column what it holds -- not data.

### Defect 3: `perplexity__reg_l2_log` is not a feature

It reached the strict pool and it is the worst column in the study.
`assets/ai_model_descriptors.md` defines it as: `-log10(C)+5` for `C`; `log10(x+eps)+5` for
`alpha`, `reg_lambda`, `weight_decay`, `lambda_sparse`; `x+1` for `shrinkage`; **averaged**
where several apply. So it is

- **pre-transformed** (defect 2 again, plus an arbitrary +5 shift),
- **sign-flipped between mechanisms** -- `C` is inverse-regularisation, so its contribution
  runs the opposite way to `alpha`'s. This is precisely what `descriptors.INADMISSIBLE`
  already rejects `perplexity__class_weight_signal` for by name,
- **negative** on part of its range (-7 .. +7), which locks out every log transform,
- **an average of incommensurable quantities**, so no single sentence describes it,
- and **zero is a legitimate value of a log**, so its applicability sentinel cannot be told
  apart from a real reading of 1.0.

It must be removed from every pool, and `admissibility._strict` needs the sign and
reachable-zero rules that would have caught it.

### Defect 4: `Active Regularization Mechanisms` is not model-constant

It varies within **6 of 25 models** (`FT-Transformer` takes 3 values, `XGBoost`, `LR`, `LDA`,
`SGD`, `Perceptron` two each). A count of a learner's regularisation mechanisms should not
depend on which dataset it ran on. `Processing Units Number`, `Training Operations` and
`Prediction Operations` vary too, but for them that is expected and documented -- capacity
and cost scale with the data. For `ARM` it is a defect, and it explains why the column is
worthless in every measurement here (0 terms in the published equation; removing it gains
+0.015 LOMO). Only `Robust to Outliers` and `Model Capability` are true per-model constants.

## The smallest set that identifies the models

Identity is a fit-free property, so this is an exhaustive search rather than a heuristic one:
every column encoded to integer codes, then a count of how many rows sit in a
(dataset, feature-vector) cell shared with a different model. Searched over all 73 numeric
columns the corpus and the three AI documents supply, plus the four proposed below.

### With what exists today

| target | best set | size | ambiguous rows |
|---|---|---:|---:|
| 10 families | `Model Capability` | 1 | **0** -- but circular, see below |
| 10 families, measured only | `PUN` + `TrOp` + `RtO` | 3 | 62 |
| 10 families, measured only | shipped-5 (no `MC`) | 5 | 50 (`linear` vs `naive bayes` only) |
| 25 models | greedy over every admissible column | 9 | 14 (`DT`/`ExtraTree`, `LightGBM_RF`/`LightGBM_ExtraTrees`) |

**`Model Capability` identifies the families perfectly because it *is* the families.**
`data.MODEL_CAPABILITY` maps 10 families to 10 distinct rungs, so the column is a bijection
with `MODEL_FAMILY` -- a ten-level categorical wearing an ordinal's clothes, which is the
standing worry `CLAUDE.md` already records about it and which this makes exact. Any claim
that "capability" explains something is, at this corpus's resolution, a claim that *family*
explains it.

Two findings from the measured-only rows. At family level the shipped features fail on
exactly **one** pair, `linear` versus `naive bayes`. At model level nothing available
separates `DT` from `ExtraTree` or `LightGBM_RF` from `LightGBM_ExtraTrees` -- the tree
variants, as expected.

### Four ordinals that close it

Each is asserted taxonomy in the same sense `Model Capability` is, and carries the same ch. 8
caveat. What distinguishes them from every AI-proposed column: each is **defined for all 25
learners**, so there is no applicability sentinel, no zero, and no support floor to clear.
Each is an ordered degree with a stated direction, which is what the grammar needs.

| ordinal | rungs | grounding |
|---|---|---|
| **Solution Stochasticity** | 1 deterministic; 2 stochastic optimisation (init/shuffling); 3 randomised over examples (bootstrap); 4 randomised over features (subspace/colsample); 5 randomised over split or parameter values | Breiman (1996) bagging; Ho (1998) random subspace; Breiman (2001) random forests; **Geurts, Ernst & Wehenkel (2006) extremely randomized trees** |
| **Loss Margin Behaviour** | 1 squared; 2 logistic / cross-entropy; 3 hinge; 4 perceptron; 5 passive-aggressive | the standard robustness ordering -- how hard the loss penalises points far from the boundary |
| **Input Distribution Modelling** | 1 discriminative; 2 instance-based; 3 generative, conditional independence; 4 generative, shared covariance; 5 generative, per-class covariance | Ng & Jordan (2001), discriminative versus generative classifiers; the LDA/QDA covariance hierarchy |
| **Fitting Regime** | 1 closed form; 2 batch iterative; 3 mini-batch stochastic; 4 per-sample online; 5 amortised / in-context | -- |

Measured on the corpus:

| feature set | size | families | models |
|---|---:|---:|---:|
| `MC` + `PUN` + Stochasticity | 3 | **0** | 120 |
| `MC` + `PUN` + Stochasticity + **Loss** | **4** | **0** | **0** |
| `MC` + `PUN` + Stoch + Loss + Regime | 5 | 0 | 0 |
| `PUN` + Stoch + InputDist + Loss + Regime (**no `MC`**) | 5 | 2 | 2 (`DNN`/`TabNet`) |
| the four ordinals alone | 4 | 85 | 193 |

**Exhaustively minimal.** No pair and no triple among all 77 columns -- everything the corpus
has, everything three AIs proposed, and these four -- identifies the 25 models. Exactly one
4-subset of defensible columns does:

> **`Model Capability` + `Processing Units Number` + `Solution Stochasticity` + `Loss Margin
> Behaviour`** -- 0 of 476 rows ambiguous, at both family and model resolution.

`Solution Stochasticity` is what resolves both tree pairs, which is the point it was designed
for: `DT` versus `ExtraTree` and `LightGBM_RF` versus `LightGBM_ExtraTrees` are the same
distinction twice, best-split versus random-split, and one column states it. `Loss Margin
Behaviour` is what splits the six linear learners.

Note the non-circular row: **five features with no `Model Capability` at all** reach 2
ambiguous rows. That matters because it offers a route out of the bijection problem -- a
model-side description built from mechanisms rather than from an asserted capability rank,
which ch. 8 could then compare against `MC` instead of resting on it.

### Caveats before any of this is used

- **The rung assignments are drafts and need a careful pass.** Two known errors in the
  version measured above: `AdaBoost` minimises *exponential* loss, not hinge, and tree
  learners minimise an impurity criterion rather than squared error. They do not change the
  identification result but they would be wrong in a chapter.
- **Identification is necessary, not sufficient.** A set that separates every model still has
  to fit, transfer and read in plain language. These four have not been fitted yet.
- **`Loss Margin Behaviour`'s ordering is a claim**, and a weaker one than the stochasticity
  ladder: ordering losses by margin behaviour is standard, but placing perceptron above hinge
  and passive-aggressive above both is a choice. Check it against per-rung mean MCC the way
  ch. 8 checks `MC` -- and expect it to do no better, since `MC` itself matches at only 6 of
  9 steps.
- **This does not lift the standing decision against re-running experiments.** All four are
  asserted from published descriptions of the learners, which is exactly what `CLAUDE.md`
  says is still available.

## Re-grading the ordinals (answering "can `RtO` and `ARM` be made like `MC`?")

`Model Capability` is 1..10 with no zero, which is why the grammar can take its log. `RtO`
(0..2) and `ARM` (0..3) start at zero, so they can only ever enter as `f` and `f^2` -- the
same narrowness that rules out binary indicators. Re-basing to 1..3 and 1..4 costs nothing
and was measured over the same 60-cell grid:

| change | in-sample | LOMO | LODO |
|---|---:|---:|---:|
| `RtO` 0..2 -> **1..3** | **+0.0065** (52/60) | -0.0060 | **+0.0253** (43/60) |
| `ARM` 0..3 -> 1..4 | +0.0063 (50/60) | -0.0041 | **-0.0435** |
| both | +0.0086 (55/60) | -0.0069 | +0.0132 |

**Yes for `RtO`, no for `ARM`.** `MC`+`PUN`+`RtO(1..3)` reaches **in-sample 0.6411, LODO
0.5435, LOMO 0.5458** -- the best LODO of any gated set measured anywhere here, with
in-sample up as well. `ARM` gets worse, consistently with Defect 4.

Re-basing is a relabelling, so it changes no collision and no ceiling; it only widens the
transforms available. A **finer** grading is the open question and is a different proposal:
`RtO`'s three rungs are coarse, and a capability-style ladder over loss robustness (unbounded
squared loss -> bounded/hinge -> rank or median based -> split based) would be asserted
taxonomy in the same sense `MC` is, with the same caveat ch. 8 already applies to `MC`. It
would also help Defect 1, since loss function is exactly what separates the six linear
learners.

## Applied to `src/` (2026-09-05) -- the pipeline is now stable

Four changes, gate green throughout: **459 tests, ruff, basedpyright, vulture all pass**, and
`python -m ml_meta_perf` still runs the whole study in 24s.

### 1. `terms.build_library` is order-independent -- this was the real cause

The ordering band was **not** the beam's tie-breaking. Making the tie-break deterministic
(below) changed the spread by nothing at all: 0.177 before, 0.177 after. The cause was the
library itself.

`build_library` iterated features in the caller's declaration order, and `pairwise_terms`
names a product after whichever operand it meets first. So `A * B` and `B * A` -- the same
column, multiplication being commutative -- entered under two different names, and `Library`
then de-duplicated by correlation *against whatever it had already kept*, so which of a
near-collinear pair survived also depended on order. Across five orderings of one six-feature
set the library came out at **270, 270, 272, 270 and 271 terms**, with **24 of 257 pool slots
differing**.

Sorting `dataset_features` and `model_features` at the top of `build_library` makes the
library a function of the feature *sets*. Measured over seven orderings each:

| case | LODO spread before | after | LOMO spread after | in-sample spread after |
|---|---:|---:|---:|---:|
| shipped-6 @ `DEFAULT_E3` | 0.0207 | **0.0000** | 0.0000 | 0.0000 |
| shipped-6 @ grid-best | 0.1770 | **0.0000** | 0.0000 | 0.0000 |
| proposed-6 @ 25/23/4.25/2 | 0.2488 | **0.0000** | 0.0000 | 0.0000 |
| `MC`+`PUN` @ 25/23/4.25/2 | 0.0772 | **0.0000** | 0.0000 | 0.0000 |

**`ORDER_SENSITIVITY` is now a historical record, not a live caveat.** Every equation is a
function of its feature set.

**And it re-scores the external results.** `ReferenceSubset` gained `canonical_*` fields and
an `order_premium` property; the published scores are kept as the historical record and the
tests now pin the canonical ones:

| reference | published LODO | canonical LODO | order premium |
|---|---:|---:|---:|
| baseline (shipped six) | 0.4779 | 0.4743 | 0.004 |
| **`atari_best`** | 0.5938 | **0.5854** | **0.008** |
| `previous_best_lodo` | 0.5816 | 0.4875 | **0.094** |
| `previous_best_mean` | 0.5672 | 0.4535 | **0.114** |

`atari_best` survives -- its margin was real. **The two earlier external solutions were
selected on a margin that was mostly their declaration order**, and that is pinned as a test
rather than tidied away.

### 2. `validate` clips each fold to its training range

`model.predict` clips to `[MCC_LOWER, MCC_UPPER] = [-1, 1]`, correct for the equation as
published and far too loose inside a fold: this corpus has one negative row (-0.29), so a
floor at -1 let `ASNM-CDX-2009` be predicted at -1.0 and reach 89% of total squared error on
its own. `_clip_to_training` bounds a fold by the target range it was trained on -- no
held-out information, exactly what the equation was shown. It rescues affected cells
(`MC`+`PUN` -0.383 to +0.260) and leaves unaffected ones identical to four decimals.

### 3. `CrossValidation.dispersion()` -- and the published equation looks different through it

Median and worst per-fold R2, from the `per_fold` scores that were already computed and
discarded, now on every `EquationReport.cross_validated` entry. For the **published E3**:

| protocol | pooled R2 | median fold | worst fold |
|---|---:|---:|---:|
| leave-one-dataset-out | 0.4743 | **0.2435** | **-9.57** |
| leave-one-model-out | 0.4283 | **0.2752** | -1.48 |

Pooled R2 scores against the *global* mean, and most variance here is between datasets --
which the dataset features get nearly free. The equation explains about **0.24-0.28 of the
variance within an average fold**, and one dataset fold is still catastrophic. Both numbers
belong in ch. 6.

### 4. `fit._descending` -- a real latent bug that fixed nothing

`np.argsort(scores)[::-1]` is an unstable sort, then reversed -- and reversing even a *stable*
ascending sort inverts tie order. Replaced with `np.argsort(-scores, kind="stable")`. Kept
because it is correct and because ties should not resolve by library position, but **recorded
honestly: it changed no measured number.** It was the plausible explanation and it was wrong.

### What moved in the published study

| | before | after |
|---|---|---|
| E3 in-sample | 0.6141 | 0.6134 |
| E3 LODO | 0.4779 | **0.4743** |
| E3 LOMO | 0.4276 | 0.4283 |
| E1 | 0.3485 / 7 terms | unchanged, same 7 terms |
| E2 / E3 equations | -- | **7 of 20 E3 terms and 7 of 12 E2 terms changed** |

The headline scores move by under 0.004, but a third of the terms changed -- so the library
held many near-equivalent terms and the old equation was one arbitrary pick among them.
**`CLAUDE.md`'s table and `assets/docs/` need the new numbers.**

### Still not done

`candidates` is still unreachable from `Configuration`; `guided_screen` still cuts silently
past `pool_size`; `build_library` still applies the z-cap outside the fold loop; and the two
LOO protocols are still only in `scripts/`, not `validate`. The **k-sweep jaggedness is
unchanged** (max adjacent jump 0.297) -- that is genuinely different equations at different
lengths, and it is what stability selection in Stage 5 is for, not a bug to patch.

## Cleanup and documentation plan (written 2026-09-05)

Six stages, in dependency order. **A and B are safe; C deletes ~80 tests and eight modules and
should be done on its own commit; D-F change published numbers.** Nothing below has been
executed.

### Stage A — one corpus, six model features

Today there are three CSVs. `src/ml_meta_perf/meta_dataset.csv` is the shipped corpus (12
dataset features + the 6 *old* model features); `assets/meta_dataset_all_descriptors.csv` adds
~55 AI candidates; `assets/meta_dataset_proposed.csv` adds the four ordinals on top of that.
The end state is **one file**: identifiers + 12 dataset features + 6 model features + `MCC`.

| column group | keep | note |
|---|---|---|
| `Dataset`, `Model`, `MCC` | yes | |
| the 12 dataset features | yes | unscreened, see Open work — but not this plan's job |
| `Model Capability`, `Processing Units Number` | yes | both load-bearing in every ablation |
| `Solution Stochasticity`, `Loss Margin Behaviour`, `Input Distribution Modelling`, `Fitting Regime` | yes | the four asserted ordinals |
| `Training Operations`, `Prediction Operations` | **decide** | dropped by every search, but they are *measured* quantities and chapters 1, 6 and 9 cite them. Dropping them makes the corpus unable to reproduce the published E2/E3 at all |
| `Active Regularization Mechanisms`, `Robust to Outliers` | **decide** | same, plus `ARM` varies within a model, which is a defect |
| the ~55 AI candidates | **no** | none admissible; the finding is written up, the columns are not needed to state it |

**The decision to make first.** Dropping the four old model features means the repository can
no longer reproduce its own published E1/E2/E3. Two honest options: (i) keep all ten model
columns in the corpus and let `data.MODEL_FEATURES` name only the six — the corpus stays a
record, the equation uses a subset; or (ii) drop them and accept that the published numbers
become historical. **(i) is strongly preferred** and costs four columns of disk.

Then: `scripts/build_proposed_frame.py` becomes the generator of the shipped corpus rather
than of a side file, and writes `src/ml_meta_perf/meta_dataset.csv` directly.

### Stage B — feature validity, already verified

Measured on all 476 rows; record it as a test rather than a claim.

| feature | levels | range | spacing | positive | gaps |
|---|---:|---|---|---|---|
| `Model Capability` | 10 | 1 .. 10 | even | yes | none |
| `Processing Units Number` | 169 | 1.79 .. 20.4 | continuous | yes | — |
| `Solution Stochasticity` | 5 | 1 .. 5 | even | yes | none |
| `Loss Margin Behaviour` | 5 | 1 .. 5 | even | yes | none |
| `Input Distribution Modelling` | 5 | 1 .. 5 | even | yes | none |
| `Fitting Regime` | 5 | 1 .. 5 | even | yes | none |

All six are positive, ordered, and every rung is occupied — so every transform in the grammar
(`log`, `sqrt`, `1/f`, `f^2`) is defined on all of them, which is what the old 0-based
`Robust to Outliers` and `Active Regularization Mechanisms` could not offer.

**One caveat to carry into the docs.** `Input Distribution Modelling` is lopsided: rung 1
holds **376 of 476 rows (79%)** and rungs 2-5 hold 20-40 each (4-8%). That is not a sentinel —
every value is a real reading — but three of its five rungs sit below the 10%-of-rows floor
the z-cap imposes elsewhere, and it is one of the two ordinals the search *kept*. Say so in
the chapter rather than letting a reader discover it.

Rung occupancy, for the record:

| feature | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| `Solution Stochasticity` | 180 | 179 | 40 | 37 | 40 |
| `Loss Margin Behaviour` | 140 | 236 | 20 | 60 | 20 |
| `Input Distribution Modelling` | **376** | 20 | 40 | 20 | 20 |
| `Fitting Regime` | 120 | 177 | 105 | 40 | 34 |

Add `tests/test_model_features.py` asserting positivity, integer rungs with no gaps, and
model-constancy — the properties the grammar and the identification argument both rely on.

### Stage C — delete the student analysis, and the subsystem built to read it

The three research inputs and the ~55-column CSV go:

```
assets/ai_model_descriptors.md              the ~61 AI proposals
assets/best_lodo_vs_best_mean_r2.md         the student's result summary
assets/subset_search_semantic_10v_24h_atari.zip   1.2 MB, over the hook's 512 KB cap
assets/meta_dataset_all_descriptors.csv     the ~55 candidates
```

**They are read by eleven files, so this is not four `rm`s.** Deleting the CSV means deleting
the subsystem that exists to search it:

| module | tests | why it goes |
|---|---:|---|
| `descriptors.py` | 19 | the candidate registry |
| `admissibility.py` | — | the gate over those candidates |
| `feature_search{,_cli}.py` | 14 | superseded by `equation_search` |
| `cluster_search{,_cli}.py` | — | the descriptor-tier search, finished |
| `ess_sampler.py`, `ess_search.py` | 23 | the ESA sampler, superseded |
| `reference_subsets.py` | 13 | the external run's record |
| `scripts/proposed_ordinals.py`, `scripts/controls.py` | — | one-shot measurement harnesses |

That is **~80 tests and eight modules**, and it is the right call: the search is *finished* and
its answer is written up. Two things must survive it:

1. **`joint_search.SearchPoint`** — `equation_search` imports it. Move the dataclass into
   `equation_search.py` and delete the rest of `joint_search`.
2. **The findings, as prose.** `reference_subsets.order_premium` pinned that two external
   solutions owed ~0.10 leave-one-dataset-out to declaration order. That belongs in
   `08-limitations.md`, not only in a deleted test.

Do this as its own commit, with the test count drop stated in the message.

### Stage D — the documentation

**A new chapter is needed**, not just edits: nothing in `assets/docs/` explains where a model
feature comes from. Suggested `11-model-features.md`, or fold into `01-problem.md`:

- **The intent of each of the six.** For each: what it claims about a learner, its rungs and
  what each rung means, its published grounding, and its weakness. The gradings are asserted
  taxonomy in the same sense `Model Capability` is, so they carry the same ch. 8 caveat and
  must not be quoted as measurements.
  - `Solution Stochasticity` — how deep randomisation reaches into the fit. Breiman (1996,
    2001), Ho (1998), Geurts, Ernst & Wehenkel (2006).
  - `Loss Margin Behaviour` — how hard the objective penalises points far from the boundary.
  - `Input Distribution Modelling` — how much of P(x) the learner commits to modelling.
    Ng & Jordan (2001), and the LDA/QDA covariance hierarchy.
  - `Fitting Regime` — how the parameters are reached, closed form to in-context.
  - `Model Capability` — carry forward ch. 8's existing caveat, **and state the new fact**:
    it is a *bijection* with `MODEL_FAMILY`, so it is a ten-level categorical wearing an
    ordinal's clothes and anything read off it is a claim about family.
  - `Processing Units Number` — and say that its log is by definition, encoding bounded
    returns to capacity, so a term reading `log(Processing Units Number)` is a second log.
- **The identification argument.** No feature set the corpus can build separates `DT` from
  `ExtraTree` or the two LightGBM variants without an asserted mechanism ordinal; one column
  resolves both. The chosen set identifies all 10 families with zero ambiguity and leaves
  134/476 rows ambiguous at model resolution — which is the family-versus-model decision in
  Open work item 1 and must be stated as a choice.
- **The negative result.** ~61 AI-proposed descriptors, none admissible, for one generalising
  reason: they are hyperparameter *values*, and a hyperparameter a learner does not have has
  no value. Their use is identity, not magnitude. This is what replaces the deleted files.
- **The two protocols.** `01-problem.md` or `04-evaluation.md` must define `nested` and
  `fixed_form` and say which every number uses. This is the single largest source of
  misreading in the current material.

Chapters needing edits because they name the old model features: **01, 02, 06, 07, 08, 09**;
**10** regenerates.

### Stage E — the plots

Two figures exist only under `results/cluster/` and are not in the study:

- `binary_decision.png` — accuracy and F1 against threshold, over a grey best-baseline line.
- `binary_average_precision.png` — per-dataset MAP against threshold.

Move both into `plots.py` / `figures.py` with captions, and add a third the branch has not
drawn: **ranking quality against a baseline** (MRR, hit@1, MAP@3 as grouped bars, per
protocol). Then `report.py` needs generators for both tests so the sentences stay derived
arithmetically rather than written — `CLAUDE.md`'s central claim.

`experiment.py` still calls the old `validate.ranking_report`, which returns Spearman and
top-1 regret only, and **Spearman was measured to separate nothing** (0.63-0.73 across every
predictor and every baseline). Replace it; do not extend it.

The eight existing figures regenerate unchanged in form.

### Stage F — regenerate, re-tune, gate

1. Re-sweep `DEFAULT_E2` / `DEFAULT_E3` for the new `MODEL_FEATURES` — `CLAUDE.md` is explicit
   that the penalty must be re-swept whenever the feature set changes, and this branch has
   measured that warning coming true twice.
2. `python -m ml_meta_perf`, check all seventeen tables and eight figures.
3. Update the headline tables in `README.md` and `CLAUDE.md` **stating the protocol**.
4. `venv/bin/pre-commit run --all-files`, and confirm CI still reproduces the study after
   `pip install .`.

### Order and risk

| stage | risk | reversible |
|---|---|---|
| B (validity test) | none | yes |
| A (one corpus, option (i)) | low | yes |
| D, E (docs and plots) | low | yes |
| F (re-tune, regenerate) | changes published numbers | yes, via git |
| C (deletions) | ~80 tests, 8 modules | only via git |

Do **B → A → D → E → F → C**. C last: once the descriptor subsystem is gone, any claim in the
new chapter that needs re-checking against the candidate pool cannot be re-measured without
reverting.

## Job 15333 result: the search picks **four** features, not six

48,576 points, 67 minutes on `atari`, `results/cluster/equation_search.csv` (gitignored, one
row per point with its full term list so the log re-ranks under other weights without
refitting).

**The winner:** `Model Capability`, `Processing Units Number`, `Fitting Regime`,
`Input Distribution Modelling` — **16 terms, lambda 25, z 4.5, arity 2**.

| | in-sample | LODO | LOMO | binary | ranking | stability | objective |
|---|---:|---:|---:|---:|---:|---:|---:|
| **winner-4** | 0.664 | **0.641** | **0.627** | 0.876 | **0.904** | **0.872** | **0.7469** |
| best 5-feature | 0.670 | 0.647 | 0.631 | 0.879 | 0.869 | 0.850 | 0.7394 |
| best 6-feature | 0.670 | 0.647 | 0.631 | 0.879 | 0.869 | 0.779 | 0.7288 |
| best 3-feature | 0.664 | 0.639 | 0.619 | 0.879 | 0.852 | 0.828 | 0.7267 |
| best 2-feature (`MC`+`PUN`) | 0.629 | 0.614 | 0.583 | 0.877 | 0.816 | 0.840 | 0.7134 |

All fixed-form, so **not comparable with the published 0.613 / 0.474 / 0.428**.

Three things worth carrying forward:

1. **The search dropped `Solution Stochasticity` and `Loss Margin Behaviour`** — precisely the
   two columns that make the set identify every model. Adding them back costs stability
   (0.872 → 0.779) and ranking (0.904 → 0.869) and buys +0.006 LODO. The fit does not want
   them.
2. **Identification and fit disagree, and this is now a decision rather than a measurement.**
   winner-4 identifies **all 10 families with 0 ambiguous rows**, but leaves **134 of 476
   rows ambiguous at model resolution** (`DT`/`ExtraTree`, `LR`/`LinearSVC`,
   `LightGBM_RF`/`LightGBM_ExtraTrees`). Its in-sample ceiling is therefore below 1.0. Either
   the study's claim is at *family* resolution — in which case winner-4 is complete and the
   two dropped ordinals are unnecessary — or it is at model resolution, in which case they
   must be carried despite costing on every fitted axis. **Decide this explicitly; do not let
   the objective decide it silently.**
3. **Stability is now high**: 0.872 against 0.614 for the six-feature equation at a comparable
   cell. The fixed-form protocol is much better justified for this equation than for anything
   measured earlier.

### The equation reads, mostly

16 terms, arity 2, largest standardised weight 0.129 and the smallest 0.020 — no term
dominates but they do separate. **No reciprocal pairs**, which the 23-term six-feature form
had. Eight of twelve dataset features are used, against six for the earlier candidate.

Two things a chapter would still have to answer:

- **`log(gravity)` appears in five of sixteen terms**, paired with three different model
  features. The fan-out is smaller than before but it is the same shape, and it reads as a
  per-model-feature slope on one dataset feature rather than as a relationship.
- **One weight at -2.3081** on `[nr_cor_attr] * [log(Processing Units Number)]`, two orders of
  magnitude above the smallest. Large weights on product terms are what sent held-out
  datasets to the clipping floor before the train-range clip landed.

## Session-close audit (2026-09-05)

### Is more cluster time needed?

**No, not for a smaller or more interpretable equation — job 15333 already searches those
axes.** Length runs 6 to 28 in steps of 1, feature subsets are searched (all 16 that keep
`MC`+`PUN`), and `brevity` is a scored component. A shorter equation does not need a second
run, it needs the front of this one read properly: sort by `objective`, then re-sort by
`n_terms` among points within a small tolerance of the best, and take the shortest that holds.

Three things *would* justify another run, none of them yet:

1. **A different weighting.** `OBJECTIVE_WEIGHTS` is a values statement, and the log records
   all seven components per row, so re-ranking is free and needs no cluster time at all. Do
   that first and only search again if the front turns out to be thin.
2. **The dataset side.** `DATASET_FEATURES` is 12 inherited columns that have never been
   screened. If candidates are added there, the search must be rerun — and that is a *bigger*
   job than this one, because the dataset half enters far more terms.
3. **Nested confirmation**, if the study ever wants to report the discovery procedure's own
   generalisation alongside the fixed-form numbers. Deliberately not scored now.

### Is the code clean?

**The separation is right.** No branch tooling is reachable from `cli` or `experiment` — 17
modules serve the study pipeline, and the 8 added here (`equation_search{,_cli}`,
`admissibility`, `reference_subsets`, `joint_search`, `cluster_search{,_cli}`, `ess_*`) sit
outside it, so `pip install .` and CI are unaffected. Full gate green: **459 tests, ruff,
basedpyright, vulture, and `pre-commit run --all-files`.**

Fixed this session: **`joblib` was undeclared.** `equation_search_cli` and every `scripts/`
driver parallelise over it while it arrived only transitively through `pyBlindOpt`. Now
explicit in the `search` extra.

Still untidy, and deliberately left:

- **Superseded search tooling**: `ess_sampler`, `ess_search`, `feature_search`,
  `feature_search_cli`, `cluster_search{,_cli}` are all superseded by `equation_search`. They
  carry ~90 tests and the record of what was tried. Delete them at merge, not before.
- **`requirements.txt` contradicts `CLAUDE.md`**, which says `pyproject.toml` is the only
  dependency source. Unresolved.
- **`assets/subset_search_semantic_10v_24h_atari.zip` is 1.2 MB** against the 512 KB hook.
  Currently untracked so the hook does not fire; decide before merge.

### Are the plots and documentation updated?

**No, and this is the largest single gap.** The two new evaluations are *not* in the study
pipeline:

| | where it lives | writes to | in `report.py` / `plots.py`? |
|---|---|---|---|
| binary decision | `scripts/downstream_tests.py` | `results/cluster/` | no |
| ranking (MAP/MRR/hit@1) | `scripts/downstream_tests.py` | `results/cluster/` | no |
| `binary_decision.png` | `scripts/downstream_tests.py` | `results/cluster/` | not in `assets/figures/` |
| `binary_average_precision.png` | `scripts/downstream_tests.py` | `results/cluster/` | not in `assets/figures/` |

`experiment.py` still calls the *old* `validate.ranking_report`, which returns Spearman and
top-1 regret only — and Spearman was measured to separate nothing (0.63-0.73 across every
predictor **and** every baseline). So chapter 9's ranking evidence currently rests on the one
metric shown to be uninformative here.

Wiring them in is real work, not a rename: `CLAUDE.md` requires every sentence of the report
to be derived arithmetically, so `report.py` (853 lines) needs generators for both tests, and
`plots.py` / `figures.py` need the two figures plus captions. It is item 2 of "Open work" for
a reason.

What *is* updated: `README.md` (headline numbers, plus a "Work in progress" section pointing
here and warning that the two protocols are not comparable), `CLAUDE.md`'s results table, and
this file.

## The protocol decision (2026-09-05): fixed form, and why

**Settled: the equation is fitted once and only its weights are refit per fold. The
re-selecting ("nested") protocol is not scored.** This is a position about what the artefact
*is*, not a concession to a number, and it is worth writing down because the opposite
convention was assumed for most of this branch.

The form of the equation is the conceptual claim -- a statement about which quantities govern
how well a learner does on a dataset. For a simpler problem one would write that form down
from domain expertise and never search for it. What cross-validation then tests is whether the
claim survives data it has not seen, with its constants recalibrated. Re-deriving the form
inside every fold tests something else entirely: whether the *discovery procedure* is stable.
That is a question about the search, not about the science, and averaging twenty different
equations is what made leave-one-dataset-out jagged in `k`.

**The obligation this creates.** A form arrived at by searching all 476 rows is only a claim
about the phenomenon if it would have been arrived at from a different sample. That is not
assumed here, it is measured: `stability` is the fraction of the equation's terms that
selection picks again when a fifth of the data is removed, and it carries **0.15 of the
objective** -- the same weight as the ranking test. A form that churns fold to fold is a
property of these rows and the objective says so. This replaces the nested R2 and carries the
same information in the form the study actually cares about.

The long-run version of the same idea is the interesting one: **an equation asserted from the
literature, then only fitted.** That would need no stability argument at all, because its form
would never have touched the data. Nothing here does that yet, and the four mechanism ordinals
are the first step towards being able to -- they are asserted taxonomy, so a term over them is
a mechanism claim rather than a fitted coincidence.

### What the objective scores

`equation_search.OBJECTIVE_WEIGHTS`, all components on a 0-1 scale:

| component | weight | why |
|---|---:|---|
| leave-one-dataset-out R2 | 0.20 | transfer to a new dataset, fixed form |
| leave-one-model-out R2 | 0.20 | transfer to a new learner, fixed form |
| ranking | 0.15 | MAP, MRR, hit@1, 1-regret@1 -- what the equation is for |
| binary | 0.15 | accuracy, F1 and per-dataset MAP over all five thresholds |
| **stability** | **0.15** | term reselection frequency across the 20 dataset folds |
| in-sample R2 | 0.10 | fit, deliberately the smallest R2 weight |
| brevity | 0.05 | 1.0 at 6 terms, 0.0 at 30 -- the shorter equation wins ties |

At a common cell (lambda 25, 14 terms, z 4.25, arity 2) the three candidates rank:

| set | objective | in-sample | LODO | LOMO | binary | ranking | stability |
|---|---:|---:|---:|---:|---:|---:|---:|
| **proposed-6** | **0.701** | 0.649 | 0.616 | 0.617 | 0.867 | **0.895** | 0.614 |
| shipped-6 | 0.682 | 0.625 | 0.608 | 0.578 | 0.872 | 0.824 | **0.629** |
| `MC`+`PUN` | 0.671 | 0.617 | 0.587 | 0.577 | 0.867 | 0.849 | 0.575 |

**These leave-one-out numbers are not comparable with the published 0.474 / 0.428**, which
were produced by the re-selecting protocol. Under the fixed form the shipped six themselves
reach 0.608 / 0.578, so most of the jump is the protocol, not the features. Any table mixing
the two is wrong; the study has to restate its headline under one convention and say which.

### The cluster batch

`scripts/equation_search.sbatch` -> `ml_meta_perf.equation_search_cli`. Searches feature
subsets as well as configuration -- requirement 3 says no equation is expected to use every
feature -- over the 16 subsets that keep `MC` and `PUN` (the ablation settled those two:
removing either costs 0.077 and 0.065 LODO), 11 penalties, 23 lengths from 6 to 28, 6 z-caps
and 2 arities: **48,576 points at 2.2s each, about 30 minutes on 62 cores.** Job 15331.

Every point records all seven components plus the term list, so the log re-ranks under
different weights without refitting -- the weights are a statement of values and should be
re-examined against the front, not trusted blindly.

## The revised objective (2026-09-05)

Six requirements, replacing "find the feature set with the best LOMO". They change what is
being built, not just what is being measured.

1. **Two candidate sets, screened and documented** -- one for datasets, one for models. The
   write-up has to show that candidates *were measured*, not that a set was asserted.
2. **No equation is expected to use every feature** from either set. Shorter is better; using
   all of them would be a result, not the goal.
3. **All three of in-sample, LODO and LOMO should go up.** The premise is that an additive
   form of this size cannot overfit a problem this complex, so a fit/transfer trade should be
   treated as a defect to explain rather than a price to pay. Mostly borne out below -- with
   one real exception that is extrapolation rather than overfitting.
4. **A binary-decision test**: given a dataset and a threshold t in {0.5 .. 0.9}, does the
   equation put each model on the right side of t?
5. **A ranking test**: given a dataset, does the equation order the models correctly? Scored
   with MAP and friends, not only rank correlation.
6. **A unified equation for the LOO protocols**: fix the *terms* once, refit only the
   *weights* per fold. The terms must be stable across folds for this to be honest.

Requirement 6 is the one that also repairs the instability documented above: the jaggedness
in LODO comes from re-running term *selection* inside every fold, so different folds fit
different equations and the pooled R2 is an average over 20 different models. Fixing the form
removes that source of variance entirely.

**It also introduces a leak, and the two protocols must be named differently.**
`validate.cross_validate_path`'s docstring says so already: choosing terms against all 476
rows and cross-validating only the weights is "the standard way to leak a held-out fold into
the model". Both questions are legitimate and they are not the same question:

| protocol | terms chosen | answers | status |
|---|---|---|---|
| **nested** (today) | inside each fold | does the *discovery method* generalise? | honest, pessimistic, jagged |
| **fixed-form** (requirement 6) | once, on all rows | does *this published equation* transfer? | leaks, stable, the number a reader wants |

Report both, always labelled. The fixed-form number is the one a practitioner cares about
because the published equation is what they would use; the nested number is the one that
keeps the paper honest. Neither may be quoted as the other.

## Is a unified equation possible? Term stability across folds

Measured by re-running selection in every fold and asking how often each term of the
full-data equation comes back.

| equation | terms | reselected in **every** LODO fold | in every LOMO fold | median reselection rate |
|---|---:|---:|---:|---:|
| proposed-6, k=13 | 13 | **0 / 13** | 2 / 13 | 75% (LODO) |
| proposed-6, k=23 | 23 | 7 / 23 | 7 / 23 | 85% |
| shipped-6, k=20 | 20 | 6 / 20 | 6 / 20 | 85% |

**No equation measured here is stable as a whole.** Every one has a core of about **6-7 terms
that survive all 20 dataset folds and all 25 model folds**, and a tail that churns. The
13-term equation has *no* term common to all folds, which is worse than the longer ones --
a shorter equation is not automatically a more stable one.

This is a usable result rather than a blocker, and it points the same way as "shorter is
better": **the stable core is the unified equation.** Selecting terms by fold-reselection
frequency rather than by in-sample gain is a documented, non-circular criterion -- it is
stability selection, which `feature_search` already implements for features and nothing yet
applies to terms.

### Fixed-form LOO, measured

Terms fixed from the full fit, weights refit per fold:

| equation | LODO nested | **LODO fixed-form** | LOMO nested | **LOMO fixed-form** |
|---|---:|---:|---:|---:|
| proposed-6, k=13 | 0.3863 | **0.4937** | 0.5543 | **0.6107** |
| proposed-6, k=23 | 0.5261 | **-0.1784** | 0.5821 | 0.6214 |
| shipped-6, k=20 | 0.4743 | **0.6173** | 0.4283 | 0.5868 |

Fixing the form lifts and stabilises almost everything -- and the exception is instructive.
proposed-6 at k=23 collapses to **-0.178** because the fixed form carries the -1.7675 weight
on `[nr_cor_attr] * [log(PUN)]`, which extrapolates to the clipping floor on `ASNM-CDX-2009`.
That is **not overfitting** -- the same form scores 0.62 on LOMO -- it is a single product
term leaving its training domain. Requirement 3's premise holds; the failure mode it has to
survive is extrapolation, and the train-range clip already measured is the mitigation.

## The two downstream tests, with baselines

`scripts/downstream_tests.py` runs both under three protocols and writes
`results/cluster/{binary_decision,ranking}.csv` plus `binary_decision.{png,pdf}`.
proposed-6 at k=13, shipped-6 at k=20, lambda 25 / z 4.25 / arity 2.

**The baselines are the point.** An accuracy of 0.87 means nothing until it is put beside
what predicting a group mean would have scored, and the per-model mean is the hard one --
`CLAUDE.md` already records it tying E3 on Spearman.

### Binary decision: accuracy and F1 per threshold

| protocol | t=0.5 | t=0.6 | t=0.7 | t=0.8 | t=0.9 |
|---|---|---|---|---|---|
| proposed-6 in-sample | 0.88 / 0.93 | 0.90 / 0.93 | 0.86 / 0.89 | 0.81 / 0.83 | 0.76 / 0.70 |
| proposed-6 LODO fixed-form | 0.87 / 0.91 | 0.84 / 0.88 | 0.80 / 0.84 | 0.75 / 0.76 | 0.71 / 0.62 |
| proposed-6 LOMO fixed-form | 0.87 / 0.92 | 0.88 / 0.91 | 0.85 / 0.88 | 0.79 / 0.81 | 0.74 / 0.67 |
| shipped-6 LODO fixed-form | 0.89 / 0.93 | 0.87 / 0.91 | 0.85 / 0.88 | 0.80 / 0.81 | 0.76 / 0.70 |
| *global mean* | 0.77 / 0.87 | 0.73 / 0.85 | 0.67 / 0.80 | 0.37 / **0.00** | 0.49 / **0.00** |
| *per-model mean* | 0.78 / 0.86 | 0.73 / 0.80 | 0.72 / 0.77 | 0.72 / 0.76 | 0.65 / 0.55 |
| *per-dataset mean* | 0.82 / 0.89 | 0.80 / 0.87 | 0.71 / 0.77 | 0.70 / 0.71 | 0.70 / 0.62 |

*(accuracy / F1)*

**The R2 is sufficient for the decision, and the honest margin is 0.05-0.10.** The equation
clears every baseline at every threshold, and the figure draws the best baseline as a grey
reference line so the gap is visible rather than asserted. Two things to state plainly:

- **shipped-6 is marginally better than proposed-6 on this test** (0.89/0.93 against
  0.87/0.91 under LODO). The binary decision does not distinguish the feature sets; the
  ranking test does.
- **At t=0.9 the margin nearly closes**: LODO F1 0.62 against the per-dataset mean's 0.62.
  At that cut the equation is separating "excellent" from "near-perfect" and adds nothing
  over a per-dataset constant. Report the threshold curve, never a single number.

### Ranking, scored as a search engine

Only the head matters, so the metrics are head-weighted. Relevance is "within 0.01 MCC of
this dataset's best model" rather than a fixed top-k, because **9.3 of 25 models are tied at
the top on average** -- 80 of 476 rows sit at exactly MCC 1.0, and a top-3 cut would score a
correct answer as a miss. Random guessing gives hit@1 of about 9.3/25 = 0.37.

| protocol | MRR | hit@1 | hit@3 | MAP@3 | P@3 | regret@1 | regret@3 | rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proposed-6 in-sample | **0.893** | **0.850** | 0.900 | 0.838 | 0.700 | 0.010 | 0.006 | 0.702 |
| proposed-6 LODO fixed-form | **0.893** | **0.850** | 0.900 | 0.829 | 0.700 | 0.010 | 0.006 | 0.697 |
| proposed-6 LOMO fixed-form | **0.893** | **0.850** | 0.900 | 0.829 | 0.683 | 0.021 | 0.006 | 0.676 |
| shipped-6 LODO fixed-form | 0.783 | 0.650 | 0.950 | 0.775 | 0.767 | 0.021 | 0.002 | 0.696 |
| shipped-6 LOMO fixed-form | 0.771 | 0.650 | 0.900 | 0.750 | 0.717 | 0.034 | 0.002 | 0.632 |
| *global / per-dataset mean* | 0.684 | 0.500 | 0.850 | 0.675 | 0.450 | 0.090 | 0.009 | 0.000 |
| *per-model mean* | 0.868 | 0.800 | 0.900 | **0.850** | **0.767** | **0.006** | 0.004 | **0.725** |

**This is a split decision, and the earlier claim of a clean win was made before the
per-model-mean baseline was run.** proposed-6 leads on the two most head-weighted metrics --
**MRR 0.893 against 0.868 and hit@1 0.850 against 0.800** -- and the per-model mean leads on
**MAP@3 (0.850 against 0.829), P@3, regret@1 (0.006 against 0.010) and Spearman**. Against
shipped-6 the win is unambiguous: +0.11 MRR, +0.20 hit@1, half the regret@1.

Two consequences for the write-up:

- **Beating the per-model mean at the head of the list is the bar**, and proposed-6 clears it
  only on MRR and hit@1. That baseline has no LOMO defence -- it is empty for a held-out
  model -- which is the argument for the equation, and it has to be made explicitly rather
  than by quoting a metric the baseline happens to lose.
- **regret@3 is 0.002-0.009 for everything including the global mean.** With 9.3 tied-best
  models the top-3 almost always contains one, so regret@3 is saturated and useless here, as
  NDCG@3 was. **Report MRR, hit@1, MAP@3 and regret@1; do not report regret@3 or NDCG.**

### Which dataset features get selected -- not yet properly

| equation | dataset features used | terms in every LODO fold | most-used |
|---|---:|---:|---|
| proposed-6, k=13 | **6 / 12** | **0 / 13** | `gravity` in 5 of 13 terms |
| shipped-6, k=20 | 8 / 12 | 6 / 20 | `gravity` in 4 of 20 terms |

Unused by **both**: `class_ent`, `nr_inst`, `nr_outliers`, `ns_ratio`.

This is the expectation that a better optimiser should make dataset-feature selection
sensible, and it is currently **not met**. Half the dataset side is untouched, `gravity`
carries five of thirteen terms in one equation -- the fan-out already flagged as a
readability problem -- and the shorter equation is the *less* stable one. Whether that is the
optimiser or the features cannot be told apart until Stages 1-2 land, which is the reason
they come first.

## The six-feature set

`scripts/build_proposed_frame.py` writes `assets/meta_dataset_proposed.csv`: the corpus plus
the four ordinals, each asserted from published descriptions of the learners, each defined
for **all 25 models**, none with a zero anywhere. `PROPOSED_MODEL_FEATURES` is the six under
test -- `Model Capability`, `Processing Units Number`, and the four.

One correction to the taxonomy since it was first drafted, and it mattered: `AdaBoost`
minimises exponential loss (rung 3), tree learners minimise an impurity criterion (rung 1),
and `PassiveAggressive` minimises **hinge** like `LinearSVC` and `SGD` (rung 4) -- only
`Perceptron` uses the perceptron criterion (rung 5). With that fixed:

| set | size | families ambiguous | models ambiguous |
|---|---:|---:|---:|
| `MC` + `PUN` | 2 | 0 | 174 |
| shipped-6 | 6 | 0 | 121 |
| `MC`+`PUN`+Stoch+Loss | 4 | 0 | 40 (`PassiveAggressive`/`SGD`) |
| `MC`+`PUN`+Stoch+Loss+Regime | 5 | **0** | **0** |
| **proposed-6** | 6 | **0** | **0** |

So the identifying core is **five**; `Input Distribution Modelling` is redundant for identity
and earns its place only if it earns it in the fit.

### It is the best result measured anywhere in this study

Same 112-cell grid as every other comparison, sorted feature order:

| set | size | in-sample | LODO | LOMO | best cell |
|---|---:|---:|---:|---:|---|
| shipped-6 (published cell) | 6 | 0.6134 | 0.4743 | 0.4283 | 20/20/3.0/3 |
| shipped-6 (its own best) | 6 | 0.6719 | 0.5153 | 0.5309 | 10/20/4.25/3 |
| `MC`+`PUN` | 2 | 0.6264 | 0.5275 | 0.5558 | 20/17/4.25/2 |
| proposed-4 | 4 | 0.6498 | 0.4951 | 0.5565 | 40/23/4.25/2 |
| proposed-5 | 5 | 0.6697 | 0.5182 | 0.5419 | 20/23/4.25/2 |
| **proposed-6** | **6** | **0.6751** | **0.5261** | **0.5821** | **25/23/4.25/2** |
| `ATARI_BEST` (inadmissible) | 10 | 0.7058 | 0.5854 | 0.5857 | 10/23/5.0/3 |

**proposed-6 is the best admissible set on every axis at once.** It answers the in-sample
complaint -- 0.6751 beats the shipped six's own optimum, and its identification ceiling is
**1.0** against 0.909 for `MC`+`PUN` and 0.988 for the shipped six. It reaches LOMO 0.5821
against `ATARI_BEST`'s 0.5857 while being fully admissible and fully identifying, and it does
it at arity 2.

Paired against shipped-6 over the 112 identical cells: in-sample **+0.0217 (110/112)**, LOMO
**+0.0523 (107/112)**, LODO -0.0488 (32/112). Paired against `MC`+`PUN`: in-sample **+0.0415
in 112 of 112 cells**.

That LODO column is the problem the next section is about, and it is not what it looks like.

## How long is the equation, and does it still read?

Length sweep at the chosen cell (lambda 25 / z 4.25 / arity 2), k from 5 to 30,
`results/cluster/knee.csv`.

### The headline length is 23, and it is a lucky draw

| k | in-sample | LODO | LOMO |
|---:|---:|---:|---:|
| 13 | 0.6430 | 0.3863 | 0.5543 |
| 17 | 0.6666 | 0.2157 | 0.5747 |
| 19 | 0.6712 | 0.3120 | 0.5753 |
| 21 | 0.6729 | 0.4990 | 0.5765 |
| **23** | **0.6751** | **0.5261** | **0.5821** |
| 25 | 0.6773 | 0.5149 | 0.5721 |
| 27 | 0.6791 | **0.2179** | 0.5735 |

Largest change between adjacent k: **in-sample 0.032, LOMO 0.025, LODO 0.298**. In-sample and
LOMO are smooth and saturating; LODO is jagged, and shipped-6 is worse -- it goes 0.508 at
k=21 to **-0.360** at k=23 to 0.459 at k=25. So the 0.5261 headline is a spike on a curve
that moves by 0.3 when k moves by 2. It is the same one-fold defect, seen along a different
axis, and it is not a property of the feature set.

### Most of the value arrives by k = 13

| k | in-sample, as % of k=23 | LOMO, as % of k=23 |
|---:|---:|---:|
| 9 | 92.7% | 94.8% |
| **13** | **95.2%** | **95.2%** |
| 17 | 98.7% | 98.7% |
| 21 | 99.7% | 99.0% |

Ten extra terms buy +0.032 in-sample and +0.028 LOMO. A 13-term equation is **shorter than
the published 20** and keeps 95% of both.

### Readability: no at 23, yes at 13

The 23-term equation (22 after `prune`) has three problems a chapter cannot write around:

- **Reciprocal pairs.** `[log(nr_class)] / [log(PUN)]` and `[log(PUN)] / [log(nr_class)]` are
  both selected, as are `[nr_cor_attr] * [log(PUN)]` and `[nr_cor_attr] / [log(PUN)]`. Using
  x and 1/x on the same pair is a flexible basis, not a relationship. The published E3 has no
  such pair.
- **One weight at -1.7675**, an order of magnitude above the rest -- and large weights on
  product terms are exactly what makes a held-out dataset extrapolate to the clipping floor.
- **`log(gravity)` fans out over six terms**, paired with four different model features. That
  is a per-model-feature slope on one dataset feature, which `CLAUDE.md` already identifies as
  the failure mode that rules out indicator features.
- Standardised weights are uniformly small (0.02-0.12): no term dominates, so no term is
  individually quotable.

The 13-term equation (12 after `prune`) has none of them: no reciprocal pair, largest weight
0.24, and standardised weights that actually separate (0.014 to 0.238, so four terms carry the
equation). `log(gravity)` still appears in four terms -- milder, but the fan-out is the
feature to watch.

**One catch.** At k=13 only four of the six features appear: `Fitting Regime`,
`Input Distribution Modelling`, `Model Capability` and `Processing Units Number`.
**`Solution Stochasticity` and `Loss Margin Behaviour` contribute nothing** -- and those are
precisely the two columns that do the identification work. All six appear at k=23. So the
identification argument and the fit argument currently point at different columns, and that
tension has to be resolved rather than papered over: either the short equation is the
deliverable and identification is a separate claim about the *feature set* rather than about
the equation, or the length has to be justified by something better than a spike in LODO.

## Why adding features "hurts" -- it does not, LODO is a one-fold statistic

The complaint was right and my first explanation of it was wrong. Recorded because the wrong
answer is the plausible one and someone will reach for it again.

**The wrong answer.** `Selector.search` expands only the top `candidates` terms per beam
parent per step; `CANDIDATE_POOL_DEFAULT = 24`, `Configuration` cannot set it, and
`validate.cross_validate_path` never passes it. A *fixed count* over a library that grows with
feature count does shrink the visible fraction -- 24 of 175 terms for `MC`+`PUN`, 24 of 316 for
six features -- so it looked like the mechanism. Likewise `guided_screen` is a univariate
marginal filter with a global top-K cut, which `CLAUDE.md` already records as destructive, and
it silently switches on when the library passes `pool_size`.

**Measured, neither is the mechanism.** Sweeping `candidates` over 24 / 48 / 96 / 192 / 384:

| set | in-sample across all five | LODO across all five | LOMO |
|---|---|---|---|
| `MC`+`PUN` | 0.627 (**identical to 3 dp**) | **-0.383 .. +0.485** | 0.549-0.553 |
| shipped-6 | 0.653 (identical) | -0.360 .. +0.487 | 0.509-0.518 |
| proposed-6 | 0.675 (identical) | +0.292 .. +0.526 | 0.570-0.582 |

In-sample does not move at all under a 16x change in search breadth, so the beam is **not**
starved -- it already finds the fit optimum. And `guided_screen` is a no-op at z 3.0 and 4.25
for every set here (libraries 175-316 against `pool_size` 600); it only cuts at z 5.0. LODO
meanwhile swings by 0.87 with no monotone trend, which is not a search-quality signature.

**The mechanism is that pooled LODO is dominated by one fold.** Decomposing the
`MC`+`PUN` case that scores -0.383, as each fold's share of the total sum of squares:

| fold | share of SST |
|---|---:|
| **`ASNM-CDX-2009`** | **0.8915** |
| `X-IIoTID` | 0.0763 |
| `MQTTEEB-D` | 0.0623 |
| all 17 others together | 0.2176 |

One dataset, 25 rows of 476, is 89% of the error. `ASNM-CDX-2009` has mean MCC 0.370 against
the corpus's 0.731; held out, the equation predicts it at the **clipped floor of -1.0**.
Predicting it at the global mean would have cost 0.113 of SST, so the equation is doing eight
times worse than a constant on that fold alone. Whether a given configuration happens to
handle it is close to a coin flip, and *that* is what the ordering band, the `candidates`
swing, and the "adding features hurts LODO" pairing are all measuring.

### Two fixes, both measured

**Fix 1 -- clip to the training fold's observed range.** `model.predict` already clips to
`[MCC_LOWER, MCC_UPPER] = [-1, 1]`, the theoretical MCC range. But this corpus has exactly one
negative row (-0.29), so a floor at -1 leaves an equation free to predict -1 on a dataset it
extrapolates badly on. The training fold's own observed range uses no held-out information
and is the range the equation was ever shown:

| set | LODO pooled, clip [-1,1] | LODO pooled, clip to train range |
|---|---:|---:|
| `MC`+`PUN` | **-0.3827** | **+0.2600** |
| shipped-6 | -0.3602 | +0.2824 |
| proposed-6 | 0.5261 | 0.5261 (unchanged) |

It rescues the blown-up cells and changes nothing else. **proposed-6 never needed it** -- it
does not predict outside the training range, which is independent evidence that the
six-feature set is the more stable object, not merely the higher-scoring one.

**Fix 2 -- stop quoting pooled R2 alone.** Pooled LODO measures predictions against the
*global* mean, and most of this corpus's variance is between datasets -- which the dataset
features get almost for free. Per-fold R2, against each dataset's own mean, is the honest
within-dataset number, and it is far lower for everyone:

| set | LODO pooled | LODO median per fold | LODO worst fold |
|---|---:|---:|---:|
| `MC`+`PUN` | 0.2600 | 0.1707 | -9.21 |
| shipped-6 | 0.2824 | 0.1724 | -16.66 |
| **proposed-6** | **0.5261** | **0.2857** | **-8.68** |

The ranking is unchanged and proposed-6's margin *widens* on the robust metric: **+0.11
median per-fold R2** against both alternatives, where the pooled gap is smaller. `CrossValidation.per_fold`
already computes these; nothing reports them.

Both fixes are small, both are defensible on their own terms rather than because they raise a
number, and neither has been applied to `src/` yet.

## Point 2 answered: the winning descriptors are **not** solid

Audited before planning anything else, because the answer gated everything after it, and
the gate held: the strict pool built from this table is the set that survived the controls
above. Kept in full because it is the evidence for why `ATARI_BEST` is a ceiling rather than
a candidate. Support = fraction of the 476 rows where the column is non-zero.

| descriptor | levels | support | terms @ z=3 | terms @ z=5 | verdict |
|---|---:|---:|---:|---:|---|
| `Processing Units Number` | 169 | 100% | 58 | 236 | solid: continuous, positive, no zeros |
| `Model Capability` | 10 | 100% | 40 | 150 | ordered, no zeros (asserted ladder, ch. 8) |
| `Robust to Outliers` | 3 | 47.9% | 26 | 93 | shipped; coarse but well supported |
| `chatgpt__l2_coefficient` | 9 | 27.9% | 0 | 9 | zero-sentinel |
| `perplexity__iters_log` | 4 | 20.4% | 23 | 83 | zero-sentinel |
| `chatgpt__smoothing_coefficient` | 8 | 8.4% | 3 | 33 | zero-sentinel, thin |
| `perplexity__max_units` | 5 | 7.1% | 0 | 7 | zero-sentinel, thin |
| `claude1__no_tunable_hp` | **2** | 7.1% | 15 | 91 | **binary indicator** |
| `chatgpt__covariance_shrinkage` | 4 | **1.7%** | 22 | 80 | 8 non-zero rows |
| `chatgpt__elasticnet_coefficient` | 3 | **0.6%** | 26 | 84 | **3 non-zero rows** |

Every column is non-negative and no column is continuous except the two already shipped.
Seven of the ten are 2-to-9-level ordinals that are zero on 72-99% of rows, and in each
the zero means *"this hyperparameter does not exist for this learner"* rather than *"its
value is zero"* -- the applicability/value collapse that `INADMISSIBLE` already rejects
`perplexity__class_weight_signal` and `chatgpt__inference_work_proxy` for. The manifest
set `exclude_applicability: true`, which removed the explicit applicability *flags* but
left the sentinel inside every value column.

**The gain is mechanically tied to this.** Raising the z-cap from 3 to 5 grows the library
291 -> 1058 terms, and the growth is concentrated in exactly the thinnest columns:
`elasticnet_coefficient`, non-zero on **3 rows**, contributes **84 terms**;
`covariance_shrinkage`, non-zero on 8, contributes 80. The published equation carries them
as additive spikes -- `([nr_cor_attr] + [chatgpt__elasticnet_coefficient]) /
[log(Processing Units Number)]` is `nr_cor_attr / log(PUN)` everywhere except three rows.
That is a per-model indicator wearing a ratio's clothing, and it is why the same ten
features collapse to 0.344 at z-cap 3.

Two standing decisions in `CLAUDE.md` this collides with head-on: **"no binary indicator
features" (2026-08-17)** -- `claude1__no_tunable_hp` is literally binary -- and
**"explainability is the constraint, not a cost to trade against R2 ... sessions keep
drifting the same way: find a gain, then quote the interpretability loss as a price. Do
not."** The +0.111 LODO is real and reproduces exactly; it is also, as it stands, bought
with the thing the project has already decided twice not to buy.

This does not mean discard the result. It means the search has to be re-run under a gate,
and the honest comparison is *gated best* against *gated baseline*. That is what the cluster
run did, and the gated set kept the LOMO half of the margin while the LODO half turned out
not to have been a feature effect at all.

## Plan

Restructured around the six requirements of 2026-09-05. Stages 1-3 repair the pipeline and
stand whatever happens to the features; 4-6 build the two candidate sets and the unified
equation; 7-9 are the new evaluations; 10 is blocking.

### Stage 1 -- make LODO a statistic worth comparing

Still the priority, and requirement 6 does half of it. Nothing downstream is trustworthy
until this lands.

1. **Clip predictions to the training fold's observed target range** rather than the
   theoretical `[-1, 1]`. Measured: rescues `MC`+`PUN` from -0.383 to +0.260, and is the
   mitigation for the k=23 fixed-form collapse to -0.178. Test that no fold's predictions
   leave its training range.
2. **Report per-fold dispersion wherever LODO appears** -- median and worst fold alongside
   pooled. `CrossValidation.per_fold` already holds them and nothing reads them. Pooled R2
   measures against the *global* mean and most variance here is between datasets, so pooled
   flatters the design; median-per-fold is the within-dataset truth (0.29 against a pooled
   0.53 for proposed-6).
3. **Implement both LOO protocols side by side** -- `nested` (selection inside the fold) and
   `fixed_form` (terms fixed, weights refit) -- as an explicit argument, never a default that
   has to be inferred. Every reported number carries its protocol name.

### Stage 2 -- close the search defects

1. **Deterministic tie-break**: `np.argsort(-scores, kind="stable")`. Direct cause of the
   0.286 ordering band. Verify with `joint_search.evaluate_over_orders`.
2. **Expose `candidates`** on `Configuration` and pass it through `cross_validate_path`,
   which ignores it today.
3. **Make `guided_screen`'s activation explicit** -- it is the marginal filter `CLAUDE.md`
   records as destructive, and it switches on silently past `pool_size`.
4. **Fix the library-construction leak**: `build_library` applies the z-cap over all 476 rows
   outside the fold loop, so which terms *exist* is decided with held-out rows in view. Expect
   nested LODO to get worse; report it anyway.

### Stage 3 -- settle the pre-logged column names

`Processing Units Number` is `log(units)` **by definition**, and the definition is the
bounded-returns argument the paper makes. Keep the transform, rename the column so
`log(Processing Units Number)` stops reading as a first log. Un-logging is not the fix
(-0.046 in-sample, measured).

### Stage 4 -- the two candidate sets, screened and documented (requirement 1)

The dataset side has never been screened at all -- `DATASET_FEATURES` is 12 columns inherited
from the corpus, and no chapter shows a candidate was ever rejected. That is the same gap the
model side is being criticised for.

- **Dataset candidates.** Enumerate what the corpus and the standard meta-feature literature
  (`pymfe`'s statistical / information-theoretic / model-based groups) offer beyond the 12,
  audit them through `admissibility.audit`, and record what was rejected and why. No
  re-running of models is needed -- these are functions of the data files.
- **Model candidates.** Already done and documented: ~61 AI-proposed columns, all rejected as
  magnitudes with one generalising reason, plus the four asserted ordinals that replace them.
  Write it up as a screen, not as a search.
- **Both sets get the same audit table**: support, levels, sign, zero semantics, spacing,
  model- or dataset-constancy, and the identification contribution.

### Stage 5 -- select the unified equation by fold stability (requirement 6)

The measurement says every equation here has a **6-7 term core** that survives all 20 dataset
folds and all 25 model folds, and a churning tail. So:

1. Run selection in every LODO and LOMO fold; record each term's reselection frequency.
2. Keep terms above a stated frequency floor -- start at 100% (the ~6-7 core) and relax until
   the curve of (terms kept, fixed-form LODO/LOMO) turns over.
3. Fit the weights of that fixed form on all rows for the published equation, and refit only
   weights per fold for the LOO numbers.
4. This is stability selection applied to terms; `feature_search` already implements it for
   features. It replaces "the beam's top-k on all rows", which is what makes the current
   equation length a lucky draw.

Note the tension to resolve here: at k=13 `Solution Stochasticity` and `Loss Margin
Behaviour` contribute **no terms**, yet they are the two columns that make the set identify
every model. Either identification is a claim about the *feature set* and the equation is
allowed to be shorter, or the stability floor has to be low enough to admit them. Decide it
explicitly.

### Stage 6 -- re-fit the candidate sets on the repaired pipeline

proposed-6, proposed-5, `MC`+`PUN`, shipped-6, over the same grid, reporting in-sample,
nested LODO/LOMO, fixed-form LODO/LOMO, and per-fold dispersion. Requirement 3 says all three
R2 should rise together; where they do not, the reason must be named -- and the one case seen
so far is extrapolation of a large-weight product term, not overfitting.

### Stage 7 -- the binary-decision test (requirement 4) -- prototyped

`scripts/downstream_tests.py` runs it: confusion matrix, accuracy, **F1**, precision, recall,
balanced accuracy and the decision's own MCC, per threshold, under three protocols, against
three baselines, with `binary_decision.{png,pdf}` drawing accuracy and F1 against threshold
over a grey best-baseline reference line.

Measured, the claim is available: the equation clears every baseline at every threshold by
0.05-0.10. Two caveats are already visible and must be carried into the chapter -- shipped-6
is *marginally better* than proposed-6 here, so this test does not distinguish the feature
sets; and at t=0.9 the LODO F1 (0.62) ties the per-dataset mean, so the threshold curve is
the result and a single number is not.

To move it into `src/`: promote `binary_metrics` and the figure into `validate` / `plots`,
add the per-threshold table to `EquationReport`, and have `report` generate the sentences
arithmetically as `CLAUDE.md` requires.

### Stage 8 -- the ranking test (requirement 5) -- prototyped, and the bar is the per-model mean

Scored as a search engine, since only the head of the list is ever used. Implemented in
`scripts/downstream_tests.py`: MRR, hit@k, precision@k, MAP@k and regret@k for k in {1,3,5},
with relevance defined as "within 0.01 MCC of this dataset's best" rather than a fixed top-k
-- 9.3 of 25 models are tied at the top on average, so a top-k cut misscores correct answers.

**Recommended headline set: MRR, hit@1, MAP@3, regret@1.** Retire the rest --
**Spearman** separates nothing (0.63-0.73 across every predictor *and* every baseline),
**NDCG@3** sits at 0.97-0.99 for everything, and **regret@3** is 0.002-0.009 even for the
global mean.

The measured result is a split decision, not a win: proposed-6 leads the per-model mean on
MRR (0.893 vs 0.868) and hit@1 (0.850 vs 0.800) and trails it on MAP@3, P@3 and regret@1.
Against shipped-6 it wins outright. So the chapter's argument has to be the one
`CLAUDE.md` already frames -- **the per-model mean is empty under leave-one-model-out** --
and not a metric the baseline happens to lose.

To move it into `src/`: extend `validate.ranking_report`, which today returns only Spearman
and top-1 regret, and add the baselines to `EquationReport` so `guidance` can compute its
verdicts against them.

### Stage 9 -- audit the ordinal tables and read the equation

Six rung tables, 25 models each, all asserted; one pass already found three errors. Check each
against a published description, and check each ordinal's rungs against per-rung mean MCC the
way ch. 8 checks `Model Capability` -- reporting the agreement fraction, since `MC` itself
matches at only 6 of 9 steps. Then print the chosen equation and confirm every term reads.
The k=23 form fails this today: reciprocal pairs, a -1.77 weight, and `log(gravity)` fanned
across six terms.

### Stage 10 -- nested confirmation (blocking, before any claim)

Nest the whole selection -- pool, terms, configuration -- inside an outer leave-one-model-out
loop and report the result. This is the only number that may be called the headline. Two
finalists. One Slurm array.

## Next, in order

1. **Stage 1**, then **Stage 2** items 1-3. Nothing else is worth measuring first.
2. **Stage 5** -- the stability-selected unified equation. It is the deliverable, and it is
   what makes stages 7 and 8 meaningful.
3. **Stages 7 and 8** -- the two new tests *with their baselines*. The ranking result is the
   strongest evidence for the new feature set found so far, and it is currently unbaselined.
4. **Stage 4** -- the dataset-side screen. Independent of the above; can run in parallel.
5. **Stage 9**, then **Stage 10** overnight.
6. Only then, Phase B.

Explicitly *not* next: more feature search. In-sample is invariant to a 16x change in search
breadth; the beam was never the constraint.

## Housekeeping

- **`assets/subset_search_semantic_10v_24h_atari.zip` is 1.2 MB and will be rejected by the
  `check-added-large-files --maxkb=512` hook.** Decide: raise the cap, add an exception,
  keep it untracked, or commit only `finalists.csv` + `search_manifest.json` (the 10.9 MB
  `evaluations.csv` inside it is the bulk and nothing here reads it). Now easier to settle:
  the run is reproduced and controlled, and `reference_subsets.py` carries everything this
  branch reads from it.
- `results/cluster/` is 11 MB of evaluation logs and is gitignored. The three
  `status_*.json`, the three `pareto_*.csv` and the three control CSVs are small (~26 KB
  total) and are the only parts any later stage reads -- worth committing those alone.
- `requirements.txt` (added earlier on this branch) contradicts `CLAUDE.md`'s standing
  constraint that `pyproject.toml` is the only dependency source. Worth resolving before
  merge.
- The `search` extra now also carries `EmptySpaceSearch>=0.7.0`. The floor is not
  cosmetic -- the attraction-field parameters do not exist in 0.1.x.
- `scripts/search.sbatch` takes `<tier> <target-size> <hours>`; submit with
  `sbatch --job-name=srch-strict --cpus-per-task=62 scripts/search.sbatch strict 8 11.0`.
  Keep it on `cpuPartition`: the ARM node would resolve beam near-ties under a different
  BLAS, which Control 1 shows is a live risk at arity 3 and not merely theoretical.

## Phase B (not started; confirm before beginning)

Wire the winning set into `data.py` (`MODEL_FEATURES`, `FEATURE_GLOSSARY`), re-sweep
`DEFAULT_E2`/`DEFAULT_E3`, update `assets/docs/`. **No merge into
`src/ml_meta_perf/meta_dataset.csv` is needed** -- the recommendation drops four shipped
columns and adds none, so the corpus is unchanged and `E2`'s feature list gets shorter. This changes the project's published
headline numbers. Note that Stage 7 changes `DEFAULT_E3` *even if Phase B never happens*,
so the two should be decided together rather than in sequence.
