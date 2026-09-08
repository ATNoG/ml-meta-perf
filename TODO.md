# Working notes

> **You are on `wip/beam-search-op`.** It carries the audit work that used to sit on
> `fix/ci-docs-plots` — that branch and `feature/descriptor-selection` were deleted on
> 2026-09-08, local and remote, after checking that every commit on them was reachable from
> here or from `main`. **Only `main` and `wip/beam-search-op` exist now, and no branch is to be
> created without being asked for.** Nothing here changes the published equation, and
> `tests/test_beam.py` fits the real corpus both ways to prove it. Everything under the
> "audit session" headings describes what is now this branch's own history.

## The plan — 2026-09-08

Set by the owner this session. Five criteria changed or sharpened, and three of them change how
job 15337's verdicts must be read:

- **Speed is a criterion, not a tie-break.** The corpus will grow, so a *faster* beam at equal
  accuracy is a win, and a tie at 1.2x is not "more machinery for no gain". Any improvement
  across the several evaluations is also a win, however small.
- **The claim is stability across all four protocols.** In-sample, LODO, LOMO and LOO-cell,
  on R2 **and** ranking — that is what separates this from shallow learning, which collapses.
  A beam variant has to be judged there, and `evaluate` currently scores no cell protocol at
  all (see B).
- **ESS is a continuous sampler.** One-shot use "gets you nothing", which is what
  `seeding.empty_space` does today. See D.
- **One branch for this work.** Done — see below.
- **The whole text needs revision**, not only chapter 6, and it waits on final results.

### The CLI cleanup plan — 2026-09-08

**One command, one pipeline: read the corpus, search for the equations, fit their weights,
evaluate under the protocols, write the tables, figures and generated chapter sections.**
Nothing about the equations is written down; every knob that remains is a knob, with the
previously-used values as its default.

**C1. Derive the length instead of asserting it.** `experiment.py:356` is the only line that
decides -- `size = min(config.headline_terms, available)` -- and it throws away work already
done: `fit()` returns an equation at *every* length in one pass and `run_equation`
cross-validates all of them, so the consensus curve exists at that point. Replace the
assertion with the rule, carry the chosen size on `EquationReport`, and have the six read
sites (566/568 protocol scores, 757/759 decision report, 1038/1040 baselines) take it from
there. `length_comparison` already derives and keeps `headline_terms` only as a fallback; that
fallback becomes `max(path)`. `Configuration.headline_terms` then has no readers and goes.

**C2. Search the arity as well.** `--arity` becomes a list, default `(2, 3, 4)`: one
`build_library` and one `fit` per arity, then choose across the combined curves. Measured cost
about 25 s, nearly all of it the arity-4 library build, against a 282 s run.

**C3. Two E3 equations, both derived.**

- **E3-short** -- the study's equation. The shortest (arity, length) whose paired interval
  against E3-best spans zero.
- **E3-best** -- the largest consensus over all (arity, length), whatever its size. It bounds
  what the additive form can reach and is reported in the docs rather than put forward.

`selection.pareto_knee` decides nothing and stays a diagnostic: `length_comparison`'s docstring
already records that every knee detector tried puts the bend at four to eight terms, and every
one of those lengths is significantly *worse* than the published equation fold by fold.

**C4. Drop `--terms`, keep `--max-terms`.** They are different things and only one is a knob:
`--max-terms` is the search horizon and a cost control; `--terms` was the assertion C1 removes.

**C5. Drop `--quick`.** It is a preset of flags that already exist individually
(`--max-terms 3 --pool 40 --penalty 20`), and it carries two Configuration objects,
`QUICK_E1` and `QUICK_E3`, whose only job is to hold a third copy of a length. The three tests
that use it (`test_plots` x2, `test_experiment` x3) pass the flags explicitly instead --
which also makes what "quick" meant visible at the call site rather than hidden in a constant.

**C6. The remaining knobs become named constants feeding argparse**, with a comment recording
that they are where the 2026-09 sweep landed: `PENALTY`, `MAX_ABS_ZSCORE`, `POOL_SIZE`,
`BEAM_WIDTH`, `MAX_TERMS`, `ARITIES`. No `DEFAULT_E1`, `DEFAULT_E2`, `DEFAULT_E3`,
`DEFAULT_E3_CAPABILITY`, `QUICK_E1`, `QUICK_E3`.

**C7. Rename so the code says what the study claims** -- and only after C1-C6 have landed, as
its own commit with no behaviour change. `fit.fit()` performs a *search* (a beam over term
subsets) and uses a ridge fit as its scoring function; calling the whole thing `fit` blurs the
form-versus-weights line the study is built on, which `validate.cross_validate_fixed_form`
names correctly and chapter 3 calls "term selection". Split into `search.py` (`guided_screen`,
`Subset`, `Selector`, `search()`, `SearchResult`) and `fit.py` (`Standardizer`, `ridge_solve`,
`to_equation`, `prune`). `search` imports from `fit` -- that direction is correct and the
docstring should say so rather than implying a clean layering.

**How each step is checked.** C1-C3 are *meant* to move the numbers, so the snapshot check does
not apply; the guard is the owner's constraint that E3-short may not land below the 15-term
result, **measured, not assumed** -- and the trap is that the paired rule's incumbent changes
from the 15-term equation to E3-best (~0.6855), where a wider interval can admit a shorter
equation scoring below 0.6381 while still reading "not significantly worse". C4-C7 change no
numbers and are verified by snapshot-and-compare over `results/`, `assets/docs/` and all
fourteen figures, which the PDF determinism fix now makes possible.

### The next piece: derive the equations, stop writing them down — 2026-09-08

The grid search is removed: **it is not useful**, because the only two things worth searching
fall out of the fit itself. `fit()` already returns an equation at *every* length in one pass,
so a search over length costs nothing, and a search over arity costs one fit each. Measured:

| arity | library | build | fit |
|---:|---:|---:|---:|
| 2 | 220 terms | 0.03 s | 0.36 s |
| 3 | 838 terms | 0.16 s | 0.59 s |
| 4 | 4,223 terms | 22.76 s | 1.60 s |

About 25 seconds for all three, nearly all of it the arity-4 library build, against a 282 s
run. A 30-core-hour cluster grid was buying a length and an arity that three local fits give.

**`DEFAULT_E1`, `DEFAULT_E2`, `DEFAULT_E3` and `DEFAULT_E3_CAPABILITY` go.** They wrote down
`headline_terms=15` and `=23`, which `selection.best_length` already derives from the consensus
curve -- the number was both computed and asserted, and if the two ever disagreed the study
would not notice. The fit hyperparameters stay as **named module constants** (`PENALTY`,
`MAX_ABS_ZSCORE`, `POOL_SIZE`, `BEAM_WIDTH`, `MAX_TERMS`) with a comment recording that they
are where the 2026-09 sweep landed; only the length and the arity are derived.

**Two E3 equations, both derived, over arity x length:**

- **E3-short** -- the interpretable one, and the one the study puts forward. The shortest
  (arity, length) whose paired interval against E3-best spans zero: the saddle where more
  terms and higher arity stop paying. This is what the 15-term arity-2 equation already is.
- **E3-best** -- the largest consensus over all (arity, length), regardless of size. It exists
  only to bound what the additive form can reach, and is reported in the docs rather than put
  forward.

**Two constraints on the rule, both from the owner and both already the codebase's position.**

1. **Not the Pareto knee.** `length_comparison`'s docstring records why: knee detectors,
   gRDP-smoothed knee detectors and the Pareto-front knee all put the bend at four to eight
   terms, and every one of those lengths is *significantly worse* than the published equation
   fold by fold. `selection.pareto_knee` stays a reported diagnostic and decides nothing.
2. **The short equation may not land below the 15-term result.** The paired rule has that
   floor built in -- "not significantly worse" -- but **the incumbent changes**, and that is
   the trap. Today the comparison is against the 15-term equation; in the new design it is
   against E3-best (~23 terms, arity 3, 0.6855), and a paired interval against a *stronger*
   incumbent is wider, so "not significantly worse than 0.6855" can admit a length whose
   absolute score is below 0.6381. **Measure what the rule returns before trusting it.** If it
   lands below the current 15-term equation the rule needs tightening, not accepting.

### Closing decisions — 2026-09-08, set by the owner

**The beam line closes as a negative and the branch closes with it.** Nothing measured beat
`beam.VANILLA` at the published operating point: the pruners and `cap-3` return the identical
fifteen terms, `cap-2` loses 0.203 of leave-one-dataset-out and 0.213 of leave-one-cell, ESS as
a continuous sampler inside the beam returns the identical equation at 3-7x the cost, and on
the configuration grid no sampler beats stratified random by more than noise. The published
configuration is unchanged, as it has been all along.

Two decisions follow, both the owner's:

- **Strip to `VANILLA`.** Every beam variant and the whole ESS apparatus come out of the tree:
  the fifteen extra policies, `seeding`, `beam_compare`, `beam_search_cli`, the two beam
  sbatch scripts, `scripts/grid_replay.py`, their tests, and the `diversity` extra. The numbers
  stay in this file and the code stays in `git log` -- that is what history is for, and ~1000
  lines the published pipeline never executes is not evidence, it is maintenance. **The strip
  must be provably a no-op**: snapshot `results/` and `assets/docs/`, remove in one step, re-run
  and `cmp` every output. Anything that moves is a bug in the removal, not a finding.
- **Documentation: this branch's six-chapter structure, regenerated here.** `origin/main`
  carries `811253c` from a collaborator -- the pipeline re-run on their machine (the report's
  header still reads `C:\Datos\IT projects\...`) plus prose updated to match, against the
  eleven-chapter pre-audit layout. Numbers come from the pipeline in this tree, never from
  anyone's laptop, so that commit is a **checklist of numbers to verify** rather than a merge.
  Main's `07-practices`, `08-limitations`, `09-model-effects` and `10-report` are read for
  hand-written prose the consolidation did not carry over, and that prose is ported.

### Job 15337 landed 2026-09-08 15:23, 15h03m. The beam line closes

`beam_search.csv` is 883 MB on the cluster and was **not** copied down; `beam_search_verdicts.csv`
is in `results/cluster/`. The compare stage printed "5 of 14 policies beat 'vanilla' on a paired
test", and that sentence is true and does not mean what it appears to.

**Every row of that verdict table is at nine terms.** `_best_configuration` ranks by
`OBJECTIVE_WEIGHTS`, whose `stability` and `brevity` both fall with length, so each policy's
sweep-best is a short equation and the incumbent it was paired against scores **0.5751** --
0.063 below the published equation's 0.6381. A0 is confirmed on the full grid, and A0 also
measured that these verdicts do not transfer: at fifteen terms `cap-3` returns the *identical*
fifteen terms and `cap-2` loses 0.203 of leave-one-dataset-out.

| where | verdict |
|---|---|
| sweep's objective-best (9 terms) | 5 of 14 policies better than vanilla, best 0.5921 |
| **published configuration (15 terms)** | **0 of 5 better; 2 of 5 catastrophic** |
| published equation | 0.6381 |

**The best-anywhere numbers are not evidence either.** Across all 48,576 points every policy,
vanilla included, has about 2,800 points above 0.6381 -- roughly 6% of the grid -- because the
published length comes from `selection.best_length` on the consensus curve and not from
maximising leave-one-dataset-out. That is the grid-truncation finding again, not a beam finding.
The per-policy maxima run 0.6809 (`prune-relative-0.005`) to 0.6925 (`diverse-0.1`) with vanilla
at 0.6855, ninth of fifteen -- but each is the maximum of 48,576 draws, and comparing maxima is
the unpaired difference-of-two-numbers this project has a standing rule against. No paired test
supports any of it.

**So the job answered its question and the answer is no.** The question was whether giving each
policy its own configuration sweep overturns the focused run's verdict. It does not: at the
operating point the study publishes, nothing beats the published beam.

### A0. The published-point check, run on 2026-09-08 — and it changes the question

**The whole beam comparison has been happening at 6-8 terms, not at the published 15.**
`_best_configuration` ranks by `OBJECTIVE_WEIGHTS`, where `stability` (0.15) and `brevity`
(0.05) both fall with length, so every policy's objective-best in the focused sweep landed at
6, 7 or 8 terms. The incumbent it was paired against is therefore *not* the published equation:
vanilla's focused-best scores LOO-dataset **0.5658**, which is the published length-curve's own
value at 6 terms (0.5659), against **0.6381** at 15. The comparison was sound; it was answering
"which beam finds the best short equation".

So the six policies were re-scored at `DEFAULT_E3` itself — 15 terms, the published operating
point. Four seconds, locally, `compare` with no sweep file on disk:

| policy | terms shared with the published equation | LOO-dataset | delta | MAE gain | verdict |
|---|---:|---:|---:|---:|---|
| `prune-relative-0.02` | **15 of 15** | 0.6381 | 0.0000 | 0.0000 | tie |
| `prune-relative-0.05` | **15 of 15** | 0.6381 | 0.0000 | 0.0000 | tie |
| `cap-3` | **15 of 15** | 0.6381 | 0.0000 | 0.0000 | tie |
| `cap-2` | 9 of 15 | **0.4355** | **-0.2026** | -0.0364 | tie |
| `cap-2+prune-0.02` | 9 of 15 | **0.4355** | **-0.2026** | -0.0364 | tie |

Three things follow, and they are the reason the beam line cannot be closed on the focused run:

- **At the published operating point the pruners are a no-op.** Not a small gain — the same
  fifteen terms, to four decimal places, on every protocol scored. The +0.0015 MAE win at six
  terms says nothing about depth fifteen, because what a pruner removes at step 6 of a beam is
  not what it removes at step 15.
- **`cap-2` destroys the equation at fifteen terms** (-0.2026 of LOO-dataset, 6 of its 15 terms
  gone) while the *same policy* was the largest positive dR2 in the focused run. A policy's
  verdict does not transfer across lengths in either direction.
- **The paired test calls that -0.2026 collapse a "tie"** (p = 0.115, 6/20 folds, CI
  [-0.101, +0.029]). That is the test behaving correctly on per-dataset MAE and it is a
  standing warning: on this comparison the paired test can under-call a catastrophic R2 loss,
  so the R2 delta has to be read beside it, never instead of it.

**The speedups do not survive either** — 1.02x, 0.88x, 1.04x, 1.01x, 0.86x at fifteen terms
against the 1.12-1.28x measured at six. But these are single 0.35-0.41 s runs, one measurement
each, so **no speed claim from this table is admissible without repeats.** Timing repeats are
a prerequisite for any cost verdict, here and in the sweep.

**Caveat on direction.** `DEFAULT_E3` was tuned under `beam.VANILLA`, so this test is biased
*toward* the incumbent: a challenger winning here would be conclusive, a challenger losing here
is not. Nothing won. But "identical equation, four decimal places, fifteen of fifteen terms" is
not a bias artefact — it is the pruner having nothing to do at this depth.

### A. Land job 15337 and make the verdict readable

Status at 09:53: 495,884 of 728,640 points at 836/min, 9h58m elapsed of a 24h limit.
Stage 1 ends ≈14:30, stage 2 (`compare`) a few minutes after. **Do not rsync `src/` to the
cluster until stage 2 has finished** — it launches a fresh Python from the same tree.

- **A1. Land it.** The sweep CSV will be ~730 MB at the focused file's ~1 KB/row, so gzip
  before copying: `ssh playstation 'gzip -k …/beam_search.csv'`, then scp both files.
  If stage 2 died, `compare` re-runs locally — it only reads `{name}.csv` off disk.
- **A2. DONE — report the frontier, not the label.** `beam_compare` records `seconds` and `speedup`
  but collapses `verdict` to better/tie/worse on accuracy alone, which discards the cost axis
  the corpus growth makes load-bearing. Report the Pareto frontier over (speedup, mae_gain)
  and let better-or-tie with speedup > 1 count as a candidate.
- **A3. DONE — score the compare stage on all four protocols.** `validate.cross_validate_doubly_held_out`
  exists and is 476 solves against LODO's 20 — prohibitive at 728,640 sweep points, negligible
  at compare's 15 configurations. Add `loo_cell_r2`, the ranking scores under it, and a column
  for the *spread* across the four. Offline; no re-running.
- **A4. Score every policy at *both* operating points.** A0 shows one is not enough: the
  sweep's objective-best (6-8 terms) and `DEFAULT_E3` (15 terms) disagree about the sign of
  every policy tried. `compare` already has both paths — it reads the sweep when the CSV is
  there and falls back to `DEFAULT_E3` when it is not — so this is a loop over two
  configurations per policy, not new machinery. Report them side by side and treat a policy
  that only wins at one length as not having won.
- **A5. DONE — repeat the timings.** `seconds` is one measurement of a 0.35-0.41 s run, and the
  speedups it produces (0.86x to 1.04x at fifteen terms) are inside its own noise. Best of
  five, or a fixed repeat count, before any cost verdict is quoted.
- **A6. `OBJECTIVE_WEIGHTS` is now a plan item, not a footnote** (it was open item 3). It does
  not merely shortlist oddly — `stability` and `brevity` both fall with length, so it moved the
  entire beam experiment to a length the study does not publish. Either re-weight it, or make
  `_best_configuration` rank within the published length band, or state the two-operating-point
  rule of A4 as the standing answer.
- **A7. Re-run `compare` locally** on the landed sweep with A2-A6 in place. That table, not the
  sweep's `objective` column, is the decision.

### A2, A3, A5 — done 2026-09-08, and the cell protocol changes one verdict

`beam_compare` now scores every run under all four protocols, reports the spread between them,
repeats its timings, and marks a Pareto frontier. Re-running the published-point check with
that in place (`results/cluster/beam_published_point_verdicts.csv`, 13 s locally):

| policy | in-sample | LODO | LOMO | **LOO-cell** | **spread** | speedup |
|---|---:|---:|---:|---:|---:|---:|
| vanilla = `prune-relative-0.02` = `-0.05` = `cap-3` | 0.6578 | 0.6381 | 0.6218 | 0.6162 | **0.0416** | 1.00 |
| `cap-2` = `cap-2+prune-0.02` | 0.6507 | 0.4355 | 0.6190 | **0.4028** | **0.2479** | 0.98 |

**The strictest protocol is where `cap-2` actually fails.** Its leave-one-*model* R2 barely
moves (-0.003) while leave-one-dataset drops 0.203 and leave-one-cell drops **0.213** — the six
terms it loses are dataset-side, so the damage only shows when the dataset is withheld. Judged
on LOMO alone it would have looked harmless. The spread column states the whole thing in one
number: 0.0416 for the published equation, **0.2479** for `cap-2`, a six-fold widening.

**That spread is also the study's headline claim as a single figure**, and it now costs nothing
to compute: 0.042 for E3 against 0.967 for the RandomForest (0.959 in-sample to -0.008 at
leave-one-cell). Worth generating into the chapters rather than leaving in a verdicts file.

**R2 and ranking disagree, which is why both are columns.** `cap-2` loses 0.213 of
leave-one-cell R2 and its ranking MAP under the same protocol goes *up* — 0.854 against the
published equation's 0.831. A shorter, worse-calibrated equation can still order the models
within a dataset correctly. The `candidate` rule is keyed on the worst-protocol R2, so it
refuses `cap-2` as it should, but a check on ranking alone would have promoted it.

**And the speed difference at fifteen terms is zero.** With best-of-five timings the speedups
are 0.978x to 1.002x — the 1.02x and 1.04x of the single-run table were the clock. Three of
those policies return the *identical* equation, so that band is a null measurement of this
machine, which is what set `SPEED_TOLERANCE = 0.05`: a `candidate` has to beat the incumbent by
more than five percent before its speed is called a result. Under the single-run numbers
`cap-3` was flagged a candidate at 1.002x; under repeats, nothing is.

What went in:

- `PROTOCOLS`, `PolicyRun.r2_loo_cell`, `.ranking_map` (MAP under each protocol), `.r2_spread`,
  `.r2_worst`, `.ranking_worst`. Ranking is scored under all four too — a policy that traded
  ranking for R2 would otherwise pass unseen.
- `TIMING_REPEATS = 5`, `PolicyRun.seconds` (best) and `.seconds_median` and `.repeats` beside
  it. `run_policy(..., repeats=1)` is the escape hatch for an expensive fit, and the run
  records the 1 so no speed claim is made from one sample by accident.
- `SPEED_TOLERANCE = 0.05`, and two columns in `compare_all`. **`standing`** is the one to
  read: `better` (wins the paired test, gives up nothing on the strictest protocol),
  `cheaper` (same accuracy, faster by more than the timing noise), `equal` (indistinguishable
  on both axes -- more machinery for nothing) or `worse` (loses the paired test, *or* buys its
  speed with the worst protocol). **`frontier`** is the cross-row Pareto view.

  The first version of this pair was two booleans and was not readable, which is a fair
  complaint and it also hid a real bug: the frontier was computed over the *challengers only*,
  so the least-bad of a set of losing policies came out `true`. Doing nothing is always an
  option, so the incumbent now enters the Pareto set at (0.0 gain, 1.0x), and speed is compared
  in units of `SPEED_TOLERANCE` so a 1.008x from the clock cannot dominate a 1.000x. At the
  published point the table now reads **0 better, 0 cheaper, 3 equal, 2 worse**, which is the
  finding in four words.
- Fifteen tests in `tests/test_beam_compare.py`, including the trade `standing` exists to
  refuse -- twice as fast, ties on leave-one-dataset-out, pays for it on leave-one-cell, which
  the paired test alone calls a tie -- and the incumbent's place in the frontier.

### B. Can pruning and capping be combined? — partly already answered

`cap-2+prune-0.02` is in `POLICIES` and in the running sweep. On the focused grid it does not
stack, in either direction:

| policy | verdict | MAE gain | 95% CI | wins | dR2 LODO | speed | terms |
|---|---|---:|---|---:|---:|---:|---:|
| `prune-relative-0.02` | better | +0.00145 | [+0.00055, +0.00238] | 16/20 | +0.0010 | 1.28x | 6 |
| `cap-2` | tie | +0.00489 | [-0.00313, +0.01808] | 10/20 | +0.0192 | 1.22x | 7 |
| `cap-2+prune-0.02` | tie | +0.00401 | [-0.00367, +0.01640] | 9/20 | +0.0181 | 1.25x | 7 |
| `cap-1` | tie | +0.00401 | [-0.00367, +0.01640] | 9/20 | +0.0181 | 1.22x | 7 |

**The combination is numerically identical to `cap-1` on every accuracy column** — pruning on
top of a cap of 2 removes the same children a cap of 1 would have, so the pair collapses to the
stricter cap and the pruner's consistency (16/20) is lost with it. Speed does not stack either:
1.28x and 1.22x alone give 1.25x together, because both cut the same children.

**At fifteen terms the collapse is total and in the other direction** (A0): `cap-2+prune-0.02`
is identical to `cap-2` — the same 9 of 15 terms, the same 0.4355, the same -0.2026 — while
`prune-relative-0.02` alone is identical to *vanilla*. So at the published depth the pair is
"whatever the cap does", and the cap does harm.

- **B1.** Confirm or overturn that on the full sweep, where the combination gets its own
  configuration sweep rather than sharing the focused grid's operating point — and at both
  operating points, per A4.
- **B2.** If it holds, the untested pairings are the *loose* ones — `cap-3+prune-0.05`,
  `prune-0.02+diverse-0.1` — where the two policies may cut different children. One focused
  job, ~40 min. If it does not hold, the full sweep already answers the question.
- **B3.** Whatever the answer, it has to be stated per length. A0 is the counter-example to any
  sentence of the form "policy X is better here": `cap-2` is the best dR2 at seven terms and
  the worst measured anywhere at fifteen.

### C. Re-sweep the shortlist on the correct evaluation protocol

- **C1.** Take the shortlist from A4: every policy that is better-or-tie with speedup > 1.
- **C2. Measure the per-point cost of the cell protocol first**, on one grid point, before
  sizing anything. `evaluate` currently spends most of a point in `fit` and `_fold_selections`,
  so 476 extra ridge solves may cost 2x rather than 10x — but sizing a cluster job on a guess
  is how the 7h estimate became 13h.
- **C3.** Focused sbatch over the shortlist only, scoring all four protocols and the ranking
  tests under each, every policy still getting its own configuration sweep. Roughly 759 points
  per policy on the focused grid, so a five-policy shortlist is ~3,800 points before the cell
  overhead.
- **C4.** The verdict then reads on what the study claims: R2 and ranking *stable across all
  four protocols*, with speed alongside.

### D. ESS, explored properly

**First, narrow the existing negative.** `ess-seed-6` / `-12` measured worse and 1.5-2.8x
slower, and TODO recorded that as "space-filling initialisation loses". It is not.
`seeding.empty_space` calls `ess.esa(embedded, bounds, n=limit-1, epochs=…)` **once**, with no
scores, no attractiveness field and no rounds, then snaps to the nearest unused term. That is a
one-shot seeder, and the negative is about the misuse, not about ESS.

What the two working implementations do — `~/git/Optuna_ESA_Sampler/ESSSampler.py` and
`~/git/pyBlindOpt/src/pyBlindOpt/init.py` (`oblesa`, `_ess_engine`):

- the anchor set **accumulates**: each round probes against everything placed so far, and the
  probes are *scored* and join the anchors, so the next field fit sees them "at the same
  standing as the sampler's own points";
- the field is guided by **attractiveness** — `attractiveness=-scores` (ESS's contract is
  higher-is-better and OBLESA minimises), with `attraction_weight`, `k_att` and a `cauchy`
  attraction metric so the pull reaches across the space instead of only nudging;
- `rounds=1` degenerates to exactly the single-pass pipeline, i.e. to what this repo does now.

- **D1. DONE — replay harness against the oracle table.** The landed sweep is 728,640 exhaustively
  scored grid points — a lookup oracle. Run the continuous ESS loop against it with every
  query answered by lookup and zero refits, and measure *evaluations to reach the grid's best*
  against random, LHS and Sobol baselines. This is the cheapest possible test of the claim that
  matters: **ESS is slower per draw but may buy back far more in evaluations it never has to
  run.** It costs no cluster time and it is the argument for the next, larger corpus where an
  exhaustive grid is unaffordable.
- **D2. DONE — port the attractiveness pattern.** The owner has approved importing both: they are
  small, both are on PyPI (`EmptySpaceSearch` 0.7.1, `pyBlindOpt` 0.5.0) and both are already in
  the local venv. Two mechanical consequences. **The floor has to move**: `pyproject.toml` pins
  `EmptySpaceSearch>=0.2.1` and the attractiveness API (`attractiveness`, `attraction_weight`,
  `k_att`, `att_power`, `attraction_metric`, `att_model`) is what 0.7.1 exposes — pin `>=0.7.1`
  the moment any of it is used. **Placement stays the `diversity` extra for now**, with
  `pyBlindOpt` added beside it, because the extra's stated reason still holds: the published
  equation is fitted with `beam.VANILLA` and reaches none of this, so a study run without the
  extra still reproduces it exactly. Promote both to base dependencies if and only if an ESS
  policy ends up in the published pipeline — at which point the same argument that made
  scikit-learn a base dependency applies.
- **D3. DONE (ahead of D1, since it needed no sweep) — ESS inside the search.** At each beam step the candidate children
  already carry scores, so the attractiveness field is free. ESS places probes in the empty
  regions of term-embedding space and they enter the beam beside the enumerated children —
  ESS as a continuous proposal over subsets, not a seeder. Measured as a new policy, through
  the same `compare` gate.
- **D4. Reframed — see the D1 section: successive halving first, TPE when the grid stops being affordable, and constraint search for the subset lattice. Originally: only if D3 pays.** A* and constraint search consume the
  same proposals at the same interface point.

### D2, D3 — done 2026-09-08. ESS used as a sampler finds the same equation, slower

**The one-shot negative is now scoped, and the real version is measured.**
`seeding.ProbeField` runs `ess.esa` the way `pyBlindOpt.init.oblesa` and the Optuna
`ESSSampler` run it: anchors accumulate, each carries the objective the search *measured*, and
the field is refitted every step. `beam.BeamPolicy` gained `probe_terms` (probes proposed per
step, 0 = off) and `probe_attraction` (pull towards what scored well, in ESS's units), and
`fit.Selector.search` takes a field, appends its proposals as children of the current best
subset, and tells it the objective of every child it evaluated.

Four policies are catalogued, including **`probe-4-attract-0.0`, the unguided null** — the same
probes placed by repulsion alone. Without it a win could be the extra candidates rather than
the guidance, which would credit ESS for the beam merely looking at more terms.

Measured at two operating points, four features, arity 2:

| policy | 15 terms: LODO / cell / terms shared / s | 6 terms: LODO / cell / shared / s |
|---|---|---|
| vanilla | 0.6381 / 0.6162 / — / **0.36** | 0.5659 / 0.5489 / — / **0.47** |
| `probe-4-attract-0.0` | 0.6381 / 0.6162 / 15 of 15 / 1.86 | 0.5659 / 0.5489 / 6 of 6 / 1.64 |
| `probe-4-attract-0.5` | 0.6381 / 0.6162 / 15 of 15 / 1.35 | 0.5659 / 0.5489 / 6 of 6 / 3.10 |
| `probe-4-attract-1.0` | 0.6381 / 0.6162 / 15 of 15 / 1.18 | 0.5659 / 0.5489 / 6 of 6 / 1.45 |
| `probe-8-attract-0.5` | 0.6381 / 0.6162 / 15 of 15 / 1.42 | 0.5659 / 0.5489 / 6 of 6 / 1.53 |

**The identical equation at both lengths, at 3-7x the cost.** Probes are *appended* and never
displace a child, so a probing beam evaluates a superset of what it would have evaluated
anyway — it can only add. On this corpus, at these two configurations, it added nothing. That
is the same shape as the standing negative that 8x compute converges to the fourth decimal:
**the beam is not the binding constraint, and now that has been tested with a sampler that was
actually being told what the search found.**

**What this does and does not close.** It closes "ESS inside the beam, on the published
grammar". It does not close **D1** — ESS over the *configuration* grid — and that is the one
worth doing, because it is where an ESS draw would replace an expensive evaluation rather than
adding one. Two features and arity 2 is also a small library; a corpus with more datasets and
a wider feature set is where a proposal mechanism would have room to matter, and re-running
this then is cheap now that the machinery exists.

One implementation note worth not rediscovering: **the anchor set is one entry per pool
position, keeping the best objective seen.** The first version appended every measurement, and
on the published configuration that is 4,142 anchors over a pool of ~600 — seven duplicate
coordinates per term. ESS reads anchors as occupied *space*, so duplicates are not extra
evidence, they are extra repulsion: a term the beam keeps returning to because it is useful
would end up pushing probes away hardest. Deduplicating also bounds the field's cost by the
pool rather than by the length of the search, and it silenced TORANN's index-tuning chatter,
which was going to the root logger and would have flooded a cluster job's output.

### D1 — settled 2026-09-08. No sampler beats random on the configuration grid

**Final numbers**, budget 2,000 draws of 48,576, 100 seeds, `validate.paired_comparison`,
against an oracle best of 0.6855 leave-one-dataset-out:

| | mean | vs random | 95% CI | wins |
|---|---:|---|---|---:|
| random | 0.6766 | — | — | — |
| Sobol | 0.6778 | tie +0.0011 | [-0.0003, +0.0025] | 53/100 |
| ESS w=1.0 | 0.6770 | tie +0.0004 | [-0.0009, +0.0018] | 53/100 |
| LHS | 0.6759 | tie -0.0007 | [-0.0022, +0.0008] | 43/100 |

**Nothing beats uniform random.** The only significant effect measured anywhere is Sobol over
LHS (+0.0013 to +0.0018 at budgets from 1,000 up), which is the expected ordering between two
low-discrepancy methods and does not lift either above the baseline. Power-of-two sample sizes,
which scipy's Sobol asks for, change nothing (+0.0011/+0.0012/+0.0009 at 32/64/128 per stratum).

**Why, measured rather than argued.** Over 32 strata x 20 seeds, Sobol samples with 3.7x lower
discrepancy than random -- it does its job -- but the **correlation between a sample's
discrepancy and the best score it finds is +0.02**. Low discrepancy is a guarantee about
integration error; finding this grid's optimum is a hitting problem on one isolated lattice
point (1 in 48,576, on a boundary at `k=28, z=5.00`). Coverage and hitting are different
problems and this grid only rewards the second.

**Three retractions got to that answer, and each is a lesson worth more than the result.**

1. **The first harness never called ESS.** Blocks of 8 over 32 strata meant a budget of 200
   spent every draw on the random cold start; at 1,000 only 93 of 125 blocks were guided. It
   reported "ESS does not help" having measured random sampling with extra steps -- the same
   one-shot-for-continuous substitution diagnosed in `seeding.empty_space` one section earlier.
   The tell was in the output: `ESS w=0` and `ESS w=0.5` printed byte-identical means *and*
   standard deviations. **`grid_replay.py` prints the ESS call count on every row** so it
   cannot recur silently.
2. **Nearest-level snapping halved the boundary sampling rate.** Rounding a continuous draw to
   the nearest lattice level gives an axis's two extreme levels a half-width cell and every
   interior level a full one: 0.200 against 0.333 on the six-level z-cap axis. Sobol and LHS hit
   an axis extreme 21.5 times per 62 draws against random's 36.8, and **every optimum on this
   grid is on an axis extreme**. That artifact alone produced four significant "worse than
   random" verdicts. Equal-probability binning fixed it exactly (37.4 against an expected 37.3).
3. **Twelve seeds could not resolve the effect.** Random's own mean moved 0.0038 between 12 and
   100 seeds, and every effect being measured is 0.001 to 0.005 -- so a 12-seed run reported
   ESS, Sobol *and* LHS all beating random, and an earlier one reported ESS beating random by
   +0.0045. Neither survived. **Match the seed count to the effect size, not to the slowest
   sampler**: the shared baseline is where power is free.

**The prediction that opened this was right in its strongest form.** The grid is 37% categorical
by variance (arity 29.4%, feature subset 6.8%) against 5.9% for the two continuous axes, with
`penalty` at 0.3%; what continuity exists is rugged (adjacent penalties differ by 69% as much as
random pairs); and the optima are on boundaries. A continuous space-filling sampler has almost
nothing here to act on, and measures as having none.

**The actionable finding is not about samplers at all: the grid is truncating the optimum.**
Both R2 argmaxes sit at `k = 28` and `z = 5.00`, the top of both axes. Widening those axes is
worth more than any search method, and it is a one-line change to `SCOPES`. Note this does not
touch the published equation, whose length comes from `selection.best_length` on the consensus
curve and not from this argmax -- but any statement about what the grid's best configuration is
currently stops where the grid does.

**Where a smarter search would actually pay** -- all of it indifferent to continuity, which is
what bounds ESS here:

- **Successive halving / Hyperband** is the strongest fit and the one to try next. A
  configuration's cost is dominated by the fit and its cross-validation, and the protocols form
  a natural ladder: rank cheaply on in-sample, promote survivors to leave-one-dataset-out, and
  only the finalists to leave-one-cell's 476 solves. That is a real cost win, it assumes nothing
  about the geometry, and it scales with the corpus.
- **Optuna's TPE** handles mixed categorical/integer/float spaces natively, which is the shape
  this grid actually has. Worth trying when the grid stops being affordable -- at 48,576 points
  and 2.2 s it still is.
- **Constraint or branch-and-bound over the feature subsets** is the only method that addresses
  the axis that will explode: subsets grow as 2^k in the optional columns, and 4 optional
  columns give 16 while 10 would give 1,024. It needs an admissible bound over the subset
  lattice, which is real work and may not exist.
- **A\*** is rejected on the same ground as OBL was: it needs an admissible heuristic on partial
  configurations, and inventing one would be adding a knob rather than testing an idea.
- **ESS stays on the list rather than off it.** It is the only sampler measured to beat random
  here, and the reason to reach for Sobol instead is simplicity rather than evidence. If a
  future grid is large enough that 2,000 draws is a small fraction of it -- which is the regime
  where the win appeared -- ESS is the one with a paired win behind it.

### The documentation audit against `origin/main`, done 2026-09-08

Main's eleven chapters were checked topic by topic against this branch's six. **The
consolidation lost almost nothing**, and where it did the replacement is better: main's
standalone `08-limitations` was distributed into a "Limitations of ..." section inside each of
the six chapters, which puts each threat beside the thing it threatens. `09-model-effects` is
absorbed into `04-equation`, and with *current* numbers -- the per-model slope at +0.0162, CI
[+0.0049, +0.0290], 14 folds, against the +0.017 the old chapter recorded.

Topic-by-topic, only four things in main have no home here:

| orphaned | where it belongs |
|---|---|
| "Ranking: a limitation that closed, and how" | `05-evaluation`, beside the ranking metrics |
| "Status: measured, and deliberately not part of the study" (the identity work's standing) | `04-equation`, which now carries the measurement without saying what it is *for* |
| **unadjusted p-values / the multiple-comparisons disclosure** | nowhere yet -- see below |
| a single consolidated threats-to-validity statement | an editorial call, see below |

**The multiple-comparisons disclosure is a real gap and not a porting problem** -- it is absent
from main too. The ordinal, the dummies, the axes, five embeddings, `Model Capability`, and every
beam and sampler variant in this file were all tried against the same 476 rows and the same
protocols, so **the surviving result comes out of a search and its p-values are unadjusted**.
That belongs in the text, and currently is not in either branch.

**And the consolidation has one cost worth weighing.** Seven "Limitations of ..." sections read
better in place, but a reviewer looking for threats to validity has no single section to open.
A short index in `index.md` linking the seven would keep the distribution and restore the
single entry point. Left to the owner.

### E. The documentation revision

Blocked on final results — the text can only be revised against numbers that will not move
again. Three parts, and the first is a surprise:

- **E1. Reconcile `origin/main`.** It carries one commit from a collaborator, `811253c`
  "Updated documentation" (2026-09-07), touching README, TODO, `02`, `06`, `07`, `08`, `10`
  and `cli.py`. It **does not merge cleanly** into this branch: the audit restructured and
  renamed those chapters, so `06`, `07`, `08` and `10` come back as modify/delete conflicts.
  `10-report.md` is *generated*, so hand edits to it are the thing the project's central rule
  forbids. This is a reconciliation, not housekeeping, and it belongs with the revision.
- **E2.** Regenerate everything (`python -m ml_meta_perf`), then read all the text end to end —
  not only chapter 6. Chapters 0 and 2 have still never had a continuous read.
- **E3.** Push every hand-written number into a generated block. Every stale figure found in
  the audit was in prose the pipeline does not write.

### Order

A0 is done. A2, A3, A5 and A6 are offline and can start now — they are what make the landing
table readable, and A4 needs them. A1 is gated only on the job. B1 falls out of A7 for free.
C needs A7's shortlist. D1 needs A1's table but nothing else, and is the one piece that can
start the moment the CSV is on disk; D2's dependency question is settled. E waits for C.

**The beam line cannot be closed early.** A0 was the fastest available test of "is
`prune-relative` simply better?" and the answer is no — it is a no-op at the published
operating point and its focused-grid win does not transfer. What the full sweep can still find
is a policy that prefers a *different* operating point, which is exactly the question A0 cannot
answer, so the job is worth its remaining four hours.

## The three results this session produced

**1. The model-descriptor question is reopened.** Item 0 under "Open". The identity ceiling is
+0.050, not the +0.017 the chapter recorded, and the *slope* rung survives a paired test
(CI [+0.0049, +0.0290], 14/20 folds; sign test p = 0.115, so magnitude rather than
consistency). Behavioural probing is the next step, not future work.

**2. The beam variants mostly do not pay** — focused job 15336, 17 min on 32 cores, verdicts in
`results/cluster/beam_search_focused_verdicts.csv`:

| policy | verdict | MAE gain | 95% CI | wins | ΔR² | speed |
|---|---|---:|---|---:|---:|---:|
| `prune-relative-0.02` / `-0.05` | **better** | +0.0015 | [+0.0005, +0.0024] | 16/20 | +0.0010 | 1.1-1.3x |
| `cap-2`, `cap-3` | tie | +0.0049 | [-0.0031, +0.0181] | 10/20 | +0.0192 | 1.2x |
| `diverse-1.0`, `ess-seed-6`, `ess-seed-12` | **worse** | -0.0011 | [-0.0018, -0.0005] | 4/20 | -0.0012 | 0.36-1.15x |
| `prune-relative-0.005` | **worse** | -0.0137 | [-0.0235, -0.0059] | 4/20 | -0.0354 | 1.4x |

Adaptive pruning is the only winner and wins **+0.001 of R²** — not a reason to change the
published configuration on accuracy, possibly one on **cost**, which is where the whole
surviving question now sits. `cap-2` has the largest ΔR² and is a **tie**: the exact shape
`paired_comparison` exists to catch. Diversity and ESS seeding are significantly *worse*, and
ESS is 1.5-2.8x slower. That matches the existing negative that 8x compute converges to the
fourth decimal: **the beam was never the binding constraint.**

**3. Under full leakage prevention, opaque models reach nothing.** The comparison now runs all
four protocols, and the ordering of the columns is the finding:

| model | in-sample | LOO-model | LOO-dataset | **LOO-cell** |
|---|---:|---:|---:|---:|
| RandomForest (300) | 0.959 | 0.598 | 0.080 | **-0.008** |
| GradientBoosting | 0.862 | 0.571 | 0.144 | **+0.009** |
| RidgeCV | 0.473 | 0.401 | -0.584 | **-0.618** |
| **E3 (15 terms)** | 0.658 | 0.622 | 0.638 | **0.616** |

**Leave-one-cell is the only like-for-like comparison in the study.** Leave-one-dataset-out
still hands a forest the held-out learner on nineteen other problems; leave-one-model-out still
hands it the held-out dataset. Only with both removed is it denied what the equation is denied
— and it is the protocol on which the trivial per-model baselines cannot be computed at all.
On the decision at threshold 0.7, all three opaque models sit at MCC ~0.27 against the
equation's 0.695 on the *same* protocol and 0.439 for a per-model mean on an *easier* one.

**This costs four minutes a run** (522 estimator refits: one per held-out group, then one per
observed cell). The equation half is still 14 s. If that becomes intolerable, the cell loop in
`opaque._doubly_held_out` is already threaded and the honest lever is fewer trees, not fewer
protocols.



## The beam-search machinery, for when the full job lands

`ml_meta_perf.beam` makes the beam's pruning, diversity and initialisation selectable, so the
variants are *measured* rather than argued about. Fifteen policies, from the beam-search
literature, mapped onto subset selection rather than sequence decoding:

- **Adaptive pruning** (Freitag & Al-Onaizan, WMT 2017) — drop a child too far from the step's
  best. Their *relative* form prunes at a ratio of the best score, which cannot transfer:
  `Subset.rss` is penalised and ridge shrinkage drives it negative, so a ratio test on a
  sign-changing quantity is meaningless. Both forms here are differences, the relative one as a
  fraction of the target's total sum of squares — the denominator R² uses.
- **Max candidates per history** (same paper) — cap how many children one parent may place in
  the beam. The hypothesis was that a redundant library lets one parent fill the width with
  near-spellings of one idea. Measured: **a tie**.
- **Determinantal selection** (Meister et al., EMNLP 2021) over a feature-overlap kernel.
  Measured: tie at low weight, **worse** at high.
- **Space-filling initialisation** — `ml_meta_perf.seeding`, using **ESS** (`ess.esa`) and
  **TORANN**, both an optional `diversity` extra; `empty_space` degrades to classical maximin
  without them, so a study run without them produces exactly the published equation. Measured:
  **worse, and slower**.

**Two stages, and the first is not the answer.** `ml-meta-perf-beam sweep` ranks by
`OBJECTIVE_WEIGHTS`, which has already put a nine-term equation on top that a paired test
called significantly worse. `compare` is the decision: each policy's best configuration refit
and paired against the incumbent on per-dataset MAE. **A tie is not a win.**

**Every policy gets its own configuration sweep, and that is why the full job is large.**
`DEFAULT_E3` was itself chosen under the vanilla beam, so scoring another beam there compares
a tuned setup with an untuned one. Measured at 2.2 s/point:

| script | scope | points | core-hours | wall |
|---|---|---:|---:|---:|
| `scripts/beam_search_focused.sbatch` | arity 2, the two load-bearing columns | 11,385 | ~7 | 17 min on 32 cores |
| `scripts/beam_search.sbatch` | the full grid × 15 policies | 728,640 | ~445 | ~7 h on 62 cores |

```bash
rsync -av --delete --exclude='__pycache__' --exclude='*.egg-info' \
  --relative ./src ./scripts ./pyproject.toml ./requirements.txt playstation:~/aiml-model/
ssh playstation 'cd ~/aiml-model && venv/bin/pip install -e ".[search,diversity]"'
ssh playstation 'cd ~/aiml-model && sbatch --job-name=beam --cpus-per-task=62 scripts/beam_search.sbatch'
```

**What is not attempted.** OBL proper — reflecting a candidate through the domain centre — has
no obvious meaning for a subset of terms, and inventing one would be adding a knob rather than
testing an idea. ESS is used only for the beam's starting points, and lost. Its more natural
home is the **configuration sweep itself**: that grid is 48,576 points over five axes, two of
them continuous, and a space-filling design over it is a direct use of what ESS is for. **That
is the thing to try next on this branch**, and it is a different claim from the one the focused
job just rejected — the beam variants lost on *accuracy*, whereas an ESS-designed sweep would
be a claim about *cost*, which is where the one surviving win (pruning, 1.1-1.3x) also sits.

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

## What the 2026-09-07 audit session changed, part two

**scikit-learn is now a base dependency and the no-scipy rule is retired.** The
opaque-regressor comparison — the priced other side of the study's whole trade — had been
measured once by hand and quoted ever since, and the hand-copied numbers had drifted.
`ml_meta_perf.opaque` computes it on every run: ridge, forest and boosting on the same
eighteen raw columns, under the same protocols, with the same clip. **The rule it replaced
still applies to `stats.py`** — four short statistics are not a reason to reach for a library,
and `validate.paired_comparison` stays scipy-free.

That comparison produced a **finding the study did not have**. The opaque models were only
ever scored on predicting the MCC *value*. Scored on the two decisions a practitioner
actually makes, they lose to the equation **and to the trivial per-model centres**: at the
0.7 threshold the equation reaches MCC 0.70 under the strictest protocol, a per-model mean
0.44, and the forest 0.33. On ranking they reach neither the equation nor the per-model
median. The honest summary is not that opaque models are bad at this — it is that **the
accuracy the study traded away was not there to be had** under a protocol where the dataset
is genuinely unseen. The run is now 38s, most of it refitting a forest 46 times.

**Chapter 6 was inverted.** It spent 270 lines on the equation — chapter 4's job — before
reaching a practice. The practices and their verdicts now come first, immediately after the
opening; the method that produces the evidence follows them. And each practice that makes a
claim about a raw feature is now checked **against the equation's own terms**: expected
direction against measured direction, with the effect size beside it, in
`guidance.equation_evidence`. That check found that the equation *disagrees* with half of
"capacity is not free" — it says `Processing Units Number` raises MCC — while agreeing on
`Model Capability`. A verdict from family means says the advice holds on this corpus; a
term-level agreement says the equation encodes it, and that is the stronger claim.

**Citations no longer point at arXiv where a published version exists.** All ten practice
sources and the two in chapter 4 were resolved; the related-work chapter went from sixteen
arXiv references to five, each now marked `Preprint` explicitly. Two errors surfaced:
arXiv:2405.09579 is **Golden**, not "Kaptanoglu et al.", and arXiv:2601.00428 — which looked
like a placeholder ID — is real.

**Chapters 4 and 5 read as repetitive because two generated sections were spliced into both.**
`term_choice` and the length note went to chapter 4 *and* chapter 5. Section ownership is now
one topic to one chapter: the length rule lives in chapter 3 with the selection procedure that
applies it, the ceilings in chapter 4 with the equation they bound, the protocols and
decisions in chapter 5. Four hand-written tables that duplicated generated ones were removed,
including one that was a stale copy of a table 240 lines below it in the same file.

Also: the 23-term capability equation is printed in full for the first time (it was quoted as
a ceiling and never shown); the two "Spearman is deliberately absent" paragraphs are one
sentence beside the metric it belongs to; and chapter 4's inverted claim about which protocol
E1 transfers better under is corrected.

## What the 2026-09-07 audit session changed

**One finding needs a decision and is not mine to make.** The per-model identity ceiling —
chapter 4's central number, the bound on what any model descriptor could add — was
hand-written and **wrong by a factor of three**: the chapter recorded +0.017 and recorded its
two rungs as *identical*, which no run of `identity.correct_out_of_fold` can produce. Measured
now, it is **+0.050 of leave-one-dataset-out R2**, of which the per-model *level* recovers
+0.020 and the *slope* the remaining +0.030. The chapter concluded from +0.017 that "the model
side is adequately described" and "the question is closed". **That conclusion does not follow
from +0.050**, and I have marked it open rather than rewriting the study's position for you.
The slope still adding more than the level is the part that matters: it is the interaction the
mixed terms were supposed to absorb and have not.

`identity` is now wired in — `experiment.identity_ceiling` reuses the finished
`cross_validate_fixed_form` path, so it costs no extra search and cannot drift again.

**Results the chapters stated by hand are now generated**, which is what let the above be
found at all. New generated sections: `1b. The corpus` (into ch. 1), `1c. The headline` (into
`index.md`), `3b. Why a subset rather than every term` (into ch. 3), `4b. The ceiling on model
descriptors` (into ch. 4), `5c. Reading a single prediction` (into ch. 6). Six chapters now
carry generated blocks, up from three.

Other stale numbers found and corrected, each of which a reader would have hit:

- **Chapter 6's worked example** was hand-written and every figure in it was stale — the
  intercept read 1.4787 against the equation's 1.3586 — in the chapter arguing that the
  analysis is generated rather than authored. Now `report.single_prediction`, on the row at
  the equation's median absolute error, with the sum printed and the clip called out when it
  fires.
- **Chapter 3's crater illustration** named 15 terms, which is the *published* length and has
  no crater. The deepest is 24. Now generated (`report._crater_note`), so it moves when the
  curve does.
- **Chapter 5's threshold table** was a stale hand copy of the generated table 240 lines below
  it in the same file. Removed.
- **Chapter 4's headline comparison** was a hand copy of `report.comparison`. Removed.
- **The equation's flatness** was stated three times in two chapters at three different values
  (13.2, 13.8, 13.8) against a generated 12.1, with the largest term's share at 10.8% against
  13.0%. All now point at the generated section.
- **Chapter 6's stability passage** claimed "four of the sixteen major terms" and put the
  rank-1 term at 10.6% of mass in 95% of folds. It is 13.0% in 100% of folds, 10 terms are
  major, and 10.6% belongs to a different term.
- **The README documented four defaults that had all moved** — 24 terms against 15, penalty 5
  against 20, arity 3 against 2, z-cap 3.0 against 4.25. Fixed, and
  `test_experiment.TestDocumentedDefaults` now reads the README table and compares it with
  `DEFAULT_E3`, so it cannot drift again. Also: 10 figures against 7, 17 tables, and a
  `--report report.md` flag that no longer exists.

**Chapters 2 and 3's historical tables are labelled, not deleted.** They support negative
results and are internally consistent; what they lacked was a statement that they predate the
2026-09-05 protocol change. A reader was free to compare a 0.5582 baseline against a current
0.658. Chapter 4's E2-aggregation table is labelled the same way and sits directly under an
E1 table that *is* current, which is how it went unnoticed.

**Performance: 15.7s to 14.0s, while adding two new analyses to the pipeline.** Four changes,
each verified to leave every table, chapter and equation byte-identical before the next was
made:

- `stats.pearson_columns` — `guided_screen` scored the pool one `pearson` call per column,
  60k calls a study, for one matrix-vector product.
- `Library.__init__`'s collinearity de-duplication was a Python loop of dot products over
  everything kept so far (590k iterations); it is one BLAS call per candidate, as
  `guided_screen` already was.
- `Selector._collinear` — the collinearity test decided once at construction instead of per
  parent per beam step.
- `_blocked` dropped an `np.unique` that could not change an `any`, and `_refine` prices its
  leave-one-out probes in one stacked solve.

What is left is intrinsic: `_evaluate_many`'s Python overhead and the argsorts in
`_descending`, whose stable tie-breaking is load-bearing and should not be traded for
`argpartition`. The one large structural win remaining is parallelising `fold_selections`
over its 20 independent folds — it is 10s of the 14s, and it is only used for `stability`.

**The gate had a hole.** `pyproject.toml` scopes ruff to `["src", "tests"]`; the hook and CI
ran `ruff check src`. basedpyright and vulture both read `tests`. Three lint errors had
accumulated there. Hook and workflow now run `ruff check src tests`.

**Figures** are numbered by position — `01_equation_comparison.png` through
`07_decision_quality.png` — from `figures.FIGURE_ORDER`, which is the only place the order is
written down; `generate` refuses a set that does not match it. `term_effects` labels are set
inline (`a / b`, not a built-up `\frac`, which mathtext shrinks so that half the labels were
two thirds the size of the other half), logarithms are parenthesised, and products use
`\times`.

**Nothing is pushed.** The CI fix in this branch is the thing that most needs pushing, and
the workflow only runs on a push to `main` or a pull request, so it has not been verified on
a runner — only in a clean local venv built exactly the way CI builds one.

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

## Documentation, as of this session

Six chapters plus related work, and nothing else. `07-limitations.md`, `08-appendix.md` and
`10-report.md` are **gone**: limitations close the chapter they concern, the appendix is
research record and lives in this file, and the generated results are spliced into chapters
4, 5 and 6 between `report.BEGIN`/`report.END` markers. Prose outside the markers is
hand-written and never touched; everything inside is rewritten on every run, so a chapter
cannot carry a stale table. `--report PATH` became `--docs DIR`.

**Keep agent notes out of the chapters.** They had accumulated changelog narration in
fourteen places — "an earlier draft said X, that was wrong" — and it was all removed. A paper
states what is true; the record of how it got there belongs in this file.

Seven figures, down from ten. `contribution_shares`, `error_curve_mae` and `per_group_quality`
were removed; `ranking_quality` gained the MCC cost in a second panel, `decision_quality`
became F1 across the four protocols, `term_effects` renders terms as mathematics
(`plots.term_to_math`), `equation_comparison` dropped the capability bar.

---

# Remaining work

Every item needs a *measurement*, not a judgement. The paired test is the tool for the ones
that compare two configurations.

## 1. The full sweep — done, and it confirms the published configuration

Ran 2026-09-07 as Slurm job **15335** on `playstation` (node `atari`, 62 cores): 48,576
points, 71 minutes, clean exit. `results/cluster/equation_search.csv` is the output, fetched
into the repo. **The published `DEFAULT_E3` is now what a search chose rather than a length
read off a curve**, and the standing "re-sweep whenever the grammar changes" debt is paid.

What it settled:

- **`max_arity = 2` confirmed outright.** Best arity-2 point 0.7297 against 0.6866 for the
  best arity-3 point, and every configuration in the top band is arity 2.
- **Penalty 15 and z-cap 4.25 stand.** No candidate beats them significantly. The nearest,
  penalty 20 / z-cap 4.50 on a four-feature subset, reached 0.6317 LOO-dataset against the
  then-incumbent's 0.6270 and was a **tie** when paired (p = 0.115) — and on all six features the same knobs score
  0.5798, so the apparent gain is the feature drop, not the shrinkage.
- **Sixteen terms stands.** See item 2.

Two traps it walked into, both worth keeping:

**The top row is not the answer, and this is the case the warning was written for.** The
sweep ranks a 9-term equation first (objective 0.7297 against the incumbent's 0.6861). Paired
over the twenty datasets on per-dataset MAE it is **significantly worse** — incumbent wins on
16 of 20, p = 0.012, CI [+0.0082, +0.0303] entirely above zero. The rule is *the shortest
configuration that is not significantly worse*, and nine terms does not qualify.

What drives the objective there is **`stability`, not brevity**: decomposed against
`OBJECTIVE_WEIGHTS`, stability contributes +0.0525 and brevity +0.0146 against -0.029 summed
over the five accuracy components. The 9-term form reselects in 88% of folds against the
incumbent's 53%. That is a real tension — a shorter form is more stable and transfers worse —
but stability is weighted 0.15 against 0.40 for the three R² combined and should not overturn
an accuracy gap this size. **Consider re-weighting `OBJECTIVE_WEIGHTS`, or treat the objective
as a shortlisting device and the paired test as the decision.** Currently the latter.

**The sweep prefers a four-feature pool for the equation, and that is admissible.** It drops
`Solution Stochasticity` and `Loss Margin Behaviour`, keeping `Model Capability`, `Processing
Units Number`, `Fitting Regime`, `Input Distribution Modelling`. An earlier note here called
that inadmissible on identification grounds; **that was wrong, and it confused the two
stages.** The corpus is designed for identification and keeps all six columns — that is what
`load` validates and what `tests/test_model_features.py` checks. The *equation* is judged on
compression, and needing fewer features as it improves is the mechanism working. Dropping
them from the term pool does not remove them from the corpus.

On every axis the four-feature pool dominates: best objective 0.7297 against six features'
0.7222, best LOO-dataset 0.6855 against 0.6716, best LOO-model 0.6543 against 0.6404.

**Re-running it.** The cluster copy at `~/aiml-model` is an rsync of the tree, not a clone, and
it was stale by three weeks when this ran — its `src/` predated the one-term-per-feature-
combination constraint, so it would have searched the wrong grammar. Re-sync first. The sbatch
script also had no `--mem-per-cpu`, so the 4G default x 62 cores asked for 248G against
`atari`'s 246G; Slurm accepted the job and scheduled it for **2027-02-13**. Measured peak RSS
is 216 MB and the script now asks 1G.

```bash
rsync -av --delete --exclude='__pycache__' --exclude='*.egg-info' \
  --relative ./src ./scripts ./pyproject.toml ./requirements.txt playstation:~/aiml-model/
ssh playstation 'cd ~/aiml-model && sbatch --job-name=eqsrch --cpus-per-task=62 scripts/equation_search.sbatch'
```

## 2. Term count — settled at 15 by a stated rule

Knee detection was **removed** and `kneeliverse` left the dependency list with it. It was
tried properly first: three detectors on four curves, raw and gRDP-smoothed at seven
tolerances, plus the Pareto-front knee by all three standard multi-criteria rules. The
smoothing works exactly as intended — detectors that split 6/8/4 on the raw curve agree at 8
after it — and 8 is *significantly worse* than the published length when paired over the
twenty datasets. Every geometric reading of this curve lands between 4 and 8, and all of them
are rejected by the paired test. A knee measures where the marginal return per term
collapses; it does not ask whether the accuracy still being added is real.

`selection.best_length` replaced it. Every alternative is still computed and reported —
`results/term_choice.csv` carries the Pareto readings, `results/length_choice.csv` the paired
verdict for every length — because a selection rule is only defensible if what it beats is on
the page. **Do not re-open this by quoting the knee**, and do not add `paretoset`: it pulls
pandas, numba and llvmlite for six lines that already exist in `selection.py`.

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

## 4. Closed this session

- **CI was red on `main`** — `pip install .` omitted the `search` extra while basedpyright
  type-checked `equation_search_cli`'s joblib import. Fixed, plus `fail-fast: false`.
  **Verified only locally**, in a clean venv built as CI builds one; needs a push or a PR.
- **The two TODO files** are merged into this one.
- **The full sweep** ran and confirmed the configuration (item 1).
- **The documentation** is the six sections that were asked for, with the generated results
  inside chapters 4, 5 and 6, and the agent notes stripped out.
- **Seven figures reworked or removed** against a detailed review; see the Documentation
  section above.
- **The ranking evaluation was scored in-sample** while the baselines it was compared with
  were leave-one-out. Fixed, and then superseded: both decision tasks are now reported with
  the dataset and the model held out together.
- **Error metrics were scored against the wrong baseline.** MAE is minimised by the median,
  so a mean baseline is not minimising the metric it is judged on. All four trivial
  predictors are reported at both centres; the per-dataset median is the tighter opponent on
  MAE (0.192 against 0.213) and SMAPE (40.3 against 43.8).
- **Large opposing weights** — closed by measurement. E3's largest standardised weight is
  0.137 against the 3.34 recorded under the pre-2026-09-05 protocol. No gate needed.
- **Collinearity below the equation** — measured, and the answer is to leave it. Tightening
  `COLLINEARITY_TOLERANCE` to 0.999 drops 6 of 270 terms and changes nothing to four decimal
  places; 0.99 drops 27 and costs 0.029 of LOO-dataset. **Never change it while a sweep is
  running** — it alters the library, and therefore the grammar being searched.
- **The additive oracle was checked** rather than assumed: the marginal-mean construction
  agrees with a least-squares two-way fit to 0.0003, so the 24 missing cells do not bias it.
  The +0.025 between the unclipped and clipped versions is the clip to MCC's range, which any
  MCC predictor should apply. It is a ceiling for a predictor additive in group effects, and
  E3-capability passing it is the expected consequence of having interaction terms.

## 5. Open, in the order it is worth picking up

0. **The model-descriptor question is reopened, and this is now the study's live problem.**
   The identity ceiling is +0.050, not the +0.017 the chapter recorded, and it was settled by
   the study's own paired test rather than by judgement: a per-model **level** is a tie, and a
   per-model **slope** is not — bootstrap interval [+0.0049, +0.0290], entirely above zero, on
   14 winning folds of 20. (Its sign test is p = 0.115, so the gain is carried by magnitude
   rather than by consistency; the generated section reports both.) The surviving half is
   exactly the one that lets a learner's advantage depend on the data, which is what a
   *capability* descriptor would do and what nothing in this corpus records. **Behavioural
   probing therefore stops being future work and becomes the next step** — see "When the
   meta-dataset can be recomputed", Group A, which is costed and specific.
1. **Open a PR from `wip/beam-search-op` and confirm CI is green.** The workflow triggers only
   on a push to `main` or a pull request, so **nothing has run on a runner yet**. That matters
   more than before: this branch made scikit-learn a base dependency and widened `ruff` to
   `src tests`, neither of which CI has seen, and the audit it carries began as a CI fix
   verified only in a local venv. The PR is also where `origin/main`'s `811253c` has to be
   reconciled — see E1 in the plan; it does not merge cleanly.
2. **Re-read chapters 0 and 2 end to end as a reader.** Chapters 1 and 3-6 had a continuous
   read on 2026-09-07 and their numbers were audited against the generated output; 0 and 2
   were not, beyond labelling chapter 2's historical tables.
3. **`OBJECTIVE_WEIGHTS` may want re-weighting — superseded by A6 in the plan, and worse than
   recorded here.** In the sweep, `stability` at 0.15 outvoted an accuracy gap that a paired
   test called significant, and put a 9-term equation on top.
   The objective is currently a shortlisting device with the paired test as the decision;
   either re-weight it or write that division of labour down as deliberate.
4. **The equation's own `stability` is low** — terms reselect in roughly a third of folds.
   That is the safeguard licensing the fixed-form protocol, so it deserves a number in the
   chapters rather than only in the generated tables.
5. **Chapter 6's practice catalogue is the highest-value open item now.** The verdicts are
   regenerated and each practice is paired with the terms carrying it, but **only 3 of the 10
   make a claim any term can answer** — the rest are about a protocol, a metric, or a family of
   learners. The ten were chosen before that pairing existed. A catalogue chosen *for* it,
   stating claims about quantities the corpus actually records, would raise coverage from 3/10
   and make the term-level check the chapter's contribution rather than a footnote to it.
   Concretely: practices about class imbalance, feature count, class entropy, and the
   instances-per-attribute ratio are all expressible as feature claims and all have terms in
   the equation to be checked against.
6. **The per-family effect table.** The one measured model-side description that transfers to
   an *unseen* model: +0.075 LOO-model, where per-model identity gives exactly 0.000. It is a
   table rather than an equation, so it fails the single-equation gate. Publish as a second
   component or keep as a ceiling? Numbers are pre-2026-09-05 and need re-measuring.
7. **The opaque comparison costs four minutes of the ~4.2-minute run**, nearly all of it the
   522 estimator refits the leave-one-cell protocol needs. `opaque._doubly_held_out` already
   threads the cell loop. If the runtime becomes a problem the honest lever is fewer trees
   (the conclusion does not depend on 300) rather than dropping a protocol — see the habit
   about strictest columns below.
8. **Correcting `Training Operations` for the 100k cap — blocked, not rejected.** Needs each
   trained instance's tuned hyperparameters; the upstream `results_stage_ml_eval.csv` covers
   348 of 476 rows and the 128 gaps are exactly the 8 GPU-trained models x 16 datasets.

---

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

## For the next session

`wip/beam-search-op` is pushed and is the only working branch. **The plan at the top of this
file is the current ordered list**; item 5 below is the older one it supersedes for the beam
work, item 0 is the one that changes what the study claims, and job 15337 is the one that
arrives on its own. Five working habits worth keeping:

- **Audit prose numbers against `results/` mechanically, not by reading.** Harvest every
  decimal from `results/*.csv|json` and the generated blocks, then flag every decimal in
  hand-written prose that does not appear at the precision it is written. That is how the
  identity ceiling, the crater at the wrong length, and three different values for one
  flatness index were all found in one pass. Exclude `results/cluster/` — 48k sweep rows
  match almost anything by chance and the filter goes useless.
- **Verify an optimisation by byte-comparing every output, one change at a time.** Snapshot
  `results/*` and the chapters, make one change, re-run, `cmp` everything. Four vectorisations
  went in that way and every one was provably identical; a batch of four would not have been.
- **When a number belongs in a chapter, generate it.** Every stale figure found this session
  was in prose the pipeline does not write. The rule is not "check the numbers", it is "give
  the chapter nowhere to keep a number of its own".
- **Score every predictor under every protocol, including the strictest.** The opaque
  comparison ran leave-one-dataset-out only, and read that way a forest looks merely weak. Run
  under leave-one-cell as well and it reaches zero — and that is the *only* protocol on which
  it and the equation are denied the same things. A comparison missing its strictest column is
  not conservative; it is flattering whichever side had more left over.
- **Pick the unit before building the check.** The practice-to-equation comparison was built
  three times — against features, against family-level row predictions, and finally against
  terms — and only the third says anything a reader could not get from a scatter plot or from
  chapter 5. The equation is a sum of terms, so the term is the unit; asking what a *feature*
  does averages over the positions it occupies, which is precisely the information that makes
  the equation worth reading.

Two habits that cost time in earlier sessions and are worth not repeating:

- **`git checkout -- assets/docs/` twice destroyed uncommitted work.** Commit before any
  bulk restructure, and prefer moving files to a scratch directory over reverting.
- **Splicing text with `s.index(a)` / `s.index(b)` silently duplicated a hundred lines** when
  the end marker occurred *before* the start marker. Assert `end > start`.
