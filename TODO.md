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

**The suite is green (540 tests, 44 seconds, no test over 8 s).** It was 35 minutes on
2026-09-08 and 2m35s after the first pass; the second pass is described below. Nothing is
half-done in the tree; what is unfinished is unstarted, and it is listed under "The plan" below.

### Read this first

**The selection rule was replaced on 2026-09-08 and is now free of the corpus size.** The
adjusted-consensus rule that priced a feature slot by `n` is gone; `best_configuration` is a
floor-curve argmax per grammar followed by a paired test across grammars. It still derives
(2, 15) and (3, 23), it still has no free parameter, and it can no longer flip its verdict as
the corpus grows. See "The selection rule" below for the design, what was rejected on the way,
**It is wired in** -- `experiment.search_grammars` calls it, and C1 through C3 are what
connected it.

### Where this stands

**C0 through C4 are done.** Nothing in the study asserts an equation's shape any more: the
length is derived per grammar by `selection.floor_argmax`, the grammar across them by
`selection.best_configuration`, and `experiment.search_grammars` runs both.

```
search_grammars(frame)  ->  E3-Valid = arity 2, 15 terms
                            E3-MAX   = arity 3, 23 terms
```

**C0 through C7 are done, and so is the documentation pass.** The repo-side work this branch
existed for is finished: the study is derived end to end, the suite is a 44-second gate, CI
runs, and `main` reproduces its own chapters byte-for-byte. **Everything still listed under
"Open" below is research**, not cleanup.

## The documentation pass, 2026-09-09

The chapters were written against a three-protocol study with a fixed arity and a hand-set
length. Every hand-written claim was checked against the code and the generated output.

**Four hand-written copies of generated tables, all stale.** This is the failure the
`generated:` markers exist to prevent, in the half of each chapter the markers do not cover.
Every one of them is now a pointer to the generated section beside it:

| chapter | copied table | how it was wrong |
|---|---|---|
| 5 | the three equations on every metric | three protocols of four; E1 and E2 at pre-C6 values |
| 5 | the trivial predictors at both centres | correct, and would have drifted the moment the corpus did |
| 5 | random-split leakage | 0.651/0.627/0.622 against the current 0.6388/0.6381/0.6218 |
| 3 | the alternative length rules | listed a Pareto front from a configuration two changes ago |

**Two functions each called themselves "the rule".** `selection.best_length` -- the argmax of
the three-protocol median -- said "**This is the rule that chooses the length**" in its
docstring, and `experiment.run_equation` had been calling `floor_argmax` instead. The
generated sentence in chapter 3, the figure caption for `02_term_count_curve`, and the
`term_choice` row label all named `best_length`. **Nothing caught it because the two agree**
-- both select 15 under arity 2 and 23 under arity 3. They are now labelled for what they are,
`recommend` reports both rows, and `test_selection` pins that exactly one row calls itself the
rule and that it is `floor_argmax`'s.

That agreement is a result and not a guarantee, and it is the sort of thing that should be
watched: if the two ever disagree, the chapters plot one and the study publishes the other.

**Chapter 2's arity argument was rewritten.** It concluded "`max_arity = 3` is the default
because it is the only setting that is not dominated", thirteen lines below a sentence saying
`DEFAULT` sets `max_arity=2`, and supported it with "in the published 20-term E3 the
three-feature `sum_ratio` accounts for 7 terms and 37% of the standardised weight mass" --
false of a 15-term arity-2 equation, where `sum_ratio` is not in the library at all. The
measurement survives as a statement about **E3-MAX**, where `sum_ratio` carries 10 of 23 terms
and 44% of the weight mass, and it makes a better point there: the search reaches for the
third arity whenever it is offered, and still cannot be shown to predict better than the
grammar that does without it.

**Chapter 3's Pareto paragraph had a wrong premise, not just wrong numbers.** It said every
length is on the in-sample front "because fit is monotone in terms". Fit is not monotone here:
the beam is a heuristic, and its best-at-24 is 0.670655 against best-at-23's 0.670709, so 24
is off the front. Corrected rather than dropped -- it is a small standing reminder that the
search is not exhaustive.

**Also corrected:** chapter 3's "the whole study runs in 27 seconds" (24 s for everything this
study fits; a full run is about four minutes, 215 s of which is the opaque comparison);
chapter 5's protocol table (three rows, now four, with the cell protocol named as the strictest
and as what the decisions are reported under); chapter 4's "under both protocols" section and
its E1 row; the README's `--arity` default (`2`, now `2 and 3` and searched);
`README`/`index`/`CLAUDE.md` module lists for the `search.py` split; and `CLAUDE.md`'s
references to a `10-report.md` that has not existed since the report was spliced into the
chapters.

**`test_experiment.TestDocumentedDefaults` gained a check for `--arity`**, which its regex
could never have covered: it publishes a set rather than a number, and it read `2` for as long
as the arity was fixed.

## The plan — C0 to C7

**C0. Done.** `results/cluster/` deleted (79 MB of artifacts from cluster jobs that no longer
exist and nothing can regenerate; their findings are recorded below with their numbers).
CI stays push-and-PR only and gets its first runner pass whenever this branch opens a PR --
deliberately not now.

**C1. Done, 2026-09-09.** `experiment.py`'s ``size = min(config.headline_terms, available)`` is
gone. `run_equation` now cross-validates the doubly-held-out protocol at every length as well
as the two single-group ones, builds the curve, and calls `selection.floor_argmax` on it;
`EquationReport.n_terms` carries the answer and all six read sites take it from there.
`Configuration.headline_terms` is removed, and `--terms` with it (that was C4, forced by this).

**The rule reproduces every asserted length, including two it was not written against:**

| | asserted | derived | in-sample | lodo | lomo | cell |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 7 | **7** | 0.3485 | 0.3411 | 0.3062 | 0.3049 |
| E2 | 6 | **6** | 0.2485 | 0.1849 | 0.2285 | 0.1724 |
| E3 | 15 | **15** | 0.6578 | 0.6381 | 0.6218 | 0.6162 |
| E3-capability | 23 | **23** | 0.7068 | 0.6781 | 0.6512 | 0.6495 |

E1's 7 and E2's 6 are the interesting rows. The selection rule was written knowing E3's 15 and
the bound's 23 -- that is the provenance the docstring discloses -- but E1's and E2's lengths
were set separately and earlier, by different reasoning, and `floor_argmax` recovers both. It
is not the corpus-with-an-unknown-answer the disclosure still asks for, being the same 476
rows, but it is two lengths the rule's author was not aiming at. **Record it in the chapter as
that and no more.**

**Nothing published moved.** Snapshot, re-run, byte-compare: every table, every equation and
all fourteen figures are identical; the three `curve_*.csv` gain `r2_loo_cell`, `mae_loo_cell`
and `smape_loo_cell` and their existing columns are byte-for-byte unchanged. `cross_validated`
now carries three protocols instead of two, for the same reason -- the strictest one is
computed anyway.

**One bug this introduced and the fix worth keeping.** Giving the boundary functions
``n_terms: int = 0`` let `ranking_baselines` and `decision_baselines` call
`doubly_held_out_predictions(frame, config)` unchanged, fall through the default, and publish
the **25-term** equation on the loo-cell row -- AP 0.790 against 0.831, MCC@0.7 0.660 against
0.695. It passed every test and was caught only by the byte-comparison. `n_terms` is now a
**required second positional parameter** on `leakage_demonstration`, `decision_quality` and
`doubly_held_out_predictions`. *A default length is the thing C1 deletes; it must not come back
as a parameter default.*

**C1 (superseded, kept for the reasoning).** `experiment.py:356` is the only line
that *decides*, and it throws away work already done: `fit()` returns an equation at every
length in one pass and `run_equation` cross-validates all of them, so the curve exists at that
point. Replace the assertion, carry the chosen size on `EquationReport`, and have the six read
sites take it from there -- 566/568 (protocol scores), 757/759 (decision report), 1038/1040
(baselines). `length_comparison` already derives and keeps `headline_terms` only as a fallback;
that fallback becomes `max(path)`. `Configuration.headline_terms` then has no readers and goes.

**C2. Done, 2026-09-09.** `experiment.search_grammars` fits E3 once per arity and lets
`best_configuration` pick; `ARITIES = (2, 3)` is the default and `--arity` is repeatable.
`run_e3_capability`, `DEFAULT_E3_CAPABILITY` and `QUICK_E3_CAPABILITY` are gone --
`Report.e3` and `Report.e3_capability` are now two entries from the same search.

```
arity  n_terms  complexity   floor   spread   gain   paired spread   ratio   role
    2       15          30  0.6162   0.0416 0.0008          0.0049  0.1718   E3-Valid
    3       23          69  0.6213   0.0539 0.0000          0.0000  0.0000   E3-MAX
```

That table is written as `results/grammars.csv` -- the rule's working, on the page, because a
selection rule is only defensible if what it beats is visible.

**The one number this moved, and it is the point of the change.** The capability bound used to
be fitted at penalty 3 while the published equation was fitted at penalty 20, so
"arity 3 reaches further" was a statement about two hyperparameter sets as much as about two
grammars -- and `best_configuration` *compares* them. Every grammar is now fitted under one
configuration with the arity the only difference. The bound is no longer allowed its own
shrinkage and its in-sample R2 falls **0.7068 to 0.6751** (lodo 0.6781 to 0.6439, lomo 0.6512
to 0.6327, cell 0.6495 to 0.6213). Its length is still 23 and it is still E3-MAX. **Anywhere
the old capability numbers are quoted in prose, they are now wrong.**

Verified otherwise unchanged by snapshot-and-byte-compare: `comparison.csv` is the *only*
altered file, every equation and all fourteen figures identical, plus the new `grammars.csv`.

**Equations are named for their role, not for the search.** `search_grammars` fits under
`E3-arity{a}` and renames to `E3` and `E3-capability` once the rule has spoken, so `e3.json`
keeps its identity. Without that the published equation's name changed to `E3-arity2_k15`,
which the byte-comparison caught and no test would have.

**Timing:** 14.0 s for the two-grammar search, against 9.7 s for the two hand-fixed runs it
replaced. Re-timed on 2026-09-08, the arity-4 build is **1.3 s, not 22.8 s** (the old number
was measured against the six-feature model pool; the equation's pool is four):

| arity | library | build | fit | 3 protocols + cell |
|---:|---:|---:|---:|---:|
| 2 | 220 terms | 0.02 s | 0.33 s | 4.1 s |
| 3 | 838 terms | 0.16 s | 0.59 s | 4.1 s |
| 4 | 4,223 terms | 1.28 s | 1.13 s | 4.2 s |

**Arity 4 is out of the default set** (owner, 2026-09-08), and the measurement supports it:
its candidate is **(4, 15)**, floor 0.6071 -- worse than arity 2's on every one of the four
protocols, complexity 60 against 30, never selected, and ~6.6 s a run to compute. It stays
available through the flag, because `max_arity = 4` is a recorded negative that a reader may
want to reproduce, and because a search that cannot be widened is not a search.

**`max_terms` is 25, down from 32** (2026-09-08). The horizon, not the published length --
`selection` picks 15 and 23, so 26 to 32 were only ever cost, and they are the worst-behaved
part of the curve: at arity 2 the leave-one-dataset-out figure craters to 0.480, 0.517, 0.537
and 0.420 at 28, 29, 30 and 32. Verified the repo's way, by re-running and byte-comparing
every output: `e1.json`, `e2.json`, `e3.json` and every table except three are **identical**;
`curve_e3.csv`, `pareto.csv` and `length_choice.csv` lose rows 26-32 and keep rows 1-25
byte-for-byte; one figure (`02_term_count_curve`) redraws with a shorter x-axis. Both
published lengths are unmoved. The dropped `length_choice` rows were all `tie` or `worse`.

**C3. Done, 2026-09-09.** E3-MAX was already absent from `ranking_baselines` and
`decision_baselines`, so the "only one is evaluated" half held. The half that did **not** hold
was subtler and is the reason C3 was worth doing rather than asserting:

**Nine downstream helpers took `config_e3`, and several of them rebuild the library and refit
from `config.max_arity` -- the *default* arity, not the one the search chose.** With the two
coinciding at 2 on this corpus it was invisible. Force them apart with `--arity 3` and the
study published an arity-3, 23-term equation while scoring its leave-one-cell row on an
**arity-2** refit: the strictest protocol measuring a different grammar from the published one.
`EquationReport.arity` now carries the searched grammar and `run` rebuilds `config_e3` around
it before any table is built.

`TestOnlyTheValidEquationIsEvaluated` pins both halves, including the arity mismatch --
`run(quick=True, arities=(2,))` against `QUICK_E3.max_arity == 3`, so the configuration and the
search disagree and the test would fail on the old code.

**Byte-identical output on this corpus**, because the two arities agree here. It is a
correctness fix for every corpus where they do not, and the corpus is going to grow.

**C4. Done with C1**, because removing `Configuration.headline_terms` left `--terms` nothing to
set. `--max-terms` stays: different thing -- the horizon is a knob and a cost control, the
length is what C1 derives. README's flag table and its worked example move with it.

**C5. Done, 2026-09-09.** `--quick` is gone from the command line, and `quick` from
`experiment.run`. It was a preset of three flags that already exist, plus two things they did
not reach -- E2's configuration and the opaque ensemble sizes -- which is exactly how it came
to advertise "seconds" while paying 226 s for the comparison. `run` now takes `config_e2`
alongside `config_e1` and `config_e3`, and `opaque_models` covers the rest, so what a cheap
call is asking for is readable at the call site. `tests/corpus.py` writes all three out.

Verified by snapshot-and-byte-compare: `results/` (27 files), all fourteen figures, and the
six chapters are identical before and after.

**Left where it was, deliberately:** `configurations` folds the shared flags onto E1 and E3
and not onto E2, so `--penalty 3` does not reach it. That is where the three separately-tuned
configurations left things rather than a decision, and C6 is what removes the asymmetry.

**C6. Done, 2026-09-09.** One `Configuration`, `experiment.DEFAULT`, built from six named
constants -- `PENALTY`, `MAX_ABS_ZSCORE`, `POOL_SIZE`, `BEAM_WIDTH`, `MAX_TERMS`, `ARITIES` --
and fitted by all three equations. `DEFAULT_E1`, `DEFAULT_E2` and `DEFAULT_E3` are gone, and
the constants are the argparse defaults, so `--help` states the tuned values instead of
`None`. `cli.configurations` (a pair) became `cli.configuration` (one).

The three sets were each defensible and together they broke the study's own claim -- that the
equations "differ only in which features they may use", which is what makes the gaps between
them evidence about the meta-data rather than about three tuning runs. E3 had a pool three
times E1's and a horizon three times its own published length, and none of that was a finding.
The shared values are E3's, because E3's are what a full sweep chose and the other two were
swept by hand over narrower grids.

**Measured, and this is the point of the step:**

| | before | after |
|---|---|---|
| E3-Valid / E3-MAX | (2, 15) and (3, 23) | **unchanged to four decimals** |
| E2 | 6 terms, in 0.2485 / lodo 0.1849 / lomo 0.2285 / cell 0.1724 | 6 terms, **0.2515 / 0.1931 / 0.2310 / 0.1813** |
| E1 | 7 terms at arity 3, 0.3485 / 0.3411 / 0.3062 / 0.3049 | 10 terms at arity 2, 0.3506 / 0.3383 / 0.3077 / 0.3034 |
| E1 fraction of own ceiling | 0.9848 | **0.9906** |
| E2 fraction of own ceiling | 0.8808 | **0.8915** |

E2's own tuning was costing it on every protocol. E1 moves by under 0.003 either way -- it is
within 0.006 of its structural ceiling of 0.354 both before and after, which is what "E1 is
done" means -- and its complexity `arity * n_terms` falls from 21 to 20. **Both controls got
closer to their own ceilings, which is the comparable quantity** (`index.md` says so, and
`CLAUDE.md` says so). Verified idempotent: two consecutive runs give identical chapters.

**Found while doing it, not fixed:** `assets/docs/02-additive-model.md` argues in hand-written
prose that "`max_arity = 3` is the default because it is the only setting that is not
dominated", citing "the published 20-term E3" -- against a line 13 above it saying `DEFAULT`
sets `max_arity=2`, and against an E3 that is 15 terms and whose arity is now *searched*
rather than set. The chapter contradicts itself and the code. It predates the arity search
(C2) and the derived length (C1); rewriting an argument a chapter makes is not a cleanup step
and wants its own pass. It is the same failure the generated-section markers exist to prevent,
in the half of the chapter the markers do not cover.

**C7. Done, 2026-09-09.** `fit.py` is split in two along the line the study already draws
between a form and its weights.

* **`search.py`** -- choosing which terms enter: `transform_gap`, `guided_screen`, `Selector`,
  `search()`, `SearchResult`, `prune`, `selected_terms`, and the beam constants.
* **`fit.py`** -- fitting weights to a form already chosen: `Standardizer`, `Subset`,
  `ridge_solve`, `to_equation`, `RIDGE_DEFAULT`.

`fit.fit()` became `search.search()` and `FitResult` became `SearchResult`. A function
performing a beam search called `fit`, in a module called `fit`, blurred exactly the
distinction `validate.cross_validate_fixed_form` exists to make and chapter 3 has always
called "term selection".

**Two placements differ from the plan above, and the callers decided both.**

`Selector` is in `search.py`, not `fit.py`. Its `_evaluate` is a ridge fit, but the class is
a Gram matrix, a collinearity matrix, a group-membership matrix and a cache, all built by one
constructor to be called several hundred thousand times by the beam. Splitting it would put
two halves of one constructor in two files to make a naming point. Nothing outside the search
wants it: `cross_validate_fixed_form` -- the protocol this whole distinction is *for* --
solves its own ridge inline and never touches `Selector`. Only `validate.term_stability`
does, and that runs a search per fold.

`prune` is in `search.py`, not `fit.py`. It drops terms and simplifies them, which are
decisions about *form*; the refit afterwards is the part it delegates. It also needs
`Selector`, so the planned placement was a cycle.

`tests/test_fit.py` split the same way into `test_fit.py` and `test_search.py`.
`TestToEquation` now solves its subset directly instead of running a beam to get one -- what
`to_equation` folds back out is indices and weights, and where they came from was never its
business.

Verified: `results/` (27 files), all fourteen figures and every generated chapter block are
byte-identical. The only chapter lines that moved are two hand-written module lists.

**How each step is checked.** C1-C3 and C6 are *meant* to move numbers, so for those the
snapshot check is a diff read line by line rather than a byte compare. C4, C5 and C7 change
nothing and are verified by byte-compare over `results/`, `assets/docs/` and all fourteen
figures -- which the PDF determinism fix makes possible. C5 came out identical on all three.

## The selection rule -- REVISED 2026-09-08, and now n-free

```
floor(a, k) = min of the FOUR protocol R2 at arity a, k terms
              in-sample, loo-dataset, loo-model, loo-cell
candidate(a) = argmax over k of floor(a, k)          one length per grammar
E3-MAX       = the candidate with the best floor
E3-Valid     = the candidate of least complexity (a*k) that E3-MAX does not beat
               by more than the spread of that beating:

               gain  = mean per-dataset MAE E3-MAX saves over the candidate
               scale = paired bootstrap SE of that same mean
               take the larger grammar only when  gain / scale > 1
```

`selection.grammar_margin` returns `(gain, scale, ratio)`. On this corpus the ratio for
(2, 15) against E3-MAX is **0.17** -- E3-MAX's advantage is a sixth of the spread of that
advantage -- so the smaller grammar stands, and would stand at any bar from 0.5 to 2.

On this corpus: candidates `{2: 15, 3: 23, 4: 15}`, **E3-MAX = (3, 23)**, **E3-Valid = (2, 15)**
-- the two equations the study publishes, still derived, and now derived without a row count.

**What the revision fixed.** The old rule was `argmax of consensus discounted by adjusted R2's
degrees-of-freedom factor against p = a*k`. It priced a feature slot by the corpus size, so
holding the curve, the folds and the plateau fixed and growing `n` from 476 to 5,000 flipped
its answer from (2, 15) to (3, 23). The table that measured that flip is gone with the rule.
`best_configuration` and `most_capable` now take no `n`, and
`test_no_corpus_size_enters_the_rule` pins it as a signature check -- the required invariance
(hold the curve, vary the hypothetical corpus size, the answer must not move) is satisfied by
construction rather than by measurement.

**Two design choices carry it, and both are arguable rather than obvious.**

*The floor, not the median.* `floor_curve` scores a length by its **worst** protocol, over four
rather than three. The weakest protocol here is always `r2_loo_cell` -- the one the study's
headline is about -- and a rule that selects on a median of three looser protocols is selecting
on a different quantity from the one it publishes. It also matters arithmetically: on the
median of four, the arity-2 curve peaks at **20 terms**, not 15. The min is what puts the peak
at 15, and the reason to prefer it is the one above, which has to be argued in the chapter.

*One candidate per grammar.* The paired test is asked about three points, never about ninety-six.
That is the whole difference from the rejected `shortest whose paired CI vs E3-MAX spans zero`,
which admitted nine terms at arity 4. Measured again this session against (3, 23) on per-dataset
MAE, leave-one-cell-out: (2, 6) and (2, 8) are **rejected**, (2, 9) onward are all admissible.
**A paired test over twenty groups cannot choose a length and is never asked to** -- length is
an argmax, the test only ever chooses a grammar.

**The first version of this rule used `paired_comparison(...).significant` and that was wrong,
caught 2026-09-08 by the owner: "the rule is not a logic statement or math value to overcome".**
Significance is a failure-to-reject. It is decided by the test's power rather than by anything
about the equations, and measured on this corpus it let **every** candidate through, so
`best_configuration` returned the first one it visited and complexity decided alone --
precisely the "the statistics were decoration" failure recorded against the rule before it.

| candidate | slots | significance form | ratio form |
|---|---:|---|---:|
| (2, 15) | 30 | not significant -> passes | **0.17** |
| (4, 15) | 60 | not significant -> passes | 0.15 |
| (3, 23) | 69 | compares with itself | 0.00 |

The ratio form is a comparison of two measured quantities and it discriminates where
significance did not. Against the same reference, on arity-2 lengths that are *not* candidates:
(2, 6) 2.29, (2, 8) 1.94, (2, 9) 1.80, (2, 12) 1.16 all clear the bar and are rejected, where
significance rejected only 6 and 8. (2, 10) 0.88, (2, 18) 0.59, (2, 20) 0.57 do not.

**The one in `ratio > 1` is where signal equals noise, not a tuned constant**, and the verdict
is not knife-edge on it -- 0.17 holds at 0.5, 1 and 2. **Where the bar sits against a sign
test, measured:** the ratio reads signal-to-noise on the *mean*, so concentration lowers it
without vetoing -- the same mean gain scores 1.02 carried by one fold of twenty, 1.50 by two,
2.23 by four, unbounded when every fold carries an equal share. That is looser than a sign
test, deliberately: the standing warning here is that the paired test *under*-calls, having
once scored a 0.203 collapse of leave-one-dataset-out R2 as a tie. Read the ratio beside
`floor_curve` and `protocol_spread`, never instead of them.

**Where the answer came from, stated correctly** (this paragraph said the opposite until
2026-09-10, and the owner corrected it): the equation was **found** by the Slurm sweeps --
job 15335, 48,576 configurations, and job 15337, 728,640 beam policies. It was not known in
advance and nothing was close to it. Those sweeps are the search this study performed, and
their result is the finding.

What `best_configuration` does is *state the criterion the sweeps established*, in a form that
runs in seconds and has no free parameter, no corpus size and no threshold. Deriving the same
(2, 15) and (3, 23) is the check that the rule captures the criterion -- a rule that disagreed
with the sweep would be the thing to worry about. That is a code-level property and it is what
C1-C3 wired in.

### Why (2, 15) and not (3, 23): consistency, measured

**Corrected framing, 2026-09-08 by the owner: 0.62 is not a threshold to hit.** The claim is
that the arity-2, 15-term equation clears 0.62 on the evaluation R2 *and holds up across every
leave-one-out form*, while the arity-3, 23-term bound reaches higher on a larger grammar and
**drops further** under lomo, lodo and loo-cell. Both halves are now measured, and both hold.

| | grammar | terms | slots | in-sample | lodo | lomo | loo-cell | floor | **spread** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E3-Valid** | arity 2 | 15 | 30 | 0.6578 | 0.6381 | 0.6218 | 0.6162 | 0.6162 | **0.0416** |
| E3-MAX | arity 3 | 23 | 69 | 0.6751 | 0.6439 | 0.6327 | 0.6213 | 0.6213 | 0.0539 |
| (the arity-4 candidate) | arity 4 | 15 | 60 | 0.6508 | 0.6294 | 0.6114 | 0.6071 | 0.6071 | 0.0438 |

`spread` is `in-sample - worst protocol`, and it is `selection.protocol_spread` -- **reported,
never selected on.** The three evaluation protocols at (2, 15) read 0.6578 / 0.6381 / 0.6218,
all above 0.62; the doubly-held-out cell reads 0.6162, and that is the strictest protocol in
the study rather than a miss.

**(2, 15) has the smallest spread of any configuration on the grid whose floor is competitive.**
Among every point within 0.01 of the best floor, the ranking is 0.0416 (2, 15), 0.0499 (3, 17),
0.0539 (3, 23), 0.0562 (3, 21), 0.0598 (3, 22), 0.0643 (3, 24). And within arity 2, among the
ten lengths clearing 0.62 on all three evaluation protocols, 15 has **both** the highest floor
and the smallest spread -- the next best spread is 0.0537 at k=18, a third worse.

So E3-MAX buys **+0.0051 of floor for +0.0123 of spread and +39 feature slots**, and the
+0.0051 is inside the fold-to-fold spread twice over: the paired test cannot separate the two
(mean +0.0008 of per-dataset MAE, 7/20, p = 0.263, CI [-0.0100, +0.0092]) and the unpaired
bootstrap standard error of a pooled cell R2 is **0.059**. That is the whole argument for the
smaller grammar, and it is now three measured quantities rather than a preference.

**The spread is deliberately not in the selection score.** Combining a level and a spread needs
a weight between them; a weight is a free parameter; a free parameter is exactly what this
revision removed. The two are reported side by side and the argument is made in the chapter.
`test_it_does_not_decide_anything` pins that the spread cannot flip the rule.

**Bootstrap standard errors, measured this session** (2,000 rounds, resampling held-out groups)
-- worth keeping, because every gap argued anywhere in this file is smaller than them:

| configuration | loo-dataset | loo-model | loo-cell |
|---|---:|---:|---:|
| (2, 15) | 0.056 | 0.045 | 0.059 |
| (3, 23) | 0.056 | 0.046 | 0.059 |

That is why "within one fold-to-fold standard error of the best" was **tried and dropped**: at
a scale of 0.056 the whole plateau from ten terms up is one band and the rule selects (2, 10)
or shorter, below the floor. Pairing is what makes the spread usable, because it removes the
between-dataset variance that dominates the unpaired number.

**A penalty re-sweep is available and was not taken.** At (2, 15) with penalty 0.3-0.5 the four
protocols read 0.6716 / 0.6460 / 0.6302 / 0.6202 -- better than penalty 20 on every one of
them, by 0.004 to 0.014. Full sweep at (2, 15), cell-loo R2: 0.1 -> 0.5847, 0.3 -> 0.6202,
0.5 -> 0.6201, 1 -> 0.6197, 2 -> 0.6180, 3 -> 0.6154, 5 -> 0.6000, 10 -> 0.6032, 15 -> 0.5927,
**20 -> 0.6162**, 30 -> 0.6014, 50 -> 0.2817. Every one of those gaps is an order of magnitude
inside the 0.059 standard error, so **if penalty 0.3 is adopted it has to be on the merits**,
re-swept jointly with z-cap and length under C6 -- not because it moves a protocol across a
round number. Job 15335 chose 20 and the note against an essentially unregularised ridge on
476 rows still applies. Left unchanged.

**What it replaced**, so it is not re-attempted: "the simplest grammar whose own best length is
not significantly worse than the best overall". Same answer, three conditions and a paired test
-- and that test *rejected nothing*: all three arities came out "not significantly worse", with
arity 4 at 26 terms winning 13 of 20 folds. Earlier attempts and why each failed:

| rule | picked | LOO-dataset | |
|---|---|---:|---|
| shortest whose paired CI vs E3-MAX spans zero | arity 4, 9 terms | 0.5813 | 0.057 below the floor |
| + sign test not against it | arity 4, 14 terms | 0.6148 | 0.023 below |
| + mean not negative | arity 3, 23 terms | 0.6439 | degenerates to E3-MAX |
| Pareto knee / gRDP knee | 4 to 8 terms | -- | every one significantly worse; already rejected 2026-09-07 |
| consensus adjusted by degrees of freedom | (2, 15) | 0.6381 | right answer, priced by `n`; replaced 2026-09-08 |
| within one fold-to-fold SE of the best | (2, 10) or shorter | 0.6151 | unpaired SE is 0.056; the plateau is one band |

**The floor that rules these out:** E3-Valid may not land below the 15-term result (0.6381).
`selection.pareto_knee` decides nothing and stays a reported diagnostic.

## The test suite: 35 minutes to 2m35s, 2026-09-08

**98% of the suite was one function.** `opaque.evaluate` is 226 s a call -- the leave-one-cell
protocol refits every estimator once per observed cell, 476 of them, at 300 trees and 100
stages -- and the suite called it six times. Profiled per test: 1,365 s of a 1,397 s attributed
run, with the other 517 tests taking **32 seconds between them**. The wall time was 2,076 s;
the gap is class fixtures, which a per-test timer does not attribute.

| test | before | after |
|---|---:|---:|
| `TestCli.test_regenerating_a_chapter_is_idempotent` | 455 s | 27 s |
| `TestFigureSet.test_generate_writes_the_whole_set` | 229 s | 16 s |
| `TestCli.test_main_writes_equations_tables_and_chapter_sections` | 227 s | 14 s |
| `TestCli.test_phase_selection_limits_what_is_printed` | 227 s | 14 s |
| `TestFigureNaming.test_every_figure_has_a_caption...` | 226 s | 14 s |
| `test_guidance.TestEquationEvidence` fixture | ~230 s | 17 s |
| whole suite, wall | **35 min** | **2 min 35 s** |

Three causes, all the same shape -- **the cheap path did not reach the expensive thing**:

* **`--quick` reached the equation's knobs and nothing else.** It now also sets the opaque
  ensemble sizes (`QUICK_TREES`, `QUICK_STAGES`) and the configurations for E2 and the
  capability bound, which were built from the *tuned* ones whatever `quick` said -- the
  capability bound alone was a full arity-3 search over an 838-term library at every wiring
  check. `run(quick=True)`: 229 s to 14 s. A real run is unchanged.
* **`test_opaque` was testing scikit-learn.** The estimators are not this project's to test;
  the plumbing around them is. `evaluate` now takes the estimator factories, and the plumbing
  tests pass doubles. **This is the standing rule going forward: unit tests exercise this
  project's functions, not the libraries they call.**
* **`test_guidance` called a full `run()`** for a fixture that reads `report.e3` and nothing
  else.

**The remaining floor is the cell protocol itself**: at five trees, `evaluate` is still 9.1 s,
spread evenly across the three estimators (1.8 / 3.5 / 2.1 s), so it is the 476 refits and
their joblib tasks rather than any one estimator. Fewer trees will not help further.

## The test suite: 2m35s to 44 seconds, 2026-09-09

The five end-to-end tests the previous pass left behind are gone, and with them the reason
anything in the suite was slow. Three changes, and `tests/corpus.py` is where all three land.

**A slice of the corpus, not the corpus.** Eight datasets by ten models, 78 of 476 rows,
derived from the shipped CSV at import rather than checked in as a second copy. `KPI-KQI`
with `MLP` and `XGBoost` keeps it ragged, so the doubly-held-out protocol is still exercised
against missing cells. A unit test asks whether a function builds its table correctly, and 78
rows answer that as well as 476 -- while `opaque.evaluate` refits once per observed cell, so
the row count is a multiplier on the most expensive thing in the package.

**The CLI is tested against a mocked engine.** `main` folds flags onto a configuration, runs
the study, and writes what came back; the first and third are the command line's and the
second is not. `tests/test_cli.py` patches `cli.run`, which made the three end-to-end tests
(13, 13 and 25 s) into nineteen tests in 3.4 s -- and sharper, because asserting on the
mock's call is how you check that `--arity 3 --arity 2` reaches the engine as `(3, 2)`. An
end-to-end run only ever sees what came out the far side.

**One published fit, shared.** `run_e3(load())` under `DEFAULT_E3` is 4.6 s and five modules
were each calling it for themselves, seven times in all. `corpus.published()` fits it once.

| test | before | after |
|---|---:|---:|
| `TestCli.test_regenerating_a_chapter_is_idempotent` | 25 s | 0.3 s |
| `TestCli.test_main_writes_...` | 13 s | 0.3 s |
| `TestCli.test_phase_selection_limits_what_is_printed` | 13 s | 1.7 s |
| `TestFigureSet.test_generate_writes_the_whole_set` | 15 s | 2.5 s |
| `TestFigureNaming.test_every_figure_has_a_caption...` | 13 s | 0.4 s |
| `test_guidance`, module | 30 s | 7 s |
| `test_experiment`, module | 93 s | 12 s |
| whole suite, wall | **2 min 35 s** | **44 s** |

**`run` gained `opaque_models`**, mirroring the parameter `opaque.evaluate` already had and
for the same reason: the opaque side is scikit-learn's, the leave-one-cell refit around it is
this project's, and a caller checking the wiring should be able to exercise the second without
paying for the first. The study's own output is unchanged -- all six chapters regenerate
byte-identically.

**Where a test belongs, stated once.** On the slice if it is about a function; on the real
corpus if its assertion is a claim about the meta-data. "Dataset features explain more than
model features" inverts on the slice (0.106 against 0.362), and a test that asserted it there
would be asserting nothing. `TestWhatTheCorpusSays` in `test_experiment.py` and
`TestTheStudysOpaqueClaim` in `test_opaque.py` are where those live, and they still fit at the
cheap configuration: reproducing the tuned search to check the *direction* of a gap would be
reproducing the study, which is what CI's end-to-end step is for.

**The reduced sizes are sound because the claim is size-independent**, measured:

| trees / stages | forest in-sample | loo-dataset | loo-cell |
|---:|---:|---:|---:|
| 5 / 5 | 0.933 | 0.031 | -0.161 |
| 30 / 30 | 0.957 | 0.028 | -0.061 |
| 300 / 100 | 0.959 | 0.080 | -0.008 |

The fit-versus-transfer gap is a property of the design matrix -- twenty dataset groups with
features constant inside a group, so a flexible model identifies the dataset and looks the
answer up -- not of how many trees vote on it.

### CI's end-to-end step is broken and has never run

`.github/workflows/main.yml`'s last step passes **`--report`, which is not a flag**;
`ml-meta-perf` exits 2 with "unrecognized arguments". It has never been caught because nothing
has ever run on a runner (see "Open", item 4). The working invocation is
`--quiet --no-report --output DIR --figures DIR`, or `--docs DIR` if the generated chapter
sections are wanted. **Fix this before opening the PR**, or the first runner pass fails on a
step that has nothing to do with the branch.

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

`experiment.DEFAULT` (one configuration for all three equations since C6): arity 2 to start,
z-cap 4.25, penalty 20, horizon 25 terms, four-feature model pool. E3-Valid comes out at arity
2 and 15 terms, E3-MAX at arity 3 and 23, both from `search_grammars` over the same object.

**The equation's model-feature pool is four; the corpus keeps six.** `MODEL_FEATURES` is the
corpus schema and identification is a corpus property; `EQUATION_MODEL_FEATURES` is what the
equation may build terms from, because fitting is judged on *compression*. Dropping
`Solution Stochasticity` and `Loss Margin Behaviour` from the term pool removes nothing from
the corpus. Do not re-run the identification argument against the equation's pool — that
mistake was made and corrected this session.

**And do not read absence from the equation as evidence against a feature either way.** The
framing is from `811253c` on `main` and it is the right one; the specifics there were measured
at 16 terms and are re-measured here against the published grammar's whole curve:
`Loss Margin Behaviour` is absent up to 15 terms and present at every length from 16 to 25,
`Solution Stochasticity` appears only at 25. The published equation is 15 terms, so neither is
in it — by one term.

**The length comes from a rule, not a constant.** `selection.floor_argmax`: fit at every
length, take the *minimum* of the four protocol R² per length, publish the argmax. No
threshold, no smoothing, no sensitivity parameter, and no corpus size. It selects 15 under
arity 2 and 23 under arity 3; neither number appears anywhere in the code.
`selection.best_length` — the median of the three single-group protocols — is the second
reading, reported beside it, and agrees on this corpus.

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

1. ~~**C1 through C7** -- the CLI and selection rebuild.~~ **Done, 2026-09-09**, and merged to
   `main` in PR #2 on 2026-09-10 with CI green on 3.12 and 3.14 for the first time.
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

3. ~~**The documentation replacement.**~~ **Done, 2026-09-09.** Six chapters, everything
   regenerated, four stale hand-written copies of generated tables replaced by pointers. See
   "The documentation pass" above.
4. ~~**Open a PR and let CI run.**~~ **Done, 2026-09-10** -- PR #2, green on 3.12 and 3.14.
   The workflow had been passing `--report`, a flag that never existed, so the reproduction
   step exited 2 on argparse every time and nothing had ever run on a runner to notice.
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
