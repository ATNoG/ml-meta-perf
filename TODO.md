# Working notes

Session closed 2026-09-07 on branch `fix/ci-docs-plots`. Gate green —
`venv/bin/pre-commit run --all-files`, 484 tests. The run is ~38s and byte-reproducible:
two consecutive runs produce identical `results/*` and identical chapters. Most of the
runtime is the opaque-regressor comparison, which refits a random forest 46 times; the
equation half of the pipeline is 14s, down from 15.7s.

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

0. **Decide what +0.050 means.** The identity ceiling is three times what the chapter said,
   and the chapter's "the model side is adequately described, the question is closed" does not
   follow from it. Chapter 4 now states the gap and marks the reading open. Either the study
   argues that +0.050 is small enough — in which case say so against the measured number and
   not the old one — or the model-descriptor question reopens, and the behavioural-probing
   direction below stops being future work and becomes the next step.
1. **Push, and confirm CI is green.** Nothing else in this branch has been seen by a runner,
   and CI now runs `ruff check src tests` rather than `src`.
2. **Re-read chapters 0 and 2 end to end as a reader.** Chapters 1 and 3-6 had a continuous
   read on 2026-09-07 and their numbers were audited against the generated output; 0 and 2
   were not, beyond labelling chapter 2's historical tables.
3. **`OBJECTIVE_WEIGHTS` may want re-weighting.** In the sweep, `stability` at 0.15 outvoted
   an accuracy gap that a paired test called significant, and put a 9-term equation on top.
   The objective is currently a shortlisting device with the paired test as the decision;
   either re-weight it or write that division of labour down as deliberate.
4. **The equation's own `stability` is low** — terms reselect in roughly a third of folds.
   That is the safeguard licensing the fixed-form protocol, so it deserves a number in the
   chapters rather than only in the generated tables.
5. **Chapter 6's practice catalogue has not been re-checked** against the 15-term equation.
   The verdicts are regenerated, but the ten practices were chosen when the equation had
   different terms; whether they are still the right ten is a judgement nobody has made.
6. **The per-family effect table.** The one measured model-side description that transfers to
   an *unseen* model: +0.075 LOO-model, where per-model identity gives exactly 0.000. It is a
   table rather than an equation, so it fails the single-equation gate. Publish as a second
   component or keep as a ceiling? Numbers are pre-2026-09-05 and need re-measuring.
7. **Correcting `Training Operations` for the 100k cap — blocked, not rejected.** Needs each
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

The branch is `fix/ci-docs-plots`, unpushed. Item 5 above is the ordered list, and item 0 is
the one that needs a person. Three working habits worth keeping:

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

Two habits that cost time in earlier sessions and are worth not repeating:

- **`git checkout -- assets/docs/` twice destroyed uncommitted work.** Commit before any
  bulk restructure, and prefer moving files to a scratch directory over reverting.
- **Splicing text with `s.index(a)` / `s.index(b)` silently duplicated a hundred lines** when
  the end marker occurred *before* the start marker. Assert `end > start`.
