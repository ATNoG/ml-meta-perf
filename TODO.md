# Working notes

Session opened 2026-09-07 on `main`. This file is the single handoff: the untracked
`todo.md` that used to sit beside it has been folded in here and deleted. Everything below
is either *left to do* or *expensive to rediscover*.

Two things gained a home in the library on 2026-09-07 and are worth knowing about before
writing any new analysis: `validate.paired_comparison` (sign test plus bootstrap over
groups) and `validate.interaction_capture` (how much of the leading interaction pattern the
equation actually reaches — chapter 5 rests on it).

## Where the study stands

| | grammar | terms | features used | in-sample | LOO-dataset | LOO-model |
|---|---|---:|---:|---:|---:|---:|
| E1 — dataset features | arity 3 | 7 | | 0.349 | 0.341 | 0.306 |
| E2 — model features | arity 2 | 6 | | 0.248 | 0.185 | 0.228 |
| **E3 — both (published)** | **arity 2** | **15** | **12 of 16** | **0.658** | **0.638** | **0.622** |
| E3 — capability | arity 3 | 23 | | 0.707 | 0.678 | 0.651 |

`DEFAULT_E3`: arity 2, z-cap 4.25, penalty 20, 15 terms, four-feature model pool.
`DEFAULT_E3_CAPABILITY`: arity 3, z-cap 4.25, penalty 3, 23 terms, same pool.

**The equation's model-feature pool is four, and the corpus keeps six.** `MODEL_FEATURES` is
the corpus schema and identification is a corpus property; `EQUATION_MODEL_FEATURES` is what
the equation may build terms from, because fitting is judged on *compression*. Dropping
`Solution Stochasticity` and `Loss Margin Behaviour` from the term pool removes nothing from
the corpus. The sweep chose the subset and it dominates the full six on every axis.

**The length comes from a rule, not a constant.** `selection.best_length` is the argmax of the
consensus curve — no threshold, no smoothing, no sensitivity parameter — and it selects 15
under arity 2 and 23 under arity 3. Neither number is written down anywhere.

**Knee detection was removed on 2026-09-07**, and `kneeliverse` left the dependency list with
it. It was tried properly first: three detectors on four curves, raw and gRDP-smoothed at
seven tolerances, plus the Pareto-front knee by three standard rules. The smoothing works —
detectors that split 6/8/4 raw agree at 8 after it — and 8 is *significantly worse* than 15
when paired over the twenty datasets. The alternatives are all still computed and reported
(`results/term_choice.csv`, `results/length_choice.csv`); a selection rule is only defensible
if what it beats is on the page.

**The protocol gained a rule on 2026-09-07: one term per combination of raw features.**
`terms.Library.feature_groups` groups terms by their feature set and `fit.Selector` refuses
a second term from a group. It is a constraint on form, not on fit, and it is measured to cost
nothing (paired over the 20 folds: p = 0.503, CI spanning zero).

The corpus is one file, `src/ml_meta_perf/meta_dataset.csv`: 2 identifiers, 12 dataset
features, 6 model features, `MCC`.

## Resolution: family or individual model? — settled

Both, at different stages, and the two stages have different criteria. Written up in
`data`'s module docstring and pinned by `tests/test_model_features.py`.

*Designing the corpus* requires **identification**: two rows sharing a feature vector are
two rows no equation over those features can ever distinguish. The corpus meets it — 20
distinct dataset vectors for 20 datasets, and 0 ambiguous rows of 476 on the model side.

*Fitting the equation* requires **compression**. An equation is a statement about families,
so it is expected to use fewer features as it improves. E3 uses 12 of the 16 it may draw on.

So `Solution Stochasticity` and `Loss Margin Behaviour` stay, earning their place at the
first stage: without them 134 of 476 rows stop being identifiable. Do not resurrect the
argument from their absence in the fit — and note that absence is not even robust. At 16
terms neither appears; at 12, 20 and 24 the fit uses one or both. It is a property of one
length.

Two caveats to carry with the identification claim. It is **joint** on the model side: five
of the six columns are constant per model and separate only 19 of 25 learners alone
(`FT-Transformer`/`TabNet`/`TabTransformer`, `LightGBM_RF`/`XGBoost`, `DNN`/`MLP`,
`TabICL`/`TabPFN`, `BernoulliNB`/`GaussianNB` collide); `Processing Units Number`, which
varies with the dataset, breaks the ties. And it was bought with redundancy — `nr_attr` and
`nr_outliers` correlate at 0.9995, and `log(inst_to_attr) + log(nr_attr)` **is**
`log(nr_inst)` to 2e-15.

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

## 2. Term count — settled at 16, on a paired test rather than a knee

The machinery was fixed first, and both fixes were needed before any answer meant anything:

- **The detector no longer runs on in-sample R² alone.** In-sample is monotone in the number
  of terms, so it can only ever say "more". `selection.consensus_curve` combines every
  protocol per length and `knee_terms` runs on that; the **median** is the default because it
  is robust to the craters item 3 explains — at 15 terms the three protocols read
  0.659 / 0.393 / 0.616 and the median ignores the crater. `recommend` still reports each
  protocol separately so disagreement stays visible.
- **The curve is reported at every length.** The old `(2, 4, 8, 12, 16, 20, 24, 26, 28, 32)`
  skipped 13, 15, 17 and 31 — exactly the four craters — and the detector was partly reporting
  the grid: 4 on the ragged grid, 6 on the dense one. `SWEEP_SIZES` is now `None`; the beam
  search already builds the whole path.

**All four detectors now agree at 6 terms**, where they previously split 4 / 8. That agreement
is the evidence the grid was the problem.

**Six is not the published length, and neither is the sweep's nine.** Both are significantly
worse than sixteen when paired over the twenty held-out datasets. The published equation is
justified by that test, not by a knee detector and not by the objective's top row — and the
chapters now say so. **Do not re-open this by quoting the knee.**

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

## 4. Closed since 2026-09-07

- **CI was red on `main`.** `pip install .` omitted the `search` extra while basedpyright
  type-checked all of `src/`, so `equation_search_cli`'s joblib import failed on a runner but
  not locally. Fixed, plus `fail-fast: false` so a 3.14 failure stops cancelling 3.12.
- **The two TODO files** are merged into this one.
- **Seven figures that misread.** See the commit; the knee label, the "additive ceiling" that
  E3 legitimately passes, Spearman sharing an axis with regret, a confidence legend where
  every bar had the same opacity, truncated term names, a rug that read as data, and
  overlapping bar series.
- **The documentation** is restructured as a paper, chapters 0–8 plus the generated 10.
- **The README** carried two wrong claims, not merely stale ones: that the equation captures
  *none* of the leading interaction component (it reaches about a third, 0.31 in-sample) and
  that the model features reach 96% of their ceiling (0.248/0.282 is 88%).
- **Large opposing weights** — closed by measurement. E3's largest standardised weight is
  **0.137**, against the 3.34 recorded under the pre-2026-09-05 protocol. The fixed-form
  protocol and the one-term-per-feature-combination rule removed whatever was producing
  cancelling near-collinear pairs. No weight-magnitude gate is needed.
- **Collinearity below the equation** — measured, and the answer is to leave it. Tightening
  `COLLINEARITY_TOLERANCE` from `1 - 1e-9` to 0.999 drops 6 of 270 terms and changes
  in-sample, LOO-dataset and LOO-model by **nothing to four decimal places**. Tightening to
  0.99 drops 27 terms and *costs* 0.029 of LOO-dataset (0.627 to 0.598). The pool was never
  the binding constraint, which agrees with the beam-width result. **Do not change it while a
  sweep is running** — it alters the library, and therefore the grammar being searched.
- **Error metrics were scored against the wrong baseline.** MAE is minimised by the median,
  so a mean baseline is not minimising the metric it is judged on. All four trivial
  predictors are now reported at both centres, and it matters: the per-dataset median is the
  tighter opponent on MAE (0.192 against 0.213) and SMAPE (40.3 against 43.8).
- **The ranking and threshold decisions now have real rivals**, not just a majority-class
  floor. The result is worth knowing: on ranking the equation is **indistinguishable** from
  ordering models by how well they usually do, against either centre (AP paired p = 0.63 vs
  the mean, p = 0.33 vs the median, both intervals spanning zero) — and note the median
  baseline's *mean* AP is the higher one, which is exactly the trap. On the threshold
  decision it wins clearly: MCC 0.683 against 0.439 and 0.304 at a threshold of 0.7.

## 5. Still open

- **Chapter 9's fate is decided** — folded into chapter 7 as the bound on what better model
  descriptors could buy, which is what it measures. `identity` is still wired into nothing.
  Remaining question: the headroom a per-model table recovers is +0.016 (LOO-dataset 0.627 to
  0.643 with `identity.correct_out_of_fold`). Publish or keep as a ceiling?
- **The per-family effect table.** The one measured model-side description that transfers to
  an *unseen* model: **+0.075 LOO-model**, where per-model identity gives exactly 0.000 by
  construction. It is a table rather than an equation, so it fails the single-equation gate.
  **Open question for the author:** publish as a second component alongside E3, or keep as a
  ceiling? Numbers are pre-2026-09-05 and need re-measuring either way.
- **Correcting `Training Operations` for the 100k cap — blocked, not rejected.** Every model
  trained on a stratified sample capped at 100,000 rows and ten of twenty datasets exceed it,
  but the column was computed from the *source* `nr_inst`. Recomputing needs each trained
  instance's tuned hyperparameters; the upstream `results_stage_ml_eval.csv` covers **348 of
  476 rows**, and the 128 gaps are exactly the 8 GPU-trained models x 16 datasets.

---

# Facts worth not rediscovering

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
venv/bin/python -m ml_meta_perf        # the whole study, ~15s
venv/bin/pre-commit run --all-files    # the gate
```

`ml-meta-perf-search` (`src/ml_meta_perf/equation_search_cli.py`,
`scripts/equation_search.sbatch`) is how the equation's features, length and configuration
were chosen. It scores seven weighted components — three R², the threshold decision, the
ranking, term stability and brevity. Needs the `search` extra.

**It does not currently reproduce the shipped E3, and that is what job 15334 is for.** It did
under the pre-2026-09-07 grammar. The shipped configuration is a length read off a curve with
the other knobs inherited, so the sweep has to land before anyone can say the published
equation is what a search chose.
