# Working notes

Session opened 2026-09-07 on `main`. This file is the single handoff: the untracked
`todo.md` that used to sit beside it has been folded in here and deleted. Everything below
is either *left to do* or *expensive to rediscover*.

Two things gained a home in the library on 2026-09-07 and are worth knowing about before
writing any new analysis: `validate.paired_comparison` (sign test plus bootstrap over
groups) and `validate.interaction_capture` (how much of the leading interaction pattern the
equation actually reaches — chapter 5 rests on it).

## Where the study stands

| | terms | in-sample | LOO-dataset | LOO-model |
|---|---:|---:|---:|---:|
| E1 — dataset features | 7 | 0.349 | 0.341 | 0.306 |
| E2 — model features | 6 | 0.248 | 0.185 | 0.228 |
| **E3 — both** | **16** | **0.665** | **0.627** | **0.622** |

`DEFAULT_E3`: arity 2, z-cap 4.25, penalty 15, 16 terms, 270-term library.

**The protocol gained a rule on 2026-09-07: one term per combination of raw features.**
`terms.Library.feature_groups` groups terms by their feature set and `fit.Selector` refuses
a second term from a group. It is a constraint on form, not on fit — the pairs it removed
sat at 0.891 and 0.786, inside `COLLINEARITY_LIMIT`, in a design conditioned at 7.8 — and
it is measured to cost nothing (paired over the 20 folds: p = 0.503, CI spanning zero).
E1 was unaffected; E2 lost 8 terms for 6 and scores below its old form on every metric.

`MODEL_FEATURES` is `Processing Units Number`, `Model Capability`, `Solution Stochasticity`,
`Loss Margin Behaviour`, `Input Distribution Modelling`, `Fitting Regime`. The corpus is one
file, `src/ml_meta_perf/meta_dataset.csv`: 2 identifiers, 12 dataset features, 6 model
features, `MCC`.

## Resolution: family or individual model? — settled

Both, at different stages, and the two stages have different criteria. Written up in
`data`'s module docstring and pinned by `tests/test_model_features.py`.

*Designing the corpus* requires **identification**: two rows sharing a feature vector are
two rows no equation over those features can ever distinguish. The corpus meets it — 20
distinct dataset vectors for 20 datasets, and 0 ambiguous rows of 476 on the model side.

*Fitting the equation* requires **compression**. An equation is a statement about families,
so it is expected to use fewer features as it improves. E3 uses 13 of 18.

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

## 1. The full sweep, on Slurm — running

Submitted 2026-09-07 as job **15334** on `playstation` (`cpuPartition`, 62 cores). The
cluster copy at `~/aiml-model` is an rsync of the tree, not a clone, and it was **stale by
three weeks** — its `src/` predated the one-term-per-feature-combination constraint, so a
sweep run there before the sync would have searched the wrong grammar. It has been synced;
re-sync `src/`, `scripts/` and `pyproject.toml` before any future submission.

```bash
rsync -av --delete --exclude='__pycache__' --exclude='*.egg-info' \
  --relative ./src ./scripts ./pyproject.toml ./requirements.txt playstation:~/aiml-model/
ssh playstation 'cd ~/aiml-model && sbatch --job-name=eqsrch --cpus-per-task=62 scripts/equation_search.sbatch'
```

- **48,576 points**: 16 feature subsets x 11 penalties x 23 lengths (6-28) x 6 z-caps x 2
  arities. Measured on the cluster at **5.66 s/point** single-threaded, so ~76 core-hours
  and about 1.2 h wall-clock on 62 cores. `--time=06:00:00` is ample.
- Writes `results/cluster/equation_search.csv`, 25 columns, sorted by `objective`.

What it settles: whether `penalty=15`, `max_abs_zscore=4.25`, `max_arity=2` and the
six-feature set survive a real search, and what the length curve looks like under the
current grammar. The z-cap and the penalty were both tuned when three now-retired columns
were in the pool.

**Do not read the top row as the answer.** `objective` is a weighted sum over seven
components computed from the same twenty folds, so neighbouring rows are ties. Take the top
band, pair the candidates against the incumbent with `validate.paired_comparison`, and
prefer the shortest configuration that is not significantly worse.

## 2. Term count: the reported knee is wrong, and wrongly sourced

**`results/term_choice.csv` reports a knee of 4 while the study publishes 16, and
`assets/figures/error_curve_mae.png` draws a line labelled "knee (4 terms)".** A reviewer
finds that contradiction immediately. Three separate defects sit under it:

- **It is detected on in-sample R² alone.** `selection.knee_terms` defaults to
  `r2_in_sample` and `selection.recommend` reports in-sample and loo-dataset separately.
  Choosing a published length on the fit alone is not defensible. **Decide what the detector
  runs on** — LOO-dataset, LOO-model, in-sample, or an aggregate of the three (sum, mean or
  median across protocols). The median is the robust choice on this data: at k=15 the three
  protocols read 0.659 / 0.393 / 0.616, and a median ignores the crater that item 3 explains.
- **The grid biases it.** `experiment.SWEEP_SIZES` is `(2, 4, 8, 12, 16, 20, 24, 26, 28, 32)`
  — non-uniform, and it **skips 13, 15, 17 and 31, which are exactly the four lengths where
  the LOO-dataset curve craters** (0.532, 0.393, 0.562, 0.539). The published curve therefore
  looks far smoother than the real one. The full path over every length 1..32 is already
  computed — `run_equation` subsamples it for E3 only — so a dense curve is free.
- **The detectors disagree and the docstring is stale.** On the dense curve, raw:
  autoelbow 6, kneedle 8, lmethod 4. RDP-smoothed at t=0.01: 4, 8, 8. `simplify_curve`'s
  docstring claims "run raw, the three report 12, 13 and 5 ... on the simplified curve they
  all report 12", which is from an older configuration and no longer true.

Nothing here can be settled until the sweep lands: the curve is a property of the
configuration, and the configuration is what the sweep is choosing. **Re-derive the knee on
the swept configuration, on a dense uniform grid, against an agreed aggregate.**

## 3. The LOO-dataset craters, diagnosed

Two different things, previously conflated as "E3's worst fold is -10.62".

**The negative per-fold R² is a metric artefact, not a fit failure.** `5G_Slicing` has a
within-dataset MCC standard deviation of **0.0516** — every one of its 25 models scores about
0.986. R² divides by that variance, so a fold whose target is nearly constant returns a large
negative number for an almost-perfect prediction. `NSR` (0.0551) and `DeepSlice` (0.0639) are
the same, and they are the only strongly negative folds. **Report per-fold MAE, not per-fold
R², and say why.** Per-fold R² is not meaningful on a group with no variance to explain.

**The pooled craters are real, and they are extrapolation.** At k=15 the held-out
`ASNM-CDX-2009` fold is predicted at **-2.41** — raw pooled LOO-dataset R² **-2.06** — and
`validate._clip_to_training` pins the whole fold to the training floor of -0.29, recovering
0.393. The design is *well* conditioned there (cond 10, max |w| 0.14), so this is not a
numerical failure: a held-out dataset can sit outside the convex hull of the other 19 in term
space, and a linear equation extrapolates without limit. k=13, 17 and 31 crater for the same
reason with milder magnitudes (raw and clipped agree to ~0.005 there, so those are genuine
fold sensitivity rather than clipping).

**The clip is load-bearing everywhere, not just in the pathological case.** Between **50 and
78 of 476 predictions** hit the training-range bound at *every* length. That is 10-16% of the
reported LOO-dataset predictions being decided by a clip rather than by the equation, and the
chapters do not currently say so. Either justify it prominently or bound extrapolation some
other way — but do not leave it undisclosed.

## 4. Documentation — restructure as a paper

`assets/docs/` currently reads as a research log of additive-modelling pitfalls. It should
read as a paper. Target structure, in order:

1. **Dataset** — the corpus, the features used, and what each one means.
2. **Methodology** — the additive model (a linear model over terms); what a term is.
3. **Model generation** — the optimisation that filters and selects terms under the grammar;
   term-count selection by knee detection (item 2).
4. **The equation** — presented and discussed for interpretability; linked forward to (6),
   since best practices are matched onto its terms. Plus the heuristic ceiling implied by the
   grammar and by raw-feature correlation with MCC.
5. **Evaluation** — R² for E1/E2/E3; MAE and SMAPE against the median-of-means baseline;
   binary and rank-based evaluation against mean/median baselines.
6. **Best-practice evaluation** — published ML practices matched against E3's terms.

Keep related work and limitations as bookends; they are standard and they are where the
negative results belong (see *What the negatives are worth* below). **The report is
generated, never narrated** — that constraint is unchanged, and a chapter that copies a
generated table goes stale silently.

## 5. README

Substantially out of step with the work. Bring it to the current equation, the current
protocol and the restructured chapters.

## 6. Plots

Specific defects, each independent of the configuration:

- `error_curve_mae.png` — marks "knee (4 terms)" against a published 16 (item 2).
- `term_count_curve.png` / `equation_comparison.png` — E3 (0.665) is drawn above the
  "additive ceiling" (0.661) with no explanation. That is not an error: `additive_oracle` is
  the ceiling for a model that is *additive in dataset and model effects*, and E3 carries
  mixed interaction terms, so exceeding it is the headline result. **Relabel and say so.**
- `per_group_quality.png` — plots Spearman and top-1 regret on one shared x-axis. They are
  not commensurable, and this file already records that **Spearman is useless here** (0.63 to
  0.73 for every predictor *and* every baseline, including a constant). Use AP, MRR, hit@1
  and regret.
- `ranking_quality.png` — AP and reciprocal-rank bars overlap rather than group; the "best
  model ranked first" stars sit outside the axis at x≈1.02.
- `practice_effects.png` — the legend advertises strong/moderate/weak in grey, and the bars
  are coloured by sign. The legend describes nothing on the plot.
- `term_effects.png` — term labels are truncated mid-name ("[log(gravity)] / [log(Processing
  Units Number…"), and only 12 of 16 terms are shown.
- `predicted_vs_actual.png` — the rug at y=0 reads as a row of data points at zero. The one
  negative row (-0.29) is off-axis.

## 7. Everything else

- **Decide whether chapter 9 survives.** `identity` is wired into nothing and now backs a
  much smaller claim: the model side reaches 88% of its ceiling, not 58%, so the headroom a
  per-model table can recover is +0.016, measured: leave-one-dataset-out goes 0.627 to 0.643
  with `identity.correct_out_of_fold` applied. Fold into chapter 6 or keep.
- **Collinearity below the equation is still unaddressed.** The constraint fixed the
  *equation*; the pool and the raw features are untouched. Of 270 library terms, 399 pairs
  correlate above 0.95, 13 above 0.999, one at 0.99998, because `COLLINEARITY_TOLERANCE`
  drops only exact duplicates at `1 - 1e-9`. Tightening it toward 0.999 would reclaim ~13
  pool slots; measure against a matched baseline before adopting. The raw-feature redundancy
  is a corpus property and is documented rather than fixed.
- **Large opposing weights, never investigated.** Under the *previous* protocol the largest
  standardised weight was **3.34** at penalty 5 and 1.05 at penalty 20. A standardised weight
  of 3.34 moves the prediction by ±3.34 target-scales across ±1 sd of its own term, which
  MCC's range cannot absorb — so it is being cancelled by another near-collinear term. An
  equation whose terms cancel at that magnitude is harder to reason about term by term, which
  is the one property this study cannot compromise. `fit.is_admissible` guards the extreme
  case (a near-constant term once produced a weight of -1.5e9 against an intercept of
  +1.5e9); nothing guards the moderate case. **Re-measure under the current protocol** — the
  one-term-per-feature-combination rule may already have fixed it — and if it has not, decide
  whether a weight-magnitude gate belongs in `Selector`.
- **The per-family effect table.** The one measured model-side description that transfers to
  an *unseen* model: **+0.075 LOO-model**, where per-model identity gives exactly 0.000 by
  construction (a held-out model has no training row). It is still a table rather than an
  equation, so it fails the single-equation gate. **Open question for the author:** publish it
  as a second component alongside E3, or keep it as a ceiling measurement the way chapter 9
  keeps its own? Numbers are pre-2026-09-05 and would need re-measuring either way.
- **Correcting `Training Operations` for the 100k cap — blocked, not rejected.** Every model
  trained on a stratified sample capped at 100,000 rows and ten of twenty datasets exceed it,
  but the column was computed from the *source* `nr_inst`, so above the cap it carries a
  dataset-size signal corresponding to no training run. Recomputing needs each trained
  instance's tuned hyperparameters; the committed upstream `results_stage_ml_eval.csv` covers
  **348 of 476 rows**, and the 128 gaps are exactly the 8 GPU-trained models x 16 datasets.
  Approximating it would put a hand-made formula inside the study's only cost column.

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
now bitten four times. `DEFAULT_E3` tuned on the re-selecting metric scored 0.246 under
fixed form. The fourth is the sweep now running.

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
venv/bin/python -m ml_meta_perf        # the whole study, ~5s
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
