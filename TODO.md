# Working notes

Session closed 2026-09-07. Branch `feature/descriptor-selection`. Working tree clean, gate
green — 416 tests, ruff, basedpyright, vulture, `venv/bin/pre-commit run --all-files`.

The exploration that produced this state has been deleted along with its research inputs;
what it found is in the commit messages and in `assets/docs/`. This file is what is *left to
do*, plus the things that would be expensive to rediscover.

Six commits on this branch, oldest first: the protocol constraint (`a2159c3`), the paired
test and three verdicts that asserted retired columns (`af8fbe8`), stale docstrings and one
duplicated vocabulary (`d761ce4`), chapter 5's measurement and the chapters brought to the
equation (`e995bd1`), this handoff (`50a85cf`), the Slurm sweep spec (`5217cb7`). Each
message carries the numbers behind its change.

Two things gained a home in the library this session and are worth knowing about before
writing any new analysis: `validate.paired_comparison` (sign test plus bootstrap over
groups) and `validate.interaction_capture` (how much of the leading interaction pattern the
equation actually reaches — chapter 5 now rests on it).

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

## The decision that was open, and is now settled

**Is the study's claim at learner-*family* resolution or individual-*model* resolution?**
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

## Remaining work

Every item below needs a *measurement*, not a judgement. The paired test is the tool for
the ones that compare two configurations.

### 1. The full sweep, on Slurm — do this first

The constraint added on 2026-09-07 changed the grammar, and the standing rule is to
re-sweep whenever the grammar or the feature set changes. **Only `headline_terms` was
re-derived, and only locally off the curve.** Penalty, z-cap, arity and the feature subset
are still the values a sweep chose under the *previous* grammar, so the published
configuration is currently a local reading rather than a searched optimum.

```bash
sbatch --job-name=eqsrch --cpus-per-task=62 scripts/equation_search.sbatch
```

- **48,576 points**: 16 feature subsets x 11 penalties x 23 lengths (6-28) x 6 z-caps x 2
  arities. `equation_search_cli.feature_subsets` keeps `Model Capability` and
  `Processing Units Number` in every subset, since the ablation settled both.
- **~65 core-hours.** Measured at 4.85 s/point single-threaded under the new constraint, so
  about an hour wall-clock on 62 cores. The `--time=06:00:00` in the script is ample.
- Writes `results/cluster/equation_search.csv`, 25 columns, sorted by `objective`.
- The whole path was smoke-tested end to end after the constraint landed — `evaluate`,
  `as_row`, the joblib fan-out and the CSV write all run clean, and `Selector` gets
  `library.feature_groups` inside the folds, so the sweep searches under the same rule the
  study publishes under. It is a long job, not an untested one.

What it settles, and what to check in the output:

- whether `penalty=15`, `max_abs_zscore=4.25`, `max_arity=2` are still right. The z-cap and
  the penalty were both tuned when three now-retired columns were in the pool.
- whether 16 terms survives a real search. Locally 16 and 20 are indistinguishable on
  accuracy and 16 wins on brevity; `OBJECTIVE_WEIGHTS` puts them at 0.689 and 0.679, which
  is inside the same twenty-fold noise and should not be treated as decisive on its own.
- whether the six-feature set is still the one to publish. The sweep searches subsets, and
  a subset that scores as well with fewer columns would be worth knowing about — though
  note that dropping `Solution Stochasticity` or `Loss Margin Behaviour` costs
  *identification* regardless of what the objective says, and the objective cannot see that.

**Do not read the top row as the answer.** `objective` is a weighted sum over seven
components computed from the same twenty folds, so neighbouring rows are ties. Take the top
band, pair the candidates against the incumbent with `validate.paired_comparison`, and
prefer the shortest configuration that is not significantly worse.

### 2. Everything else

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
- **E3's worst leave-one-dataset-out fold is -10.62**, against -2.54 before the constraint,
  while the pooled figure moved 0.025. Find which fold and why. The dispersion columns matter
  more than they did, and a single fold at -10 is the kind of thing a reviewer finds first.

## Facts worth not rediscovering

**The protocol is fixed-form and there is no second one.** `cross_validate_fixed_form` fits
the equation once and refits only its weights per fold. `fold_selections` re-runs selection
inside the folds and returns **term names and no predictions**, so the re-selecting protocol
cannot produce a reported score; `term_stability` over it is the safeguard that licenses
fixing the form. `tests/test_model_features.py::TestReportedProtocol` pins this.

**Numbers from the two protocols differ by up to 0.3 and must never share a table.** The old
published 0.474 / 0.428 were re-selecting; anything current is fixed-form.

**Re-sweep the configuration whenever the feature set *or the grammar* changes.** This has
now bitten four times. `DEFAULT_E3` tuned on the re-selecting metric scored 0.246 under
fixed form. The fourth is open right now: the 2026-09-07 constraint changed the grammar and
only length has been re-derived — see Remaining work, item 1.

**Library construction is order-independent as of 2026-09-05** — `build_library` sorts both
feature groups. Before that, `A * B` and `B * A` entered as two names for one column and the
de-duplication kept different terms depending on argument order, moving leave-one-dataset-out
by up to 0.286. If a future change reintroduces order sensitivity, that is the cause.

**Measured and rejected — do not re-attempt:**

- **The sixty-one AI-proposed hyperparameter descriptors.** None is admissible. A
  hyperparameter a learner does not have has no value, and encoding that absence as zero
  collapses applicability into magnitude — seven of the best-scoring ten were zero on 72–99%
  of rows. Their use is *identity*, not magnitude, and one asserted ordinal replaces all of
  them there.
- **The four retired model columns.** Removing each of them *helps* leave-one-model-out;
  three varied within a model; two were zero-based so no log, root or reciprocal applied.
- **Widening the beam.** In-sample is identical to three decimal places across a 16× change
  in the candidate frontier. The search was never the binding constraint.
- **Spearman as a ranking metric here.** It sits between 0.63 and 0.73 for every predictor
  *and* every baseline, including a constant. Use AP, MRR, hit@1 and regret.
- **NDCG@3 and regret@3.** Saturated: 0.97–0.99 and 0.002–0.009 for everything, because 9.3
  of 25 models are tied at the top on average.

**E3's model-only terms carry almost none of its output variance, and that is expected.**
Currently 0.047 of the share across 2 terms, against 0.42 for the 4 dataset-only terms and
0.53 for the 10 mixed ones. It is not a defect and does not need fixing: a model feature on
its own can only shift every row of a dataset by the same amount, so the work it does is
*conditional* on the data and lands in the mixed terms by construction. That is precisely
why E3 beats E1 and E2 combined, and why the mixed block is the largest of the three. The
shares are a covariance decomposition and sum to 1, so a small model-only share means those
terms are correlated with the mixed block, not that they are idle. Do not re-open this as an
anomaly.

**Never read a difference of two means over twenty folds as a result.** This has now
produced three wrong conclusions in one session: that the equation out-ranked the trivial
baseline (paired, it wins on 7 of the 17 datasets that differ, every interval spanning
zero); that ranking quality falls with equation length (it is two datasets flipping their
top pick, and hit@1 can only move in steps of 0.05 on twenty folds); and that a stricter
form of the new constraint was catastrophic (one bad landing at one length). Use
`validate.paired_comparison` — exact sign test plus a bootstrap over groups, no scipy.

**The group-mean baselines need a leave-one-out correction.** With 17–25 rows per group,
including the row being predicted inflates the per-model mean's R² by 0.082.
`validate.baseline_group_mean` does it correctly; a naive group mean does not.

## Reproducing

```bash
venv/bin/python -m ml_meta_perf        # the whole study, ~5s
venv/bin/pre-commit run --all-files    # the gate
```

`ml-meta-perf-search` (`src/ml_meta_perf/equation_search_cli.py`,
`scripts/equation_search.sbatch`) is how the equation's features, length and configuration
were chosen. It scores seven weighted components — three R², the threshold decision, the
ranking, term stability and brevity. A full grid is 48,576 points, about an hour on 62
cores. Needs the `search` extra.

**It no longer reproduces the shipped E3, and that is the open item.** It did under the
pre-2026-09-07 grammar. The current configuration is a length read off a curve with the
other knobs inherited, so the sweep has to be re-run before anyone can say the published
equation is what a search chose. Remaining work, item 1.
