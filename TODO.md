# Working notes

Session closed 2026-09-07. Branch `feature/descriptor-selection`. Working tree clean, gate
green — 416 tests, ruff, basedpyright, vulture, `venv/bin/pre-commit run --all-files`.

The exploration that produced this state has been deleted along with its research inputs;
what it found is in the two commit messages and in `assets/docs/`. This file is what is
*left to do*, plus the things that would be expensive to rediscover.

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

1. **Re-sweep the configuration under the new constraint.** Only *length* was re-derived,
   locally off the curve. Penalty, z-cap and arity are still the 2026-09-05 values, and the
   standing rule — re-sweep whenever the feature set or grammar changes, which has bitten
   three times — applies. `ml-meta-perf-search`, ~48k points, about an hour on 60 cores.
2. **Decide whether chapter 9 survives.** `identity` is wired into nothing and now backs a
   much smaller claim: the model side reaches 88% of its ceiling, not 58%, so the headroom
   a per-model table can recover is roughly +0.017. Fold into chapter 6 or keep.
3. **Collinearity below the equation is still unaddressed.** The constraint fixed the
   *equation*; the pool and the raw features are untouched. Of 270 library terms, 399 pairs
   correlate above 0.95, 13 above 0.999, one at 0.99998, because `COLLINEARITY_TOLERANCE`
   drops only exact duplicates at `1 - 1e-9`. Tightening it toward 0.999 would reclaim ~13
   pool slots; measure against a matched baseline before adopting. The raw-feature
   redundancy above is a corpus property and is documented rather than fixed.
4. **E3's worst leave-one-dataset-out fold is -10.62**, against -2.54 before the constraint,
   while the pooled figure moved 0.025. Worth understanding which fold and why; the
   dispersion columns matter more than they did.
5. **E3's model-only terms carry 0.047 of output variance while moving predicted MCC by
   0.066.** They largely cancel against the mixed terms. Not wrong — the shares are a
   covariance decomposition and sum to 1 — but chapter 6 should say something about it.

## Facts worth not rediscovering

**The protocol is fixed-form and there is no second one.** `cross_validate_fixed_form` fits
the equation once and refits only its weights per fold. `fold_selections` re-runs selection
inside the folds and returns **term names and no predictions**, so the re-selecting protocol
cannot produce a reported score; `term_stability` over it is the safeguard that licenses
fixing the form. `tests/test_model_features.py::TestReportedProtocol` pins this.

**Numbers from the two protocols differ by up to 0.3 and must never share a table.** The old
published 0.474 / 0.428 were re-selecting; anything current is fixed-form.

**Re-sweep the configuration whenever the feature set changes.** This has now bitten three
times. `DEFAULT_E3` tuned on the re-selecting metric scored 0.246 under fixed form.

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
ranking, term stability and brevity — and reproduces the shipped E3 exactly. A full grid is
48,576 points, about an hour on 60 cores. Needs the `search` extra.
