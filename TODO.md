# Working notes

Session closed 2026-09-05. Branch `feature/descriptor-selection`, two commits pushed:
`3fc5c27` (code) and `3a147cb` (study and documentation). Working tree clean, gate green —
389 tests, ruff, basedpyright, vulture, `venv/bin/pre-commit run --all-files`.

The exploration that produced this state has been deleted along with its research inputs;
what it found is in the two commit messages and in `assets/docs/`. This file is what is
*left to do*, plus the things that would be expensive to rediscover.

## Where the study stands

| | terms | in-sample | LOO-dataset | LOO-model |
|---|---:|---:|---:|---:|
| E1 — dataset features | 7 | 0.349 | 0.341 | 0.306 |
| E2 — model features | 8 | 0.271 | 0.203 | 0.242 |
| **E3 — both** | **16** | **0.672** | **0.652** | **0.633** |

`DEFAULT_E3`: arity 2, z-cap 4.25, penalty 15, 16 terms, 270-term library.

`MODEL_FEATURES` is `Processing Units Number`, `Model Capability`, `Solution Stochasticity`,
`Loss Margin Behaviour`, `Input Distribution Modelling`, `Fitting Regime`. The corpus is one
file, `src/ml_meta_perf/meta_dataset.csv`: 2 identifiers, 12 dataset features, 6 model
features, `MCC`.

## The one decision that is genuinely open

**Is the study's claim at learner-*family* resolution or individual-*model* resolution?**

It decides whether two of the six model features stay:

- At **family** resolution the set is complete. `Model Capability` alone separates all ten
  families with zero ambiguity, and the equation is finished as it is.
- At **model** resolution the set is *not* complete without `Solution Stochasticity` and
  `Loss Margin Behaviour`. Those two are what separate `DT` from `ExtraTree`,
  `LR` from `LinearSVC`, and `LightGBM_RF` from `LightGBM_ExtraTrees` — pairs no measured
  descriptor in this corpus, and none of sixty-one proposed ones, can tell apart. Without
  them 134 of 476 rows are ambiguous between models.

**And the fit does not want them.** At 16 terms both contribute *zero* terms; the equation
uses 12 of the 18 available features. So the identification argument and the fit argument
point at different columns, and the objective must not be allowed to settle it silently.

Two honest resolutions: state the claim at family level and drop the two columns, or keep
them and say plainly that they earn their place on identification rather than on fit. Either
is defensible. Leaving it unstated is not.

## Remaining work

1. **Settle the resolution question above.** Everything else is smaller.
2. **`assets/docs/07-practices.md` still describes an older equation in places.** The block
   table was removed and pointed at the generated chapter 10; the surrounding prose about
   "three movements" and the term-count arithmetic was written for a 24-term equation and
   should be re-read against the current one.
3. **Chapter 5's rank-1 oracle is unrevisited.** It says one interaction component is worth
   +0.122 R² and the equation captures none of it. E3 now has nine mixed terms and crosses
   the additive oracle, so that claim needs re-measuring — it may be substantially closed,
   as chapter 9's was.
4. **`identity.py` is wired into nothing** and now backs a much smaller claim (+0.017,
   down from +0.106). Decide whether chapter 9 still needs its own chapter or folds into 6.
5. **`requirements.txt` contradicts `CLAUDE.md`**, which says `pyproject.toml` is the only
   dependency source. Unresolved, and harmless.

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
