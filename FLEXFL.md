# FlexFL targets

`ml-meta-perf fit` and `ml-meta-perf sweep` also work on FlexFL's per-run meta-dataset, one row per federated learning (FL) run,
assembled by FlexFL's `scripts/assemble_meta_dataset.py`. The abbreviations used below are
coefficient of determination (**R²**), mean absolute error (**MAE**), leave-one-dataset-out
(**LODO**), leave-one-model-out (**LOMO**) and doubly held out (**DHO**).

## One study file per target

A target is a study file in `config/flexfl/`, and the file holds everything about the problem:
the corpus and how to read it, the target and the range it can take, the features, which
equation to fit and the tuned hyperparameters. Bounds and tuned values differ per target, so
each target has a complete file of its own.

| study file | target | bounds | notes |
|---|---|---|---|
| `total_time_s.json` | total wall time of the run | `[0, null]` | |
| `comm_bytes_total.json` | bytes sent plus received | `[0, null]` | |
| `n_epochs.json` | last epoch logged | `[0, null]` | at most the run's `epoch_cap` |
| `compute_time_total_s.json` | summed worker work time | `[0, null]` | empty rows dropped |
| `compute_time_max_s.json` | work time of the busiest worker | `[0, null]` | empty rows dropped |
| `comm_time_total_s.json` | summed worker communication time | `[0, null]` | empty rows dropped |
| `validation_time_s.json` | summed validation time | `[0, null]` | empty rows dropped |
| `performance-classification.json` | MCC on the classification runs | `[-1, 1]` | keeps `is_classification = true` |
| `performance-regression.json` | SMAPE on the regression runs | `[0, 2]` | keeps `is_classification = false` |

The corpus is `dataset/flexfl_meta_dataset.csv`, or any CSV given with `--data`. A column that
is nobody's target has no file: `comm_bytes_sent` and `comm_bytes_recv` are not targets, since
each is half of `comm_bytes_total` to within 1%, so a fit on either repeats that fit. To model a
new quantity, copy the closest file, change `response`, and run the sweep to tune it.

FlexFL's assembler produces the decomposition columns, clipped to the master's first `start`
and last `end`. `compute_time_total_s` sums worker work time, while
`compute_time_max_s` is the work time of the busiest worker. `comm_time_total_s` sums
worker communication time. Its timings pair a master and a worker clock, so positive
clock skew is not detectable. `validation_time_s` sums validation time and mostly follows
validation-set size and the number of validations. The assembler leaves the decomposition
columns empty for a run with no worker logs or no work inside that window. The study files of
these four targets set `corpus.drop_missing_response`: the rows whose target is empty are dropped
after any row filter, and the number dropped is printed to stderr, with counts by dataset and by
`fl_algo`; the load fails if dropping removes every row. An empty feature, or an empty value in
any other target, still fails the load. Constant-feature removal (`corpus.drop_constant_features`)
runs after the drop, so for these targets it sees only the kept rows.

A corpus that mixes early-stopping rules is rejected (`corpus.single_valued` lists
`early_stop_on` and `min_epochs`): fit the runs of one rule at a time.

## Log scale

`--response.transform log1p` on a study file fits `log(1 + y)` of the target, so R² and MAE are on
the log scale (chapter 7, section 7.5). It needs a target that cannot be negative, which the study
file states as a low bound of 0 or more (or no bounds): every file above qualifies except
`performance-classification.json`, whose MCC bounds include negative values. A log scale on SMAPE
(`performance-regression.json`) passes that check but rarely makes sense, since the error is
already relative. Equations are labelled `log1p(<target>)` and named `E3_log1p_k<n>`, and the output
folder gets a `-log1p` suffix. A transform that does not apply is refused when the command starts.

## What is fitted

A FlexFL study fits the all-features equation (`equations` is `["E3", "E3-MAX"]`) over every FlexFL
dataset and model feature, under its own tuned configuration, and publishes two variants of it.
**E3-Valid** is the smaller, explainable equation that performs similarly: the study's length rule
(`selection.method`, `pareto_knee` here) reads the Pareto front of cost (arity times terms) and the
worst-protocol score over arities 1 to `search.max_arity`, and takes its knee, so both the arity and
the number of terms come from it; any other rule, or `auto`, can be named in the file (chapter 7, section 7.6). **E3-MAX** is the best score the
configuration reaches: the raw maximum of the floor at `selection.capability_arity`; it bounds what
the additive form can do and is written to `e3-max/`. There is no E1 or E2.
Features constant in the
loaded frame are removed before the term library is built.
The saved `equation.json` carries the equation's `target` label and its `bounds`.

A target open on one side (every cost) prunes terms whose contribution spans less than 0.002 times
the target's 1st-to-99th percentile spread divided by 2; a target bounded on both sides scales 0.002
by its span over MCC's.

`epoch_cap` is a FlexFL model feature: the global epoch cap each run trained under. Loading a
FlexFL CSV without the column fails with a stale-corpus error; re-assemble it with FlexFL's
`scripts/assemble_meta_dataset.py`. In a corpus where every run shares one cap it is constant
and removed before the term library is built.

**LOMO and DHO do not apply to FlexFL, so the study files do not run them.** FlexFL fits one
learner family (neural networks); `fl_algo` is how the run was trained (centralised or
decentralised, synchronous or asynchronous), not a model. The four `fl_algo_*` indicators are
a one-hot encoding of the very group that LOMO holds out. In each fold the held-out algorithm's
indicator is zero on every training row, so its ridge weight is exactly zero and the held-out
runs would be predicted as "none of the trained algorithms": the score would measure how the other
terms extrapolate with that mode's own effect removed, not transfer to a new learner. DHO inherits
this. Every file in `config/flexfl/` therefore declares `"protocols": ["IS", "LODO"]`, and the
length of an equation is judged on the worse of those two (chapter 7, section 7.2a).

**A cost is floored at zero and never capped in cross-validation.** The prediction of a held-out
dataset is clipped at the bounds the study file gives and nowhere else, so a dataset costlier than
every training dataset can be predicted above them. Only a target bounded on both sides (MCC,
SMAPE) is limited to the range of each fold's training rows. The rule is written out in
[chapter 7](assets/docs/07-generic-targets.md).

**Indicator columns are offsets, not independent features.** See the note in the README and
chapter 7: `fl_algo_*`, `strategy_*`, `is_classification` and `is_categorical` are accepted, and
are read as group offsets.

## Commands

```bash
# E3-Valid and E3-MAX: <output>/<csv stem>/<target>/{e3,e3-max}/, scores.csv (fit scale and the
# target's own scale), term_effects.csv, group_shares.csv, report.md and manifest.json
ml-meta-perf fit --config config/flexfl/total_time_s.json --data meta_dataset.csv

# the same target on the log scale
ml-meta-perf fit --config config/flexfl/total_time_s.json --response.transform log1p --data meta_dataset.csv

# tune its hyperparameters; writes a complete proposed study file for review
ml-meta-perf sweep --config config/flexfl/total_time_s.json --data meta_dataset.csv
```

The sweep fits a grid of `max_abs_zscore`, `penalty` and `max_arity`, and writes to
`<output>/<csv stem>/<target>/sweep/`, beside the result folder of the same study. Its
`proposed_study.json` is the study file with the tuned values, ready to replace the one it came
from. FlexFL targets have no go/no-go threshold and no model ranking to score, so the DHO
task-metric columns of `candidates.csv` are empty, and among the configurations tied on R² at a
readable length the proposal goes to fewer terms, then lower arity, then the heavier penalty.
