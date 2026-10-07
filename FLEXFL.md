# FlexFL targets

Both commands also fit FlexFL's per-run meta-dataset, one row per federated learning (FL) run,
assembled by FlexFL's `scripts/assemble_meta_dataset.py`. The abbreviations used below are
coefficient of determination (**R²**), mean absolute error (**MAE**), leave-one-dataset-out
(**LODO**), leave-one-model-out (**LOMO**) and doubly held out (**DHO**).

## Targets

Use `--target performance`, `--target total_time_s`, `--target comm_bytes_total`,
`--target n_epochs`, `--target compute_time_total_s`, `--target compute_time_max_s`,
`--target comm_time_total_s`, or `--target validation_time_s` with a FlexFL CSV supplied
by `--data`. Performance also requires
`--task-type classification` or `--task-type regression`. A task type on any other target
filters the input rows as well. `n_epochs` is the last epoch a run logged before it
stopped, at most the run's global epoch cap, the `epoch_cap` column.
`comm_bytes_sent` and `comm_bytes_recv` are not targets:
each is half of `comm_bytes_total` to within 1%, so a fit on either repeats the
`comm_bytes_total` fit.

FlexFL's assembler produces the decomposition columns, clipped to the master's first `start`
and last `end`. `compute_time_total_s` sums worker work time, while
`compute_time_max_s` is the work time of the busiest worker. `comm_time_total_s` sums
worker communication time. Its timings pair a master and a worker clock, so positive
clock skew is not detectable. `validation_time_s` sums validation time and mostly follows
validation-set size and the number of validations. The assembler leaves the decomposition
columns empty for a run with no worker logs or no work inside that window. Loading one of
these four targets drops the rows whose target is empty, after any task-type filter, and
prints the number dropped to stderr, with counts by dataset and by `fl_algo`; it fails if
dropping removes every row. An empty feature, or an empty value in any other selected
target, still fails the load. Constant-feature removal runs after the drop, so for these
targets it sees only the kept rows.

A corpus that mixes early-stopping rules is rejected: fit the runs of one rule at a time.

## Log scale

`--log-target` with `total_time_s`, `comm_bytes_total`, `compute_time_total_s`,
`compute_time_max_s`, `comm_time_total_s`, or `validation_time_s` fits `log1p` of the target,
so R² and MAE are on the log scale. It is rejected for `mcc`, `performance` and `n_epochs`.
Equations are labelled `log1p(<target>)` and named `E3_log1p_k<n>`.

## What is fitted

A FlexFL target runs E3 only, over every FlexFL dataset and model feature, under the
configuration in `config/study.json` and the study's length rule. There is no E1, E2 or E3-MAX,
and none of the MCC study's baselines, practices, figures or chapters. Features constant in the
loaded frame are removed before the term library is built.

Unbounded targets, the cost targets and `n_epochs`, prune terms whose contribution spans
less than 0.002 times the target's 1st-to-99th percentile spread divided by 2, while
bounded targets keep 0.002.

`epoch_cap` is a FlexFL model feature: the global epoch cap each run trained under. Loading a
FlexFL CSV without the column fails with a stale-corpus error; re-assemble it with FlexFL's
`scripts/assemble_meta_dataset.py`. In a corpus where every run shares one cap it is constant
and removed before the term library is built.

LOMO holds out one `fl_algo` value at a time, while the four `fl_algo_*` indicators are
model features. In each held-out fold the held-out algorithm's indicator is zero on every
training row, so its terms get no weight there. FlexFL LOMO R² therefore measures transfer
to an unseen algorithm without per-algorithm offsets, and it is not comparable with the MCC
study's LOMO, whose model features are descriptors shared across models.

## Commands

```bash
# one E3 fit; writes equation.json, equation.txt, curve.csv, term_effects.csv and
# group_shares.csv under results/flexfl/<slug>/
ml-meta-perf --target total_time_s --log-target --data meta_dataset.csv

# the configuration sweep on the same target
ml-meta-perf-search --target total_time_s --log-target --data meta_dataset.csv --output results/sweep_time
```

The sweep fits the same grid as for MCC. FlexFL targets have no go/no-go threshold and no
model ranking to score, so the DHO task-metric columns of `candidates.csv` are empty, and
among the configurations tied on R² at a readable length the proposal goes to fewer terms,
then lower arity, then the heavier penalty.
