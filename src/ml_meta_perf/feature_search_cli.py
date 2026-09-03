"""``ml-meta-perf-select-features`` -- run the descriptor search end to end.

A separate entry point from `ml_meta_perf.cli`, deliberately: the main `ml-meta-perf`
command reproduces the published study from a fixed `MODEL_FEATURES`, and its exact
invocation is what CI checks (`.github/workflows/main.yml`'s "study reproduces end to
end" step). This command answers a different question -- *which* features belong in
`MODEL_FEATURES` -- against `assets/meta_dataset_all_descriptors.csv`, the candidate pool,
not the shipped corpus, and needs the `search` extra (`pip install ml-meta-perf[search]`)
the main command does not.

Writes, to ``--output`` (default ``results/feature_search``):

- ``dropped.csv`` -- every candidate excluded before any search ran, and why
  (`descriptors.Registry.dropped`)
- ``optuna_front.csv`` / ``pyblindopt_front.csv`` -- each optimizer's Pareto front over
  (feature count, leave-one-dataset-out R2), from one search over the full corpus
- ``stability.csv`` -- selection frequency per candidate across the resampled repeats
- ``recommendation.md`` -- the features surviving `--threshold`, one line each, for a
  paper table

This is Phase A of the two-phase plan: it recommends a `MODEL_FEATURES` replacement and
writes the evidence for it. Nothing here edits `data.py`, the shipped `meta_dataset.csv`,
or `experiment.py`'s tuned defaults -- that is Phase B, done separately once the
recommendation has been read.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

from ml_meta_perf.data import FEATURE_GLOSSARY
from ml_meta_perf.descriptors import EXCLUSION_GROUPS, build_registry, compute_strengths
from ml_meta_perf.feature_search import (
    optuna_search,
    pyblindopt_search,
    recommend_features,
    stability_selection,
)

DEFAULT_CANDIDATE_PATH = "assets/meta_dataset_all_descriptors.csv"


def _group_of(feature: str) -> str | None:
    for group, members in EXCLUSION_GROUPS.items():
        if feature in members:
            return group
    return None


def _justification(feature: str, frequency: float) -> str:
    """One line per recommended feature: what it means, plus why it's here."""
    described = FEATURE_GLOSSARY.get(feature)
    if described is None:
        source, _, name = feature.partition("__")
        described = f"{source}'s `{name}`" if source else feature
    group = _group_of(feature)
    concept = f", the survivor of exclusion group `{group}`" if group else ""
    return f"{described}{concept} -- selected in {frequency:.0%} of stability-selection repeats"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-meta-perf-select-features",
        description="Search the candidate descriptor pool down to a MODEL_FEATURES recommendation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default=DEFAULT_CANDIDATE_PATH, help="candidate meta-dataset CSV")
    parser.add_argument("--output", default="results/feature_search", help="directory for the written tables")
    parser.add_argument("--trials", type=int, default=300, help="Optuna trials for the full-corpus front")
    parser.add_argument("--pyblindopt-iter", type=int, default=25, help="DE iterations per lambda")
    parser.add_argument("--repeats", type=int, default=30, help="stability-selection repeats")
    parser.add_argument(
        "--repeat-trials", type=int, default=60, help="Optuna trials per stability-selection repeat"
    )
    parser.add_argument(
        "--hold-out-fraction", type=float, default=0.25, help="fraction of datasets held out per repeat"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="minimum stability-selection frequency to recommend a feature"
    )
    parser.add_argument("--seed", type=int, default=0, help="base seed for every stochastic step")
    parser.add_argument("--skip-pyblindopt", action="store_true", help="skip the DE cross-check (Optuna only)")
    parser.add_argument(
        "--no-representatives",
        action="store_true",
        help="search the full admissible pool instead of one strongest-correlated representative per group",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    frame = pl.read_csv(arguments.data)
    registry = build_registry(frame)
    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)

    if not arguments.no_representatives:
        strengths = compute_strengths(frame, registry.admissible)
        registry = registry.reduce_to_representatives(strengths)

    print(f"{len(registry.admissible)} admissible candidates, {len(registry.dropped)} dropped before any search")
    pl.DataFrame(
        [{"feature": name, "reason": reason} for name, reason in registry.dropped.items()]
    ).write_csv(destination / "dropped.csv")

    started = time.perf_counter()
    print(f"Optuna search: {arguments.trials} trials over the full corpus...")
    optuna_front = optuna_search(registry, frame, n_trials=arguments.trials, seed=arguments.seed)
    optuna_front.write_csv(destination / "optuna_front.csv")
    print(f"  {optuna_front.height} Pareto-optimal subsets, {time.perf_counter() - started:.1f}s")

    if not arguments.skip_pyblindopt:
        started = time.perf_counter()
        print("pyBlindOpt (OBLESA-seeded DE) cross-check sweep...")
        pyblindopt_front = pyblindopt_search(registry, frame, n_iter=arguments.pyblindopt_iter, seed=arguments.seed)
        pyblindopt_front.write_csv(destination / "pyblindopt_front.csv")
        print(f"  done, {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    print(f"Stability selection: {arguments.repeats} repeats x {arguments.repeat_trials} trials...")
    frequency = stability_selection(
        registry, frame,
        repeats=arguments.repeats, trials_per_repeat=arguments.repeat_trials,
        hold_out_fraction=arguments.hold_out_fraction, seed=arguments.seed,
    )
    frequency.write_csv(destination / "stability.csv")
    print(f"  done, {time.perf_counter() - started:.1f}s")

    recommended = recommend_features(frequency, threshold=arguments.threshold)
    frequency_by_name = dict(zip(frequency["feature"].to_list(), frequency["frequency"].to_list(), strict=True))

    lines = [
        "# Recommended `MODEL_FEATURES`",
        "",
        f"{len(recommended)} features selected in >= {arguments.threshold:.0%} of "
        f"{arguments.repeats} stability-selection repeats, out of {len(registry.admissible)} "
        "admissible candidates.",
        "",
    ]
    for feature in recommended:
        lines.append(f"- `{feature}` -- {_justification(feature, frequency_by_name[feature])}")
    report_path = destination / "recommendation.md"
    report_path.write_text("\n".join(lines) + "\n")

    print(f"\n{len(recommended)} recommended features (see {report_path}):")
    for feature in recommended:
        print(f"  - {feature} ({frequency_by_name[feature]:.0%})")
    print(f"\nall tables written to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
