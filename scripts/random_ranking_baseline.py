"""Compute the exact Study 5 random-ranking baseline from the saved meta-dataset.

No model is fitted and no random permutations are sampled. For each dataset, relevance is
defined exactly as in Study 5: a classifier is relevant when its observed MCC is within 0.01
of that dataset's best observed MCC. Expected Hit@1, MRR, and AP are then evaluated
analytically under a uniformly random permutation of the available classifiers. Expected
top-choice regret is the best observed MCC minus the dataset's mean observed MCC.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import fmean
from typing import cast

RELEVANCE_TOLERANCE = 0.01
EXPECTED_OBSERVATIONS = 476
EXPECTED_DATASETS = 20
EXPECTED_CLASSIFIERS = 25

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPOSITORY_ROOT / "src" / "ml_meta_perf" / "meta_dataset.csv"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results" / "random_ranking_baseline.csv"


@dataclass(frozen=True)
class DatasetBaseline:
    """Expected random-ranking metrics for one dataset."""

    dataset: str
    n: int
    relevant: int
    hit_at_1: Fraction
    mrr: Fraction
    ap: Fraction
    regret: float


def expected_hit_at_1(n: int, relevant: int) -> Fraction:
    """Expected Hit@1 under a uniformly random ranking."""
    return Fraction(relevant, n)


def expected_mrr(n: int, relevant: int) -> Fraction:
    """Expected reciprocal rank of the first relevant classifier."""
    denominator = math.comb(n, relevant)
    return sum(
        (Fraction(math.comb(n - rank, relevant - 1), denominator * rank) for rank in range(1, n - relevant + 2)),
        start=Fraction(),
    )


def expected_average_precision(n: int, relevant: int) -> Fraction:
    """Expected non-interpolated average precision under a random ranking."""
    if n == 1:
        return Fraction(1)
    harmonic = sum((Fraction(1, rank) for rank in range(1, n + 1)), start=Fraction())
    return harmonic / n + Fraction(relevant - 1, n * (n - 1)) * (n - harmonic)


def load_observations(path: Path) -> dict[str, list[tuple[str, float]]]:
    """Load and validate the final 476-pair Study 5 corpus."""
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"Dataset", "Model", "MCC"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    observations: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    pairs: set[tuple[str, str]] = set()
    classifiers: set[str] = set()
    for row_number, untyped_row in enumerate(rows, start=2):
        row = cast(dict[str, str], untyped_row)
        dataset = row["Dataset"].strip()
        classifier = row["Model"].strip()
        if not dataset or not classifier or not row["MCC"].strip():
            raise ValueError(f"missing Dataset, Model, or MCC at CSV row {row_number}")
        try:
            mcc = float(row["MCC"])
        except ValueError as error:
            raise ValueError(f"invalid MCC at CSV row {row_number}: {row['MCC']!r}") from error
        if not math.isfinite(mcc):
            raise ValueError(f"non-finite MCC at CSV row {row_number}: {row['MCC']!r}")

        pair = (dataset, classifier)
        if pair in pairs:
            raise ValueError(f"duplicate dataset-classifier pair: {dataset!r}, {classifier!r}")
        pairs.add(pair)
        classifiers.add(classifier)
        observations[dataset].append((classifier, mcc))

    if len(rows) != EXPECTED_OBSERVATIONS:
        raise ValueError(f"expected {EXPECTED_OBSERVATIONS} observations, found {len(rows)}")
    if len(pairs) != EXPECTED_OBSERVATIONS:
        raise ValueError(f"expected {EXPECTED_OBSERVATIONS} unique pairs, found {len(pairs)}")
    if len(observations) != EXPECTED_DATASETS:
        raise ValueError(f"expected {EXPECTED_DATASETS} datasets, found {len(observations)}")
    if len(classifiers) != EXPECTED_CLASSIFIERS:
        raise ValueError(f"expected {EXPECTED_CLASSIFIERS} classifiers, found {len(classifiers)}")
    return dict(observations)


def calculate(observations: dict[str, list[tuple[str, float]]]) -> list[DatasetBaseline]:
    """Calculate the exact expectation for every dataset."""
    baselines: list[DatasetBaseline] = []
    for dataset in sorted(observations):
        scores = [score for _, score in observations[dataset]]
        n = len(scores)
        best = max(scores)
        relevant = sum(score >= best - RELEVANCE_TOLERANCE for score in scores)
        if not 1 <= relevant <= n:
            raise ValueError(f"invalid relevance count for {dataset}: R={relevant}, N={n}")
        baselines.append(
            DatasetBaseline(
                dataset=dataset,
                n=n,
                relevant=relevant,
                hit_at_1=expected_hit_at_1(n, relevant),
                mrr=expected_mrr(n, relevant),
                ap=expected_average_precision(n, relevant),
                regret=best - fmean(scores),
            )
        )

    if sum(row.n for row in baselines) != EXPECTED_OBSERVATIONS:
        raise ValueError("the per-dataset classifier counts do not sum to 476")
    return baselines


def write_csv(rows: list[DatasetBaseline], destination: Path) -> None:
    """Write the 20 dataset rows and their paper-style unweighted mean."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(["dataset", "N", "R", "random_Hit@1", "random_MRR", "random_AP", "random_regret"])
        for row in rows:
            writer.writerow(
                [
                    row.dataset,
                    row.n,
                    row.relevant,
                    f"{float(row.hit_at_1):.10f}",
                    f"{float(row.mrr):.10f}",
                    f"{float(row.ap):.10f}",
                    f"{row.regret:.10f}",
                ]
            )
        writer.writerow(
            [
                "MEAN",
                f"{fmean(row.n for row in rows):.4f}",
                f"{fmean(row.relevant for row in rows):.4f}",
                f"{fmean(float(row.hit_at_1) for row in rows):.10f}",
                f"{fmean(float(row.mrr) for row in rows):.10f}",
                f"{fmean(float(row.ap) for row in rows):.10f}",
                f"{fmean(row.regret for row in rows):.10f}",
            ]
        )


def parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=DEFAULT_DATA, help="final 476-pair meta-dataset CSV")
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination CSV")
    return result


def main() -> None:
    """Run the exact calculation and report the global expectations."""
    arguments = parser().parse_args()
    rows = calculate(load_observations(arguments.data))
    write_csv(rows, arguments.output)
    print(f"Wrote {len(rows)} dataset rows and one MEAN row to {arguments.output}")
    print(f"Verified {sum(row.n for row in rows)} unique pairs across {len(rows)} datasets")
    print(f"Random Hit@1: {fmean(float(row.hit_at_1) for row in rows):.4f}")
    print(f"Random MRR: {fmean(float(row.mrr) for row in rows):.4f}")
    print(f"Random MAP: {fmean(float(row.ap) for row in rows):.4f}")
    print(f"Random mean top-choice regret: {fmean(row.regret for row in rows):.4f}")


if __name__ == "__main__":
    main()
