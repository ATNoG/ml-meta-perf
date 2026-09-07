"""The handful of statistics the project needs, written out rather than imported.

scipy would supply all of this, but it is a large dependency for four short functions,
and the rank machinery here has to be exercised by the test suite regardless.

Study chapter: [5. Evaluation](../../assets/docs/05-evaluation.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, with ties receiving their average rank.

    Average ranks matter here: MCC piles up at exactly 1.0 (17% of the meta-dataset),
    and ordinal ranks would invent an ordering among tied perfect scores.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if unique.shape[0] != values.shape[0]:
        sums = np.zeros(unique.shape[0], dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def rank_columns(matrix: np.ndarray) -> np.ndarray:
    """`rankdata` applied down every column at once, with the same tie handling.

    `ml_meta_perf.fit.guided_screen` needs the ranks of every candidate term in the library, in
    every fold. Doing that a column at a time cost 32k calls and a tenth of the study's
    runtime; the work is identical but the Python loop is not. ``test_stats`` asserts the
    two agree column by column, so `rankdata` stays the definition and this stays a
    restatement of it.

    Ties are averaged by locating each run of equal values in the sorted order: a run
    spanning sorted positions ``i..j`` takes rank ``(i + j) / 2 + 1``, which is the mean of
    the consecutive ranks it would otherwise receive.
    """
    rows = matrix.shape[0]
    order = np.argsort(matrix, axis=0, kind="stable")
    ordered = np.take_along_axis(matrix, order, axis=0)

    starts = np.empty(matrix.shape, dtype=bool)
    starts[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=starts[1:])
    ends = np.empty(matrix.shape, dtype=bool)
    ends[-1] = True
    ends[:-1] = starts[1:]

    position = np.arange(rows, dtype=np.float64)[:, None]
    first = np.maximum.accumulate(np.where(starts, position, -1.0), axis=0)
    last = np.minimum.accumulate(np.where(ends, position, float(rows))[::-1], axis=0)[::-1]

    ranks = np.empty(matrix.shape, dtype=np.float64)
    np.put_along_axis(ranks, order, (first + last) / 2.0 + 1.0, axis=0)
    return ranks


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Linear correlation, returning 0.0 when either side is constant."""
    a = first - first.mean()
    b = second - second.mean()
    scale = float(np.sqrt(float(a @ a) * float(b @ b)))
    return float(a @ b / scale) if scale > 1e-15 else 0.0


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Rank correlation: monotone association, insensitive to the MCC ceiling."""
    if first.shape[0] < 2:
        return 0.0
    return pearson(rankdata(first), rankdata(second))


def r2_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Coefficient of determination against the mean of ``truth``.

    Note the denominator is always the variance of the *evaluated* rows, so an R2 from
    the 20 aggregated dataset means is not comparable with one from the 476 raw rows.
    That is exactly why E1 is reported on both scales.
    """
    residual = float(((truth - prediction) ** 2).sum())
    total = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - residual / total if total > 1e-15 else 0.0


def mae(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Mean absolute error, in MCC units."""
    return float(np.abs(truth - prediction).mean())


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Root mean squared error, in MCC units."""
    return float(np.sqrt(float(((truth - prediction) ** 2).mean())))


def smape(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Symmetric mean absolute percentage error, as a percentage in [0, 200].

    **Read this one with care on MCC.** SMAPE divides by ``|truth| + |prediction|``, and
    15 of the 476 rows here have MCC exactly 0. Every one of those contributes the full
    200% unless the prediction is also exactly 0, so the metric is dominated by the rows
    the equation is already known to handle worst rather than by its typical error. It is
    reported because it was asked for and because it is scale-free, but MAE is the
    honest headline for a target that legitimately passes through zero.

    Rows where both truth and prediction are zero contribute 0, not a division by zero.
    """
    denominator = np.abs(truth) + np.abs(prediction)
    safe = np.where(denominator > 0.0, denominator, 1.0)
    ratio = np.where(denominator > 0.0, 2.0 * np.abs(truth - prediction) / safe, 0.0)
    return float(100.0 * ratio.mean())
