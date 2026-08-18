"""Correlation screening: which terms carry signal, and which merely repeat each other.

Two numbers are reported per term. The plain correlation against MCC is the obvious
one. The within-group correlation is the one that actually decides whether a term is
worth anything: both the term and the target are centred inside each group before
correlating, so a term is credited only for variance that group identity does not
already explain. A dataset feature scored within-model, or a model feature scored
within-dataset, has to earn its correlation the hard way.

Study chapter: [3. Search and fitting](../../assets/docs/03-search-and-fitting.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ml_meta_perf.stats import pearson, spearman
from ml_meta_perf.terms import Library


def _center_within_groups(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's own mean, leaving only within-group variation."""
    centered = values.astype(np.float64).copy()
    for label in np.unique(groups):
        mask = groups == label
        centered[mask] -= centered[mask].mean()
    return centered


def screen(
    library: Library,
    target: np.ndarray,
    groups: np.ndarray | None = None,
) -> pl.DataFrame:
    """Rank every term in the library by its association with the target.

    ``groups`` enables the within-group columns. Sorting is by absolute within-group
    Pearson when it is available, and by absolute Pearson otherwise.
    """
    matrix = library.matrix
    rows: list[dict[str, object]] = []
    within_target = _center_within_groups(target, groups) if groups is not None else None

    for index, name in enumerate(library.names):
        column = matrix[:, index]
        record: dict[str, object] = {
            "term": name,
            "pearson": pearson(column, target),
            "spearman": spearman(column, target),
        }
        if within_target is not None:
            within_column = _center_within_groups(column, groups)  # pyright: ignore[reportArgumentType]
            record["within_pearson"] = pearson(within_column, within_target)
        rows.append(record)

    frame = pl.DataFrame(rows)
    key = "within_pearson" if within_target is not None else "pearson"
    return frame.with_columns(pl.col(key).abs().alias("strength")).sort("strength", descending=True)




def redundancy_groups(library: Library, threshold: float = 0.99) -> list[list[str]]:
    """Cluster terms that are near-duplicates of one another.

    Purely diagnostic: it explains why a library of a thousand terms behaves like a much
    smaller one, and why the collinearity guard during selection is not optional.
    """
    matrix = library.matrix
    normalized = matrix - matrix.mean(axis=0)
    scale = np.sqrt((normalized**2).sum(axis=0))
    scale = np.where(scale < 1e-12, 1.0, scale)
    normalized = normalized / scale
    correlation = np.abs(normalized.T @ normalized)

    names = library.names
    unassigned = set(range(len(names)))
    clusters: list[list[str]] = []
    while unassigned:
        seed = min(unassigned)
        members = sorted(index for index in unassigned if correlation[seed, index] >= threshold)
        unassigned -= set(members)
        if len(members) > 1:
            clusters.append([names[index] for index in members])
    return clusters
