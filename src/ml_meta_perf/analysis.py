"""Correlation screening: which terms carry signal, and which merely repeat each other.

Two numbers are reported per term. The plain correlation against MCC is the obvious
one. The within-group correlation is the one that actually decides whether a term is
worth anything: both the term and the target are centred inside each group before
correlating, so a term is credited only for variance that group identity does not
already explain. A dataset feature scored within-model, or a model feature scored
within-dataset, has to earn its correlation the hard way.

Study chapter: [3. Term generation and selection][study-chapter] -- the rationale, in
prose, with the figures.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/03-term-selection.md
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ml_meta_perf.fit import Standardizer
from ml_meta_perf.stats import pearson, r2_score, spearman
from ml_meta_perf.terms import Library
from ml_meta_perf.validate import leave_one_group_out


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


def feature_reach(
    library: Library,
    target: np.ndarray,
    features: tuple[str, ...],
) -> pl.DataFrame:
    """What each raw feature is worth on its own, raw and after the grammar has worked on it.

    Two questions the equation itself cannot answer, because by the time an equation exists
    every feature is entangled with the others:

    * **How much of MCC does this feature carry at all?** ``r2_raw`` is the squared
      correlation of the untransformed column with the target -- what a straight line through
      that one feature would explain. It is NaN when the grammar's outlier guard rejects the
      raw column while retaining a compressed transform.
    * **How much of that does the grammar unlock?** ``r2_best`` is the same quantity for the
      best *single-feature* term the vocabulary can build over it -- ``log(f)``, ``1/f``,
      ``f^2`` and the rest. ``gain`` is the difference, and it is the only place the value of
      the transforms is visible in isolation.

    Terms touching more than one feature are excluded on purpose: a product of two features
    would be credited to both, and the point here is the marginal worth of each one.

    This is a **heuristic**, not a bound. It ignores every interaction the equation is built
    out of, so an equation can and does exceed the sum of these parts -- which is the useful
    reading, since the excess is exactly what the multi-feature terms contribute.
    """
    rows: list[dict[str, object]] = []
    columns = {name: library.matrix[:, index] for index, name in enumerate(library.names)}

    for feature in features:
        candidates = [
            term.name for term in library.terms if term.features == (feature,) or set(term.features) == {feature}
        ]
        raw = columns.get(feature)
        raw_r = pearson(raw, target) if raw is not None else float("nan")
        best_name, best_r = "", 0.0
        for name in candidates:
            value = pearson(columns[name], target)
            if abs(value) > abs(best_r):
                best_name, best_r = name, value
        rows.append(
            {
                "feature": feature,
                "n_terms": len(candidates),
                "raw_pearson": raw_r,
                "best_term": best_name,
                "best_pearson": best_r,
                "r2_raw": raw_r**2 if raw_r == raw_r else float("nan"),
                "r2_best": best_r**2,
                "gain": best_r**2 - (raw_r**2 if raw_r == raw_r else 0.0),
            }
        )
    return pl.DataFrame(rows).sort("r2_best", descending=True)


def grammar_ceiling(
    library: Library,
    target: np.ndarray,
    features: tuple[str, ...],
    *,
    penalty: float = 0.0,
) -> dict[str, float]:
    """A heuristic estimate of how far the grammar reaches, before any search runs.

    Three levels, each a least-squares fit, each computable from the library alone:

    * ``r2_raw_additive`` -- every untransformed feature term admitted by the grammar's
      outlier guard. This is the raw-term rung of the vocabulary, not an external model.
    * ``r2_best_per_feature`` -- the best admissible single-feature term per feature, one
      each. What the vocabulary adds over its raw-term rung, with no interactions yet.
    * ``r2_all_single_feature`` -- *every* single-feature term at once. The most an equation
      with no cross-feature terms could explain, and therefore the level above which an
      equation's accuracy has to be coming from interactions.

    The gap between the last of these and the fitted equation is the part of the study that
    the additive mean-based reference cannot see: `validate.additive_mean_reference` combines a per-dataset value plus
    a per-model value, whereas this bounds a sum of per-*feature* functions. An equation above
    ``r2_all_single_feature`` is one whose mixed terms are doing real work.

    Fitted in-sample (IS) and unregularised by default: it is a description of the vocabulary's
    reach on this data, not a predictor, and shrinkage would understate it.
    """
    index = {name: position for position, name in enumerate(library.names)}

    def fit_r2(names: list[str]) -> float:
        present = [index[name] for name in names if name in index]
        if not present:
            return float("nan")
        design = library.matrix[:, present]
        design = design - design.mean(axis=0)
        scale = design.std(axis=0)
        design = design / np.where(scale < 1e-12, 1.0, scale)
        gram = design.T @ design + penalty * np.eye(design.shape[1])
        offset = float(target.mean())
        weights = np.linalg.solve(gram, design.T @ (target - offset))
        return r2_score(target, design @ weights + offset)

    single = [term for term in library.terms if len(set(term.features)) == 1]
    reach = feature_reach(library, target, features)
    return {
        "r2_raw_additive": fit_r2([name for name in features if name in index]),
        "r2_best_per_feature": fit_r2([str(name) for name in reach["best_term"].to_list() if name]),
        "r2_all_single_feature": fit_r2([term.name for term in single]),
        "n_single_feature_terms": float(len(single)),
    }


def saturated_fit(
    library: Library,
    target: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """What happens if every candidate term is handed to least squares at once.

    The control for the whole selection stage. If an unpenalised fit over the entire library
    transferred well, the beam search and the length rule would be machinery in search of a
    problem, and the honest thing would be to delete them. It does not: the design is far
    wider than 20 held-out groups can support, so the fit is excellent and the transfer is
    catastrophic.

    Reported at both bounds because the gap between them is itself the finding. Unclipped, a
    held-out dataset outside the convex hull of the others is extrapolated without limit and
    the pooled R2 goes to three or four figures negative; clipped to the training fold's own
    observed range -- exactly what `validate._clip_to_training` does for every reported
    number -- it lands somewhere merely bad. Quoting only the clipped figure would hide how
    far the unconstrained fit actually goes.

    No ridge penalty and no screening: this is the null procedure the study is measured
    against, so giving it either would be measuring something else.
    """
    matrix = library.matrix
    standardizer = Standardizer.fit(matrix)
    offset = float(target.mean())
    weights = np.linalg.lstsq(standardizer.apply(matrix), target - offset, rcond=None)[0]
    in_sample = r2_score(target, standardizer.apply(matrix) @ weights + offset)

    held = np.zeros_like(target)
    bounded = np.zeros_like(target)
    for _, train, test in leave_one_group_out(groups):
        fold = Standardizer.fit(matrix[train])
        design = fold.apply(matrix[train])
        centre = float(target[train].mean())
        fitted = np.linalg.lstsq(design, target[train] - centre, rcond=None)[0]
        held[test] = fold.apply(matrix[test]) @ fitted + centre
        bounded[test] = np.clip(held[test], float(target[train].min()), float(target[train].max()))
    return {
        "terms": float(len(library)),
        "r2_in_sample": in_sample,
        "r2_loo_dataset_clipped": r2_score(target, bounded),
        "r2_loo_dataset_unclipped": r2_score(target, held),
    }
