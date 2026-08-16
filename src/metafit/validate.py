"""Validation: leave-one-group-out protocols, baselines and the term-count sweep.

Two protocols are reported, and they answer different questions.

* Leave-one-dataset-out holds out all rows of one dataset. It answers "what MCC will
  these models reach on a dataset nobody has run yet", which is the meta-learning use
  case, and it is the harder number because dataset features are constant within a
  fold and so the held-out dataset is genuinely unseen.
* Leave-one-model-out holds out all rows of one model. It answers "what will a new
  model reach on datasets we know".

A random k-fold split is deliberately *not* offered as a headline protocol. With
dataset features constant across a dataset's rows, a random split puts the same dataset
on both sides of the fold and the equation can memorise dataset identity; measured on
this data that inflates R2 from 0.28 to 0.50 without changing the model at all.
``random_kfold_groups`` exists so that the README can show that gap, not to score with.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from metafit.fit import Selector, Standardizer, guided_screen, to_equation
from metafit.model import MCC_LOWER, MCC_UPPER
from metafit.stats import mae, r2_score, rmse, smape, spearman
from metafit.terms import Library


def leave_one_group_out(groups: np.ndarray) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(held-out label, train mask, test mask)`` for each distinct group."""
    for label in np.unique(groups):
        test = groups == label
        yield str(label), ~test, test


def random_kfold_groups(size: int, folds: int = 10, seed: int = 42) -> np.ndarray:
    """Fold labels for a random split -- used only to demonstrate the leakage it causes."""
    return np.random.default_rng(seed).integers(0, folds, size=size).astype(str)


@dataclass(frozen=True)
class Scores:
    """The metric set reported for every equation and every baseline."""

    r2: float
    mae: float
    rmse: float
    smape: float
    spearman: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "r2": self.r2,
            "mae": self.mae,
            "rmse": self.rmse,
            "smape": self.smape,
            "spearman": self.spearman,
            "n": self.n,
        }


def score(truth: np.ndarray, prediction: np.ndarray) -> Scores:
    return Scores(
        r2=r2_score(truth, prediction),
        mae=mae(truth, prediction),
        rmse=rmse(truth, prediction),
        smape=smape(truth, prediction),
        spearman=spearman(truth, prediction),
        n=int(truth.shape[0]),
    )


@dataclass
class CrossValidation:
    """Out-of-fold predictions plus the terms each fold chose."""

    predictions: np.ndarray
    per_fold: dict[str, Scores] = field(default_factory=dict)
    selected: list[list[str]] = field(default_factory=list)

    def scores(self, truth: np.ndarray) -> Scores:
        return score(truth, self.predictions)

    def stability(self) -> pl.DataFrame:
        """How often each term was selected across folds.

        A term chosen in 19 of 20 folds is a finding. A term chosen in 3 is an artefact
        of which datasets happened to be in the training split, and reporting the final
        all-data equation without this column would present the two identically.
        """
        counts: dict[str, int] = {}
        for names in self.selected:
            for name in names:
                counts[name] = counts.get(name, 0) + 1
        total = max(len(self.selected), 1)
        return (
            pl.DataFrame({"term": list(counts), "folds": list(counts.values())})
            .with_columns((pl.col("folds") / total).alias("frequency"))
            .sort("folds", descending=True)
        )


def cross_validate_path(
    library: Library,
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    *,
    max_terms: int,
    penalty: float,
    pool_size: int = 250,
    beam_width: int = 6,
) -> dict[int, CrossValidation]:
    """Cross-validate every equation size at once, from a single search per fold.

    The beam already records its best subset at each size, so scoring sizes 1..k costs
    one search rather than k of them. That is what makes an honest penalty-by-size sweep
    affordable, and the sweep is not optional: on this data the in-sample optimum and
    the cross-validated optimum sit at opposite ends of the penalty range.

    Term *selection* happens inside the fold, not once outside it. Screening the library
    against the full target and then cross-validating only the weights is the standard
    way to leak a held-out fold into the model, and it would flatter these numbers
    considerably.
    """
    results: dict[int, CrossValidation] = {
        size: CrossValidation(predictions=np.zeros_like(target)) for size in range(1, max_terms + 1)
    }

    for label, train, test in leave_one_group_out(groups):
        train_matrix = library.matrix[train]
        standardizer = Standardizer.fit(train_matrix)
        design = standardizer.apply(train_matrix)
        pool = guided_screen(_view(library, train), target[train], keep=pool_size)
        selector = Selector(design, target[train], penalty)
        subsets = selector.search(pool, max_terms, beam_width=beam_width)
        held = {name: values[test] for name, values in columns.items()}
        fallback = float(target[train].mean())

        for size, result in results.items():
            if size not in subsets:
                result.predictions[test] = fallback
                continue
            equation = to_equation(library, subsets[size], standardizer, selector.offset, f"fold_{label}")
            result.predictions[test] = equation.predict(held)
            result.per_fold[label] = score(target[test], result.predictions[test])
            result.selected.append([term.name for term in equation.terms])

    return results


def cross_validate(
    library: Library,
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    *,
    n_terms: int,
    penalty: float,
    pool_size: int = 250,
    beam_width: int = 6,
) -> CrossValidation:
    """Cross-validate a single equation size."""
    path = cross_validate_path(
        library,
        columns,
        target,
        groups,
        max_terms=n_terms,
        penalty=penalty,
        pool_size=pool_size,
        beam_width=beam_width,
    )
    return path[n_terms]


def _view(library: Library, mask: np.ndarray) -> Library:
    """A library restricted to a row subset, without re-evaluating any term."""
    clone = object.__new__(Library)
    clone.terms = library.terms
    clone.matrix = library.matrix[mask]
    return clone




def baseline_group_mean(
    target: np.ndarray,
    outer: np.ndarray,
    inner: np.ndarray | None = None,
) -> np.ndarray:
    """Predict the training mean, optionally conditioned on a second grouping.

    With ``inner`` set to the model column under a leave-one-dataset-out split, this is
    "what does this model usually score", the baseline any meta-model has to beat to be
    worth writing down.
    """
    predictions = np.zeros_like(target)
    for _, train, test in leave_one_group_out(outer):
        fallback = float(target[train].mean())
        if inner is None:
            predictions[test] = fallback
            continue
        for label in np.unique(inner[test]):
            selected = test & (inner == label)
            source = train & (inner == label)
            predictions[selected] = float(target[source].mean()) if source.any() else fallback
    return np.clip(predictions, MCC_LOWER, MCC_UPPER)


def additive_oracle(target: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The best any purely additive equation could do, given perfect group effects.

    Fit in-sample with the true per-group means, so it is not a predictor -- it is the
    ceiling that says how much of MCC is additive in dataset and model at all, and
    therefore how much of the remaining error no amount of term engineering can remove.
    """
    grand = float(target.mean())
    prediction = np.full_like(target, grand)
    for group in (first, second):
        for label in np.unique(group):
            mask = group == label
            prediction[mask] += float(target[mask].mean()) - grand
    return np.clip(prediction, MCC_LOWER, MCC_UPPER)


def interaction_oracle(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    rank: int,
) -> np.ndarray:
    """The additive oracle plus the best rank-``rank`` approximation of what it misses.

    ``additive_oracle`` answers "how much of MCC is dataset effect plus model effect".
    The obvious next question is what the leftover looks like, and the leftover is not
    noise: it is a (dataset x model) matrix of interactions, and interaction matrices are
    usually dominated by a few components.

    So the residual is decomposed by SVD and its leading components added back. This is
    the AMMI model -- additive main effects, multiplicative interaction -- long used for
    genotype-by-environment trials, which is structurally the same problem: a grid of
    subjects crossed with conditions where some pairings suit each other.

    The result is a *ladder* rather than a single ceiling. ``rank=0`` is the additive
    oracle; each further component is one more pattern of "this kind of model suits this
    kind of dataset". At full rank it reproduces every observed cell and R2 is 1, which is
    why the interesting question is how fast the ladder climbs, not where it ends.

    Like ``additive_oracle`` this is fitted with the true values and predicts nothing.
    Unobserved cells (24 of the 500 here) contribute zero residual, so they neither
    distort the decomposition nor are counted in any score.
    """
    rows = np.unique(first)
    columns = np.unique(second)
    row_index = {label: position for position, label in enumerate(rows)}
    column_index = {label: position for position, label in enumerate(columns)}

    grid = np.full((rows.shape[0], columns.shape[0]), np.nan)
    for value, row, column in zip(target, first, second, strict=True):
        grid[row_index[row], column_index[column]] = value
    observed = ~np.isnan(grid)

    grand = float(np.nanmean(grid))
    row_effect = np.nanmean(grid, axis=1) - grand
    column_effect = np.nanmean(grid, axis=0) - grand
    additive = grand + row_effect[:, None] + column_effect[None, :]

    if rank > 0:
        residual = np.where(observed, grid - additive, 0.0)
        left, values, right = np.linalg.svd(residual, full_matrices=False)
        additive = additive + (left[:, :rank] * values[:rank]) @ right[:rank]

    prediction = np.array(
        [additive[row_index[row], column_index[column]] for row, column in zip(first, second, strict=True)]
    )
    return np.clip(prediction, MCC_LOWER, MCC_UPPER)


def oracle_ladder(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    ranks: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8),
) -> pl.DataFrame:
    """How much each additional interaction component would be worth."""
    rows: list[dict[str, object]] = []
    previous: float | None = None
    for rank in ranks:
        value = r2_score(target, interaction_oracle(target, first, second, rank))
        rows.append(
            {
                "interaction_rank": rank,
                "r2": value,
                "gain": value - previous if previous is not None else float("nan"),
            }
        )
        previous = value
    return pl.DataFrame(rows)


def ranking_report(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
) -> pl.DataFrame:
    """Per-group rank correlation and top-1 regret.

    Regret is the practical question: if you pick the model this equation ranks first,
    how much MCC do you give up against the best model you could have picked?
    """
    rows: list[dict[str, object]] = []
    for label in np.unique(groups):
        mask = groups == label
        truth, predicted = target[mask], prediction[mask]
        rows.append(
            {
                "group": str(label),
                "spearman": spearman(predicted, truth),
                "regret": float(truth.max() - truth[int(np.argmax(predicted))]),
            }
        )
    return pl.DataFrame(rows).sort("group")
