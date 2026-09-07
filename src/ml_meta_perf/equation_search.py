"""Search for the smallest equation that holds up on every test at once.

The earlier searches on this branch maximised one R2 at a time and produced equations that
were long, unstable across folds, and good at a number nobody uses. This one scores a
candidate the way the study will actually judge it:

- **three R2 values** -- in-sample, leave-one-dataset-out, leave-one-model-out;
- **the binary decision** -- can it place a model above or below a threshold, averaged over
  the five thresholds the study reports;
- **the ranking** -- treated as a search engine, since only the head of the list is ever used;
- **fold stability** -- what fraction of the equation's terms survive every fold, because a
  unified equation whose terms churn is not a unified equation;
- **length**, penalised, because the shorter equation wins ties.

Every component is on a 0-1 scale before weighting, so `OBJECTIVE_WEIGHTS` is readable as a
statement of what the study values rather than as arbitrary scaling.

Scoring uses the **fixed-form** protocol throughout: the terms are chosen once, and only the
weights are refit per fold. That is not a concession, it is the study's position on what an
equation *is*. The form is the conceptual claim -- a statement about which quantities govern
how well a learner does on a dataset -- and for a simpler problem one would write it down from
domain expertise and never search for it at all. What cross-validation then tests is whether
that claim holds on data it has not seen, with its constants recalibrated; it is not a test of
the search that proposed it.

A re-selecting ("nested") protocol answers a different question -- whether the *discovery
procedure* generalises -- and averaging twenty different equations is what made
leave-one-dataset-out jagged in `k`. It is deliberately not scored here.

**The safeguard that replaces it is `stability`.** A form derived by search rather than
asserted from expertise is only a claim about the phenomenon if the same terms keep being
chosen when the sample changes; a form that churns fold to fold is a property of these 476
rows. So term reselection frequency carries real weight in `OBJECTIVE_WEIGHTS`, and it is what
licenses fixing the form in the first place.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ml_meta_perf.data import (
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_COLUMN,
    columns_as_arrays,
    groups,
    target,
)
from ml_meta_perf.experiment import Configuration
from ml_meta_perf.fit import Selector, Standardizer, fit, guided_screen, prune
from ml_meta_perf.terms import Library, build_library
from ml_meta_perf.validate import _view, leave_one_group_out

#: `Configuration.pool_size` for every point. Not searched: it caps how many terms the beam
#: may consider, so lowering it only removes candidates the larger pool also had.
DEFAULT_POOL_SIZE = 600


@dataclass(frozen=True)
class SearchPoint:
    """A feature set together with the configuration it is to be fitted under."""

    features: tuple[str, ...]
    penalty: float = 20.0
    headline_terms: int = 20
    max_abs_zscore: float = 3.0
    max_arity: int = 3

    def configuration(self, *, pool_size: int = DEFAULT_POOL_SIZE) -> Configuration:
        """The `experiment.Configuration` this point stands for.

        ``max_terms`` is derived rather than searched separately: it is the longest
        equation the path is built out to, and it only has to reach ``headline_terms``.
        The external run's grid ties them the same way -- its `n_terms` is the reported
        length -- so searching both would be searching one quantity twice.
        """
        return Configuration(
            max_abs_zscore=self.max_abs_zscore,
            penalty=self.penalty,
            pool_size=pool_size,
            max_terms=max(self.headline_terms + 4, 8),
            headline_terms=self.headline_terms,
            max_arity=self.max_arity,
        )


#: The thresholds the binary decision is averaged over -- the study reports all five, so the
#: search must not be allowed to win by being good at one.
THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)

#: A model is a right answer for the ranking if it is this close to its dataset's best. Not a
#: fixed top-k: 9.3 of 25 models are tied at the top on average, so a top-k cut scores correct
#: answers as misses.
RELEVANCE_TOLERANCE = 0.01

#: What the study values, on one 0-1 scale. Transfer outweighs fit; the two downstream tests
#: together outweigh either R2, because they are what the equation is *for*; stability and
#: brevity are real terms rather than tie-breaks.
OBJECTIVE_WEIGHTS: dict[str, float] = {
    "in_sample_r2": 0.10,
    "loo_dataset_r2": 0.20,
    "loo_model_r2": 0.20,
    "binary": 0.15,
    "ranking": 0.15,
    "stability": 0.15,
    "brevity": 0.05,
}

#: Equations longer than this score zero for brevity; at `MIN_TERMS` they score one.
MIN_TERMS, MAX_TERMS = 6, 30


@dataclass(frozen=True)
class EquationScore:
    """Everything one candidate equation reached, on every axis the study reports."""

    point: SearchPoint
    n_terms: int
    in_sample_r2: float
    loo_dataset_r2: float
    loo_model_r2: float
    binary_accuracy: float
    binary_f1: float
    binary_map: float
    ranking_map: float
    ranking_mrr: float
    ranking_hit1: float
    ranking_regret1: float
    stability: float
    median_fold_r2: float
    worst_fold_r2: float
    terms: tuple[str, ...] = field(default=())

    @property
    def brevity(self) -> float:
        return float(np.clip((MAX_TERMS - self.n_terms) / (MAX_TERMS - MIN_TERMS), 0.0, 1.0))

    @property
    def binary(self) -> float:
        """The binary test as one number: accuracy, F1 and MAP carry equal weight."""
        return (self.binary_accuracy + self.binary_f1 + self.binary_map) / 3

    @property
    def ranking(self) -> float:
        """The ranking test as one number, head-weighted and with regret on the same scale."""
        return (self.ranking_map + self.ranking_mrr + self.ranking_hit1 + (1.0 - min(self.ranking_regret1, 1.0))) / 4

    @property
    def objective(self) -> float:
        parts = {
            "in_sample_r2": max(self.in_sample_r2, 0.0),
            "loo_dataset_r2": max(self.loo_dataset_r2, 0.0),
            "loo_model_r2": max(self.loo_model_r2, 0.0),
            "binary": self.binary,
            "ranking": self.ranking,
            "stability": self.stability,
            "brevity": self.brevity,
        }
        return float(sum(OBJECTIVE_WEIGHTS[name] * value for name, value in parts.items()))

    def as_row(self) -> dict[str, object]:
        return {
            "features": ", ".join(self.point.features),
            "n_features": len(self.point.features),
            "penalty": self.point.penalty,
            "headline_terms": self.point.headline_terms,
            "max_abs_zscore": self.point.max_abs_zscore,
            "max_arity": self.point.max_arity,
            "n_terms": self.n_terms,
            "in_sample_r2": self.in_sample_r2,
            "loo_dataset_r2": self.loo_dataset_r2,
            "loo_model_r2": self.loo_model_r2,
            "binary_accuracy": self.binary_accuracy,
            "binary_f1": self.binary_f1,
            "binary_map": self.binary_map,
            "ranking_map": self.ranking_map,
            "ranking_mrr": self.ranking_mrr,
            "ranking_hit1": self.ranking_hit1,
            "ranking_regret1": self.ranking_regret1,
            "stability": self.stability,
            "median_fold_r2": self.median_fold_r2,
            "worst_fold_r2": self.worst_fold_r2,
            "binary": self.binary,
            "ranking": self.ranking,
            "brevity": self.brevity,
            "objective": self.objective,
            "terms": " | ".join(self.terms),
        }


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if not labels.any():
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = np.cumsum(labels[order])
    precision = hits / np.arange(1, labels.size + 1)
    return float((precision * labels[order]).sum() / labels.sum())


def binary_scores(truth: np.ndarray, prediction: np.ndarray, datasets: np.ndarray) -> tuple[float, float, float]:
    """Accuracy, F1 and per-dataset MAP, averaged over `THRESHOLDS`."""
    accuracies, f1s, maps = [], [], []
    for threshold in THRESHOLDS:
        actual, predicted = truth >= threshold, prediction >= threshold
        tp = float((actual & predicted).sum())
        tn = float((~actual & ~predicted).sum())
        fp = float((~actual & predicted).sum())
        fn = float((actual & ~predicted).sum())
        precision, recall = tp / max(tp + fp, 1.0), tp / max(tp + fn, 1.0)
        accuracies.append((tp + tn) / truth.size)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
        per_dataset = [
            _average_precision(truth[datasets == label] >= threshold, prediction[datasets == label])
            for label in np.unique(datasets)
        ]
        usable = [value for value in per_dataset if not np.isnan(value)]
        maps.append(float(np.mean(usable)) if usable else 0.0)
    return float(np.mean(accuracies)), float(np.mean(f1s)), float(np.mean(maps))


def ranking_scores(
    truth: np.ndarray, prediction: np.ndarray, datasets: np.ndarray
) -> tuple[float, float, float, float]:
    """MAP, MRR, hit@1 and top-1 regret, averaged over the datasets that discriminate."""
    maps, mrrs, hits, regrets = [], [], [], []
    for label in np.unique(datasets):
        mask = datasets == label
        actual, predicted = truth[mask], prediction[mask]
        if actual.size < 3 or float(actual.max() - actual.min()) < 1e-12:
            continue
        relevant = actual >= actual.max() - RELEVANCE_TOLERANCE
        order = np.argsort(-predicted, kind="stable")
        maps.append(_average_precision(relevant, predicted))
        ranks = [position for position, index in enumerate(order, 1) if relevant[index]]
        mrrs.append(1.0 / ranks[0] if ranks else 0.0)
        hits.append(float(relevant[order[0]]))
        regrets.append(float(actual.max() - actual[order[0]]))
    return float(np.mean(maps)), float(np.mean(mrrs)), float(np.mean(hits)), float(np.mean(regrets))


def _fixed_form(library: Library, index: list[int], truth: np.ndarray, group: np.ndarray, penalty: float) -> np.ndarray:
    """Out-of-fold predictions with the terms held fixed and only the weights refit."""
    matrix = library.matrix[:, index]
    prediction = np.zeros_like(truth)
    for _, train, test in leave_one_group_out(group):
        standardizer = Standardizer.fit(matrix[train])
        design = standardizer.apply(matrix[train])
        gram = design.T @ design + penalty * np.eye(len(index))
        weights = np.linalg.solve(gram, design.T @ (truth[train] - truth[train].mean()))
        held = standardizer.apply(matrix[test]) @ weights + truth[train].mean()
        prediction[test] = np.clip(held, float(truth[train].min()), float(truth[train].max()))
    return prediction


def _fold_selections(
    library: Library,
    truth: np.ndarray,
    group: np.ndarray,
    point: SearchPoint,
    size: int,
) -> list[set[str]]:
    """Which terms selection picks when it is re-run inside each fold.

    Not a scoring protocol -- the equation is fitted with its form fixed, see the module
    docstring. This exists only to ask whether that form is a property of the phenomenon or of
    these 476 rows: if the same terms come back when a fifth of the data is removed, fixing
    the form is a claim about the mechanism; if they churn, it is a claim about the sample.
    """
    config = point.configuration()
    picks: list[set[str]] = []
    for _, train, _ in leave_one_group_out(group):
        standardizer = Standardizer.fit(library.matrix[train])
        selector = Selector(
            standardizer.apply(library.matrix[train]), truth[train], config.penalty, library.feature_groups
        )
        pool = guided_screen(_view(library, train), truth[train], keep=config.pool_size)
        subsets = selector.search(pool, config.max_terms, beam_width=config.beam_width)
        if size in subsets:
            picks.append({library.terms[i].name for i in subsets[size].indices})
    return picks


def evaluate(point: SearchPoint, frame: pl.DataFrame) -> EquationScore | None:
    """Fit one point and score it on every axis. ``None`` if the library cannot be built."""
    config = point.configuration()
    truth = target(frame)
    datasets, models = groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)
    columns = columns_as_arrays(frame, tuple(DATASET_FEATURES) + tuple(point.features))
    try:
        library = build_library(
            tuple(DATASET_FEATURES),
            tuple(point.features),
            columns,
            max_abs_zscore=config.max_abs_zscore,
            max_arity=config.max_arity,
        )
        result = fit(
            library,
            truth,
            max_terms=config.max_terms,
            penalty=config.penalty,
            pool_size=config.pool_size,
            beam_width=config.beam_width,
        )
    except ValueError:
        return None
    if config.headline_terms not in result.equations:
        return None
    equation = prune(result.equations[config.headline_terms], columns, truth, penalty=config.penalty)
    index = [library.names.index(term.name) for term in equation.terms]
    if not index:
        return None

    in_sample = np.clip(equation.predict(columns), truth.min(), truth.max())
    lodo = _fixed_form(library, index, truth, datasets, config.penalty)
    lomo = _fixed_form(library, index, truth, models, config.penalty)

    picks = _fold_selections(library, truth, datasets, point, config.headline_terms)
    names = [term.name for term in equation.terms]
    stability = float(np.mean([sum(name in pick for pick in picks) / len(picks) for name in names])) if picks else 0.0

    def r2(prediction: np.ndarray) -> float:
        return 1 - float(((truth - prediction) ** 2).sum()) / float(((truth - truth.mean()) ** 2).sum())

    per_fold = [
        1
        - float(((truth[datasets == q] - lodo[datasets == q]) ** 2).sum())
        / max(float(((truth[datasets == q] - truth[datasets == q].mean()) ** 2).sum()), 1e-12)
        for q in np.unique(datasets)
    ]
    # Scored on the leave-one-dataset-out predictions of the fixed form: the equation as
    # published, with its weights recalibrated on the training folds. That is the object the
    # study puts forward, so it is the object the practical tests must grade.
    accuracy, f1, binary_map = binary_scores(truth, lodo, datasets)
    ranking_map, mrr, hit1, regret1 = ranking_scores(truth, lodo, datasets)
    return EquationScore(
        point=point,
        n_terms=len(equation.terms),
        in_sample_r2=r2(in_sample),
        loo_dataset_r2=r2(lodo),
        loo_model_r2=r2(lomo),
        binary_accuracy=accuracy,
        binary_f1=f1,
        binary_map=binary_map,
        ranking_map=ranking_map,
        ranking_mrr=mrr,
        ranking_hit1=hit1,
        ranking_regret1=regret1,
        stability=stability,
        median_fold_r2=float(np.median(per_fold)),
        worst_fold_r2=float(min(per_fold)),
        terms=tuple(names),
    )


def grid(
    features: Sequence[tuple[str, ...]],
    penalties: Sequence[float],
    term_counts: Sequence[int],
    zscores: Sequence[float],
    arities: Sequence[int],
) -> list[SearchPoint]:
    """Every combination, features sorted so the point is canonical."""
    return [
        SearchPoint(
            features=tuple(sorted(subset)),
            penalty=penalty,
            headline_terms=terms,
            max_abs_zscore=zscore,
            max_arity=arity,
        )
        for subset in features
        for penalty in penalties
        for terms in term_counts
        for zscore in zscores
        for arity in arities
    ]
