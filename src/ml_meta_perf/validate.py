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

Study chapter: [5. Evaluation](../../assets/docs/05-evaluation.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from math import comb

import numpy as np
import polars as pl

from ml_meta_perf.fit import Selector, Standardizer, guided_screen
from ml_meta_perf.model import MCC_LOWER, MCC_UPPER, Equation
from ml_meta_perf.stats import mae, pearson, r2_score, rmse, smape, spearman
from ml_meta_perf.terms import Library


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
    #: The equation each fold fitted, keyed by the label it held out. Kept so that a
    #: correction fitted on a fold's training rows -- `ml_meta_perf.identity` fits one -- can
    #: reuse the search this path already paid for instead of repeating it.
    equations: dict[str, Equation] = field(default_factory=dict)

    def scores(self, truth: np.ndarray) -> Scores:
        return score(truth, self.predictions)

    def dispersion(self) -> dict[str, float]:
        """How the per-fold R2 is spread, next to the pooled number.

        `scores` pools every out-of-fold prediction and scores it against the *global* mean.
        On this corpus most of the variance is between datasets, which the dataset features
        capture almost for free, so pooled R2 flatters a leave-one-dataset-out split: a set
        reaching a pooled 0.53 was explaining 0.29 of the variance *within* the average
        dataset. Pooling also hides that a single fold can dominate -- one held-out dataset
        has reached 89% of the total squared error, taking the pooled figure negative while
        eighteen folds were fine.

        So every reported leave-one-group-out R2 needs these beside it. ``worst`` is the one
        that catches the failure the pooled number hides.
        """
        values = [fold.r2 for fold in self.per_fold.values()]
        if not values:
            return {"median_fold_r2": float("nan"), "worst_fold_r2": float("nan"), "folds": 0.0}
        return {
            "median_fold_r2": float(np.median(values)),
            "worst_fold_r2": float(min(values)),
            "folds": float(len(values)),
        }

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


def fold_selections(
    library: Library,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    n_terms: int,
    penalty: float,
    pool_size: int = 250,
    beam_width: int = 6,
) -> list[list[str]]:
    """Which terms selection picks when it is re-run inside each fold.

    **This returns term names and nothing else, deliberately.** Re-selecting inside the folds
    and scoring the result answers a question about the *discovery procedure*, not about the
    equation, and the study does not report it; `cross_validate_fixed_form` produces every
    cross-validated number. Returning no predictions means the re-selecting protocol cannot
    produce a reported score even by accident.

    What it is for is `CrossValidation.stability`. Fixing an equation's form is a claim that
    the form describes the phenomenon rather than these 476 rows, and the way to check that is
    to remove a fifth of the data and see whether the same terms come back. A form whose terms
    churn fold to fold has not earned the fixed-form protocol.
    """
    selections: list[list[str]] = []
    for _, train, _ in leave_one_group_out(groups):
        train_matrix = library.matrix[train]
        standardizer = Standardizer.fit(train_matrix)
        design = standardizer.apply(train_matrix)
        pool = guided_screen(_view(library, train), target[train], keep=pool_size)
        selector = Selector(design, target[train], penalty, library.feature_groups)
        subsets = selector.search(pool, n_terms, beam_width=beam_width)
        if n_terms in subsets:
            selections.append([library.terms[index].name for index in subsets[n_terms].indices])
    return selections


def term_stability(selections: list[list[str]]) -> pl.DataFrame:
    """How often each term was selected across folds.

    A term chosen in 19 of 20 folds is a finding. A term chosen in 3 is an artefact of which
    datasets happened to be in the training split, and reporting the final all-data equation
    without this column would present the two identically.
    """
    counts: dict[str, int] = {}
    for names in selections:
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    total = max(len(selections), 1)
    return (
        pl.DataFrame({"term": list(counts), "folds": list(counts.values())})
        .with_columns((pl.col("folds") / total).alias("frequency"))
        .sort("folds", descending=True)
    )


def cross_validate_fixed_form(
    library: Library,
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    equations: dict[int, Equation],
    *,
    penalty: float,
) -> dict[int, CrossValidation]:
    """Cross-validate each equation with its **form fixed**: only the weights are refit.

    **This is the study's cross-validation protocol.** Every reported leave-one-group-out
    number comes from here. The choice is a position about what the artefact is rather than a
    statistical convenience. The form of an equation is its
    conceptual claim -- a statement about which quantities govern how well a learner does on a
    dataset -- and for a simpler problem one would write that form down from domain expertise
    and never search for it. What cross-validation then tests is whether the claim survives
    data it has not seen, with its constants recalibrated.

    `fold_selections` re-runs selection inside every fold and so answers a different
    question -- whether the *discovery procedure* generalises. **It cannot be used for a reported score** -- it
    returns no predictions. Twenty folds fit twenty different equations, so its pooled R2 is an
    average over twenty models: it moved by up to 0.3 when the requested length changed by
    two, while the fixed form varies by 0.014 over the same range. Its one remaining job is
    `CrossValidation.stability`.

    The obligation this creates is that the form must not be an artefact of the sample, and
    `term_stability` over a `fold_selections` run is how that is checked --
    separately, and not as part of the reported score.

    Predictions are bounded by each training fold's own target range, as in
    `fold_selections`; see `_clip_to_training`.
    """
    index = {name: position for position, name in enumerate(library.names)}
    results: dict[int, CrossValidation] = {}
    for size, equation in equations.items():
        positions = [index[term.name] for term in equation.terms]
        if not positions:
            continue
        matrix = library.matrix[:, positions]
        outcome = CrossValidation(predictions=np.zeros_like(target))
        for label, train, test in leave_one_group_out(groups):
            standardizer = Standardizer.fit(matrix[train])
            design = standardizer.apply(matrix[train])
            gram = design.T @ design + penalty * np.eye(len(positions))
            offset = float(target[train].mean())
            weights = np.linalg.solve(gram, design.T @ (target[train] - offset))
            held = standardizer.apply(matrix[test]) @ weights + offset
            outcome.predictions[test] = _clip_to_training(held, target[train])
            outcome.per_fold[label] = score(target[test], outcome.predictions[test])
            outcome.selected.append([term.name for term in equation.terms])
            # The same terms every fold, carrying that fold's weights, with the standardisation
            # folded back in exactly as `fit.to_equation` does. Recorded so a downstream
            # correction -- `identity.correct_out_of_fold` fits one -- can reuse the fold's own
            # equation instead of the all-data one.
            raw = weights / standardizer.scale
            outcome.equations[label] = Equation(
                intercept=offset - float(raw @ standardizer.mean),
                terms=tuple(equation.terms),
                weights=tuple(float(value) for value in raw),
                standardized_weights=tuple(float(value) for value in weights),
                name=f"fold_{label}",
            )
        results[size] = outcome
    return results


def _clip_to_training(prediction: np.ndarray, training_target: np.ndarray) -> np.ndarray:
    """Bound a fold's predictions by the target range that fold was trained on.

    `model.predict` already clips to ``[MCC_LOWER, MCC_UPPER]``, the range MCC can take in
    principle. That is the right bound for the equation as published and too loose here: this
    corpus has exactly one negative row (-0.29), so a floor at -1.0 leaves a held-out fold free
    to be predicted at a value nothing in training ever took.

    It happens. Held out, `ASNM-CDX-2009` -- 25 rows of 476, mean MCC 0.370 against the
    corpus's 0.731 -- was predicted at the -1.0 floor by several configurations, and that one
    fold alone reached **89% of the total squared error**, taking pooled leave-one-dataset-out
    R2 negative. Predicting it at the global mean would have cost 0.113 of the total.

    The training fold's own observed range uses no held-out information -- it is exactly what
    the fitted equation was shown -- so this tightens the bound without leaking. Measured, it
    moves an affected configuration from -0.383 to +0.260 and leaves unaffected ones untouched
    to four decimal places.
    """
    return np.clip(prediction, float(training_target.min()), float(training_target.max()))


def _view(library: Library, mask: np.ndarray) -> Library:
    """A library restricted to a row subset, without re-evaluating any term."""
    clone = object.__new__(Library)
    clone.terms = library.terms
    clone.matrix = library.matrix[mask]
    return clone


def baseline_group_centre(
    target: np.ndarray,
    outer: np.ndarray,
    inner: np.ndarray | None = None,
    *,
    centre: str = "mean",
) -> np.ndarray:
    """Predict a training-fold centre, optionally conditioned on a second grouping.

    With ``inner`` set to the model column under a leave-one-dataset-out split, this is
    "what does this model usually score", the baseline any meta-model has to beat to be
    worth writing down.

    ``centre`` picks the summary. **Which one is the fair comparison depends on the metric
    being reported, and using the mean for all of them understates the baseline.** The mean
    minimises squared error, so it is the right opponent for R2 and RMSE. The *median*
    minimises absolute error, so an MAE quoted against a mean baseline is quoted against a
    baseline that is not even trying -- and SMAPE, being an absolute-error ratio, behaves the
    same way. Both are reported so each metric is read against the baseline that is hardest
    to beat on it.

    The leave-one-group-out loop is not decoration: with 17-25 rows per group, including the
    row being predicted inflates the per-model mean's R2 by 0.082. A naive group centre
    computed over all rows is not this function.
    """
    summarise = np.median if centre == "median" else np.mean
    predictions = np.zeros_like(target)
    for _, train, test in leave_one_group_out(outer):
        fallback = float(summarise(target[train]))
        if inner is None:
            predictions[test] = fallback
            continue
        for label in np.unique(inner[test]):
            selected = test & (inner == label)
            source = train & (inner == label)
            predictions[selected] = float(summarise(target[source])) if source.any() else fallback
    return np.clip(predictions, MCC_LOWER, MCC_UPPER)


def baseline_group_mean(
    target: np.ndarray,
    outer: np.ndarray,
    inner: np.ndarray | None = None,
) -> np.ndarray:
    """`baseline_group_centre` at the mean. Kept as the name the rest of the study uses."""
    return baseline_group_centre(target, outer, inner, centre="mean")


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


#: A model is a right answer for the ranking if its MCC is within this of its dataset's best.
#: See `ranking_report`: a fixed top-k relevance set misscores the datasets whose best models
#: are genuinely tied.
RELEVANCE_TOLERANCE = 0.01


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve, by the step-wise sum.

    ``sum (R_n - R_{n-1}) * P_n`` over the ranking, the definition that does not interpolate
    and so cannot flatter a short list. Returns NaN when no label is positive, which is a
    real state here: at a threshold of 0.9 some datasets have no model above it.
    """
    if not labels.any():
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = np.cumsum(labels[order])
    precision = hits / np.arange(1, labels.size + 1)
    return float((precision * labels[order]).sum() / labels.sum())


def decision_report(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray | None = None,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> pl.DataFrame:
    """Quality of the go/no-go decision the equation supports.

    An R2 of 0.47 sounds too weak to act on, and read as "how precisely can I state the
    MCC" it is. But the question a practitioner actually asks is coarser -- *will this
    model work on this data* -- and a regression too imprecise for the first question can
    be accurate on the second, because thresholding discards exactly the precision it
    lacks.

    Each row thresholds both the truth and the prediction at the same value and scores the
    resulting binary decision. ``majority`` is the accuracy of always answering with the
    larger class, which is the bar any such rule has to clear to be worth running.

    ``f1`` is reported beside accuracy because the classes are far from balanced at the outer
    thresholds -- at 0.9 a constant predictor reaches 0.49 accuracy and an F1 of exactly zero,
    and only the pair distinguishes a rule that works from one that has guessed the majority.

    ``map`` needs ``groups`` and is the mean over datasets of the average precision of the
    ranking the prediction induces. It is the threshold-free companion: accuracy grades one
    cut of an ordering, average precision grades the whole ordering. The mean is taken over
    datasets rather than pooled over all rows because the question is always "which model for
    *this* data" -- pooling would let a correct ordering between datasets hide a wrong one
    within a dataset, and nobody ever chooses between datasets.
    """
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        actual = target >= threshold
        predicted = prediction >= threshold
        hits = int(np.sum(actual & predicted))
        correct_rejections = int(np.sum(~actual & ~predicted))
        false_alarms = int(np.sum(~actual & predicted))
        misses = int(np.sum(actual & ~predicted))

        positives = hits + false_alarms
        actual_positives = hits + misses
        denominator = float(
            np.sqrt(
                float(positives)
                * float(actual_positives)
                * float(correct_rejections + false_alarms)
                * float(correct_rejections + misses)
            )
        )
        share = float(np.mean(actual))
        rows.append(
            {
                "threshold": threshold,
                "accuracy": (hits + correct_rejections) / target.shape[0],
                "majority": max(share, 1.0 - share),
                "precision": hits / positives if positives else float("nan"),
                "recall": hits / actual_positives if actual_positives else float("nan"),
                "mcc": (
                    (hits * correct_rejections - false_alarms * misses) / denominator
                    if denominator > 0.0
                    else float("nan")
                ),
                "f1": (
                    2 * hits / (2 * hits + false_alarms + misses)
                    if (2 * hits + false_alarms + misses)
                    else float("nan")
                ),
                "map": _grouped_average_precision(target, prediction, groups, threshold),
                "n_positive": actual_positives,
            }
        )
    return pl.DataFrame(rows)


def _grouped_average_precision(
    target: np.ndarray, prediction: np.ndarray, groups: np.ndarray | None, threshold: float
) -> float:
    if groups is None:
        return float("nan")
    scores = [
        average_precision(target[groups == label] >= threshold, prediction[groups == label])
        for label in np.unique(groups)
    ]
    usable = [value for value in scores if not np.isnan(value)]
    return float(np.mean(usable)) if usable else float("nan")


def ranking_report(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
) -> pl.DataFrame:
    """Per-group ranking quality, scored the way a search engine is scored.

    Only the head of the list is ever used: a practitioner tries the top few models and never
    sees the tail, so a metric rewarding a correct rank twenty is measuring something nobody
    reads. Hence ``mrr``, ``hit@1`` and ``regret`` -- all head-weighted -- with ``ap`` grading
    the whole ordering and ``spearman`` kept for continuity.

    **Read ``ap``, ``mrr``, ``hit@1`` and ``regret``; treat ``spearman`` as weak evidence.**
    Measured on this corpus it sits between 0.63 and 0.73 for every predictor *and* every
    baseline, including a constant, so it cannot separate the things this study compares.

    A model counts as a right answer if its MCC is within ``RELEVANCE_TOLERANCE`` of the best
    on its dataset, rather than by a fixed top-k cut. That is not a convenience: 80 of the 476
    rows sit at exactly MCC 1.0, so many datasets have several genuinely tied best models, and
    a top-3 rule would score a correct answer as a miss.

    Regret is the practical question in the target's own units: if you pick the model this
    equation ranks first, how much MCC do you give up against the best you could have picked?
    """
    rows: list[dict[str, object]] = []
    for label in np.unique(groups):
        mask = groups == label
        truth, predicted = target[mask], prediction[mask]
        relevant = truth >= truth.max() - RELEVANCE_TOLERANCE
        order = np.argsort(-predicted, kind="stable")
        ranks = [position for position, index in enumerate(order, 1) if relevant[index]]
        rows.append(
            {
                "group": str(label),
                "ap": average_precision(relevant, predicted),
                "mrr": 1.0 / ranks[0] if ranks else 0.0,
                "hit_at_1": float(relevant[order[0]]),
                "regret": float(truth.max() - truth[int(order[0])]),
                "spearman": spearman(predicted, truth),
            }
        )
    return pl.DataFrame(rows).sort("group")


#: Resamples for the paired bootstrap. Twenty groups is a small enough sample that the
#: interval is the point of the exercise, and 20k resamples costs microseconds.
BOOTSTRAP_ROUNDS = 20_000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class PairedResult:
    """Whether a per-group difference is real, or inside the spread between folds.

    Every headline in this study is a mean over twenty held-out datasets, and a mean over
    twenty groups is not a measurement until something says how much of it one group could
    have produced. Two cases from this corpus make the point, and both looked conclusive
    as single numbers:

    * a 16-term equation ranks at average precision 0.819 against a trivial baseline's
      0.798 -- and is the better of the two on **7 of the 17** datasets where they differ
      at all. The favourable mean comes from a few large wins, not from being better;
    * lengthening that equation to 20 terms drops hit@1 from 0.800 to 0.700, which reads
      as a trend until one notices that hit@1 over twenty folds moves only in steps of
      0.05, and that the drop is two datasets changing their top pick.

    So: a **sign test** on how many groups improved, which no single group can swing, and
    a **paired bootstrap** interval on the mean, which shows what the mean is worth. The
    two answer different questions and disagreeing is informative -- a significant sign
    test with an interval spanning zero means a consistent but tiny effect.

    ``mean`` and the interval are in the metric's own units, positive meaning the first
    argument is better. Errors must therefore be negated by the caller; `paired_comparison`
    does it for you when told the metric is an error.
    """

    mean: float
    wins: int
    losses: int
    n: int
    p_value: float
    low: float
    high: float

    @property
    def significant(self) -> bool:
        """Whether the bootstrap interval excludes zero."""
        return self.low > 0.0 or self.high < 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "wins": self.wins,
            "losses": self.losses,
            "n_differing": self.n,
            "p_value": self.p_value,
            "ci_low": self.low,
            "ci_high": self.high,
        }


def sign_test(differences: np.ndarray) -> tuple[int, int, float]:
    """Wins, comparisons that differ, and the exact two-sided binomial p-value.

    Exact rather than normal-approximate because the sample is twenty groups, where the
    approximation is poor exactly when the answer matters. Summing binomial coefficients
    directly is also why this needs no scipy: ``math.comb`` is in the standard library and
    twenty choose ten is not a large number.

    Ties are dropped rather than split, which is the conservative convention: a fold where
    two predictors score identically is evidence for neither.
    """
    wins = int((differences > 0.0).sum())
    n = int((differences != 0.0).sum())
    if n == 0:
        return 0, 0, 1.0
    tail = sum(comb(n, k) for k in range(min(wins, n - wins) + 1))
    return wins, n, min(1.0, 2.0 * tail / 2**n)


def paired_comparison(
    first: np.ndarray,
    second: np.ndarray,
    *,
    lower_is_better: bool = False,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> PairedResult:
    """Compare two predictors group by group, on the same folds.

    ``first`` and ``second`` are per-group scores in the same group order -- a column of
    `ranking_report`, or per-fold errors. Pass ``lower_is_better`` for an error metric and
    the sign is handled here, so a positive ``mean`` always means ``first`` won.

    The bootstrap resamples *groups*, not rows. Rows within a dataset are not exchangeable
    with rows in another one, and the question is whether the result survives a different
    draw of datasets, which is the sampling this study's twenty-dataset corpus is a draw
    from.
    """
    if first.shape != second.shape:
        raise ValueError("paired comparison needs one score per group on both sides")
    differences = (second - first) if lower_is_better else (first - second)
    wins, n, p_value = sign_test(differences)
    losses = int((differences < 0.0).sum())
    if differences.size == 0:
        return PairedResult(float("nan"), 0, 0, 0, 1.0, float("nan"), float("nan"))
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, differences.size, size=(rounds, differences.size))
    means = differences[draws].mean(axis=1)
    return PairedResult(
        mean=float(differences.mean()),
        wins=wins,
        losses=losses,
        n=n,
        p_value=p_value,
        low=float(np.percentile(means, 2.5)),
        high=float(np.percentile(means, 97.5)),
    )


def interaction_capture(
    target: np.ndarray,
    prediction: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    rank: int = 1,
) -> dict[str, float]:
    """How much of the leading interaction pattern the equation actually reaches.

    `interaction_oracle` measures what a rank-``k`` interaction would be *worth* if
    someone could predict it. It says nothing about whether the fitted equation gets any
    of it, and chapter 5 originally answered that by comparing two R2 values -- which
    cannot distinguish an equation that misses the pattern from one that finds it and is
    inaccurate elsewhere.

    This compares the two interaction *structures* directly. Both the truth and the
    prediction are laid on the (dataset x model) grid and stripped of their own additive
    part, leaving each side's interaction residual; the truth's is then reduced to its
    leading ``rank`` components. ``alignment`` is the squared correlation between the two
    over observed cells: 1.0 means the equation's interactions lie exactly along the
    pattern the oracle found, 0.0 means they are unrelated to it.

    Only observed cells count. The 24 absent ones contribute to neither side.
    """
    rows, columns = np.unique(first), np.unique(second)
    row_index = {label: position for position, label in enumerate(rows)}
    column_index = {label: position for position, label in enumerate(columns)}

    def grid_of(values: np.ndarray) -> np.ndarray:
        grid = np.full((rows.shape[0], columns.shape[0]), np.nan)
        for value, row, column in zip(values, first, second, strict=True):
            grid[row_index[row], column_index[column]] = value
        return grid

    def interaction(grid: np.ndarray) -> np.ndarray:
        grand = float(np.nanmean(grid))
        row_effect = np.nanmean(grid, axis=1) - grand
        column_effect = np.nanmean(grid, axis=0) - grand
        return grid - (grand + row_effect[:, None] + column_effect[None, :])

    truth_grid = grid_of(target)
    observed = ~np.isnan(truth_grid)
    truth_interaction = np.where(observed, interaction(truth_grid), 0.0)
    model_interaction = np.where(observed, interaction(grid_of(prediction)), 0.0)

    left, values, right = np.linalg.svd(truth_interaction, full_matrices=False)
    leading = (left[:, :rank] * values[:rank]) @ right[:rank]

    wanted, reached = leading[observed], model_interaction[observed]
    total = float((truth_interaction[observed] ** 2).sum())
    return {
        "alignment": float(pearson(wanted, reached) ** 2),
        "leading_share": float((wanted**2).sum() / total) if total > 0.0 else float("nan"),
        "interaction_share": total / float(((truth_grid[observed] - np.nanmean(truth_grid)) ** 2).sum())
        if total > 0.0
        else float("nan"),
    }
