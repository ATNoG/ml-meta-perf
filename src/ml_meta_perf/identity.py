"""Model identity: the part of a classifier the meta-features do not describe.

Chapter 4 measures a gap. Dataset identity explains 0.354 of MCC variance and the twelve
dataset meta-features recover 99% of that; model identity explains 0.282 and the six model
meta-features recover 92%. The residual 8% is capability this corpus records nowhere --
the difference between *what a model costs*, which the features state, and *what a model
is good at*, which they do not.

The remaining headroom for a per-model table is small, but still useful as a ceiling:
it bounds what any model-side proposal could recover beyond the generated descriptors.
The module is kept for that measurement, not because the correction it computes is worth
publishing -- see the standing objections in chapter 4.

This module writes that difference down. Under leave-one-dataset-out (LODO), every one of the 25
models appears in every training fold, so the residual of the fitted equation can be
averaged per model and the average used on the held-out dataset. Nothing about the
held-out dataset enters it, and the result is two numbers per model:

    MCC = [the equation]  +  b_model  +  c_model * log(gravity)

``b`` is a level -- this model runs above or below what its meta-features predict -- and
``c`` is a slope, how that offset changes with how separable the data is. The pair is a
*table*, not an equation, and that is the honest cost of the accuracy it buys.

**Two properties of this construction decide how it may be used.**

It is not a predictor for an unseen *model*. A model with no training rows has no
effect, `ModelEffects.apply` returns zero for it, and the correction degenerates to the
equation it corrects. The leave-one-model-out (LOMO) column measures exactly that and shows no
gain, by construction rather than by accident.

It is fitted on the equation's residual, not jointly with it. Alternating the two would
turn the measurement into a backfitted model with a different training procedure
(Hastie & Tibshirani, 1990). The reported experiment does not fit or evaluate that
variant. The equation remains a statement about MCC, and the table explains only what
the finished equation leaves behind.

The construction is a **factorial regression** in the sense used for genotype-by-
environment trials: a two-way table modelled with covariates on one side and free
coefficients on the other, one of which multiplies an environmental covariate (Denis,
1988; van Eeuwijk, Denis & Kang, 1996). Where the AMMI oracle of chapter 4 estimates
both sides freely and therefore predicts nothing, this estimates the dataset side from
meta-features and keeps the model side free -- which is what makes it usable on a
dataset nobody has run. In machine-learning terms it is the collaborative half of a
hybrid recommender: the equation is content-based over meta-features, the table is a
per-item effect learned from the observed grid, and algorithm selection has been posed
as collaborative filtering before (Misir & Sebag, 2017; Fusi et al., 2018; Yang et al.,
2019).

Study chapter: [4. The equation][study-chapter]
-- the rationale, in prose, with the figures.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/04-equation.md#limitations-of-the-equation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ml_meta_perf.model import MCC_LOWER, MCC_UPPER, Equation
from ml_meta_perf.terms import Atom, composition_atom
from ml_meta_perf.validate import CrossValidation, leave_one_group_out

#: Denominator added to a model's row count when averaging its residuals. A model with
#: 19 rows is shrunk by 19/24 toward zero, which is the empirical-Bayes estimator for a
#: group mean under a common prior (Efron & Morris, 1975) with the prior-to-noise ratio
#: fixed rather than estimated. Fixed because estimating it from 20 groups is noisier
#: than choosing it: LODO R² moves by 0.015 across 0 to 20, which is
#: less than the fold noise it would be tuned against.
INTERCEPT_SHRINKAGE = 5.0

#: The same idea for the slopes, in units of the standardised carrying feature. R2 moves
#: by 0.005 across 0.5 to 10.
SLOPE_SHRINKAGE = 1.0


@dataclass(frozen=True)
class ModelEffects:
    """Per-model level and slope, in the units the equation is written in.

    ``intercepts`` and ``slopes`` are keyed by model name and read directly: a prediction
    for model *m* on a dataset is the equation's value plus ``intercepts[m]`` plus
    ``slopes[m]`` times the carrying feature. Models absent from the table contribute
    nothing, so an equation corrected this way never becomes *worse* than uncorrected on
    a model it has not seen -- it simply stops being corrected.
    """

    #: The dataset feature the slope multiplies, and the transform it enters under.
    #: ``None`` when the effects are levels only.
    atom: Atom | None
    intercepts: dict[str, float]
    slopes: dict[str, float]

    @property
    def n_models(self) -> int:
        return len(self.intercepts)

    def apply(self, columns: dict[str, np.ndarray], models: np.ndarray) -> np.ndarray:
        """The correction to add to the equation's output, row by row."""
        levels = np.array([self.intercepts.get(str(name), 0.0) for name in models])
        if self.atom is None:
            return levels
        carrier = self.atom.evaluate(columns)
        gradient = np.array([self.slopes.get(str(name), 0.0) for name in models])
        return levels + gradient * carrier

    def table(self) -> pl.DataFrame:
        """One row per model, largest level first.

        ``effect_at_median`` is the whole correction evaluated at the median of the
        carrying feature, which is the number to compare against MCC: a level and a
        slope in different units cannot be ranked against each other, and reading the
        level alone overstates models whose slope cancels it.
        """
        names = sorted(self.intercepts, key=lambda name: -self.intercepts[name])
        return pl.DataFrame(
            {
                "model": names,
                "level": [self.intercepts[name] for name in names],
                "slope": [self.slopes.get(name, 0.0) for name in names],
            }
        )

    def __str__(self) -> str:
        carrier = "" if self.atom is None else f" + c_model * {self.atom.name}"
        lines = [f"correction = b_model{carrier}"]
        for row in self.table().iter_rows(named=True):
            lines.append(f"      {row['model']:<22s} b={row['level']:+.4f}  c={row['slope']:+.4f}")
        return "\n".join(lines)


def carrying_atoms(features: tuple[str, ...], columns: dict[str, np.ndarray]) -> list[Atom]:
    """The candidate carriers for the slope, one per dataset feature.

    Each feature enters under the library's own compression rule -- log when strictly
    positive, raw otherwise -- so the slope is read in the same units as any composite
    term in the equation, and no second convention is introduced here.
    """
    return [composition_atom(feature, columns) for feature in features]


def fit_effects(
    residual: np.ndarray,
    columns: dict[str, np.ndarray],
    models: np.ndarray,
    features: tuple[str, ...] = (),
    *,
    intercept_shrinkage: float = INTERCEPT_SHRINKAGE,
    slope_shrinkage: float = SLOPE_SHRINKAGE,
) -> ModelEffects:
    """Fit per-model levels, and one per-model slope, to an equation's residual.

    Two stages, in this order. The level is the shrunk mean residual of the model's own
    rows. The slope is then fitted to what the levels leave behind, and the feature that
    carries it is chosen by training residual sum of squares over every candidate in
    ``features`` -- which is a search, so it happens inside the fold like any other.

    Pass no ``features`` for levels only.
    """
    names = [str(name) for name in np.unique(models)]
    labels = np.array([str(name) for name in models])
    intercepts = {
        name: float(residual[labels == name].sum() / (int(np.sum(labels == name)) + intercept_shrinkage))
        for name in names
    }
    inner = residual - np.array([intercepts[name] for name in labels])

    candidates = carrying_atoms(features, columns) if features else []
    best: tuple[float, Atom | None, dict[str, float]] = (float(inner @ inner), None, {})
    for atom in candidates:
        with np.errstate(all="ignore"):
            carrier = atom.evaluate(columns)
        if not np.all(np.isfinite(carrier)):
            continue
        # Centred and scaled so the shrinkage means the same thing whichever feature is
        # carrying, then folded back into raw units below.
        centre, spread = float(carrier.mean()), float(carrier.std())
        if spread < 1e-12:
            continue
        scaled = (carrier - centre) / spread
        gradient = {
            name: float(
                scaled[labels == name]
                @ inner[labels == name]
                / (float(scaled[labels == name] @ scaled[labels == name]) + slope_shrinkage)
            )
            for name in names
        }
        fitted = np.array([gradient[name] for name in labels]) * scaled
        rss = float((inner - fitted) @ (inner - fitted))
        if rss < best[0]:
            raw = {name: value / spread for name, value in gradient.items()}
            best = (rss, atom, raw)

    _, atom, slopes = best
    if atom is None:
        return ModelEffects(atom=None, intercepts=intercepts, slopes={})
    # The slope's centring belongs in the level, so that both numbers read in raw units.
    centre = float(atom.evaluate(columns).mean())
    shifted = {name: intercepts[name] - slopes[name] * centre for name in names}
    return ModelEffects(atom=atom, intercepts=shifted, slopes=slopes)


def correct_out_of_fold(
    fold: CrossValidation,
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    models: np.ndarray,
    features: tuple[str, ...] = (),
    *,
    intercept_shrinkage: float = INTERCEPT_SHRINKAGE,
    slope_shrinkage: float = SLOPE_SHRINKAGE,
) -> np.ndarray:
    """Re-score a finished cross-validation with per-model effects fitted in each fold.

    The effects are cheap and the beam search is not, so this consumes the per-fold
    equations a completed `ml_meta_perf.validate.cross_validate_fixed_form` already recorded rather
    than searching again. Each fold's effects are fitted on that fold's training rows
    only, from that fold's own equation, and applied to the rows it held out -- so the
    protocol is the one the path was run under, unchanged.
    """
    predictions = np.zeros_like(target)
    for label, train, test in leave_one_group_out(groups):
        equation = fold.equations.get(label)
        if equation is None:
            predictions[test] = fold.predictions[test]
            continue
        trained = {name: values[train] for name, values in columns.items()}
        held = {name: values[test] for name, values in columns.items()}
        effects = fit_effects(
            target[train] - equation.evaluate(trained),
            trained,
            models[train],
            features,
            intercept_shrinkage=intercept_shrinkage,
            slope_shrinkage=slope_shrinkage,
        )
        corrected = equation.evaluate(held) + effects.apply(held, models[test])
        predictions[test] = np.clip(corrected, MCC_LOWER, MCC_UPPER)
    return predictions


def predict(
    equation: Equation,
    effects: ModelEffects,
    columns: dict[str, np.ndarray],
    models: np.ndarray,
) -> np.ndarray:
    """Equation plus table, clipped into the range MCC can take."""
    return np.clip(equation.evaluate(columns) + effects.apply(columns, models), MCC_LOWER, MCC_UPPER)


def carrier_stability(
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    models: np.ndarray,
    fold: CrossValidation,
    features: tuple[str, ...],
) -> pl.DataFrame:
    """Which feature carried the slope, counted across folds.

    The carrier is chosen inside each fold, so it is a fitted quantity like any selected
    term and reporting it without its selection frequency would present a choice made 20
    times out of 20 identically to one made 6 times.
    """
    counts: dict[str, int] = {}
    for label, train, _ in leave_one_group_out(groups):
        equation = fold.equations.get(label)
        if equation is None:
            continue
        trained = {name: values[train] for name, values in columns.items()}
        effects = fit_effects(target[train] - equation.evaluate(trained), trained, models[train], features)
        name = "none" if effects.atom is None else effects.atom.name
        counts[name] = counts.get(name, 0) + 1
    total = max(sum(counts.values()), 1)
    return (
        pl.DataFrame({"carrier": list(counts), "folds": list(counts.values())})
        .with_columns((pl.col("folds") / total).alias("frequency"))
        .sort("folds", descending=True)
    )
