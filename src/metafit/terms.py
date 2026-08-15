"""The term library: the vocabulary the equations are allowed to use.

A term is a small, named, serialisable expression over one to three raw features --
``f``, ``log(f)``, ``1/f``, ``f^2``, ``f1/f2``, ``(f1+f2)/f3`` and friends. Terms are
built once from the training columns, evaluated into a design matrix, and the selected
ones are what the published equation prints. Keeping them as data (rather than as
closures) is what lets a fitted equation round-trip through JSON.

The raw features span wildly different magnitudes -- ``gravity`` runs from 1.6 to 1e16 --
so composite terms are built over log-compressed operands wherever the feature is
strictly positive. Dividing raw ``gravity`` by anything produces a term whose weight is
1e-16 and whose meaning is unreadable.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np

Transform = Literal["id", "log", "sqrt", "inv", "sq"]
Operation = Literal["atom", "ratio", "product", "sum_ratio"]

DENOMINATOR_FLOOR = 1e-9

_TRANSFORM_FORMAT: dict[Transform, str] = {
    "id": "{0}",
    "log": "log({0})",
    "sqrt": "sqrt({0})",
    "inv": "1/{0}",
    "sq": "{0}^2",
}


@dataclass(frozen=True)
class Atom:
    """One raw feature under one elementary transform."""

    feature: str
    transform: Transform = "id"

    @property
    def name(self) -> str:
        return _TRANSFORM_FORMAT[self.transform].format(self.feature)

    def evaluate(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        values = columns[self.feature]
        match self.transform:
            case "id":
                return values
            case "log":
                return np.log(values)
            case "sqrt":
                return np.sqrt(values)
            case "inv":
                return 1.0 / values
            case "sq":
                return values * values

    def is_defined_on(self, columns: dict[str, np.ndarray]) -> bool:
        """Whether the transform is real-valued and finite over these columns."""
        values = columns.get(self.feature)
        if values is None:
            return False
        if self.transform in ("log", "sqrt", "inv"):
            return bool(np.all(values > 0.0))
        return True


@dataclass(frozen=True)
class Term:
    """A named expression over one to three atoms."""

    operation: Operation
    operands: tuple[Atom, ...]

    @property
    def name(self) -> str:
        parts = [operand.name for operand in self.operands]
        match self.operation:
            case "atom":
                return parts[0]
            case "ratio":
                return f"[{parts[0]}] / [{parts[1]}]"
            case "product":
                return f"[{parts[0]}] * [{parts[1]}]"
            case "sum_ratio":
                return f"([{parts[0]}] + [{parts[1]}]) / [{parts[2]}]"

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(operand.feature for operand in self.operands)

    def evaluate(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        values = [operand.evaluate(columns) for operand in self.operands]
        match self.operation:
            case "atom":
                return values[0]
            case "ratio":
                return values[0] / _guard(values[1])
            case "product":
                return values[0] * values[1]
            case "sum_ratio":
                return (values[0] + values[1]) / _guard(values[2])

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "operands": [{"feature": a.feature, "transform": a.transform} for a in self.operands],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Term:
        operands = payload["operands"]
        if not isinstance(operands, list):
            raise ValueError("term payload has no operand list")
        atoms = tuple(Atom(str(item["feature"]), item["transform"]) for item in operands)  # pyright: ignore[reportIndexIssue, reportArgumentType, reportUnknownArgumentType]
        return cls(payload["operation"], atoms)  # pyright: ignore[reportArgumentType]


def _guard(denominator: np.ndarray) -> np.ndarray:
    """Keep a denominator away from zero without changing its sign."""
    return np.where(denominator >= 0.0, 1.0, -1.0) * (np.abs(denominator) + DENOMINATOR_FLOOR)


def composition_atom(feature: str, columns: dict[str, np.ndarray]) -> Atom:
    """The operand used when a feature appears inside a ratio, product or sum.

    Strictly positive features enter log-compressed; the rest enter raw. This is the
    single decision that keeps composite terms on a comparable scale.
    """
    values = columns[feature]
    return Atom(feature, "log" if bool(np.all(values > 0.0)) else "id")


def unary_terms(features: tuple[str, ...], columns: dict[str, np.ndarray]) -> list[Term]:
    """``f``, ``log(f)``, ``sqrt(f)``, ``1/f`` and ``f^2`` for each feature that admits them."""
    terms: list[Term] = []
    for feature in features:
        for transform in ("id", "log", "sqrt", "inv", "sq"):
            atom = Atom(feature, transform)
            if atom.is_defined_on(columns):
                terms.append(Term("atom", (atom,)))
    return terms


def pairwise_terms(
    left: tuple[str, ...],
    right: tuple[str, ...],
    columns: dict[str, np.ndarray],
    *,
    both_directions: bool,
) -> list[Term]:
    """Ratios and products over two feature groups.

    ``both_directions`` controls whether ``a/b`` and ``b/a`` are both emitted. Within a
    single group the pairs are unordered and one direction suffices; across the dataset
    and model groups both directions carry different meaning, so both are kept.
    """
    pairs = itertools.product(left, right) if both_directions else itertools.combinations(left, 2)
    terms: list[Term] = []
    seen: set[str] = set()
    for first, second in pairs:
        if first == second:
            continue
        a, b = composition_atom(first, columns), composition_atom(second, columns)
        candidates = [Term("ratio", (a, b)), Term("product", (a, b))]
        if both_directions:
            candidates.append(Term("ratio", (b, a)))
        for term in candidates:
            if term.name not in seen:
                seen.add(term.name)
                terms.append(term)
    return terms


def sum_ratio_terms(features: tuple[str, ...], columns: dict[str, np.ndarray]) -> list[Term]:
    """``(f1 + f2) / f3`` over distinct features, with ``f1`` and ``f2`` unordered."""
    terms: list[Term] = []
    for first, second in itertools.combinations(features, 2):
        for third in features:
            if third in (first, second):
                continue
            operands = (
                composition_atom(first, columns),
                composition_atom(second, columns),
                composition_atom(third, columns),
            )
            terms.append(Term("sum_ratio", operands))
    return terms


MAX_ABS_ZSCORE = 8.0
MIN_RELATIVE_SPREAD = 1e-6
MIN_DENOMINATOR_MARGIN = 0.05


def is_admissible(values: np.ndarray, max_abs_zscore: float = MAX_ABS_ZSCORE) -> bool:
    """Whether a term is stable enough to put in an equation.

    Three rejections, each earned by a failure mode this data actually produces.

    *Not finite, or constant.* The obvious one.

    *Effectively constant.* A term whose spread is negligible against its own magnitude
    survives an absolute variance check but destroys the published equation: fitting
    happens on standardised terms, so folding the standardisation back into raw units
    divides the weight by that spread. One near-constant term produced a weight of
    -1.5e9 against an intercept of +1.5e9 -- algebraically fine, arithmetically fine,
    and completely unreadable, which defeats the purpose of the exercise.

    *Driven by one row.* A term is rejected when a single row sits more than
    ``max_abs_zscore`` standard deviations from the mean. Several features here contain
    exact zeros -- ``nr_norm``, ``nr_bin``, ``nr_outliers`` -- and a ratio dividing by
    one lands on the denominator floor and spikes. Standardisation hides this while
    fitting, since the spike simply becomes the scale, but the term then explodes on a
    held-out dataset outside the training range. Left unfiltered these terms drive
    leave-one-dataset-out R2 to about -1.7 while in-sample R2 still reads 0.61.

    The cap is sample-size dependent and has to be chosen with that in mind: a single
    outlier among ``n`` rows can reach a z-score of at most about ``sqrt(n)``, so a cap
    of 8 constrains the 476-row fit but would be inert on the 20-row aggregated fit,
    which is why the two configurations do not share a value.
    """
    if not np.all(np.isfinite(values)):
        return False
    spread = float(np.std(values))
    if spread <= 1e-12:
        return False
    if spread < MIN_RELATIVE_SPREAD * (1.0 + abs(float(np.mean(values)))):
        return False
    return float(np.max(np.abs(values - np.mean(values)))) / spread <= max_abs_zscore


def denominator_is_safe(term: Term, columns: dict[str, np.ndarray]) -> bool:
    """Whether a ratio's denominator stays clear of zero across the data.

    ``_guard`` keeps division finite, but finite is not the same as meaningful. A
    denominator that approaches zero somewhere in the data -- ``log(f)`` where ``f``
    reaches 1, or any count feature that reaches 0 -- turns the ratio into a spike whose
    weight carries no interpretation. Requiring the smallest denominator to stand clear
    of zero by a fraction of its own spread removes those terms up front, rather than
    letting the selector discover them and the equation inherit them.
    """
    if term.operation not in ("ratio", "sum_ratio"):
        return True
    with np.errstate(all="ignore"):
        denominator = term.operands[-1].evaluate(columns)
    if not np.all(np.isfinite(denominator)):
        return False
    spread = float(np.std(denominator))
    floor = MIN_DENOMINATOR_MARGIN * spread if spread > 0.0 else DENOMINATOR_FLOOR
    return bool(np.min(np.abs(denominator)) > floor)


class Library:
    """A set of terms together with the design matrix they produce."""

    def __init__(
        self,
        terms: list[Term],
        columns: dict[str, np.ndarray],
        *,
        max_abs_zscore: float = MAX_ABS_ZSCORE,
    ) -> None:
        kept: list[Term] = []
        vectors: list[np.ndarray] = []
        seen: set[str] = set()
        for term in terms:
            if term.name in seen or not denominator_is_safe(term, columns):
                continue
            with np.errstate(all="ignore"):
                values = term.evaluate(columns)
            if not is_admissible(values, max_abs_zscore):
                continue
            seen.add(term.name)
            kept.append(term)
            vectors.append(values)
        if not kept:
            raise ValueError("term library is empty after filtering")
        self.terms: list[Term] = kept
        self.matrix: np.ndarray = np.column_stack(vectors)

    @property
    def names(self) -> list[str]:
        return [term.name for term in self.terms]

    def __len__(self) -> int:
        return len(self.terms)

    def design(self, columns: dict[str, np.ndarray], indices: list[int]) -> np.ndarray:
        """Re-evaluate a subset of terms on fresh columns, in the given order."""
        with np.errstate(all="ignore"):
            return np.column_stack([self.terms[index].evaluate(columns) for index in indices])


def build_library(
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
    columns: dict[str, np.ndarray],
    *,
    include_sum_ratio: bool = True,
    max_abs_zscore: float = MAX_ABS_ZSCORE,
) -> Library:
    """Assemble the full candidate vocabulary.

    With ``model_features`` empty this yields the E1 vocabulary: unary terms over the
    dataset features plus their internal ratios, products and sums. With model features
    supplied it additionally yields the cross terms, which is where E2 gets its lift --
    dataset features alone cannot express anything that varies within a dataset.
    """
    features = dataset_features + model_features
    terms = unary_terms(features, columns)
    terms += pairwise_terms(dataset_features, dataset_features, columns, both_directions=False)
    if model_features:
        terms += pairwise_terms(model_features, model_features, columns, both_directions=False)
        terms += pairwise_terms(dataset_features, model_features, columns, both_directions=True)
    if include_sum_ratio:
        terms += sum_ratio_terms(dataset_features, columns)
    return Library(terms, columns, max_abs_zscore=max_abs_zscore)
