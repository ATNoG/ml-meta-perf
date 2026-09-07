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

**This vocabulary assumes strictly positive, continuous features, and degrades quietly
when given anything else.** The assumption is worth stating because nothing here raises
when it is violated -- the affected terms are simply never generated, and a caller adding
a feature sees a smaller library rather than an error. For a feature that reaches zero,
``log``, ``sqrt`` and ``1/f`` are all undefined (`Atom.is_defined_on`) and it can never be
a denominator (`denominator_atom`), which already excludes ``nr_norm``, ``nr_bin`` and
``nr_outliers`` from three quarters of the grammar. For a **binary** feature only ``f`` and
``f^2`` survive, and those are the same column twice -- see `Library`. What is left to a
0/1 column is addition and multiplication, and the two are not equally safe: inside a sum
it contributes a level to every row, while inside a product it zeroes the term entirely on
the rows where it is off, which is a per-group slope rather than a relationship. Prefer a
continuous descriptor that grades the same distinction.

Study chapter: [2. The additive model](../../assets/docs/02-additive-model.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np

Transform = Literal["id", "log", "sqrt", "inv", "sq"]
Operation = Literal["atom", "ratio", "product", "sum_ratio", "ratio_of_sums"]

DENOMINATOR_FLOOR = 1e-9
MAX_ABS_ZSCORE = 8.0
MIN_RELATIVE_SPREAD = 1e-6
MAX_DENOMINATOR_RANGE = 20.0

#: Two terms count as the same column when their centred correlation reaches this. At
#: ``1 - 1e-9`` the pair is indistinguishable to the solver anyway, so the threshold
#: separates "algebraically identical" from "merely similar" rather than imposing a taste.
COLLINEARITY_TOLERANCE = 1.0 - 1e-9

#: Every elementary transform a unary term may use.
TRANSFORMS: tuple[Transform, ...] = ("id", "log", "sqrt", "inv", "sq")

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
    """A named expression over one to three operands.

    An operand is an `Atom` or, when the term is nested, another `Term`. Nesting lets one
    term carry a deep expression -- ``([log(a)] / [b]) * [log(c)]`` -- which matters
    because the equation's length is counted in *terms*, not in operations. A short
    equation over deep terms can be easier to read than a long one over shallow terms,
    and ``depth`` is what keeps that trade under control.
    """

    operation: Operation
    operands: tuple[Atom | Term, ...]

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
            case "ratio_of_sums":
                return f"([{parts[0]}] + [{parts[1]}]) / ([{parts[2]}] + [{parts[3]}])"

    @property
    def features(self) -> tuple[str, ...]:
        collected: list[str] = []
        for operand in self.operands:
            if isinstance(operand, Term):
                collected.extend(operand.features)
            else:
                collected.append(operand.feature)
        return tuple(collected)

    @property
    def depth(self) -> int:
        """1 for a term over atoms, one more for each level of nesting."""
        nested = [operand.depth for operand in self.operands if isinstance(operand, Term)]
        return 1 + max(nested, default=0)

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
            case "ratio_of_sums":
                return (values[0] + values[1]) / _guard(values[2] + values[3])

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "operands": [
                operand.to_dict()
                if isinstance(operand, Term)
                else {"feature": operand.feature, "transform": operand.transform}
                for operand in self.operands
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Term:
        operands = payload["operands"]
        if not isinstance(operands, list):
            raise ValueError("term payload has no operand list")
        restored: list[Atom | Term] = []
        for item in operands:  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(item, dict):
                raise ValueError("term operand is not an object")
            if "operation" in item:
                restored.append(cls.from_dict(item))  # pyright: ignore[reportUnknownArgumentType]
            else:
                restored.append(Atom(str(item["feature"]), item["transform"]))  # pyright: ignore[reportArgumentType]
        return cls(payload["operation"], tuple(restored))  # pyright: ignore[reportArgumentType]


def simplify(term: Term) -> Term:
    """Apply algebraic identities so a term prints in its shortest equivalent form.

    Nesting produces expressions that are correct but redundant: ``([a] / [b]) * [b]``
    computes ``a`` and says so in six symbols instead of one. Since the whole claim of
    this project is that the equation can be read, a term that is longer than it needs to
    be is a defect, not a cosmetic issue.

    Only exact identities are applied -- cancellation of a matching factor against a
    matching divisor, and division of a term by itself. Nothing is dropped on numerical
    grounds here; that is ``fit.prune``'s job, and keeping the two separate means this
    function never changes what a term computes.
    """
    operands = tuple(simplify(operand) if isinstance(operand, Term) else operand for operand in term.operands)
    term = Term(term.operation, operands)

    if term.operation == "product":
        left, right = term.operands
        # (x / y) * y  ->  x
        for numerator_side, other in ((left, right), (right, left)):
            if isinstance(numerator_side, Term) and numerator_side.operation == "ratio":
                inner_numerator, inner_divisor = numerator_side.operands
                if inner_divisor == other:
                    return _as_term(inner_numerator)
    elif term.operation == "ratio":
        numerator, divisor = term.operands
        # x / x  ->  1, which is collinear with the intercept and carries no information
        if numerator == divisor:
            return term
        # (x * y) / y  ->  x
        if isinstance(numerator, Term) and numerator.operation == "product":
            first, second = numerator.operands
            if second == divisor:
                return _as_term(first)
            if first == divisor:
                return _as_term(second)
    return term


def is_trivial(term: Term) -> bool:
    """Whether a term reduces to a constant, and so duplicates the intercept."""
    return term.operation == "ratio" and term.operands[0] == term.operands[1]


def _as_term(operand: Atom | Term) -> Term:
    return operand if isinstance(operand, Term) else Term("atom", (operand,))


def _guard(denominator: np.ndarray) -> np.ndarray:
    """Keep a denominator away from zero without changing its sign.

    A backstop only. Terms whose denominator can actually approach zero are never
    generated (see ``denominator_atom``), so in a library built by ``build_library``
    this floor is not what any published term relies on.
    """
    return np.where(denominator >= 0.0, 1.0, -1.0) * (np.abs(denominator) + DENOMINATOR_FLOOR)


def composition_atom(feature: str, columns: dict[str, np.ndarray]) -> Atom:
    """The operand used when a feature appears as a numerator or inside a sum.

    Strictly positive features enter log-compressed; the rest enter raw. This is the
    single decision that keeps composite terms on a comparable scale.
    """
    values = columns[feature]
    return Atom(feature, "log" if bool(np.all(values > 0.0)) else "id")


def denominator_atom(feature: str, columns: dict[str, np.ndarray]) -> Atom | None:
    """The operand to divide by, or ``None`` when this feature cannot safely be one.

    Division is the one operation in the vocabulary that can manufacture a term no
    linear solver can use: as a denominator approaches zero the term's magnitude runs
    away, one row comes to hold all of its variance, and the fitted weight becomes an
    artefact of that row rather than a relationship. The library therefore decides
    eligibility per feature, before any term is built, rather than generating the term
    and screening it afterwards.

    Two candidate forms are tried in order of preference and the first that qualifies is
    used:

    * ``log(f)`` -- compresses scale, but is zero at ``f == 1`` and negative below it,
      so it is only eligible when the feature stays clear of 1;
    * ``f`` itself -- eligible when the feature stays clear of 0.

    Eligibility is a bound on **dynamic range**, ``max|d| / min|d| <=
    MAX_DENOMINATOR_RANGE``, not merely ``min|d| > 0``. That is the criterion that
    matches the failure: a divisor spanning three orders of magnitude makes the ratio
    span three orders of magnitude too, so one row holds nearly all of the term's
    variance and its fitted weight describes that row. Requiring only a non-zero
    minimum would admit exactly those terms.

    Features that qualify under neither form are never used as denominators. In this
    meta-dataset that rules out ``nr_norm``, ``nr_bin`` and ``nr_outliers`` (which reach
    0), the log of anything reaching 1, and wide-range counts such as ``nr_inst``
    (165 to 7.1M) in raw form -- though ``log(nr_inst)`` qualifies comfortably, which is
    the compression the preference order exists to find.
    """
    for candidate in (Atom(feature, "log"), Atom(feature, "id")):
        if not candidate.is_defined_on(columns):
            continue
        with np.errstate(all="ignore"):
            evaluated = np.abs(candidate.evaluate(columns))
        if not np.all(np.isfinite(evaluated)):
            continue
        smallest = float(np.min(evaluated))
        if smallest <= DENOMINATOR_FLOOR:
            continue
        if float(np.max(evaluated)) / smallest <= MAX_DENOMINATOR_RANGE:
            return candidate
    return None


def unary_terms(features: tuple[str, ...], columns: dict[str, np.ndarray]) -> list[Term]:
    """``f``, ``log(f)``, ``sqrt(f)``, ``1/f`` and ``f^2`` for each feature that admits them."""
    terms: list[Term] = []
    for feature in features:
        for transform in TRANSFORMS:
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
        # Products are always safe; ratios only exist when the divisor is eligible.
        candidates = [Term("product", (a, b))]
        denominator_b = denominator_atom(second, columns)
        if denominator_b is not None:
            candidates.append(Term("ratio", (a, denominator_b)))
        if both_directions:
            denominator_a = denominator_atom(first, columns)
            if denominator_a is not None:
                candidates.append(Term("ratio", (b, denominator_a)))
        for term in candidates:
            if term.name not in seen:
                seen.add(term.name)
                terms.append(term)
    return terms


def sum_ratio_terms(features: tuple[str, ...], columns: dict[str, np.ndarray]) -> list[Term]:
    """``(f1 + f2) / f3`` over distinct features, with ``f1`` and ``f2`` unordered.

    Pass **every** feature, not one group. An earlier version was called with the dataset
    features alone, which silently made the highest-arity operation the only one unable to
    mix a dataset feature with a model one -- precisely the combination that carries E2's
    entire lift over E1.
    """
    terms: list[Term] = []
    for first, second in itertools.combinations(features, 2):
        for third in features:
            if third in (first, second):
                continue
            divisor = denominator_atom(third, columns)
            if divisor is None:
                continue
            operands = (
                composition_atom(first, columns),
                composition_atom(second, columns),
                divisor,
            )
            terms.append(Term("sum_ratio", operands))
    return terms


def ratio_of_sums_terms(features: tuple[str, ...], columns: dict[str, np.ndarray]) -> list[Term]:
    """``(f1 + f2) / (f3 + f4)`` -- the four-feature operation.

    Included so that "three features per term is enough" can be **measured** rather than
    assumed. A sum is a safe divisor far more often than a single feature is, since adding
    two log-compressed operands moves the result away from zero, so this reaches
    combinations ``sum_ratio`` cannot.
    """
    terms: list[Term] = []
    for numerator in itertools.combinations(features, 2):
        rest = [feature for feature in features if feature not in numerator]
        for divisor in itertools.combinations(rest, 2):
            operands = (
                composition_atom(numerator[0], columns),
                composition_atom(numerator[1], columns),
                composition_atom(divisor[0], columns),
                composition_atom(divisor[1], columns),
            )
            with np.errstate(all="ignore"):
                bottom = operands[2].evaluate(columns) + operands[3].evaluate(columns)
            if not _is_safe_sum(bottom):
                continue
            terms.append(Term("ratio_of_sums", operands))
    return terms


def _is_safe_sum(values: np.ndarray) -> bool:
    """Whether a summed divisor stays clear of zero with a bounded dynamic range."""
    magnitude = np.abs(values)
    if not np.all(np.isfinite(magnitude)):
        return False
    smallest = float(np.min(magnitude))
    if smallest <= DENOMINATOR_FLOOR:
        return False
    return float(np.max(magnitude)) / smallest <= MAX_DENOMINATOR_RANGE


def is_admissible(values: np.ndarray, max_abs_zscore: float = MAX_ABS_ZSCORE) -> bool:
    """Whether a term is numerically usable as a column of a linear system.

    This is a backstop, not the main line of defence. The terms that used to fail here
    were unstable by construction -- ratios whose denominator could reach zero -- and
    those are no longer generated at all (see ``denominator_atom``). What remains is a
    cheap check that a caller assembling a ``Library`` by hand, or a transform applied
    to an unexpected feature, cannot smuggle in a column that breaks the solve.

    Three rejections:

    *Not finite, or constant.* A constant column is collinear with the intercept.

    *Effectively constant.* A term whose spread is negligible against its own magnitude
    passes an absolute variance check but destroys the published equation: fitting
    happens on standardised terms, so folding the standardisation back into raw units
    divides the weight by that spread. One near-constant term produced a weight of
    -1.5e9 against an intercept of +1.5e9 -- algebraically fine, arithmetically fine,
    and completely unreadable, which defeats the purpose of the exercise.

    *Driven by one row.* A term is rejected when a single row sits more than
    ``max_abs_zscore`` standard deviations from the mean, because its fitted weight
    would describe that row rather than a relationship.

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


def _unit(values: np.ndarray) -> np.ndarray:
    """The centred column scaled to unit length -- the form collinearity is judged in.

    Centring first is what makes the test match the fit: fitting happens on standardised
    terms, so two columns differing by an additive or multiplicative constant are one
    column as far as the solve is concerned, however different their raw values look.
    """
    centred = values - np.mean(values)
    norm = float(np.linalg.norm(centred))
    return centred / norm if norm > 0.0 else centred


class Library:
    """A set of terms together with the design matrix they produce.

    Terms are dropped for three reasons, in order: a repeated *name*, failing
    `is_admissible`, and -- the subtlest -- being the same column as a term already kept.

    That last check cannot be done on names. ``inst_to_attr`` is ``nr_inst / nr_attr`` in
    this meta-dataset, so ``[log(inst_to_attr)] + [log(nr_attr)]`` **is**
    ``log(nr_inst)``, exactly, and the grammar generates both. Three such pairs exist in
    the published 281-term library. A binary feature produces them too and more bluntly:
    ``log``, ``sqrt`` and ``1/f`` are all undefined at zero, leaving only ``f`` and
    ``f^2``, which for a 0/1 column are the same numbers under two names.

    Keeping both members of a pair is wasteful rather than dangerous, and the distinction
    is worth being precise about. The beam search already refuses a candidate whose
    correlation with a selected term exceeds ``ml_meta_perf.fit.COLLINEARITY_LIMIT`` (0.95), so
    a duplicate pair cannot both be selected on that path and no singular system arises
    there. What the duplicates cost is candidate-pool slots, search time, and a place in
    the reported term rankings, where they appear as two independent findings. The check
    here also covers the paths ``_blocked`` does not: a ``Library`` assembled by hand, and
    any consumer reading `terms` or `matrix` directly.

    The first term of a pair wins. Generation order runs simple to complex, so the survivor
    is the shorter form: ``[log(nr_inst)] / [log(Training Operations)]`` is kept and
    ``([log(inst_to_attr)] + [log(nr_attr)]) / [log(Training Operations)]`` is dropped.
    """

    def __init__(
        self,
        terms: list[Term],
        columns: dict[str, np.ndarray],
        *,
        max_abs_zscore: float = MAX_ABS_ZSCORE,
    ) -> None:
        kept: list[Term] = []
        vectors: list[np.ndarray] = []
        units: list[np.ndarray] = []
        seen: set[str] = set()
        for term in terms:
            if term.name in seen:
                continue
            with np.errstate(all="ignore"):
                values = term.evaluate(columns)
            if not is_admissible(values, max_abs_zscore):
                continue
            unit = _unit(values)
            if any(abs(float(other @ unit)) > COLLINEARITY_TOLERANCE for other in units):
                continue
            seen.add(term.name)
            kept.append(term)
            vectors.append(values)
            units.append(unit)
        if not kept:
            raise ValueError("term library is empty after filtering")
        self.terms: list[Term] = kept
        self.matrix: np.ndarray = np.column_stack(vectors)

    @property
    def names(self) -> list[str]:
        return [term.name for term in self.terms]

    @property
    def feature_groups(self) -> np.ndarray:
        """Which terms describe the *same combination of raw features*, as group ids.

        One id per term, ``-1`` for a term over a single feature. Two terms share an id
        when their feature multisets reduce to the same set, however differently they
        arrange it: ``[log(a)] / [log(b)]`` and ``[log(b)] / [log(a)]`` are one group, and
        so are ``[a] * [log(b)]`` and ``[a] / [log(b)]``.

        `ml_meta_perf.fit.Selector` refuses to place two terms of one group in the same
        equation. That is a **readability** rule, not a numerical one, and the distinction
        matters because the numerical guard already passes: the two mirrored pairs this
        removed from the previous 16-term equation correlated at 0.891 and 0.786, both
        under ``fit.COLLINEARITY_LIMIT``, in a design conditioned at 7.8. Nothing was
        ill-posed. What was wrong is that the equation spent two of its sixteen slots
        writing one relationship both ways up -- ``log(PUN)/log(nr_class)`` beside
        ``log(nr_class)/log(PUN)``, *both* carrying negative weight -- and the term table
        then reported them as two independent findings, each reading "lowers MCC". Jointly
        they encode a curvature in one ratio; separately neither sentence is true.

        An equation this study cannot reason about term by term has failed its purpose,
        so the constraint is part of the protocol rather than an option. It costs nothing
        measurable: paired over the twenty leave-one-dataset-out folds it moves mean
        absolute error by +0.0018 at twenty terms, worse on ten of twenty datasets, sign
        test p = 1.000 and a bootstrap interval of [-0.0039, +0.0084] that spans zero.

        A term over one feature is left ungrouped, so ``1/f`` may still sit beside
        ``f^2``. Those are two points of one curve over one feature, which reads as a
        shape; the constraint is about a *relationship* being stated twice.
        """
        identifiers: dict[frozenset[str], int] = {}
        groups = np.full(len(self.terms), -1, dtype=np.int64)
        for index, term in enumerate(self.terms):
            features = frozenset(term.features)
            if len(features) < 2:
                continue
            groups[index] = identifiers.setdefault(features, len(identifiers))
        return groups

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
    max_arity: int = 3,
    max_abs_zscore: float = MAX_ABS_ZSCORE,
) -> Library:
    """Assemble the full candidate vocabulary.

    With ``model_features`` empty this yields the E1 vocabulary: unary terms over the
    dataset features plus their internal ratios, products and sums. With model features
    supplied it additionally yields the cross terms, which is where E2 gets its lift --
    dataset features alone cannot express anything that varies within a dataset.

    ``max_arity`` caps how many distinct raw features one term may combine: 2 for atoms,
    ratios and products only, 3 to add ``(f1+f2)/f3``, 4 to add ``(f1+f2)/(f3+f4)``. It is
    a parameter rather than a constant because "three is enough" is a claim that has to be
    measured, and measuring it means being able to build the alternatives.
    """
    # Sorted, so the library is a function of the feature *sets* and not of the order the
    # caller happened to pass them in. It was not, and the consequence was not cosmetic:
    # `pairwise_terms` names a product after whichever operand it sees first, so `A * B` and
    # `B * A` -- the same column, multiplication being commutative -- entered under two names
    # depending on declaration order. `Library` then de-duplicated by *correlation with terms
    # already kept*, so which of a near-collinear pair survived also depended on order, and
    # across five orderings of one six-feature set the library came out at 270, 270, 272, 270
    # and 271 terms with 24 of 257 pool slots differing. That is the whole of the
    # leave-one-dataset-out ordering band recorded in the branch notes;
    # it was never the beam's tie-breaking.
    dataset_features = tuple(sorted(dataset_features))
    model_features = tuple(sorted(model_features))
    features = dataset_features + model_features
    terms = unary_terms(features, columns)
    terms += pairwise_terms(dataset_features, dataset_features, columns, both_directions=False)
    if model_features:
        terms += pairwise_terms(model_features, model_features, columns, both_directions=False)
        terms += pairwise_terms(dataset_features, model_features, columns, both_directions=True)
    if include_sum_ratio and max_arity >= 3:
        # Every feature, not just the dataset ones: the highest-arity operation must be
        # able to mix the two groups like every other operation can.
        terms += sum_ratio_terms(features, columns)
    if max_arity >= 4:
        terms += ratio_of_sums_terms(features, columns)
    return Library(terms, columns, max_abs_zscore=max_abs_zscore)
