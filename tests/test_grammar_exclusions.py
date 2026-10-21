"""What a study may forbid a feature: the excluded operation never appears in any library term."""

import dataclasses
import itertools
import unittest
from collections.abc import Iterator

import numpy as np
import polars as pl

from ml_meta_perf.config import EXCLUDABLE as CONFIG_EXCLUDABLE
from ml_meta_perf.config import Schema, default_configuration
from ml_meta_perf.experiment import fit_path
from ml_meta_perf.terms import EXCLUDABLE, Atom, Term, build_library, composition_atom, denominator_atom, unary_terms

FEATURES = ("a", "b", "c", "d", "e")
TINY = dataclasses.replace(default_configuration(), max_terms=3, pool_size=30, beam_width=1, max_arity=3)


def make_columns() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(23)
    return {name: rng.uniform(2.0, 30.0, 60) for name in FEATURES}


def atoms(term: Term) -> Iterator[Atom]:
    for operand in term.operands:
        if isinstance(operand, Term):
            yield from atoms(operand)
        else:
            yield operand


def denominators(term: Term) -> Iterator[Atom]:
    """The atoms a term divides by: the divisor of a ratio or sum-ratio, both of a summed divisor."""
    divisors = {"ratio": slice(1, 2), "sum_ratio": slice(2, 3), "ratio_of_sums": slice(2, 4)}.get(term.operation)
    for operand in term.operands[divisors] if divisors else ():
        if isinstance(operand, Term):
            yield from atoms(operand)
        else:
            yield operand
    for operand in term.operands:
        if isinstance(operand, Term):
            yield from denominators(operand)


def breaks(term: Term, feature: str, operation: str) -> bool:
    """Whether a term takes ``feature`` through the forbidden ``operation``."""
    if operation == "inv":
        return any(atom.feature == feature for atom in denominators(term)) or any(
            atom == Atom(feature, "inv") for atom in atoms(term)
        )
    return Atom(feature, operation) in set(atoms(term))  # pyright: ignore[reportArgumentType]


class TestVocabulary(unittest.TestCase):
    def test_the_file_and_the_grammar_agree_on_what_can_be_forbidden(self) -> None:
        self.assertEqual(CONFIG_EXCLUDABLE, EXCLUDABLE)


class TestLibraryExclusions(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = make_columns()
        self.dataset, self.model = FEATURES[:3], FEATURES[3:]

    def build(self, exclude: dict[str, tuple[str, ...]] | None, arity: int = 4):
        return build_library(self.dataset, self.model, self.columns, max_arity=arity, exclude=exclude)

    def test_an_excluded_operation_appears_in_no_term_of_the_library(self) -> None:
        for feature, operation in itertools.product(FEATURES, ("log", "sqrt", "inv", "sq")):
            library = self.build({feature: (operation,)})
            with self.subTest(feature=feature, operation=operation):
                self.assertGreater(len(library.terms), 50)
                self.assertFalse([term.name for term in library.terms if breaks(term, feature, operation)])

    def test_the_unforbidden_library_does_use_every_operation(self) -> None:
        # Guards the check above: it would pass vacuously if `breaks` never matched anything.
        library = self.build(None)
        for feature, operation in itertools.product(FEATURES, ("log", "sqrt", "inv", "sq")):
            with self.subTest(feature=feature, operation=operation):
                self.assertTrue(any(breaks(term, feature, operation) for term in library.terms))

    def test_several_exclusions_hold_at_once(self) -> None:
        exclude = {"a": ("log", "inv"), "d": ("sq", "sqrt", "log"), "e": ("inv",)}
        library = self.build(exclude)
        for feature, operations in exclude.items():
            for operation in operations:
                self.assertFalse(any(breaks(term, feature, operation) for term in library.terms))

    def test_no_exclusion_is_the_library_without_the_argument(self) -> None:
        names = [term.name for term in self.build(None).terms]
        self.assertEqual([term.name for term in self.build({}).terms], names)
        self.assertEqual([term.name for term in self.build({"a": ()}).terms], names)

    def test_a_forbidden_log_enters_a_composite_term_raw(self) -> None:
        self.assertEqual(composition_atom("a", self.columns), Atom("a", "log"))
        self.assertEqual(composition_atom("a", self.columns, {"a": ("log",)}), Atom("a", "id"))
        self.assertEqual(composition_atom("a", self.columns, {"b": ("log",)}), Atom("a", "log"))

    def test_a_forbidden_inverse_is_never_a_divisor(self) -> None:
        self.assertIsNotNone(denominator_atom("a", self.columns))
        self.assertIsNone(denominator_atom("a", self.columns, {"a": ("inv",)}))

    def test_a_divisor_falls_back_to_the_raw_feature_when_only_its_log_is_forbidden(self) -> None:
        # A narrow range keeps both forms eligible, so forbidding the log leaves the raw divisor.
        narrow = {"a": np.linspace(10.0, 20.0, 40)}
        self.assertEqual(denominator_atom("a", narrow), Atom("a", "log"))
        self.assertEqual(denominator_atom("a", narrow, {"a": ("log",)}), Atom("a", "id"))

    def test_the_unary_terms_of_a_feature_lose_only_what_is_forbidden(self) -> None:
        kept = {term.name for term in unary_terms(("a",), self.columns, {"a": ("log", "sq")})}
        self.assertEqual(kept, {"a", "sqrt(a)", "1/a"})


class TestFitHonoursExclusions(unittest.TestCase):
    def test_no_equation_of_the_path_uses_a_forbidden_operation(self) -> None:
        rng = np.random.default_rng(5)
        rows = [
            {"d": f"d{i}", "m": f"m{j}", "size": s, "depth": w, "y": 1.0 + np.log(s) + 0.3 * w + 0.01 * rng.normal()}
            for i, s in enumerate(rng.uniform(2.0, 40.0, 7))
            for j, w in enumerate(rng.uniform(1.0, 6.0, 3))
        ]
        frame = pl.DataFrame(rows)
        exclusions = (("size", ("log",)),)
        free = Schema("d", "m", ("size",), ("depth",), "y", None)
        schema = dataclasses.replace(free, exclusions=exclusions)
        free_path = fit_path(frame, ("size",), ("depth",), TINY, "E3", schema=free)
        path = fit_path(frame, ("size",), ("depth",), TINY, "E3", schema=schema)
        self.assertTrue(any(breaks(t, "size", "log") for e in free_path.equations.values() for t in e.terms))
        for equation in path.equations.values():
            for term in equation.terms:
                self.assertFalse(breaks(term, "size", "log"), term.name)


if __name__ == "__main__":
    unittest.main()
