"""The fitted object: an equation of the form ``MCC = w0 + w1*t1 + w2*t2 + ...``.

Selection runs on standardised terms, because greedy selection compares candidates by
correlation with the residual and raw scales would make that comparison meaningless.
What gets published, though, must be evaluable as written, so the weights are folded
back into raw units before they are stored. Both sets are kept: the raw weights are the
equation, the standardised weights are how the terms rank against each other.

Study chapter: [2. The additive model](../../assets/docs/02-additive-model.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml_meta_perf.terms import Term

MCC_LOWER = -1.0
MCC_UPPER = 1.0


@dataclass(frozen=True)
class Equation:
    """A sparse linear equation over named terms."""

    intercept: float
    terms: tuple[Term, ...]
    weights: tuple[float, ...]
    standardized_weights: tuple[float, ...]
    name: str = "equation"

    def __post_init__(self) -> None:
        if not (len(self.terms) == len(self.weights) == len(self.standardized_weights)):
            raise ValueError("terms and weights must have the same length")

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    def evaluate(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        """Apply the equation as written, without clipping."""
        n = len(next(iter(columns.values())))
        total = np.full(n, self.intercept, dtype=np.float64)
        with np.errstate(all="ignore"):
            for term, weight in zip(self.terms, self.weights, strict=True):
                total += weight * term.evaluate(columns)
        return total

    def predict(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        """Apply the equation and clip into the range MCC can actually take.

        A linear form has no idea that MCC stops at 1.0, and 17% of the meta-dataset
        sits exactly there. Clipping is the one piece of domain knowledge imposed on the
        output, and it is applied identically during cross-validation.
        """
        return np.clip(self.evaluate(columns), MCC_LOWER, MCC_UPPER)

    def ranked_terms(self) -> list[tuple[Term, float, float]]:
        """Terms ordered by standardised weight magnitude: strongest contributor first."""
        triples = list(zip(self.terms, self.weights, self.standardized_weights, strict=True))
        return sorted(triples, key=lambda item: abs(item[2]), reverse=True)

    def __str__(self) -> str:
        lines = [f"MCC = {self.intercept:+.6g}"]
        for term, weight, standardized in self.ranked_terms():
            # Long terms overflow the column rather than being truncated, so the padding
            # has to keep a separator of its own -- ljust alone glues the comment to a
            # name that is already past the stop.
            lines.append(f"      {weight:+.6g} * {term.name}".ljust(64) + f"  # beta={standardized:+.4f}")
        return "\n".join(lines)

    def to_latex(self) -> str:
        """A LaTeX rendering of the equation, for dropping straight into the paper."""
        parts = [f"{self.intercept:+.4g}"]
        for term, weight, _ in self.ranked_terms():
            parts.append(f"{weight:+.4g} \\cdot \\mathrm{{{_escape(term.name)}}}")
        return "\\mathrm{MCC} = " + " ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "intercept": self.intercept,
            "terms": [term.to_dict() for term in self.terms],
            "weights": list(self.weights),
            "standardized_weights": list(self.standardized_weights),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Equation:
        terms = payload["terms"]
        weights = payload["weights"]
        standardized = payload["standardized_weights"]
        if not isinstance(terms, list) or not isinstance(weights, list) or not isinstance(standardized, list):
            raise ValueError("malformed equation payload")
        return cls(
            intercept=float(payload["intercept"]),  # pyright: ignore[reportArgumentType]
            terms=tuple(Term.from_dict(item) for item in terms),  # pyright: ignore[reportArgumentType]
            weights=tuple(float(value) for value in weights),  # pyright: ignore[reportArgumentType]
            standardized_weights=tuple(float(value) for value in standardized),  # pyright: ignore[reportArgumentType]
            name=str(payload.get("name", "equation")),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Equation:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _escape(name: str) -> str:
    return name.replace("_", r"\_").replace("^", r"\^{}")


#: How a term's direction is written in every exported table.
#:
#: One spelling, in one place, because `attribution.term_effects` and `report` both write a
#: ``direction`` column and used to disagree -- "increases MCC" in one CSV, "raises MCC" in
#: the next -- which reads as two different quantities to anyone joining them.
RAISES, LOWERS = "raises MCC", "lowers MCC"


def direction(signed: float) -> str:
    """The word for the sign of a standardised weight or effect."""
    return RAISES if signed > 0.0 else LOWERS
