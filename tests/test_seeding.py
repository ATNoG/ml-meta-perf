"""Space-filling selection of the beam's starting terms.

Two claims to pin. The embedding has to put behaviourally similar terms near each other, or
"spread out in term space" means nothing; and the selection has to be measurably more spread
than taking the strongest terms, or it is doing nothing the beam did not already do.
"""

import unittest

import numpy as np

from ml_meta_perf.seeding import (
    EMBEDDING_DIMENSIONS,
    ProbeField,
    _toroidal_distance,
    embed_terms,
    empty_space,
    maximin,
    seed_positions,
)


def _design(rows: int = 80, groups: int = 5, per_group: int = 4, seed: int = 0) -> np.ndarray:
    """A design with known structure: ``groups`` distinct signals, each with near-duplicates.

    This is the shape the real library has -- `nr_attr` and `nr_outliers` correlate at 0.9995
    and every ratio has a near-twin -- so a seeding rule that cannot separate these groups will
    not separate the real ones either.
    """
    generator = np.random.default_rng(seed)
    signals = generator.normal(size=(rows, groups))
    columns = [
        signals[:, index] + 0.01 * generator.normal(size=rows)
        for index in range(groups)
        for _ in range(per_group)
    ]
    matrix = np.column_stack(columns)
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)


class TestEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.design = _design()
        self.pool = list(range(self.design.shape[1]))
        self.embedded = embed_terms(self.design, self.pool)

    def test_lands_in_the_unit_cube(self) -> None:
        """ESA and TORANN work on the unit torus, so anything outside it is out of domain."""
        self.assertGreaterEqual(self.embedded.min(), 0.0)
        self.assertLess(self.embedded.max(), 1.0)

    def test_one_row_per_pool_term(self) -> None:
        self.assertEqual(self.embedded.shape, (len(self.pool), EMBEDDING_DIMENSIONS))

    def test_near_duplicate_columns_land_near_each_other(self) -> None:
        """The claim the whole module rests on. Columns 0-3 are one signal plus noise; if they
        are not closer to each other than to column 4, the embedding is not measuring
        similarity and 'spread' is meaningless."""
        distances = _toroidal_distance(self.embedded, self.embedded)
        within = distances[0, 1:4].mean()
        across = distances[0, 4:].mean()
        self.assertLess(within, across)

    def test_uses_ranks_so_one_extreme_term_cannot_dominate(self) -> None:
        """The pool reliably contains a column dominated by one extreme row. Under min-max
        such a column pushes everything else into a corner; under ranks it cannot."""
        design = _design()
        design[0, 0] = 500.0
        embedded = embed_terms(design, list(range(design.shape[1])))
        for axis in range(embedded.shape[1]):
            with self.subTest(axis=axis):
                self.assertGreater(float(np.ptp(embedded[:, axis])), 0.5)

    def test_fewer_dimensions_than_terms_is_respected(self) -> None:
        design = _design(rows=10, groups=2, per_group=1)
        embedded = embed_terms(design, [0, 1], dimensions=8)
        self.assertLessEqual(embedded.shape[1], 8)


class TestSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.design = _design()
        self.pool = list(range(self.design.shape[1]))
        self.embedded = embed_terms(self.design, self.pool)
        # Strength that deliberately favours one redundancy group, which is the situation the
        # seeding exists for: the strongest terms are near-duplicates of each other.
        self.strength = np.array([1.0, 0.99, 0.98, 0.97] + [0.5] * (len(self.pool) - 4))

    def _separation(self, chosen: list[int]) -> float:
        picked = self.embedded[chosen]
        distances = _toroidal_distance(picked, picked)
        np.fill_diagonal(distances, np.inf)
        return float(distances.min())

    def test_both_rules_start_from_the_strongest_term(self) -> None:
        """There is no argument for starting a beam anywhere else, and it makes the two rules
        a comparison of what the *rest* of the seeds do."""
        self.assertEqual(maximin(self.embedded, self.strength, 4)[0], 0)
        self.assertEqual(empty_space(self.embedded, self.strength, 4)[0], 0)

    def test_both_rules_beat_taking_the_strongest_terms(self) -> None:
        """The whole point: the top four by strength are one idea four times over."""
        by_strength = list(np.argsort(-self.strength)[:4])
        self.assertGreater(self._separation(maximin(self.embedded, self.strength, 4)), self._separation(by_strength))
        self.assertGreater(
            self._separation(empty_space(self.embedded, self.strength, 4)), self._separation(by_strength)
        )

    def test_selections_are_distinct(self) -> None:
        for rule in (maximin, empty_space):
            with self.subTest(rule=rule.__name__):
                chosen = rule(self.embedded, self.strength, 6)
                self.assertEqual(len(chosen), len(set(chosen)))

    def test_asking_for_more_than_the_pool_holds_returns_the_pool(self) -> None:
        small = self.embedded[:3]
        self.assertLessEqual(len(maximin(small, self.strength[:3], 10)), 3)

    def test_zero_or_empty_returns_nothing(self) -> None:
        self.assertEqual(maximin(self.embedded, self.strength, 0), [])
        self.assertEqual(empty_space(np.zeros((0, 4)), np.zeros(0), 4), [])

    def test_empty_space_is_deterministic_at_one_seed(self) -> None:
        first = empty_space(self.embedded, self.strength, 5, seed=7)
        self.assertEqual(first, empty_space(self.embedded, self.strength, 5, seed=7))


class TestSeedPositions(unittest.TestCase):
    def test_returns_library_indices_not_pool_positions(self) -> None:
        """The beam takes library indices. Returning pool positions would seed the wrong terms
        and do it silently, since both are small integers."""
        design = _design()
        pool = [3, 7, 11, 15, 19]
        strength = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        chosen = seed_positions(design, pool, strength, 3)
        self.assertTrue(set(chosen) <= set(pool))

    def test_no_seeds_when_none_are_asked_for(self) -> None:
        self.assertEqual(seed_positions(_design(), [0, 1], np.array([1.0, 0.5]), 0), [])

    def test_no_seeds_from_an_empty_pool(self) -> None:
        self.assertEqual(seed_positions(_design(), [], np.zeros(0), 4), [])

    def test_both_strategies_are_reachable(self) -> None:
        design = _design()
        pool = list(range(design.shape[1]))
        strength = np.linspace(1.0, 0.1, len(pool))
        for strategy in ("maximin", "empty_space"):
            with self.subTest(strategy=strategy):
                chosen = seed_positions(design, pool, strength, 4, strategy=strategy)
                self.assertEqual(len(chosen), 4)


class TestProbeField(unittest.TestCase):
    """ESS used as a sampler rather than as a design.

    The distinction is the whole point of this class, and it is not visible in a score: a
    one-shot placement and a continuous one both return coordinates. What separates them is
    whether the search's measurements reach the field, so that is what these pin.
    """

    def _field(self, points: int = 60, **kwargs: object) -> ProbeField:
        rng = np.random.default_rng(0)
        return ProbeField(rng.random((points, 4)), rng.random(points), seed=1, **kwargs)  # type: ignore[arg-type]

    def test_keeps_the_best_objective_per_position(self) -> None:
        """One anchor per term. A beam tries the same term against many parents, and piling
        every measurement in would read as occupied space rather than as evidence -- a term the
        beam keeps returning to would push probes away hardest, which is backwards."""
        field = self._field()
        field.tell([1, 2, 1, 3], [0.9, 0.5, 0.4, 0.7])
        self.assertEqual(field.measured, [1, 2, 3])
        self.assertEqual(field.objectives, [0.4, 0.5, 0.7])

    def test_the_anchor_set_is_bounded_by_the_pool(self) -> None:
        """What stops the field's cost growing with the length of the search."""
        field = self._field(points=10)
        for _ in range(50):
            field.tell([3, 4], [0.5, 0.6])
        self.assertEqual(len(field.measured), 2)

    def test_a_worse_measurement_does_not_overwrite_a_better_one(self) -> None:
        field = self._field()
        field.tell([5], [0.2])
        field.tell([5], [0.9])
        self.assertEqual(field.objectives, [0.2])

    def test_ignores_a_measurement_that_is_not_finite(self) -> None:
        """A subset the solver could not fit must not become an anchor at nan."""
        field = self._field()
        field.tell([1, 2], [float("nan"), 0.5])
        self.assertEqual(field.measured, [2])

    def test_a_cold_field_still_proposes(self) -> None:
        """Before anything is measured there is no field to fit, so it degrades to the
        classical rule rather than refusing -- a policy that asks for probes has to run."""
        self.assertEqual(len(self._field().propose(3, taken=[0])), 3)

    def test_never_proposes_something_already_taken(self) -> None:
        field = self._field()
        field.tell(list(range(12)), [1.0 - 0.05 * index for index in range(12)])
        taken = list(range(12))
        self.assertFalse(set(field.propose(5, taken=taken)) & set(taken))

    def test_proposals_are_distinct(self) -> None:
        field = self._field()
        field.tell(list(range(12)), [0.5] * 12)
        proposed = field.propose(6, taken=list(range(12)))
        self.assertEqual(len(proposed), len(set(proposed)))

    def test_the_attractiveness_field_is_scaled_and_signed(self) -> None:
        """Negated because the caller minimises and ESS's contract is higher-is-better, then
        scaled to unit range so `attraction_weight` means the same thing at every step --
        `rss` shrinks as the equation grows, and an unscaled field would quietly change what
        the policy's weight is worth."""
        field = self._field()
        field.tell([1, 2, 3], [1.0, 0.5, 0.0])
        values = field._attractiveness()
        self.assertAlmostEqual(float(values.min()), 0.0)
        self.assertAlmostEqual(float(values.max()), 1.0)
        self.assertLess(float(values[0]), float(values[2]))

    def test_a_flat_field_does_not_divide_by_zero(self) -> None:
        field = self._field()
        field.tell([1, 2, 3], [0.5, 0.5, 0.5])
        self.assertTrue(np.isfinite(field._attractiveness()).all())

    def test_attraction_changes_where_the_probes_go(self) -> None:
        """The null the guided policies have to beat: same probes, placed by repulsion alone.
        If these agreed, `probe_attraction` would be an inert knob."""
        measured = list(range(14))
        objectives = [1.0 - 0.05 * index for index in measured]
        unguided, guided = self._field(), self._field(attraction_weight=2.0)
        unguided.attraction_weight = 0.0
        for field in (unguided, guided):
            field.tell(measured, objectives)
        self.assertNotEqual(
            unguided.propose(6, taken=measured), guided.propose(6, taken=measured)
        )


if __name__ == "__main__":
    unittest.main()
