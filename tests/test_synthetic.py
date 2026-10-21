"""The planted corpora: reproducible, shaped as advertised, and recoverable by the search."""

import unittest

import numpy as np
import polars as pl

from ml_meta_perf.data import columns_as_arrays, target
from ml_meta_perf.search import search
from ml_meta_perf.terms import build_library
from ml_meta_perf.validate import score
from tests.synthetic import DATASET_FEATURES, MODEL_FEATURES, cost_shaped, planted


class TestPlanted(unittest.TestCase):
    def test_the_same_seed_gives_the_same_corpus(self) -> None:
        self.assertTrue(planted(3, seed=5).frame.equals(planted(3, seed=5).frame))
        self.assertFalse(planted(3, seed=5).frame.equals(planted(3, seed=6).frame))

    def test_it_has_one_row_per_dataset_and_model_and_the_stated_groups(self) -> None:
        corpus = planted(2, datasets=5, models=4)
        self.assertEqual(corpus.frame.height, 20)
        self.assertEqual((corpus.frame["d"].n_unique(), corpus.frame["m"].n_unique()), (5, 4))
        self.assertEqual(corpus.k, 2)

    def test_dataset_features_are_constant_within_a_dataset_and_model_features_within_a_model(self) -> None:
        frame = planted(2).frame
        for name in DATASET_FEATURES:
            self.assertTrue((frame.group_by("d").agg(pl.col(name).n_unique()).to_numpy()[:, 1] == 1).all())
        for name in MODEL_FEATURES:
            self.assertTrue((frame.group_by("m").agg(pl.col(name).n_unique()).to_numpy()[:, 1] == 1).all())

    def test_an_unplantable_k_is_an_error(self) -> None:
        for k in (0, 7):
            with self.assertRaises(ValueError):
                planted(k)

    def test_without_noise_the_target_is_a_sum_of_k_unit_variance_terms_and_one_term_is_not_enough(self) -> None:
        one = planted(1, noise=0.0).frame["y"].to_numpy()
        many = planted(4, noise=0.0).frame["y"].to_numpy()
        self.assertAlmostEqual(float(one.std()), 1.0, places=6)
        self.assertGreater(float(many.std()), 1.0)

    def curve(self, k: int, noise: float) -> list[float]:
        corpus = planted(k, noise=noise)
        schema = corpus.schema()
        columns = columns_as_arrays(corpus.frame, schema.features)
        library = build_library(schema.dataset_features, schema.model_features, columns, max_arity=2)
        truth = target(corpus.frame, schema)
        found = search(library, truth, max_terms=6, penalty=1e-4, pool_size=300, beam_width=6, bounds=None)
        return [score(truth, found.equations[n].predict(columns)).r2 for n in sorted(found.equations)]

    def test_the_planted_signal_is_found_without_noise_and_needs_its_terms(self) -> None:
        two, three = self.curve(2, 0.0), self.curve(3, 0.0)
        self.assertGreater(two[1], 0.999)
        self.assertLess(two[0], 0.9)
        self.assertGreater(three[-1], 0.999)
        self.assertLess(three[0], 0.6)

    def test_noise_sets_the_ceiling_of_the_curve_at_one_over_one_plus_noise_squared(self) -> None:
        ceiling = 1.0 / (1.0 + 0.3**2)
        self.assertAlmostEqual(self.curve(2, 0.3)[-1], ceiling, delta=0.03)

    def test_the_cost_shape_is_positive_and_spans_orders_of_magnitude(self) -> None:
        cost = cost_shaped(3).frame["y"].to_numpy()
        self.assertTrue((cost > 0.0).all())
        self.assertGreater(float(np.log10(cost.max() / cost.min())), 1.0)


if __name__ == "__main__":
    unittest.main()
