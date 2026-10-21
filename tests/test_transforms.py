"""The target scales a study can fit on, and the scores reported on the target's own scale."""

import dataclasses
import json
import unittest

import numpy as np
import polars as pl

from ml_meta_perf.config import DEFAULT_CONFIG_PATH, ConfigError, Schema, default_configuration, parse_config
from ml_meta_perf.data import SchemaError, from_fit_scale, target, to_fit_scale
from ml_meta_perf.experiment import finalise, fit_path, original_scale_scores
from ml_meta_perf.validate import score
from tests.synthetic import DATASET_FEATURES, MODEL_FEATURES, cost_shaped

CONFIG = dataclasses.replace(default_configuration(), max_terms=3, pool_size=40, beam_width=1, max_arity=1)


def schema(transform: str, bounds: tuple[float | None, float | None] | None = None, **kwargs: object) -> Schema:
    return Schema("d", "m", DATASET_FEATURES, MODEL_FEATURES, "y", bounds, transform=transform, **kwargs)  # pyright: ignore[reportArgumentType]


class TestScales(unittest.TestCase):
    def test_each_scale_is_undone_by_its_inverse(self) -> None:
        values = np.array([0.5, 1.0, 3.0, 40.0, 1.0e6])
        for transform in ("none", "log1p", "log"):
            with self.subTest(transform):
                scale = schema(transform)
                np.testing.assert_allclose(from_fit_scale(to_fit_scale(values, scale), scale), values, rtol=1e-12)

    def test_the_scales_are_the_logarithms_they_name(self) -> None:
        values = np.array([1.0, np.e, 10.0])
        np.testing.assert_allclose(to_fit_scale(values, schema("log")), np.log(values))
        np.testing.assert_allclose(to_fit_scale(values, schema("log1p")), np.log1p(values))
        np.testing.assert_array_equal(to_fit_scale(values, schema("none")), values)

    def test_a_value_outside_the_domain_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(SchemaError, "zero or negative"):
            to_fit_scale(np.array([1.0, 0.0]), schema("log"))
        with self.assertRaisesRegex(SchemaError, "negative values"):
            to_fit_scale(np.array([1.0, -0.1]), schema("log1p"))
        for transform in ("log", "log1p"):
            with self.subTest(transform), self.assertRaisesRegex(SchemaError, "non-finite"):
                to_fit_scale(np.array([1.0, np.inf]), schema(transform))
        self.assertEqual(float(to_fit_scale(np.array([0.0]), schema("log1p"))[0]), 0.0)

    def test_the_frame_target_is_read_on_the_fitted_scale(self) -> None:
        frame = pl.DataFrame({"y": [1.0, 2.0, 4.0]})
        np.testing.assert_allclose(target(frame, schema("log")), np.log([1.0, 2.0, 4.0]))

    def test_log_names_its_outputs_and_carries_its_bounds_to_the_fitted_scale(self) -> None:
        scale = schema("log", (1.0, np.e**2))
        self.assertEqual((scale.label, scale.slug, scale.tag("E3")), ("log(y)", "y-log", "E3_log"))
        bounds = scale.fit_bounds
        assert bounds is not None and bounds[0] is not None and bounds[1] is not None
        self.assertAlmostEqual(bounds[0], 0.0)
        self.assertAlmostEqual(bounds[1], 2.0)
        self.assertEqual(schema("log1p").label, "log1p(y)")
        self.assertEqual(schema("none").tag("E3"), "E3")

    def test_log_needs_a_target_that_is_always_positive(self) -> None:
        for bounds in ((0.0, None), (-1.0, 5.0), (None, 5.0)):
            with self.subTest(bounds), self.assertRaisesRegex(ValueError, "always positive"):
                schema("log", bounds)
        schema("log", None)
        schema("log", (0.001, None))


class TestStudyFile(unittest.TestCase):
    def payload(self, transform: str, bounds: object) -> dict[str, object]:
        payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        payload["response"] = {"column": "MCC", "bounds": bounds, "transform": transform}
        return payload

    def test_a_study_may_ask_for_log_on_a_positive_target(self) -> None:
        study = parse_config(self.payload("log", [0.5, None]))
        self.assertEqual(study.schema().transform, "log")

    def test_a_study_cannot_ask_for_log_on_a_target_that_may_be_zero(self) -> None:
        for bounds in ([0.0, None], [None, 3.0]):
            with self.subTest(bounds), self.assertRaisesRegex(ConfigError, "response.transform log: log needs"):
                parse_config(self.payload("log", bounds))

    def test_an_unknown_transform_lists_the_known_ones(self) -> None:
        with self.assertRaisesRegex(ConfigError, "log1p"):
            parse_config(self.payload("sqrt", None))


class TestOriginalScaleScores(unittest.TestCase):
    def fitted(self, transform: str):
        corpus = cost_shaped(3, noise=0.1, seed=1)
        scale = schema(transform, (0.0, None) if transform != "log" else None, protocols=("IS", "LODO"))
        frame = corpus.frame
        path = fit_path(frame, DATASET_FEATURES, MODEL_FEATURES, CONFIG, scale.tag("E3"), schema=scale)
        return frame, scale, finalise(path, 2)

    def test_an_untransformed_target_scores_the_same_on_both_scales(self) -> None:
        frame, scale, report = self.fitted("none")
        table = original_scale_scores(report, frame, scale, "E3-Valid")
        for protocol in ("IS", "LODO"):
            fit, original = (
                table.filter((pl.col("protocol") == protocol) & (pl.col("scale") == s)) for s in ("fit", "original")
            )
            self.assertEqual(fit["r2"][0], original["r2"][0])

    def test_only_the_declared_protocols_are_reported(self) -> None:
        frame, scale, report = self.fitted("log1p")
        table = original_scale_scores(report, frame, scale, "E3-Valid")
        self.assertEqual(set(table["protocol"]), {"IS", "LODO"})
        self.assertEqual(set(table["scale"]), {"fit", "original"})
        self.assertEqual(table.height, 4)

    def test_the_original_scale_is_scored_against_the_raw_target(self) -> None:
        frame, scale, report = self.fitted("log1p")
        table = original_scale_scores(report, frame, scale, "E3-Valid")
        truth = frame["y"].to_numpy()
        manual = score(truth, np.expm1(report.paths["loo_dataset"][report.n_terms].predictions))
        row = table.filter((pl.col("protocol") == "LODO") & (pl.col("scale") == "original")).row(0, named=True)
        self.assertAlmostEqual(row["r2"], manual.r2)
        self.assertAlmostEqual(row["mae"], manual.mae)
        fit_row = table.filter((pl.col("protocol") == "LODO") & (pl.col("scale") == "fit")).row(0, named=True)
        self.assertAlmostEqual(fit_row["r2"], report.cross_validated["loo_dataset"]["r2"])

    def test_a_log_fit_back_transforms_with_exp(self) -> None:
        frame, scale, report = self.fitted("log")
        table = original_scale_scores(report, frame, scale, "E3-Valid")
        manual = score(
            frame["y"].to_numpy(), np.exp(report.equation.predict({f: frame[f].to_numpy() for f in scale.features}))
        )
        row = table.filter((pl.col("protocol") == "IS") & (pl.col("scale") == "original")).row(0, named=True)
        self.assertAlmostEqual(row["r2"], manual.r2)


if __name__ == "__main__":
    unittest.main()
