"""``ml-meta-perf fit | sweep | init``: the commands, the result folder and the refusals."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from ml_meta_perf import cli
from ml_meta_perf.config import parse_config
from ml_meta_perf.output import file_sha256, markdown_table, study_sha256
from tests.synthetic import planted

#: One equation under two protocols: enough to exercise the writing, and fast enough to run it twice.
QUICK = ["--equations", '["E3"]', "--protocols", '["IS", "LODO"]']
SMALL = [
    "--search.max_terms",
    "3",
    "--search.pool_size",
    "40",
    "--search.beam_width",
    "1",
    "--search.max_arity",
    "2",
    "--selection.capability_arity",
    "2",
    "--selection.window",
    "1",
]


def run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as exit_:
            code = int(exit_.code or 0)
    return code, out.getvalue(), err.getvalue()


class Corpus(unittest.TestCase):
    """A planted corpus on disk, and the study file ``init`` writes for it."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv = self.root / "runs.csv"
        planted(3, noise=0.1, seed=1).frame.write_csv(self.csv)
        self.study = self.root / "study.json"

    def init(self, *more: str) -> tuple[int, str, str]:
        return run(
            [
                "init",
                str(self.csv),
                "--dataset",
                "d",
                "--model",
                "m",
                "--response",
                "y",
                "--output",
                str(self.study),
                *more,
            ]
        )

    def fit(self, *more: str) -> tuple[int, str, str]:
        return run(["fit", "--config", str(self.study), "--output", str(self.root / "out"), "--quiet", *SMALL, *more])


class TestInit(Corpus):
    def test_it_writes_a_study_file_that_parses(self) -> None:
        code, out, _ = self.init()
        self.assertEqual(code, 0)
        study = parse_config(json.loads(self.study.read_text()))
        self.assertEqual(
            (study.corpus.dataset_column, study.corpus.model_column, study.response.column), ("d", "m", "y")
        )
        self.assertIn("wrote", out)

    def test_a_column_constant_within_each_dataset_is_a_dataset_feature_and_the_rest_are_model_features(self) -> None:
        self.init()
        features = json.loads(self.study.read_text())["features"]
        self.assertEqual(sorted(features["dataset"]), ["depth", "rate", "size", "width"])
        self.assertEqual(sorted(features["model"]), ["batch", "epochs", "lr"])

    def test_the_protocols_and_equations_follow_the_groups_it_was_given(self) -> None:
        self.init()
        study = json.loads(self.study.read_text())
        self.assertEqual(
            (study["protocols"], study["equations"]), (["IS", "LODO", "LOMO", "DHO"], ["E1", "E2", "E3", "E3-MAX"])
        )

    def test_without_a_model_column_only_the_dataset_protocols_and_one_equation_are_offered(self) -> None:
        pl.read_csv(self.csv).with_columns(run=pl.int_range(pl.len())).write_csv(self.csv)
        code, _, _ = run(["init", str(self.csv), "--dataset", "d", "--response", "y", "--output", str(self.study)])
        self.assertEqual(code, 0)
        study = json.loads(self.study.read_text())
        self.assertEqual(
            (study["protocols"], study["equations"], study["corpus"]["model_column"]),
            (["IS", "LODO"], ["E3", "E3-MAX"], None),
        )
        self.assertIn("run", study["features"]["model"])

    def test_constant_and_non_numeric_columns_are_left_out_and_said_so(self) -> None:
        pl.read_csv(self.csv).with_columns(flat=pl.lit(1.0), tag=pl.lit("a")).write_csv(self.csv)
        _, out, _ = self.init()
        self.assertIn("left out (constant): flat", out)
        self.assertIn("left out (not numeric): tag", out)
        features = json.loads(self.study.read_text())["features"]
        self.assertNotIn("flat", features["dataset"] + features["model"])

    def test_it_refuses_to_overwrite_a_study_file_without_force(self) -> None:
        self.init()
        code, _, err = self.init()
        self.assertEqual(code, 2)
        self.assertIn("exists", err)
        self.assertEqual(self.init("--force")[0], 0)

    def test_a_missing_column_is_named_with_the_columns_there_are(self) -> None:
        code, _, err = run(["init", str(self.csv), "--dataset", "nope", "--response", "y", "--output", str(self.study)])
        self.assertEqual(code, 2)
        self.assertIn("nope", err)
        self.assertIn("size", err)

    def test_bounds_and_transform_reach_the_file(self) -> None:
        self.init("--bounds", "[0, null]", "--transform", "log1p")
        response = json.loads(self.study.read_text())["response"]
        self.assertEqual((response["bounds"], response["transform"]), ([0.0, None], "log1p"))

    def test_an_unknown_transform_is_refused(self) -> None:
        code, _, err = self.init("--transform", "sqrt")
        self.assertEqual(code, 2)
        self.assertIn("transform must be one of", err)

    def test_a_bounds_argument_that_is_not_a_pair_is_refused(self) -> None:
        code, _, err = self.init("--bounds", "[1, 2, 3]")
        self.assertEqual(code, 2)
        self.assertIn("pair", err)

    def test_a_transform_the_target_cannot_take_is_refused(self) -> None:
        code, _, err = self.init("--transform", "log", "--bounds", "[0, null]")
        self.assertEqual(code, 2)
        self.assertIn("always positive", err)


class TestFit(Corpus):
    def setUp(self) -> None:
        super().setUp()
        self.init()
        self.folder = self.root / "out" / "runs" / "y"

    def test_it_writes_the_result_folder_named_by_the_csv_and_the_target(self) -> None:
        code, out, _ = self.fit()
        self.assertEqual(code, 0)
        self.assertIn("files written to", out)
        for name in ("scores.csv", "term_effects.csv", "group_shares.csv", "report.md", "manifest.json"):
            self.assertTrue((self.folder / name).is_file(), name)
        for variant in ("e1", "e2", "e3", "e3-max"):
            for name in ("equation.json", "equation.txt", "curve.csv"):
                self.assertTrue((self.folder / variant / name).is_file(), f"{variant}/{name}")

    def test_the_manifest_records_the_corpus_and_the_study(self) -> None:
        self.fit()
        manifest = json.loads((self.folder / "manifest.json").read_text())
        self.assertEqual(manifest["corpus"]["sha256"], file_sha256(self.csv))
        self.assertEqual(manifest["corpus"]["rows"], 112)
        self.assertEqual(manifest["corpus"]["path"], str(self.csv.resolve()))
        self.assertEqual(manifest["equations"], ["E1", "E2", "E3", "E3-MAX"])

    def test_the_study_hash_changes_when_a_flag_changes_the_study(self) -> None:
        self.fit(*QUICK)
        first = json.loads((self.folder / "manifest.json").read_text())["study_sha256"]
        self.fit(*QUICK, "--search.penalty", "7.0")
        self.assertNotEqual(json.loads((self.folder / "manifest.json").read_text())["study_sha256"], first)

    def test_the_report_is_made_of_the_numbers_of_the_run(self) -> None:
        self.fit()
        text = (self.folder / "report.md").read_text()
        self.assertTrue(text.startswith("# y"))
        self.assertIn("112 rows, 14 datasets, 8 models", text)
        for heading in (
            "## Equations",
            "### E3-Valid",
            "### E3-MAX",
            "## Scores",
            "## What the terms of E3-Valid are worth",
        ):
            self.assertIn(heading, text)
        scores = pl.read_csv(self.folder / "scores.csv")
        row = scores.filter(
            (pl.col("variant") == "E3-Valid") & (pl.col("protocol") == "LODO") & (pl.col("scale") == "fit")
        ).row(0, named=True)
        self.assertIn(f"{row['r2']:.4g}", text)

    def test_a_second_run_on_the_same_corpus_replaces_the_folder(self) -> None:
        self.assertEqual(self.fit(*QUICK)[0], 0)
        self.assertEqual(self.fit(*QUICK)[0], 0)

    def test_a_different_corpus_content_is_not_overwritten_without_force(self) -> None:
        self.fit(*QUICK)
        planted(3, noise=0.1, seed=2).frame.write_csv(self.csv)
        code, _, err = self.fit(*QUICK)
        self.assertEqual(code, 2)
        self.assertIn("different corpus", err)
        self.assertEqual(self.fit(*QUICK, "--force")[0], 0)
        self.assertEqual(
            json.loads((self.folder / "manifest.json").read_text())["corpus"]["sha256"], file_sha256(self.csv)
        )

    def test_a_flag_that_makes_the_study_unusable_exits_two(self) -> None:
        code, _, err = self.fit("--equations", '["E1"]')
        self.assertEqual(code, 2)
        self.assertIn("equations must be", err)

    def test_data_overrides_the_corpus_path(self) -> None:
        other = self.root / "other.csv"
        planted(3, noise=0.1, seed=1).frame.write_csv(other)
        self.assertEqual(self.fit(*QUICK, "--data", str(other))[0], 0)
        self.assertTrue((self.root / "out" / "other" / "y" / "manifest.json").is_file())

    def test_a_log_scale_adds_the_scores_on_the_target_itself(self) -> None:
        self.init("--force", "--bounds", "[0, null]", "--transform", "log1p")
        pl.read_csv(self.csv).with_columns((pl.col("y") - pl.col("y").min() + 1.0).alias("y")).write_csv(self.csv)
        self.assertEqual(self.fit()[0], 0)
        text = (self.root / "out" / "runs" / "y-log1p" / "report.md").read_text()
        self.assertIn("Scores (on the target itself)", text)
        self.assertIn("## Scores on the fitted scale", text)


class TestSweep(Corpus):
    def test_the_sweep_subcommand_writes_beside_the_result_folder(self) -> None:
        self.init()
        code, out, _ = run(
            ["sweep", "--config", str(self.study), "--output", str(self.root / "out"), "--jobs", "1", *SMALL,
             "--sweep.zscores", "[3.0]", "--sweep.penalties", "[1.0]", "--sweep.arities", "[1]"]
        )  # fmt: skip
        self.assertEqual(code, 0)
        self.assertIn("proposed for y", out)
        self.assertTrue((self.root / "out" / "runs" / "y" / "sweep" / "proposed_study.json").is_file())


class TestParser(unittest.TestCase):
    def test_a_command_is_required(self) -> None:
        code, _, err = run([])
        self.assertEqual(code, 2)
        self.assertIn("fit", err)

    def test_an_unknown_command_is_refused(self) -> None:
        self.assertEqual(run(["plot"])[0], 2)

    def test_the_version_is_printed(self) -> None:
        code, out, _ = run(["--version"])
        self.assertEqual(code, 0)
        self.assertIn("ml-meta-perf", out)


class TestWindowsOutput(unittest.TestCase):
    def test_a_legacy_windows_console_is_switched_to_utf8_and_other_platforms_are_left_alone(self) -> None:
        from unittest import mock

        stream = mock.Mock()
        with mock.patch.object(cli.sys, "platform", "win32"), mock.patch.object(cli.sys, "stdout", stream):
            cli._configure_windows_output()  # pyright: ignore[reportPrivateUsage]
        stream.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")
        other = mock.Mock()
        with mock.patch.object(cli.sys, "platform", "linux"), mock.patch.object(cli.sys, "stdout", other):
            cli._configure_windows_output()  # pyright: ignore[reportPrivateUsage]
        other.reconfigure.assert_not_called()


class TestHelpers(unittest.TestCase):
    def test_a_markdown_table_has_a_header_a_rule_and_a_row_per_row(self) -> None:
        table = markdown_table(pl.DataFrame({"a": [1, 2], "b": [0.123456, float("nan")]})).split("\n")
        self.assertEqual(table[:2], ["| a | b |", "|---|---|"])
        self.assertEqual(table[2:], ["| 1 | 0.1235 |", "| 2 |  |"])

    def test_the_study_hash_depends_on_the_study_only(self) -> None:
        from ml_meta_perf.config import load_config

        study = load_config()
        self.assertEqual(study_sha256(study), study_sha256(load_config()))


if __name__ == "__main__":
    unittest.main()
