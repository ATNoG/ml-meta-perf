"""The command line: the flags, and what each one causes to be written.

**The engine is mocked here.** `main` does three separable things -- fold the flags onto a
configuration, run the study, and write what came back into files -- and only the first and
third are the command line's. Letting a CLI test run the study meant every one of them paid
for a fit and a set of cross-validation folds to find out whether `--no-tables` skips the
tables, which is not a question about fitting; three of these tests were 13, 13 and 25 seconds
for that reason. With `cli.run` patched they assert something sharper as well as faster: that
the flags reach the engine as the right arguments, which no end-to-end run can check because
an end-to-end run only sees what came out the far side.

The report handed back by the mock is the shared one from `tests.corpus`, fitted on the sample
slice, and `--data` points at the same slice so that the frame `main` reloads for the figures
and the chapters matches the report it was given.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from ml_meta_perf.cli import _configure_windows_output, arities, build_parser, configuration, main, render
from ml_meta_perf.experiment import ARITIES, DEFAULT
from ml_meta_perf.model import Equation
from ml_meta_perf.report import BEGIN, END
from tests import corpus


class TestFlags(unittest.TestCase):
    """Parsing and folding, with nothing else involved."""

    def test_a_bare_run_is_the_tuned_configuration(self) -> None:
        self.assertEqual(configuration(build_parser().parse_args([])), DEFAULT)

    def test_flags_override_the_tuned_configuration(self) -> None:
        parser = build_parser()
        config = configuration(parser.parse_args(["--penalty", "3", "--arity", "2", "--max-terms", "40"]))
        self.assertEqual(config.penalty, 3.0)
        self.assertEqual(config.max_arity, 2)
        self.assertEqual(config.max_terms, 40)
        self.assertEqual(config.pool_size, DEFAULT.pool_size, "an unmentioned knob keeps its tuned value")

    def test_the_grammar_a_non_searching_equation_gets_is_the_smallest_searched(self) -> None:
        """E1 and E2 are fitted once rather than searched, so they need an arity from
        somewhere. It is the most parsimonious of the ones asked for, not the first written."""
        parser = build_parser()
        self.assertEqual(configuration(parser.parse_args(["--arity", "3", "--arity", "2"])).max_arity, 2)
        self.assertEqual(configuration(parser.parse_args(["--arity", "3"])).max_arity, 3)

    def test_arity_is_repeatable_and_de_duplicated_in_order(self) -> None:
        parser = build_parser()
        self.assertEqual(arities(parser.parse_args(["--arity", "3", "--arity", "2", "--arity", "3"])), (3, 2))
        self.assertEqual(arities(parser.parse_args([])), ARITIES)

    def test_the_published_length_is_not_a_flag(self) -> None:
        """The selected length is derived from the curve rather than set by a flag."""
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--terms", "12"])

    def test_windows_output_is_reconfigured_for_unicode_tables(self) -> None:
        stream = mock.Mock()
        with mock.patch("ml_meta_perf.cli.sys.platform", "win32"), mock.patch(
            "ml_meta_perf.cli.sys.stdout", stream
        ):
            _configure_windows_output()
        stream.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")


class CliTestCase(unittest.TestCase):
    """`main` with the engine replaced, over a temporary output directory.

    `self.engine` is the patched `cli.run`; asserting on its call is how these tests check
    that a flag arrived, rather than inferring it from a number in a file.
    """

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="ml-meta-perf-cli-"))
        self.docs = self.directory / "docs"
        self.docs.mkdir()
        patcher = mock.patch("ml_meta_perf.cli.run", return_value=corpus.report())
        self.engine = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.directory, ignore_errors=True)

    def run_main(self, *flags: str) -> str:
        """`main` over the sample slice, returning what it printed. Asserts it exited 0."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(
                [
                    "--data",
                    str(corpus.sample_path()),
                    "--output",
                    str(self.directory),
                    "--docs",
                    str(self.docs),
                    "--no-figures",
                    *flags,
                ]
            )
        self.assertEqual(code, 0)
        return buffer.getvalue()

    def chapter(self, name: str = "05-evaluation.md", prose: str = "Prose that must survive.") -> Path:
        path = self.docs / name
        path.write_text(f"# 5. Evaluation\n\n{prose}\n")
        return path


class TestWhatTheFlagsReachTheEngineAs(CliTestCase):
    def test_the_default_run_is_the_tuned_one_over_the_default_arities(self) -> None:
        self.run_main("--quiet", "--no-report", "--no-tables")
        self.engine.assert_called_once()
        arguments, keywords = self.engine.call_args
        self.assertEqual(arguments[0], str(corpus.sample_path()))
        self.assertEqual(keywords["config_e1"], DEFAULT)
        self.assertEqual(keywords["config_e2"], DEFAULT)
        self.assertEqual(keywords["config_e3"], DEFAULT)
        self.assertEqual(keywords["arities"], ARITIES)

    def test_a_repeated_arity_becomes_the_searched_set(self) -> None:
        self.run_main("--quiet", "--no-report", "--no-tables", "--arity", "3", "--arity", "2", "--arity", "3")
        self.assertEqual(self.engine.call_args.kwargs["arities"], (3, 2))

    def test_there_is_no_preset_that_sets_knobs_the_flags_do_not(self) -> None:
        """`--quick` was three flags that already existed plus two that did not: it also chose
        E2's configuration and the opaque ensemble sizes, which is how it came to advertise
        "seconds" while paying 226 s for the comparison. Every knob is a flag or a parameter
        now, so a cheap run says at its call site which parts of it are cheap."""
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--quick"])


class TestWhatIsWritten(CliTestCase):
    def test_the_three_equations_are_saved(self) -> None:
        self.run_main("--quiet", "--no-report", "--no-tables")
        for name in ("e1.json", "e2.json", "e3.json"):
            path = self.directory / name
            with self.subTest(name=name):
                self.assertTrue(path.is_file())
                self.assertGreater(Equation.load(path).n_terms, 0)

    def test_the_tables_are_written_as_csv(self) -> None:
        printed = self.run_main("--quiet", "--no-report")
        written = sorted(path.name for path in self.directory.glob("*.csv"))
        self.assertIn("curve_e3.csv", written)
        self.assertIn("grammars.csv", written)
        self.assertIn(f"{len(written)} tables written", printed)

    def test_no_tables_writes_the_equations_and_nothing_else(self) -> None:
        self.run_main("--quiet", "--no-report", "--no-tables")
        self.assertEqual(list(self.directory.glob("*.csv")), [])
        self.assertTrue((self.directory / "e3.json").is_file())

    def test_generated_sections_go_into_the_chapter_around_the_prose(self) -> None:
        chapter = self.chapter()
        self.run_main("--quiet", "--no-tables")
        written = chapter.read_text()
        self.assertIn("Prose that must survive.", written)
        self.assertIn(BEGIN, written)
        self.assertIn(END, written)

    def test_regenerating_a_chapter_replaces_the_block_rather_than_appending(self) -> None:
        chapter = self.chapter()
        for _ in range(2):
            self.run_main("--quiet", "--no-tables")
        written = chapter.read_text()
        self.assertEqual(written.count(BEGIN), 1)
        self.assertEqual(written.count(END), 1)
        self.assertEqual(written.count("Prose that must survive."), 1)

    def test_no_report_leaves_the_chapter_untouched(self) -> None:
        chapter = self.chapter()
        before = chapter.read_text()
        self.run_main("--quiet", "--no-tables", "--no-report")
        self.assertEqual(chapter.read_text(), before)

    def test_a_chapter_is_not_rewritten_when_its_phase_was_not_requested(self) -> None:
        """`--phase` gates the writing as well as the printing: a partial run that rewrote the
        chapters would leave them describing a study that was never finished."""
        chapter = self.chapter()
        before = chapter.read_text()
        self.run_main("--quiet", "--no-tables", "--phase", "screen")
        self.assertEqual(chapter.read_text(), before)


class TestWhatIsPrinted(CliTestCase):
    def test_quiet_writes_the_files_without_the_study(self) -> None:
        printed = self.run_main("--quiet", "--no-report", "--no-tables")
        self.assertNotIn("Correlation screening", printed)
        self.assertIn("equations written to", printed)

    def test_phase_selection_limits_what_is_printed(self) -> None:
        printed = self.run_main("--no-report", "--no-tables", "--phase", "screen")
        self.assertIn("Correlation screening", printed)
        self.assertNotIn("Oracle ladder", printed)

    def test_render_prints_every_section(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            render(corpus.report(), columns=corpus.columns())
        printed = buffer.getvalue()
        for expected in (
            "Correlation screening",
            "E1 --",
            "E2 --",
            "E3 --",
            "Where the signal lives",
            "Extracted practices",
            "Baselines",
            "Model selection",
        ):
            with self.subTest(section=expected):
                self.assertIn(expected, printed)


if __name__ == "__main__":
    unittest.main()
