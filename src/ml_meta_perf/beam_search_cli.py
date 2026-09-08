"""One Slurm job: does a better beam find a better equation?

Two stages, and they answer different questions. **Do not read the first as the answer** --
that mistake has already been made once on this project, with a nine-term equation that
topped the objective and lost a paired test.

``sweep`` crosses the beam policies of `ml_meta_perf.beam` with the *whole* configuration
grid the equation search uses. Every policy has to be given its own sweep: `DEFAULT_E3` was
itself chosen by a sweep under the vanilla beam, so scoring a different beam at that point
compares a tuned configuration against an untuned one and will always favour the incumbent.

``compare`` takes each policy's best configuration out of that sweep, refits it, and pairs it
against the incumbent on per-dataset absolute error with an exact sign test and a bootstrap
over the twenty held-out datasets. That is the decision, and a tie means the incumbent stays:
a variant that matches the published beam is more machinery for no measured gain.

Nothing here is imported by the study pipeline; `python -m ml_meta_perf` does not touch it.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import polars as pl
from joblib import Parallel, delayed

from ml_meta_perf.beam import BY_NAME, CATALOGUE, BeamPolicy
from ml_meta_perf.beam_compare import compare_all, run_policy
from ml_meta_perf.data import load
from ml_meta_perf.equation_search import SearchPoint, evaluate, grid
from ml_meta_perf.equation_search_cli import feature_subsets
from ml_meta_perf.experiment import Configuration

#: The configuration axes, matching the equation search so the two sweeps are comparable.
#: Crossing them with the policy catalogue is what makes this job large.
PENALTIES = (3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0)
ZSCORES = (3.0, 3.5, 4.0, 4.25, 4.5, 5.0)
ARITIES = (2, 3)

#: The two load-bearing model columns. Every subset the equation search considers keeps them,
#: because the ablation settled it: removing either costs 0.065 or more leave-one-dataset-out.
CORE_FEATURES = ("Model Capability", "Processing Units Number")

#: How much of the grid a run covers. Three scopes, and each answers a different question --
#: naming them is what stops a forty-minute job being reported as if it were the seven-hour
#: one.
#:
#: * ``full`` -- every feature subset, arity and z-cap. "Is there a beam policy that, given
#:   its own configuration sweep, beats the published one *anywhere* in the grid?"
#: * ``focused`` -- the operating point the study already chose (arity 2, the load-bearing
#:   feature pair), with the penalty and length axes at full resolution because those are what
#:   the published configuration is most sensitive to. "At the point we already picked, does a
#:   different beam do better?"
#: * ``quick`` -- a wiring check that fits on a laptop. **Its output is not a comparison**:
#:   the grid is too coarse for several policies to find different equations at all.
SCOPES: dict[str, dict[str, tuple]] = {
    "full": {"penalties": PENALTIES, "zscores": ZSCORES, "arities": ARITIES},
    "focused": {"penalties": PENALTIES, "zscores": (4.0, 4.25, 4.5), "arities": (2,)},
    "quick": {"penalties": (10.0, 20.0, 40.0), "zscores": (4.0, 4.25), "arities": (2,)},
}


def _policies(names: list[str] | None) -> tuple[BeamPolicy, ...]:
    if not names:
        return CATALOGUE
    missing = [name for name in names if name not in BY_NAME]
    if missing:
        raise SystemExit(f"unknown beam policies: {', '.join(missing)}")
    return tuple(BY_NAME[name] for name in names)


def _best_configuration(table: pl.DataFrame, policy: str) -> tuple[Configuration, tuple[str, ...]] | None:
    """The configuration this policy scored highest at, out of a finished sweep.

    Highest by ``objective``, which is the shortlisting device rather than the decision -- the
    paired test that follows is the decision, and it needs one candidate per policy to test.
    """
    rows = table.filter(pl.col("policy") == policy).sort("objective", descending=True)
    if rows.height == 0:
        return None
    row = rows.row(0, named=True)
    config = Configuration(
        max_abs_zscore=float(row["max_abs_zscore"]),
        penalty=float(row["penalty"]),
        pool_size=600,
        max_terms=max(int(row["headline_terms"]) + 4, 8),
        headline_terms=int(row["headline_terms"]),
        max_arity=int(row["max_arity"]),
    )
    return config, tuple(name.strip() for name in str(row["features"]).split(","))


def sweep(args: argparse.Namespace) -> int:
    frame = load(args.frame)
    policies = _policies(args.policies)
    scope = SCOPES[args.scope]
    points = grid(
        features=feature_subsets() if args.scope == "full" else [tuple(sorted(CORE_FEATURES))],
        penalties=scope["penalties"],
        term_counts=tuple(range(args.min_terms, args.max_terms + 1, args.term_step)),
        zscores=scope["zscores"],
        arities=scope["arities"],
        policies=policies,
    )
    logging.info("scope %s: %d points over %d policies", args.scope, len(points), len(policies))

    started = time.time()

    def run(point: SearchPoint) -> dict[str, object] | None:
        score = evaluate(point, frame)
        return score.as_row() if score else None

    rows = Parallel(n_jobs=args.jobs or -1, verbose=5)(delayed(run)(point) for point in points)
    table = pl.DataFrame([row for row in rows if row]).sort("objective", descending=True)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{args.name}.csv"
    table.write_csv(destination)
    logging.info("%d scored points in %.0fs -> %s", table.height, time.time() - started, destination)

    best = table.group_by("policy").agg(pl.all().sort_by("objective", descending=True).first()).sort(
        "objective", descending=True
    )
    print(best.select("policy", "objective", "loo_dataset_r2", "loo_model_r2", "stability", "n_terms"))
    print("\nThe top row is a shortlist, not a verdict. Run `compare` next.")
    return 0


def compare(args: argparse.Namespace) -> int:
    frame = load(args.frame)
    policies = _policies(args.policies)
    source = args.output / f"{args.name}.csv"

    configurations: dict[str, tuple[Configuration, tuple[str, ...]]] = {}
    if source.is_file():
        table = pl.read_csv(source)
        for policy in policies:
            found = _best_configuration(table, policy.name)
            if found is not None:
                configurations[policy.name] = found
        logging.info("read best configurations for %d policies from %s", len(configurations), source)
    else:
        # No sweep to read: score every policy at the published configuration and say so. It
        # is the wrong comparison -- that configuration was tuned under the vanilla beam --
        # but it is a useful smoke test and the header records the caveat.
        from ml_meta_perf.experiment import DEFAULT_E3, EQUATION_MODEL_FEATURES

        logging.warning("no sweep at %s; scoring every policy at DEFAULT_E3, which favours vanilla", source)
        configurations = {policy.name: (DEFAULT_E3, EQUATION_MODEL_FEATURES) for policy in policies}

    runs = []
    for policy in policies:
        found = configurations.get(policy.name)
        if found is None:
            logging.warning("no configuration for %s; skipped", policy.name)
            continue
        config, features = found
        outcome = run_policy(frame, config, features, policy)
        if outcome is None:
            logging.warning("%s produced no equation at its best configuration; skipped", policy.name)
            continue
        runs.append(outcome)
        logging.info(
            "%s: in-sample %.4f loo-dataset %.4f loo-model %.4f loo-cell %.4f (spread %.4f) in %.2fs",
            policy.name,
            outcome.r2_in_sample,
            outcome.r2_loo_dataset,
            outcome.r2_loo_model,
            outcome.r2_loo_cell,
            outcome.r2_spread,
            outcome.seconds,
        )

    verdicts = compare_all(runs, incumbent_name=args.incumbent)
    if verdicts.height == 0:
        logging.error("no comparison: the incumbent %r produced no run", args.incumbent)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{args.name}_verdicts.csv"
    verdicts.write_csv(destination)
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(
            verdicts.select(
                "policy", "standing", "mae_gain", "r2_loo_dataset_delta", "r2_loo_cell_delta",
                "r2_spread", "speedup", "frontier",
            )
        )
    counts = dict(verdicts.group_by("standing").len().iter_rows())
    print(
        f"\nAgainst {args.incumbent!r}, over both axes and all four protocols: "
        + ", ".join(f"{counts.get(name, 0)} {name}" for name in ("better", "cheaper", "equal", "worse"))
        + f", of {verdicts.height}."
    )
    print(
        "  better  = wins the paired test and gives up nothing on the strictest protocol\n"
        "  cheaper = same accuracy, faster by more than the timing noise\n"
        "  equal   = indistinguishable on both axes; more machinery for nothing\n"
        "  worse   = loses the paired test, or buys speed with the worst protocol"
    )
    print(
        "\nThis is ONE operating point. A policy that wins here and not at the published "
        "configuration has not won: run the other configuration before quoting either."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("sweep", "compare"), help="sweep proposes; compare decides")
    parser.add_argument("--frame", type=Path, default=None, help="defaults to the packaged corpus")
    parser.add_argument("--output", type=Path, default=Path("results/cluster"))
    parser.add_argument("--name", default="beam_search")
    parser.add_argument("--jobs", type=int, default=0, help="0 = every core the job holds")
    parser.add_argument("--policies", nargs="*", default=None, help="default: the whole catalogue")
    parser.add_argument("--incumbent", default="vanilla", help="the policy every other is paired against")
    parser.add_argument("--min-terms", type=int, default=6)
    parser.add_argument("--max-terms", type=int, default=28)
    parser.add_argument("--term-step", type=int, default=1)
    parser.add_argument(
        "--scope",
        choices=tuple(SCOPES),
        default="full",
        help="full: every subset and arity. focused: the published operating point. quick: wiring only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return sweep(args) if args.stage == "sweep" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
