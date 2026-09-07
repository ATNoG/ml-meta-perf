#!/bin/bash
# The smallest equation that holds up on every test at once. Submit with:
#   sbatch --job-name=eqsrch --cpus-per-task=62 scripts/equation_search.sbatch
#
# 48,576 points -- 16 feature subsets x 11 penalties x 23 lengths x 6 z-caps x 2 arities --
# at about 4.85 s each, so roughly 65 core-hours and an hour of wall-clock on 62 cores.
#
# Searches under the same rules the study publishes under, including the one-term-per-feature-
# combination constraint added on 2026-09-07: `fit.Selector` receives `library.feature_groups`
# on every path, folds included. A sweep run before that date searched a different grammar and
# its results are not comparable with one run after it.
#
# `cpuPartition` only: the ARM node is armPartition-only and a different BLAS build is not
# worth the risk of a divergent result. This is caution rather than a live hazard --
# `terms.build_library` became order-independent on 2026-09-05, so an equation is now a
# function of its feature set rather than of the order it was handed them in.
#
# Reads the packaged corpus, so nothing has to be generated first. Do not read the top row of
# the output as the answer: `objective` is a weighted sum over seven components measured on
# the same twenty folds, so the leading rows are ties. Pair the top band against the incumbent
# with `validate.paired_comparison` and take the shortest that is not significantly worse.

#SBATCH --partition=cpuPartition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --output=results/cluster/slurm-%x-%j.out
#SBATCH --error=results/cluster/slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/cluster

# One BLAS thread per worker: each joblib worker runs an independent fit, and letting every
# one spawn a 62-thread pool oversubscribes the node by three orders of magnitude.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export JOBLIB_START_METHOD=loky
export PYTHONUNBUFFERED=1

echo "=== $(date -Is) node=$(hostname) job=$SLURM_JOB_ID cores=$SLURM_CPUS_PER_TASK ==="
uname -m
venv/bin/python -c "import numpy, polars; print('numpy', numpy.__version__, 'polars', polars.__version__)"

exec venv/bin/python -m ml_meta_perf.equation_search_cli \
  --jobs "$SLURM_CPUS_PER_TASK" \
  --output results/cluster \
  --name equation_search
