#!/bin/bash
# The smallest equation that holds up on every test at once. Submit with:
#   sbatch --job-name=eqsrch --cpus-per-task=62 scripts/equation_search.sbatch
#
# `cpuPartition` only: the ARM node is armPartition-only and a different BLAS build is not
# worth the risk of a divergent result. This is caution rather than a live hazard --
# `terms.build_library` became order-independent on 2026-09-05, so an equation is now a
# function of its feature set rather than of the order it was handed them in.
#
# Reads the packaged corpus, so nothing has to be generated first.

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
