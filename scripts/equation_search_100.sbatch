#!/bin/bash
# Recalibrate the E3 configuration against the current packaged meta-dataset. Submit from the
# repository root with:
#
#   sbatch scripts/equation_search_100.sbatch
#
# The Python command is resumable. Submitting the same script again after a timeout skips every
# complete shard under results/configuration_search_max100.

#SBATCH --job-name=e3-config-search-100
#SBATCH --partition=cpuPartition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

seff --monitor --interval 10 &
MONITOR_PID=$!

cleanup() {
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [[ ! -x venv/bin/python ]]; then
    echo "venv/bin/python does not exist; create the environment and install the project first" >&2
    exit 2
fi

# Every joblib worker runs independent small linear solves. Letting BLAS create another pool
# inside each worker would oversubscribe the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export JOBLIB_START_METHOD=loky
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== $(date -Is) node=$(hostname) job=$SLURM_JOB_ID cores=$SLURM_CPUS_PER_TASK ==="
venv/bin/python -c "from importlib.metadata import version; print(*(f'{name} {version(name)}' for name in ('numpy', 'polars', 'joblib')), sep='; ')"

exec venv/bin/python -m ml_meta_perf.configuration_search all \
    --data src/ml_meta_perf/meta_dataset.csv \
    --output results/configuration_search_max100 \
    --jobs "$SLURM_CPUS_PER_TASK" \
    --min-terms 1 \
    --max-terms 100 \
    --plateau-tolerance 0.001 \
    --plateau-window 3
