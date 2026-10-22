"""Command-line entry point for ``python -m ml_meta_perf``.

Delegates directly to `ml_meta_perf.cli.main`, which is also installed as the
``ml-meta-perf`` console script (with subcommands ``fit``, ``sweep`` and ``init``).

Note for anyone tempted to pin BLAS threading here: it does not work. ``python -m
ml_meta_perf`` imports the ``ml_meta_perf`` package -- and therefore NumPy, and therefore OpenBLAS --
*before* this module runs, and OpenBLAS reads its thread count when it loads. It has to be
set in the environment by the caller (e.g. ``OPENBLAS_NUM_THREADS=1``), which is a decision
left to them rather than forced from inside a library; [chapter 3][study-chapter] measures
what it is worth.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/03-term-selection.md
"""

import sys

from ml_meta_perf.cli import main

if __name__ == "__main__":
    sys.exit(main())
