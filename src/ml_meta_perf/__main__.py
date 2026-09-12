"""Allow ``python -m ml_meta_perf``.

The pipeline itself is `ml_meta_perf.cli`, which is also installed as the ``ml-meta-perf``
console script. Everything it reads is an argument and everything it writes is a
path, so ``ml-meta-perf --help`` is the complete interface.

Note for anyone tempted to pin BLAS threading here: it does not work. ``python -m
ml-meta-perf`` imports the `ml-meta-perf` package -- and therefore numpy, and therefore OpenBLAS --
*before* this module runs, and OpenBLAS reads its thread count when it loads. It has to be
set in the environment by the caller, which is a decision left to them rather than forced
from inside a library; [chapter 3][study-chapter] measures what
it is worth.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/03-term-selection.md
"""

import sys

from ml_meta_perf.cli import main

if __name__ == "__main__":
    sys.exit(main())
