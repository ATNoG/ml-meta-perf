"""Allow ``python -m metafit``.

The pipeline itself is `metafit.cli`, which is also installed as the ``metafit``
console script. Everything it reads is an argument and everything it writes is a
path, so ``metafit --help`` is the complete interface.

Note for anyone tempted to pin BLAS threading here: it does not work. ``python -m
metafit`` imports the `metafit` package -- and therefore numpy, and therefore OpenBLAS --
*before* this module runs, and OpenBLAS reads its thread count when it loads. It has to be
set in the environment by the caller, which is a decision left to them rather than forced
from inside a library; [chapter 3](../../assets/docs/03-search-and-fitting.md) measures what
it is worth.
"""

import sys

from metafit.cli import main

if __name__ == "__main__":
    sys.exit(main())
