"""Allow ``python -m metafit``.

Note for anyone tempted to pin BLAS threading here: it does not work. ``python -m
metafit`` imports the `metafit` package -- and therefore numpy, and therefore OpenBLAS --
*before* this module runs, and OpenBLAS reads its thread count when it loads. The variable
has to be set on the command line, which is what the ``Makefile`` does and what
[chapter 3](../../assets/docs/03-search-and-fitting.md) explains.
"""

import sys

from metafit.cli import main

if __name__ == "__main__":
    sys.exit(main())
