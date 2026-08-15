"""Allow ``python -m metafit``."""

import sys

from metafit.cli import main

if __name__ == "__main__":
    sys.exit(main())
