.PHONY: venv docs figures study report quick test lint clean

VENV := venv

# One BLAS thread. OpenBLAS threads by default, but this study is a million solves of
# k-by-k systems with k <= 32 -- far below the size where BLAS parallelism pays. Measured
# end to end: 1 thread 24.7s wall / 24.7s cpu, 8 threads 25.3s wall / 87.7s cpu. Threading
# buys nothing and burns 3.5x the CPU.
#
# It has to be set here rather than inside the package: `python -m metafit` imports numpy
# before __main__ runs, and OpenBLAS reads the variable when it loads.
THREADS := OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

PY := $(THREADS) PYTHONPATH=src $(VENV)/bin/python

## Create the virtualenv, building numpy against the host's OpenBLAS.
##
## The `--no-binary numpy` in requirements.txt is what does it; this target exists so that
## recreating the venv is one command rather than a remembered incantation. Needs
## the BLAS development files (an openblas.pc for pkg-config), a C compiler and ninja.
## Roughly two minutes the first time; pip caches the built wheel, so later recreates of
## the venv reuse it. See assets/docs/03-search-and-fitting.md for why, and how to opt out.
venv:
	rm -rf $(VENV)
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements-dev.txt
	@$(VENV)/bin/python -c "import numpy; b = numpy.show_config(mode='dicts')['Build Dependencies']['blas']; \
	print(f\"numpy {numpy.__version__} -> {b['name']} {b.get('version')} from {b.get('lib directory')}\")"

## Generate the API reference from the module docstrings.
##
## The narrative chapters live in assets/docs/ as markdown and are linked from each
## module docstring. pdoc renders the API; the assets tree is copied alongside so both
## the chapter links and the figure links resolve from the generated pages.
docs:
	rm -rf docs
	$(VENV)/bin/pdoc --math -d google -o docs src/metafit \
		--logo "assets/metafit_logo.svg" \
		--favicon "assets/metafit_logo.svg"
	cp -r assets docs/assets
	@echo "API reference in docs/, study chapters in docs/assets/docs/"

## The whole study: equations, validation, practices, CSV tables, report and figures.
study:
	$(PY) -m metafit

## The same run without the figures, when only the numbers are wanted.
report:
	$(PY) -m metafit --no-figures

## Figures only, from a fresh run.
figures:
	$(PY) -m metafit --quiet --no-tables --no-report

## Seconds rather than minutes; checks the wiring, not the numbers.
quick:
	$(PY) -m metafit --quick --no-figures --output /tmp/metafit-quick

test:
	$(PY) -m unittest discover -s tests

lint:
	./ci.sh

clean:
	rm -rf docs results .coverage
