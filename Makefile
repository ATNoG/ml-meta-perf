.PHONY: docs figures study report quick test lint clean

VENV := venv
PY := PYTHONPATH=src $(VENV)/bin/python

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
