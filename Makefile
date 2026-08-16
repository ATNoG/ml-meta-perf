.PHONY: docs figures study test lint clean

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

## Re-run the study and rewrite every figure.
figures:
	$(PY) -m metafit --figures assets/figures --save results

## Print the full study to stdout.
study:
	$(PY) -m metafit

test:
	$(PY) -m unittest discover -s tests

lint:
	./ci.sh

clean:
	rm -rf docs results .coverage
