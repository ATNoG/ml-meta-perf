#!/bin/bash
# Description: Project CI pipeline: Unittest, Coverage, Ruff, BasedPyright, and Vulture.
#
# The same checks pre-commit runs, in one script, for CI and for running by hand.

echo "=== RUNNING UNIT TESTS ==="
TEST_OUTPUT=$(PYTHONPATH=src venv/bin/python -m unittest discover -s tests 2>&1)
TEST_STATUS=$?

echo "$TEST_OUTPUT"

if [ $TEST_STATUS -ne 0 ]; then
  echo "--------------------------------------------------"
  echo "Tests failed (Exit code: $TEST_STATUS). Read the traceback above and forward-fix the errors."
  exit $TEST_STATUS
fi

echo -e "\n=== RUNNING COVERAGE ==="
COVERAGE_OUTPUT=$(PYTHONPATH=src venv/bin/coverage run -m unittest discover -s tests 2>&1)
COVERAGE_STATUS=$?

if [ $COVERAGE_STATUS -ne 0 ]; then
  echo "--------------------------------------------------"
  echo "Coverage collection failed (Exit code: $COVERAGE_STATUS)."
  exit $COVERAGE_STATUS
fi

venv/bin/coverage report

echo -e "\n=== RUNNING RUFF ==="
RUFF_OUTPUT=$(venv/bin/ruff check . 2>&1)
RUFF_STATUS=$?

echo "$RUFF_OUTPUT"

if [ $RUFF_STATUS -ne 0 ]; then
  echo "--------------------------------------------------"
  echo "Ruff check failed. Focus on resolving the errors above."
  exit $RUFF_STATUS
fi

echo -e "\n=== RUNNING BASEDPYRIGHT ==="
PYRIGHT_OUTPUT=$(venv/bin/basedpyright 2>&1)
PYRIGHT_STATUS=$?

echo "$PYRIGHT_OUTPUT"

if [ $PYRIGHT_STATUS -ne 0 ]; then
  echo "--------------------------------------------------"
  echo "BasedPyright check failed. Focus on resolving the errors above."
  exit $PYRIGHT_STATUS
fi

echo -e "\n=== RUNNING VULTURE ==="
VULTURE_OUTPUT=$(venv/bin/vulture 2>&1)
VULTURE_STATUS=$?

echo "$VULTURE_OUTPUT"

if [ $VULTURE_STATUS -ne 0 ]; then
  echo "--------------------------------------------------"
  echo "Vulture check failed. Focus on resolving the errors above."
  exit $VULTURE_STATUS
fi

echo -e "\nAll checks passed!"
exit 0
