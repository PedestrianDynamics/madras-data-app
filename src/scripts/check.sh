#!/bin/bash

# Define the path to the virtual environment's bin directory
VENV_BIN=".venv/bin"

if [ ! -d "$VENV_BIN" ]; then
    echo "Virtual environment not found at $VENV_BIN."
    exit 1
fi

PYTHON_FILES=$(git ls-files '*.py')

if [ -z "$PYTHON_FILES" ]; then
    echo "No Python files found to check."
    exit 0
fi

run_tool() {
    echo ">> $1..."
    $VENV_BIN/$1 $2 $PYTHON_FILES
}

# Static type checker
run_tool mypy "--strict"

# Fast, extensible Python linter
run_tool ruff "check"

# Sort imports
run_tool isort "check"

run_tool black "--check"

echo "All checks and formatting done."
