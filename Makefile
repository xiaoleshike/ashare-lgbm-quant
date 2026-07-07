.PHONY: install test lint typecheck

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check src tests

typecheck:
	python -m mypy
