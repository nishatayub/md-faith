.PHONY: install check lint format test serve demo clean

PY ?= python

install:
	$(PY) -m pip install -e ".[dev,api]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

test:
	pytest -q

check: lint test

serve:
	mdfaith serve --reload

demo:
	mdfaith demo

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info results/*.db
