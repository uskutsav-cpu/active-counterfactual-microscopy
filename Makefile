.PHONY: install test lint demo

install:
	python -m pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check .

demo:
	python scripts/demo_simulation.py
