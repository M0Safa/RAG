.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src --help

debug:
	uv run python -m pdb -m src --help

clean:
	rm -rf data/processed/* data/output/* .mypy_cache .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -r {} +

lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/
	uv run mypy src/ --strict