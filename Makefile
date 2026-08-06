.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

eval_code:
	./moulinette-ubuntu evaluate_student_search_results \
    data/output/search_code_results.json \
    data/datasets/AnsweredQuestions/dataset_code_public.json \
    --k 10 \
    --max_context_length 2000
eval_docs:
	./moulinette-ubuntu evaluate_student_search_results \
    data/output/search_docs_results.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 \
    --max_context_length 2000
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