.PHONY: install run debug clean lint lint-strict


install:
	uv sync

run:
	uv run python -m src

index:
	uv run python -m src index --max_chunck 2000
eval_code:
	./moulinette-ubuntu evaluate_student_search_results \
    data/output/dataset_code.json \
    ../Downloads/datasets_private/private/AnsweredQuestions/dataset_code_private.json \
    --k 10 \
    --max_context_length 2000
eval_docs:
	./moulinette-ubuntu evaluate_student_search_results \
    data/output/dataset_docs.json \
    ../Downloads/datasets_private/private/AnsweredQuestions/dataset_docs_private.json \
    --k 10 \
    --max_context_length 2000
eval_docs2:
	./moulinette-ubuntu evaluate_student_search_results \
    data/output/dataset_docs.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 \
    --max_context_length 2000
eval_code2:
	./moulinette-ubuntu evaluate_student_search_results \
    data/output/dataset_code.json \
    data/datasets/AnsweredQuestions/dataset_code_public.json \
    --k 10 \
    --max_context_length 2000
search_docs:
	uv run python -m src search_dataset --dataset_path ../Downloads/datasets_private/private/UnansweredQuestions/dataset_docs_private.json --k 10 --output_path data/output/dataset_docs.json
search_code:
	uv run python -m src search_dataset --dataset_path ../Downloads/datasets_private/private/UnansweredQuestions/dataset_code_private.json --k 10 --output_path data/output/dataset_code.json
search_docs2:
	uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --output_path data/output/dataset_docs.json
search_code2:
	uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --output_path data/output/dataset_code.json
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