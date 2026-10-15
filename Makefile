.PHONY: install run debug clean lint lint-strict
install:
	uv sync
run:
	uv run python -m src --help
debug:
	uv run python -m pdb -m src --help
clean:
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache
lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
index:
	uv run python -m src index --max_chunk_size 2000
search_docs_pub:
	uv run python -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--k 10 \
		--save_directory data/output/search_results/UnansweredQuestions
eval_docs_pub:
	./moulinette evaluate_student_search_results \
		data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
		data/datasets/AnsweredQuestions/dataset_docs_public.json \
		--k 10 --max_context_length 2000
search_code_pub:
	uv run python -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
		--k 10 \
		--save_directory data/output/search_results/UnansweredQuestions
eval_code_pub:
	./moulinette evaluate_student_search_results \
		data/output/search_results/UnansweredQuestions/dataset_code_public.json \
		data/datasets/AnsweredQuestions/dataset_code_public.json \
		--k 10 --max_context_length 2000
search_docs_pri:
	uv run python -m src search_dataset \
		--dataset_path data/datasets_private/UnansweredQuestions/dataset_docs_private.json \
		--k 10 \
		--save_directory data/output/search_results/UnansweredQuestions
eval_docs_pri:
	./moulinette evaluate_student_search_results \
		data/output/search_results/UnansweredQuestions/dataset_docs_private.json \
		data/datasets_private/AnsweredQuestions/dataset_docs_private.json \
		--k 10 --max_context_length 2000
search_code_pri:
	uv run python -m src search_dataset \
		--dataset_path data/datasets_private/UnansweredQuestions/dataset_code_private.json \
		--k 10 \
		--save_directory data/output/search_results/UnansweredQuestions
eval_code_pri:
	./moulinette evaluate_student_search_results \
		data/output/search_results/UnansweredQuestions/dataset_code_private.json \
		data/datasets_private/AnsweredQuestions/dataset_code_private.json \
		--k 10 --max_context_length 2000
lint-strict:
	uv run flake8 src
	uv run mypy src --strict
