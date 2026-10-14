*This project has been created as part of the 42 curriculum by mosafa.*

# RAG Against the Machine

## Description

This project answers questions about the vLLM source code. First, it splits Python
and documentation files into small chunks and builds a BM25 search index. For a
question, BM25 selects the most relevant chunks. A small local Qwen model then uses
the retrieved text to write an answer.

## Instructions

Python 3.10+ and `uv` are required. Put the provided vLLM corpus under
`data/raw/vllm-0.10.1/`, and question datasets under `data/datasets/`.

```bash
make install
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How does vLLM serve requests?" --k 5
uv run python -m src answer "How does vLLM serve requests?" --k 5
```

Search a full dataset and evaluate it:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Generate answers from saved search results:

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

`make lint` checks flake8 and mypy. `make clean` removes local caches.

## System architecture

`chunker.py` reads files -> `indexer.py` tokenizes and saves BM25 ->
`retriever.py` selects source spans -> `generator.py` reads those spans and
sends a prompt to Qwen3-0.6B. `cli.py` exposes commands with Python Fire.

## Chunking strategy

Python files use language-aware separators so code is split close to classes,
functions, and lines. Markdown/text uses recursive splitting by paragraphs and
lines. The maximum chunk length is 2000 characters. Documentation uses 18%
overlap and Python uses 20%. Each chunk keeps its original file path and
character offsets.

## Retrieval method

BM25 ranks the tokenized chunks; code identifiers are separated on underscores,
slashes, and camelCase. File paths are included twice to make path terms more
important. BM25 uses `k1=1.8` and `b=0.7`. Retrieval returns source locations,
not the generated answer.

## Performance analysis

Earlier local tuning reported **81% docs Recall@5** and **58% code Recall@5**
with these BM25 parameters. These are historical measurements, not a claim that
the current package has been fully benchmarked. The subject requires at least
80% docs and 50% code Recall@5, indexing below five minutes, and searching
200 questions in at most 90 seconds. Re-run the evaluation on the evaluation
machine to verify these numbers. The supplied moulinette is the official scorer.

## Design decisions and challenges

BM25 was chosen because it is lightweight, CPU friendly, and handles identifiers
well. Repeating the file path can improve searches for code entities. Overlap
helps avoid losing information at chunk boundaries, but increases index size.
The generator uses a short prompt and disables Qwen3 thinking output to keep
answers readable. Missing input files or malformed datasets should produce
clear errors instead of full Python tracebacks.

## Resources

- [BM25 introduction](https://en.wikipedia.org/wiki/Okapi_BM25)
- [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- [Transformers](https://huggingface.co/docs/transformers/)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

AI assistance was used to review subject compliance, debug the CLI and code
style, and prepare documentation. The chunking, retrieval, and generation
pipeline should be understood and checked by the student before evaluation.

## Before submission

Run `uv lock` then `uv sync` on a connected machine, and commit `uv.lock`.
Do not commit the vLLM corpus, the provided private datasets, generated
indices, model weights, or other large files.
