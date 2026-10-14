"""Command-line interface for the RAG pipeline."""

import os
import time
from pathlib import Path

import fire
from pydantic import ValidationError
from tqdm import tqdm

from src.indexer import build_and_save_index
from src.models import (RagDataset, StudentSearchResults,
                        StudentSearchResultsAndAnswer, UnansweredQuestion)
from src.retriever import BM25Retriever


class RAGCLI:
    """Index, search, answer, and evaluate source-code questions."""

    def index(self, max_chunk_size: int = 2000,
              raw_dir: str = "data/raw/vllm-0.10.1",
              processed_dir: str = "data/processed") -> None:
        """Build the search index."""
        start = time.time()
        build_and_save_index(raw_dir, processed_dir, max_chunk_size)
        print(f"Indexing took {time.time() - start:.1f}s")

    def search(self, query: str = "", k: int = 5,
               processed_dir: str = "data/processed") -> None:
        """Print sources matching one query."""
        sources = BM25Retriever(processed_dir).search_query(query, k)
        for source in sources:
            print(source.model_dump_json())

    def search_dataset(self, dataset_path: str = "", k: int = 10,
                       save_directory: str = "data/output/search_results",
                       processed_dir: str = "data/processed") -> None:
        """Save search results for a JSON dataset."""
        dataset = read_dataset(dataset_path)
        retriever = BM25Retriever(processed_dir)
        results = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            results.append(retriever.search_question(question, k))
        output = StudentSearchResults(search_results=results, k=max(k, 0))
        saved = output_file(save_directory, dataset_path)
        saved.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved results to {saved}")

    def answer(self, query: str = "", k: int = 5,
               processed_dir: str = "data/processed") -> None:
        """Answer one question using Qwen."""
        from src.generator import RAGGenerator
        generator = RAGGenerator(processed_dir=processed_dir)
        question = UnansweredQuestion(question=query)
        answer = generator.answer_question(question, k)
        print(answer.answer)

    def answer_dataset(self, student_search_results_path: str = "",
                       save_directory: str = (
                           "data/output/search_results_and_answer"
                       ),
                       processed_dir: str = "data/processed") -> None:
        """Generate answers from an existing search-results JSON file."""
        data = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text(encoding="utf-8")
        )
        from src.generator import RAGGenerator
        generator = RAGGenerator(processed_dir=processed_dir)
        answers = []
        for result in tqdm(data.search_results, desc="Answering"):
            answers.append(generator.generate_answer(result))
        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=data.k
        )
        saved = output_file(save_directory, student_search_results_path)
        saved.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved answers to {saved}")

    def evaluate(self, student_search_results_path: str = "",
                 dataset_path: str = "") -> None:
        """Print approximate local recall based on same-file span overlap."""
        from src.evaluation import evaluate_results
        print(evaluate_results(student_search_results_path, dataset_path))


def read_dataset(path: str) -> RagDataset:
    """Read and validate a question dataset."""
    content = Path(path).read_text(encoding="utf-8")
    return RagDataset.model_validate_json(content)


def output_file(directory: str, input_path: str) -> Path:
    """Create a destination with the input file's original name."""
    os.makedirs(directory, exist_ok=True)
    return Path(directory) / Path(input_path).name


def main() -> None:
    """Start Fire without exposing an unhandled traceback to the reviewer."""
    try:
        fire.Fire(RAGCLI)
    except (OSError, ValueError, TypeError, ValidationError, KeyError) as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
