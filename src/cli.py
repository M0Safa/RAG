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
        """Build the BM25 index from the source files.

        Args:
            max_chunk_size (int): Maximum chunk size in characters.
            raw_dir (str): Directory containing the source files.
            processed_dir (str): Directory where the index is saved.
        """
        start = time.time()
        build_and_save_index(raw_dir, processed_dir, max_chunk_size)
        print(f"Indexing took {time.time() - start:.1f}s")

    def search(self, query: str = "", k: int = 5,
               processed_dir: str = "data/processed") -> None:
        """Search for relevant sources and print the results.

        Args:
            query (str): Text or question to search for.
            k (int): Number of sources to retrieve.
            processed_dir (str): Directory containing the BM25 index.
        """
        sources = BM25Retriever(processed_dir).search_query(query, k)
        for source in sources:
            print(source.model_dump_json())

    def search_dataset(self, dataset_path: str = "", k: int = 10,
                       save_directory: str = "data/output/search_results",
                       processed_dir: str = "data/processed") -> None:
        """Search all questions in a dataset and save the results.

        Args:
            dataset_path (str): Path to the input question dataset.
            k (int): Number of sources to retrieve per question.
            save_directory (str): Directory for saving search results.
            processed_dir (str): Directory containing the BM25 index.
        """
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
        """Generate and print an answer to a single question.

        Args:
            query (str): Question to answer.
            k (int): Number of sources to retrieve.
            processed_dir (str): Directory containing the BM25 index.
        """
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
        """Generate answers for a dataset of retrieved questions.

        Loads existing search results, generates an answer for each
        question using Qwen, and saves the results as JSON.

        Args:
            student_search_results_path (str): Path to search results.
            save_directory (str): Directory for saving generated answers.
            processed_dir (str): Directory containing the BM25 index.
        """
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
        """Evaluate retrieved sources against the reference dataset.

        Args:
            student_search_results_path (str): Path to search results.
            dataset_path (str): Path to the answered reference dataset.
        """
        from src.evaluation import evaluate_results
        print(evaluate_results(student_search_results_path, dataset_path))


def read_dataset(path: str) -> RagDataset:
    """Read and validate a RAG question dataset.

    Args:
        path (str): Path to the dataset JSON file.

    Returns:
        RagDataset: Validated dataset containing the questions.
    """
    content = Path(path).read_text(encoding="utf-8")
    return RagDataset.model_validate_json(content)


def output_file(directory: str, input_path: str) -> Path:
    """Create an output directory and return the output file path.

    Args:
        directory (str): Directory where the file will be saved.
        input_path (str): Input file used to determine the filename.

    Returns:
        Path: Path where the output file should be saved.
    """
    os.makedirs(directory, exist_ok=True)
    return Path(directory) / Path(input_path).name


def main() -> None:
    """Start the command-line interface using Python Fire.

    Runs the RAG commands and catches common errors to avoid
    displaying unhandled tracebacks.
    """
    try:
        fire.Fire(RAGCLI)
    except (OSError, ValueError, TypeError, ValidationError, KeyError) as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
