import json
import time
from pathlib import Path
from typing import Optional
import fire
from tqdm import tqdm

from src.indexer import build_and_save_index
from src.retriever import BM25Retriever
from src.generator import RAGGenerator
from src.models import UnansweredQuestion


class RAGCLI:
    def index(
        self,
        raw_dir: str = "data/raw/vllm-0.10.1",
        processed_dir: str = "data/processed"
    ) -> None:
        print(f"Starting corpus indexing from: {raw_dir}")
        start_time = time.time()
        build_and_save_index(raw_dir=raw_dir, processed_dir=processed_dir)
        elapsed = time.time() - start_time
        print(f"Indexing completed in {elapsed:.2f} seconds.")

    def search(self, query: str, k: int = 5) -> None:
        retriever = BM25Retriever()
        sources = retriever.search_query(query=query, k=k)
        print(f"\nQuery: '{query}' (Top {k} results):")
        for i, src in enumerate(sources, 1):
            print(f"  [{i}] Path: {src.file_path} ({src.first_character_index}:{src.last_character_index})")

    def search_dataset(
        self,
        dataset_path: str,
        output_path: str = "data/output/search_results.json",
        k: int = 10,
    ) -> None:
        """Batch-search a dataset file and save formatted results."""
        retriever = BM25Retriever()
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rag_questions = data.get("rag_questions", [])
        search_results = []

        print(f"Processing {len(rag_questions)} questions from {dataset_path}...")
        for q in tqdm(rag_questions, desc="Retrieving"):
            sources = retriever.search_query(query=q["question"], k=k)
            search_results.append({
                "question_id": q["question_id"],
                "question": q["question"],
                "retrieved_sources": [s.model_dump() for s in sources]
            })

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"search_results": search_results, "k": k}, f, indent=2)

        print(f"Search results saved to: {output_path}")

    def answer(self, question: str, k: int = 5) -> None:
        generator = RAGGenerator()
        unanswered = UnansweredQuestion(question_id="cli_query", question=question)
        ans = generator.answer_question(unanswered, k=k)
        print(f"\nQuestion: {question}\n")
        print(f"Generated Answer:\n{ans.answer}")

    def answer_dataset(
        self,
        dataset_path: str,
        output_path: str = "data/output/answers.json",
        k: int = 5,
    ) -> None:
        generator = RAGGenerator()
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rag_questions = data.get("rag_questions", [])
        answers = []

        print(f"Answering {len(rag_questions)} questions from {dataset_path}...")
        for q in tqdm(rag_questions, desc="Generating"):
            unanswered = UnansweredQuestion(question_id=q["question_id"], question=q["question"])
            ans = generator.answer_question(unanswered, k=k)
            answers.append(ans.model_dump())

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"answers": answers}, f, indent=2)

        print(f"Answers saved to: {output_path}")


def main():
    fire.Fire(RAGCLI)


if __name__ == "__main__":
    main()