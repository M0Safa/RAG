from .indexer import build_and_save_index, load_index_and_chunks
import json
import sys
from pathlib import Path
from src.retriever import BM25Retriever

def chunk_index():
	build_and_save_index()
	bm25, chunks = load_index_and_chunks()
	print(f'✅ Successfully indexed {len(chunks)} chunks!')


def run_batch_search(
    dataset_path: str,
    output_path: str = "data/output/search_results.json",
    k: int = 10,
) -> None:

    retriever = BM25Retriever(processed_dir="data/processed")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rag_questions = data.get("rag_questions", [])
    search_results = []

    for q in rag_questions:
        q_id = q["question_id"]
        q_text = q["question"]

        retrieved_sources = retriever.search_query(query=q_text, k=k)

        sources_list = [source.model_dump() for source in retrieved_sources]

        search_results.append(
            {
                "question_id": q_id,
                "question": q_text,
                "retrieved_sources": sources_list,
            }
        )

    moulinette_output = {
        "search_results": search_results,
        "k": k,
    }

    # 5. Save output JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(moulinette_output, f, indent=2)

    print(f"Processed {len(search_results)} questions and saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_batch_search(
            dataset_path="data/datasets/AnsweredQuestions/dataset_code_public.json",
            output_path="data/output/search_code_results.json",
            k=10,
        )
        run_batch_search(
            dataset_path="data/datasets/AnsweredQuestions/dataset_docs_public.json",
            output_path="data/output/search_docs_results.json",
            k=10,
        )
    else:
        chunk_index()



