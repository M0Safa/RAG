import os
import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

from src.chunker import process_corpus
from src.tokenizer import tokenize


def build_and_save_index(
    raw_dir: str = "data/raw/vllm-0.10.1",
    processed_dir: str = "data/processed"
) -> None:
    print(f"--> Processing raw files from: {raw_dir}")
    chunks: List[Dict[str, Any]] = process_corpus(raw_dir)
    print(f"--> Extracted total chunks: {len(chunks)}")

    if not chunks:
        raise ValueError(f"No valid python or markdown files found in {raw_dir}")

    print("--> Tokenizing corpus chunks...")
    tokenized_corpus: List[List[str]] = [tokenize(chunk["text"]) for chunk in chunks]

    print("--> Building BM25 index...")
    bm25_index = BM25Okapi(tokenized_corpus)

    os.makedirs(processed_dir, exist_ok=True)

    chunks_path = os.path.join(processed_dir, "chunks_metadata.pkl")
    index_path = os.path.join(processed_dir, "bm25_index.pkl")

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    with open(index_path, "wb") as f:
        pickle.dump(bm25_index, f)

    print(f"--> Indexing complete! Saved artifacts to:\n    - {chunks_path}\n    - {index_path}")


def load_index_and_chunks(
    processed_dir: str = "data/processed"
) -> tuple[BM25Okapi, List[Dict[str, Any]]]:

    chunks_path = os.path.join(processed_dir, "chunks_metadata.pkl")
    index_path = os.path.join(processed_dir, "bm25_index.pkl")

    if not os.path.exists(chunks_path) or not os.path.exists(index_path):
        raise FileNotFoundError(
            "Index files not found in data/processed/. Please run indexing first."
        )

    with open(chunks_path, "rb") as f:
        chunks: List[Dict[str, Any]] = pickle.load(f)

    with open(index_path, "rb") as f:
        bm25_index: BM25Okapi = pickle.load(f)

    return bm25_index, chunks