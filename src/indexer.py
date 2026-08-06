import os
import json
import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

from src.chunker import process_corpus
from src.tokenizer import tokenize

import json
from rank_bm25 import BM25Okapi

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
    tokenized_corpus: List[List[str]] = [tokenize(f"{chunk['file_path']} {chunk['file_path']}\n{chunk['text']}") for chunk in chunks]

    print("--> Building BM25 index...")
    bm25_index = BM25Okapi(tokenized_corpus, k1=1.2, b=0.3)

    os.makedirs(processed_dir, exist_ok=True)

    chunks_path = os.path.join(processed_dir, "chunks_metadata.json")
    index_path = os.path.join(processed_dir, "bm25_index.pkl")

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    with open(index_path, "wb") as f:
        pickle.dump(bm25_index, f)


    print(f"--> Indexing complete! Saved artifacts to:\n    - {chunks_path}\n    - {index_path}")


def load_index_and_chunks(
    processed_dir: str = "data/processed"
) -> tuple[BM25Okapi, List[Dict[str, Any]]]:

    chunks_path = os.path.join(processed_dir, "chunks_metadata.json")
    index_path = os.path.join(processed_dir, "bm25_index.pkl")

    if not os.path.exists(chunks_path) or not os.path.exists(index_path):
        raise FileNotFoundError(
            "Index files not found in data/processed/. Please run indexing first."
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    with open(index_path, "rb") as f:
        bm25_index = pickle.load(f)

    return bm25_index, chunks