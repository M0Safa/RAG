import os
import json
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

from src.chunker import process_corpus
from src.tokenizer import tokenize

import json
from rank_bm25 import BM25Okapi

def save_bm25_json(bm25: BM25Okapi, filepath: str) -> None:
    data = {
        "k1": bm25.k1,
        "b": bm25.b,
        "epsilon": bm25.epsilon,
        "corpus_size": bm25.corpus_size,
        "avgdl": bm25.avgdl,
        "doc_len": list(bm25.doc_len),
        "doc_freqs": bm25.doc_freqs,
        "nd": getattr(bm25, "nd", {}),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_bm25_json(filepath: str) -> BM25Okapi:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    bm25 = BM25Okapi.__new__(BM25Okapi)
    bm25.k1 = data["k1"]
    bm25.b = data["b"]
    bm25.epsilon = data["epsilon"]
    bm25.corpus_size = data["corpus_size"]
    bm25.avgdl = data["avgdl"]
    bm25.doc_len = data["doc_len"]
    bm25.doc_freqs = data["doc_freqs"]
    bm25.nd = data["nd"]

    return bm25

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

    chunks_path = os.path.join(processed_dir, "chunks_metadata.json")
    index_path = os.path.join(processed_dir, "bm25_index.json")

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    save_bm25_json(bm25_index, index_path)


    print(f"--> Indexing complete! Saved artifacts to:\n    - {chunks_path}\n    - {index_path}")


def load_index_and_chunks(
    processed_dir: str = "data/processed"
) -> tuple[BM25Okapi, List[Dict[str, Any]]]:

    chunks_path = os.path.join(processed_dir, "chunks_metadata.json")
    index_path = os.path.join(processed_dir, "bm25_index.json")

    if not os.path.exists(chunks_path) or not os.path.exists(index_path):
        raise FileNotFoundError(
            "Index files not found in data/processed/. Please run indexing first."
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    bm25_index: BM25Okapi = load_bm25_json(index_path)

    return bm25_index, chunks