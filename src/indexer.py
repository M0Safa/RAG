"""Build and load the BM25 index."""

import json
import os
import pickle
from typing import Any

from rank_bm25 import BM25Okapi

from src.chunker import process_corpus
from src.tokenizer import tokenize


def build_and_save_index(raw_dir: str = "data/raw/vllm-0.10.1",
                         processed_dir: str = "data/processed",
                         max_chunk_size: int = 2000) -> None:
    """Index the source code and save chunks and BM25 separately."""
    chunks = process_corpus(raw_dir, max_chunk_size)
    if not chunks:
        raise ValueError(f"No supported text files found in {raw_dir}")
    print(f"Indexing {len(chunks)} chunks...")
    corpus = [tokenize(f"{c['file_path']} {c['file_path']}\n{c['text']}")
              for c in chunks]
    index = BM25Okapi(corpus, k1=1.8, b=0.7)
    os.makedirs(processed_dir, exist_ok=True)
    with open(os.path.join(processed_dir, "chunks_metadata.json"),
              "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    with open(os.path.join(processed_dir, "bm25_index.pkl"), "wb") as f:
        pickle.dump(index, f)
    print(f"Index saved to {processed_dir}")


def load_index_and_chunks(processed_dir: str = "data/processed"
                          ) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """Load the saved index and its chunk metadata."""
    with open(os.path.join(processed_dir, "chunks_metadata.json"),
              "r", encoding="utf-8") as f:
        chunks: list[dict[str, Any]] = json.load(f)
    with open(os.path.join(processed_dir, "bm25_index.pkl"), "rb") as f:
        index: BM25Okapi = pickle.load(f)
    return index, chunks
