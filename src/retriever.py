"""Rank source chunks using the saved BM25 index."""

from typing import Any

import numpy as np

from src.indexer import load_index_and_chunks
from src.models import (AnsweredQuestion, MinimalSearchResults,
                        MinimalSource, UnansweredQuestion)
from src.tokenizer import tokenize


class BM25Retriever:
    """Search indexed chunks and return their source positions."""

    def __init__(self, processed_dir: str = "data/processed") -> None:
        """Load the saved BM25 index."""
        self.bm25_index, self.chunks = load_index_and_chunks(processed_dir)

    def search_query(self, query: str, k: int = 5) -> list[MinimalSource]:
        """Return up to k ranked sources for a query."""
        tokens = tokenize(query)
        if not tokens or k <= 0:
            return []
        scores = self.bm25_index.get_scores(tokens)
        indices = np.argsort(scores)[::-1][:k]
        sources = []
        for idx in indices:
            chunk: dict[str, Any] = self.chunks[int(idx)]
            sources.append(MinimalSource(
                file_path=chunk["file_path"],
                first_character_index=chunk["first_character_index"],
                last_character_index=chunk["last_character_index"],
            ))
        return sources

    def search_question(self, question: UnansweredQuestion | AnsweredQuestion,
                        k: int = 5) -> MinimalSearchResults:
        """Search one question while retaining its ID."""
        return MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=self.search_query(question.question, k),
        )
