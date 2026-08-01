from typing import List, Dict, Any, Union
import numpy as np
from rank_bm25 import BM25Okapi

from src.models import MinimalSource, MinimalSearchResults, UnansweredQuestion, AnsweredQuestion
from src.tokenizer import tokenize
from src.indexer import load_index_and_chunks


class BM25Retriever:

    def __init__(self, processed_dir: str = "data/processed") -> None:
        self.bm25_index, self.chunks = load_index_and_chunks(processed_dir)

    def search_query(self, query: str, k: int = 5) -> List[MinimalSource]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25_index.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:k]

        retrieved_sources: List[MinimalSource] = []
        for idx in top_indices:
            chunk: Dict[str, Any] = self.chunks[int(idx)]
            retrieved_sources.append(
                MinimalSource(
                    file_path=chunk["file_path"],
                    first_character_index=chunk["first_character_index"],
                    last_character_index=chunk["last_character_index"],
                )
            )

        return retrieved_sources

    def search_question(
        self, question: Union[UnansweredQuestion, AnsweredQuestion], k: int = 5
    ) -> MinimalSearchResults:
        sources = self.search_query(question.question, k=k)
        return MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=sources,
        )