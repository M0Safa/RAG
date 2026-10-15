"""Pydantic models used to validate questions, sources, and RAG results."""

import uuid
from typing import List, Union
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Store the file path and character range of a source chunk."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question with a unique ID and question text."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with its answer and reference sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Store a collection of answered or unanswered questions."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Store a question and its retrieved source locations."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Store search results for multiple questions and the value of k."""

    answer: str


class StudentSearchResults(BaseModel):
    """Store search results for multiple questions and the value of k."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Store generated answers and retrieved sources for a dataset."""

    search_results: List[MinimalAnswer]
    k: int
