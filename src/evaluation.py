"""Simple local retrieval evaluation (not the official grader)."""

from pathlib import Path

from src.models import RagDataset, StudentSearchResults


def evaluate_results(results_path: str, dataset_path: str) -> str:
    """Calculate Recall@5 using the retrieved results and references.

    Compares retrieved source locations with the expected sources.
    A source is considered correct if the file paths match and
    the character ranges have enough overlap.

    Args:
        results_path (str): Path to the student search results.
        dataset_path (str): Path to the reference answered dataset.

    Returns:
        str: Formatted local Recall@5 score.
    """
    results = StudentSearchResults.model_validate_json(
        Path(results_path).read_text(encoding="utf-8")
    )
    dataset = RagDataset.model_validate_json(
        Path(dataset_path).read_text(encoding="utf-8")
    )
    expected = {q.question_id: q for q in dataset.rag_questions}
    found = 0
    total = 0
    for result in results.search_results:
        question = expected.get(result.question_id)
        if question is None or not hasattr(question, "sources"):
            continue
        for target in question.sources:
            total += 1
            for source in result.retrieved_sources[:5]:
                if source.file_path != target.file_path:
                    continue
                intersection = max(0, min(source.last_character_index,
                                          target.last_character_index)
                                   - max(source.first_character_index,
                                         target.first_character_index))
                union = max(source.last_character_index,
                            target.last_character_index) - min(
                                source.first_character_index,
                                target.first_character_index)
                if union > 0 and intersection / union >= 0.05:
                    found += 1
                    break
    return f"Local Recall@5: {found / total:.2%}" if total else "No references"
