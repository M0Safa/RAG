"""Tokenize code and documentation for lexical retrieval."""

import re


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase tokens for BM25.

    Splits camelCase, underscores, hyphens, and file paths into
    separate words. Removes punctuation and single-character tokens.

    Args:
        text (str): Text to tokenize.

    Returns:
        list[str]: List of normalized tokens.
    """
    if not text:
        return []
    processed = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    processed = processed.replace("_", " ").replace("-", " ")
    processed = processed.replace("/", " ")
    return [word.lower() for word in re.findall(r"\w+", processed)
            if len(word) > 1]
