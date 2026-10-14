"""Tokenize code and documentation for lexical retrieval."""

import re


def tokenize(text: str) -> list[str]:
    """Normalize punctuation, snake_case and camelCase into words."""
    if not text:
        return []
    processed = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    processed = processed.replace("_", " ").replace("-", " ")
    processed = processed.replace("/", " ")
    return [word.lower() for word in re.findall(r"\w+", processed)
            if len(word) > 1]
