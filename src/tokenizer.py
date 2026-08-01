import re


def tokenize(text: str) -> list[str]:
    if not text:
        return []

    processed = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    processed = processed.replace("_", " ").replace("-", " ")

    tokens = re.findall(r"\w+", processed.lower())

    return tokens
