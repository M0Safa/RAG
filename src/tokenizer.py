import re

def tokenize(text: str) -> list[str]:
    if not text:
        return []

    processed = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    processed = processed.replace("_", " ").replace("-", " ").replace("/", " ")

    tokens = [t.lower() for t in re.findall(r"\w+", processed) if len(t) > 1]

    return tokens