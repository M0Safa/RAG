import re

def tokenize(text: str) -> list[str]:
    if not text:
        return []

    raw_tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_-]+", text) if len(t) > 1]

    processed = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    processed = processed.replace("_", " ").replace("-", " ")
    split_tokens = [t.lower() for t in re.findall(r"\w+", processed) if len(t) > 1]

    return raw_tokens + split_tokens
