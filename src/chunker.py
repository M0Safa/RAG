"""Split vLLM source files into small chunks with source locations."""

import os
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


def split_file(file_path: str, content: str, max_chunk_size: int,
               overlap: float, language: Language | None = None
               ) -> list[dict[str, Any]]:
    """Split file content into chunks while keeping their positions.

    Args:
        file_path (str): Path to the original file.
        content (str): Full text of the file.
        max_chunk_size (int): Maximum number of characters per chunk.
        overlap (float): Overlap ratio between chunks.
        language (Language | None): Language used for splitting.

    Returns:
        list[dict[str, Any]]: Chunks with their text, file path,
        and start and end positions.
    """
    if language is None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=int(max_chunk_size * overlap),
            add_start_index=True,
        )
    else:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=max_chunk_size,
            chunk_overlap=int(max_chunk_size * overlap),
            add_start_index=True,
        )
    document = Document(
        page_content=content, metadata={"file_path": file_path}
    )
    chunks = []
    for piece in splitter.split_documents([document]):
        start = piece.metadata["start_index"]
        chunks.append({
            "file_path": file_path,
            "first_character_index": start,
            "last_character_index": start + len(piece.page_content),
            "text": piece.page_content,
        })
    return chunks


def chunk_markdown(file_path: str, content: str,
                   max_chunk_size: int) -> list[dict[str, Any]]:
    """Split Markdown or text files into smaller chunks.

    Args:
        file_path (str): Path to the source file.
        content (str): Text to split.
        max_chunk_size (int): Maximum chunk size in characters.

    Returns:
        list[dict[str, Any]]: Chunks with their source positions.
    """
    return split_file(file_path, content, max_chunk_size, 0.18)


def chunk_python(file_path: str, content: str,
                 max_chunk_size: int) -> list[dict[str, Any]]:
    """Split Python code into chunks using Python separators.

    Args:
        file_path (str): Path to the Python file.
        content (str): Python code to split.
        max_chunk_size (int): Maximum chunk size in characters.

    Returns:
        list[dict[str, Any]]: Code chunks with their source positions.
    """
    return split_file(file_path, content, max_chunk_size, 0.20,
                      Language.PYTHON)


def process_corpus(root_dir: str,
                   max_chunk_size: int = 2000) -> list[dict[str, Any]]:
    """Read supported files from a directory and split them into chunks.

    Args:
        root_dir (str): Directory containing the source files.
        max_chunk_size (int): Maximum chunk size in characters.

    Returns:
        list[dict[str, Any]]: All chunks extracted from the files.

    Raises:
        ValueError: If the chunk size is invalid.
        FileNotFoundError: If the source directory does not exist.
    """
    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("max_chunk_size must be between 1 and 2000")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Corpus directory not found: {root_dir}")

    chunks: list[dict[str, Any]] = []
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        for filename in sorted(files):
            path = os.path.join(root, filename).replace("\\", "/")
            if not filename.endswith((".py", ".md", ".txt", ".rst")):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except OSError as exc:
                print(f"Skipping {path}: {exc}")
                continue
            if filename.endswith(".py"):
                chunks.extend(chunk_python(path, content, max_chunk_size))
            else:
                chunks.extend(chunk_markdown(path, content, max_chunk_size))
    return chunks
