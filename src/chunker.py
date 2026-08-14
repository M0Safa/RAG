import ast
import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, PythonCodeTextSplitter, Language
from langchain_core.documents import Document


def chunk_markdown(file_path: str, content: str, max_chunk:int = 2000) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk, chunk_overlap=int(max_chunk * 0.14), add_start_index=True)
    initial_doc = Document(
        page_content=content, metadata={"file_path": file_path}
    )
    split_docs = splitter.split_documents([initial_doc])
    chunks = []
    for doc in split_docs:
        start_idx = doc.metadata["start_index"]
        end_idx = start_idx + len(doc.page_content)
        chunks.append(
            {
                "file_path": doc.metadata["file_path"],
                "first_character_index": start_idx,
                "last_character_index": end_idx,
                "text": doc.page_content.strip(),
            }
        )
    return chunks


def chunk_python(file_path: str, content: str, max_chunk: int = 2000) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=max_chunk,
            chunk_overlap=int(max_chunk * 0.17),
            add_start_index=True,
        )
    initial_doc = Document(
        page_content=content, metadata={"file_path": file_path}
    )
    split_docs = splitter.split_documents([initial_doc])
    chunks = []
    for doc in split_docs:
        start_idx = doc.metadata["start_index"]
        end_idx = start_idx + len(doc.page_content)
        chunks.append(
            {
                "file_path": doc.metadata["file_path"],
                "first_character_index": start_idx,
                "last_character_index": end_idx,
                "text": doc.page_content.strip(),
            }
        )
    return chunks


def process_corpus(root_dir: str, max_chunck: int = 2000) -> List[Dict[str, Any]]:
    all_chunks = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            full_path = os.path.join(root, file)
            norm_path = full_path.replace("\\", "/")
            
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
                
            if file.endswith(".py"):
                file_chunks = chunk_python(norm_path, content, max_chunck)
            elif file.endswith((".md", ".txt", ".rst")):
                file_chunks = chunk_markdown(norm_path, content, max_chunck)
            else:
                continue

            all_chunks.extend(file_chunks)
 
    return all_chunks
