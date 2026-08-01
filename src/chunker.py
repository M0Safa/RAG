import ast
import os
from typing import List, Dict, Any


def chunk_markdown(file_path: str, content: str, max_chunk:int = 2000) -> List[Dict[str, Any]]:
    chunks = []
    sections = content.split("\n\n")
    current_start = 0
    for section in sections:
        if len(section) + 2 > max_chunk:
            sub_start = current_start
            while len(section) > 0:
                sub_chunk = section[:max_chunk]
                sub_end = sub_start + len(sub_chunk)
                chunks.append({
                    "file_path": file_path,
                    "first_character_index": sub_start,
                    "last_character_index": sub_end,
                    "text": sub_chunk.strip()
                })
                section = section[max_chunk:]
                sub_start = sub_end
            current_start = sub_start
        else:
            end_idx = current_start + len(section)
            chunks.append({
                "file_path": file_path,
                "first_character_index": current_start,
                "last_character_index": end_idx,
                "text": section.strip()
            })
            current_start = end_idx + 2

    return chunks


def chunk_python(file_path: str, content: str, max_chunck: int = 2000) -> List[Dict[str, Any]]:
    chunks = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return chunk_markdown(file_path, content)

    lines = content.splitlines(keepends=True)

    for node in tree.body:
        start_line = getattr(node, "lineno", 1) - 1
        end_line = getattr(node, "end_lineno", len(lines))
        
        node_lines = lines[start_line:end_line]
        node_text = "".join(node_lines)
        
        start_char = sum(len(l) for l in lines[:start_line])
        end_char = start_char + len(node_text)

        if len(node_text) > max_chunck:
            sub_start = start_char
            while len(node_text) > 0:
                sub_chunk = node_text[:max_chunck]
                sub_end = sub_start + len(sub_chunk)
                chunks.append({
                    "file_path": file_path,
                    "first_character_index": sub_start,
                    "last_character_index": sub_end,
                    "text": sub_chunk.strip()
                })
                node_text = node_text[max_chunck:]
                sub_start = sub_end
        else:
            if node_text.strip():
                chunks.append({
                    "file_path": file_path,
                    "first_character_index": start_char,
                    "last_character_index": end_char,
                    "text": node_text.strip()
                })
                
    return chunks


def process_corpus(root_dir: str, max_chunck: int = 2000) -> List[Dict[str, Any]]:
    """Scans the vLLM corpus directory, routes files to the correct chunker,

    and collects all chunks.
    """
    all_chunks = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            full_path = os.path.join(root, file)
            # Normalize path to match expected format (e.g., data/raw/vllm-0.10.1/...)
            norm_path = full_path.replace("\\", "/")
            
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
                
            if file.endswith(".py"):
                file_chunks = chunk_python(norm_path, content, max_chunck)
            elif file.endswith(".md") or file.endswith(".txt"):
                file_chunks = chunk_markdown(norm_path, content, max_chunck)
            else:
                continue
                
            all_chunks.extend(file_chunks)
            
    return all_chunks