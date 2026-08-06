import ast
import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, PythonCodeTextSplitter, Language
from langchain_core.documents import Document


def chunk_markdown(file_path: str, content: str, max_chunk:int = 2000) -> List[Dict[str, Any]]:
    # chunks = []
    # sections = content.split("\n\n")
    # step = max_chunk - 300
    # search_pos = 0
    # for section in sections:
    #     section_text = section.strip()
    #     if not section_text:
    #         continue

    #     start_idx = content.find(section, search_pos)
    #     if start_idx == -1:
    #         start_idx = search_pos
    #     end_idx = start_idx + len(section)
    #     search_pos = end_idx

    #     if len(section) > max_chunk:
    #         sub_start = start_idx
    #         for i in range(0, len(section), step):
    #             sub_chunk = section[i:i + max_chunk]
    #             if not sub_chunk.strip():
    #                 continue
    #             sub_end = min(sub_start + len(sub_chunk), end_idx)
    #             chunks.append({
    #                 "file_path": file_path,
    #                 "first_character_index": sub_start,
    #                 "last_character_index": sub_end,
    #                 "text": sub_chunk.strip()
    #             })
    #             sub_start += step
    #     else:
    #         chunks.append({
    #             "file_path": file_path,
    #             "first_character_index": start_idx,
    #             "last_character_index": end_idx,
    #             "text": section_text
    #         })
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk, chunk_overlap=int(max_chunk * 0.13), add_start_index=True)
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
    # chunks = []
    # try:
    #     tree = ast.parse(content)
    # except SyntaxError:
    #     return chunk_markdown(file_path, content)

    # lines = content.splitlines(keepends=True)

    # for node in tree.body:
    #     start_line = getattr(node, "lineno", 1) - 1
    #     end_line = getattr(node, "end_lineno", len(lines))
        
    #     node_lines = lines[start_line:end_line]
    #     node_text = "".join(node_lines)
        
    #     start_char = sum(len(l) for l in lines[:start_line])
    #     end_char = start_char + len(node_text)

    #     if len(node_text) > max_chunk:
    #         sub_start = start_char
    #         step = max_chunk - 300
    #         for i in range(0, len(node_text), step):
    #             sub_chunk = node_text[i:i + max_chunk]
    #             if not sub_chunk.strip():
    #                 continue
    #             sub_end = min(sub_start + len(sub_chunk), end_char)
    #             chunks.append({
    #                 "file_path": file_path,
    #                 "first_character_index": sub_start,
    #                 "last_character_index": sub_end,
    #                 "text": sub_chunk.strip()
    #             })
    #             sub_start += step
    #     else:
    #         if node_text.strip():
    #             chunks.append({
    #                 "file_path": file_path,
    #                 "first_character_index": start_char,
    #                 "last_character_index": end_char,
    #                 "text": node_text.strip()
    #             })
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
            elif file.endswith(".md") or file.endswith(".txt"):
                file_chunks = chunk_markdown(norm_path, content, max_chunck)
            else:
                continue

            all_chunks.extend(file_chunks)
 
    return all_chunks