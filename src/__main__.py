from .indexer import build_and_save_index, load_index_and_chunks


build_and_save_index()
bm25, chunks = load_index_and_chunks()
print(f'✅ Successfully indexed {len(chunks)} chunks!')