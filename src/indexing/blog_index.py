from pathlib import Path

from src.embedding.embedder import Embedder
from src.ingestion.blog_corpus import build_blog_chunks
from src.retrieval.vector_store import VectorStore


def build_blog_vector_store(directory: Path, embedder: Embedder) -> VectorStore:
    chunks = build_blog_chunks(directory)
    store = VectorStore()
    if not chunks:
        return store

    embeddings = embedder.embed_texts([chunk.content for chunk in chunks])
    store.add(chunks, embeddings)
    return store
