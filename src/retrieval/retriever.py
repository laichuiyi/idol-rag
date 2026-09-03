from src.embedding.embedder import Embedder
from src.models.chunk import Chunk
from src.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedder.embed_text(query)
        return self.vector_store.search(query_embedding, top_k=top_k)
