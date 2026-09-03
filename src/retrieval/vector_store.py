import numpy as np
import faiss

from src.models.chunk import Chunk


class VectorStore:
    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must have the same length, "
                f"got {len(chunks)} chunks and {len(embeddings)} embeddings"
            )
        if not embeddings:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2D list of equal-length vectors")

        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        elif vectors.shape[1] != self._index.d:
            raise ValueError(
                f"embedding dimension {vectors.shape[1]} does not match "
                f"index dimension {self._index.d}"
            )

        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        if self._index is None or not self._chunks:
            return []

        k = min(top_k, len(self._chunks))
        query = np.asarray([query_embedding], dtype=np.float32)
        scores, indices = self._index.search(query, k)

        results: list[tuple[Chunk, float]] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            results.append((self._chunks[int(index)], float(score)))
        return results

    def __len__(self) -> int:
        return len(self._chunks)
