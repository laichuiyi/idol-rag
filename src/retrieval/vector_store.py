from datetime import datetime
from pathlib import Path
import json

import faiss
import numpy as np

from src.models.chunk import Chunk

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"


class VectorStore:
    def __init__(self) -> None:
        self._index: faiss.Index | None = None
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

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        chunks_payload = [_chunk_to_dict(chunk) for chunk in self._chunks]
        (directory / CHUNKS_FILENAME).write_text(
            json.dumps(chunks_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self._index is not None and self._chunks:
            faiss.write_index(self._index, str(directory / INDEX_FILENAME))

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        raw_chunks = json.loads((directory / CHUNKS_FILENAME).read_text(encoding="utf-8"))
        store = cls()
        store._chunks = [_chunk_from_dict(item) for item in raw_chunks]
        if not store._chunks:
            return store

        index = faiss.read_index(str(directory / INDEX_FILENAME))
        if index.ntotal != len(store._chunks):
            raise ValueError(
                f"FAISS index size {index.ntotal} does not match "
                f"chunk count {len(store._chunks)}"
            )
        store._index = index
        return store


def _chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "content": chunk.content,
        "author": chunk.author,
        "title": chunk.title,
        "published_at": chunk.published_at.isoformat(),
        "source_url": chunk.source_url,
        "source_type": chunk.source_type,
    }


def _chunk_from_dict(payload: dict) -> Chunk:
    return Chunk(
        chunk_id=payload["chunk_id"],
        document_id=payload["document_id"],
        content=payload["content"],
        author=payload["author"],
        title=payload["title"],
        published_at=datetime.fromisoformat(payload["published_at"]),
        source_url=payload["source_url"],
        source_type=payload["source_type"],
    )
