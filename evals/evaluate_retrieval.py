from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.blog_chunker import chunk_blog_document
from src.embedding.embedder import Embedder
from src.ingestion.blog_parser import parse_blog_html
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

SAMPLE_HTML = PROJECT_ROOT / "data" / "raw" / "blog" / "綴夏.html"
CASES_PATH = PROJECT_ROOT / "evals" / "retrieval_cases.json"
TOP_K = 5


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> int:
    relevant = set(relevant_ids)
    return int(any(chunk_id in relevant for chunk_id in retrieved_ids))


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    retrieved_relevant = len(relevant.intersection(retrieved_ids))
    return retrieved_relevant / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1 / rank
    return 0.0


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    document = parse_blog_html(SAMPLE_HTML)
    chunks = chunk_blog_document(document)

    embedder = Embedder()
    chunk_embeddings = embedder.embed_texts([chunk.content for chunk in chunks])

    store = VectorStore()
    store.add(chunks, chunk_embeddings)
    retriever = Retriever(embedder, store)

    hits: list[float] = []
    recalls: list[float] = []
    rrs: list[float] = []

    for case in cases:
        query = case["query"]
        relevant_ids = case["relevant_chunk_ids"]
        results = retriever.retrieve(query, top_k=TOP_K)
        retrieved_ids = [chunk.chunk_id for chunk, _score in results]

        hit = hit_at_k(retrieved_ids, relevant_ids)
        recall = recall_at_k(retrieved_ids, relevant_ids)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        hits.append(hit)
        recalls.append(recall)
        rrs.append(rr)

        print(f"query: {query}")
        print(f"expected relevant chunk IDs: {relevant_ids}")
        print(f"retrieved chunk IDs: {retrieved_ids}")
        print(f"Hit@{TOP_K}: {hit}")
        print(f"Recall@{TOP_K}: {recall:.4f}")
        print(f"RR: {rr:.4f}")
        print()

    print("aggregate:")
    print(f"Hit Rate@{TOP_K}: {mean(hits):.4f}")
    print(f"Mean Recall@{TOP_K}: {mean(recalls):.4f}")
    print(f"MRR: {mean(rrs):.4f}")


if __name__ == "__main__":
    main()
