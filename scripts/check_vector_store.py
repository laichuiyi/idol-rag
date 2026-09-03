from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.blog_chunker import chunk_blog_document
from src.embedding.embedder import Embedder
from src.ingestion.blog_parser import parse_blog_html
from src.retrieval.vector_store import VectorStore

SAMPLE_HTML = PROJECT_ROOT / "data" / "raw" / "blog" / "綴夏.html"
QUERY = "井上和有冇講過美空做center?"
PREVIEW_CHARS = 200
TOP_K = 5


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("1. Parsing...")
    document = parse_blog_html(SAMPLE_HTML)
    chunks = chunk_blog_document(document)

    embedder = Embedder()
    chunk_embeddings = embedder.embed_texts([chunk.content for chunk in chunks])
    query_embedding = embedder.embed_text(QUERY)

    store = VectorStore()
    store.add(chunks, chunk_embeddings)
    results = store.search(query_embedding, top_k=TOP_K)

    print(f"Query: {QUERY}")
    print(f"Chunks stored: {len(store)}")
    print()
    for rank, (chunk, score) in enumerate(results, start=1):
        preview = chunk.content[:PREVIEW_CHARS]
        print(f"rank: {rank}")
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"similarity: {score:.6f}")
        print(f"content: {preview}")
        print()


if __name__ == "__main__":
    main()
