from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.blog_chunker import chunk_blog_document
from src.embedding.embedder import Embedder
from src.generation.prompts import build_context
from src.ingestion.blog_parser import parse_blog_html
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

SAMPLE_HTML = PROJECT_ROOT / "data" / "raw" / "blog" / "綴夏.html"
QUERY = "井上和有冇講過美空做center？"
TOP_K = 3


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    document = parse_blog_html(SAMPLE_HTML)
    chunks = chunk_blog_document(document)

    embedder = Embedder()
    chunk_embeddings = embedder.embed_texts([chunk.content for chunk in chunks])

    store = VectorStore()
    store.add(chunks, chunk_embeddings)

    retriever = Retriever(embedder, store)
    results = retriever.retrieve(QUERY, top_k=TOP_K)
    context = build_context(results)

    print(f"Query: {QUERY}")
    print()
    print(context)


if __name__ == "__main__":
    main()
