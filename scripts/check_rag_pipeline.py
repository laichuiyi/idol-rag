from pathlib import Path
import os
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.blog_chunker import chunk_blog_document
from src.embedding.embedder import Embedder
from src.generation.generator import Generator
from src.ingestion.blog_parser import parse_blog_html
from src.rag.pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

SAMPLE_HTML = PROJECT_ROOT / "data" / "raw" / "blog" / "綴夏.html"
QUESTION = "井上和有冇講過美空做center？"
TOP_K = 3


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = _require_env("GEMINI_API_KEY")
    model_name = _require_env("GEMINI_MODEL")

    document = parse_blog_html(SAMPLE_HTML)
    chunks = chunk_blog_document(document)

    embedder = Embedder()
    chunk_embeddings = embedder.embed_texts([chunk.content for chunk in chunks])

    store = VectorStore()
    store.add(chunks, chunk_embeddings)

    retriever = Retriever(embedder, store)
    generator = Generator(api_key=api_key, model_name=model_name)
    pipeline = RAGPipeline(retriever, generator)
    answer = pipeline.answer(QUESTION, top_k=TOP_K)

    print(f"question: {QUESTION}")
    print(f"model: {generator.model_name}")
    print(f"answer: {answer}")


if __name__ == "__main__":
    main()
