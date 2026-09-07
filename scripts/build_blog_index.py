from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.embedder import Embedder
from src.indexing.blog_index import build_blog_vector_store

BLOG_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "blog"
INDEX_DIRECTORY = PROJECT_ROOT / "data" / "index" / "blog"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    embedder = Embedder()
    store = build_blog_vector_store(BLOG_DIRECTORY, embedder)
    store.save(INDEX_DIRECTORY)

    print(f"indexed chunks: {len(store)}")
    print(f"saved index directory: {INDEX_DIRECTORY}")


if __name__ == "__main__":
    main()
