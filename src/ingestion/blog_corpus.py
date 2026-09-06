from pathlib import Path

from src.chunking.blog_chunker import chunk_blog_document
from src.ingestion.blog_parser import parse_blog_html
from src.models.blog_document import BlogDocument
from src.models.chunk import Chunk


def load_blog_documents(directory: Path) -> list[BlogDocument]:
    html_paths = sorted(path for path in directory.glob("*.html") if path.is_file())
    return [parse_blog_html(path) for path in html_paths]


def build_blog_chunks(directory: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in load_blog_documents(directory):
        chunks.extend(chunk_blog_document(document))
    return chunks
