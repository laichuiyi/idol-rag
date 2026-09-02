from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.blog_chunker import chunk_blog_document
from src.chunking.token_counter import count_tokens
from src.ingestion.blog_parser import parse_blog_html

SAMPLE_HTML = PROJECT_ROOT / "data" / "raw" / "blog" / "綴夏.html"
MAX_TOKENS = 300


@pytest.fixture(scope="module")
def document():
    return parse_blog_html(SAMPLE_HTML)


@pytest.fixture(scope="module")
def chunks(document):
    return chunk_blog_document(document)


def test_sample_blog_produces_at_least_one_chunk(chunks):
    assert len(chunks) >= 1


def test_chunk_ids_are_deterministic_and_sequential(document, chunks):
    expected_ids = [f"{document.document_id}:{index}" for index in range(len(chunks))]
    assert [chunk.chunk_id for chunk in chunks] == expected_ids


def test_chunks_inherit_parent_document_fields(document, chunks):
    for chunk in chunks:
        assert chunk.document_id == document.document_id
        assert chunk.author == document.author
        assert chunk.title == document.title
        assert chunk.published_at == document.published_at
        assert chunk.source_url == document.source_url
        assert chunk.source_type == document.source_type


def test_every_chunk_has_non_empty_content(chunks):
    for chunk in chunks:
        assert chunk.content


def test_sample_chunks_stay_within_token_budget(chunks):
    for chunk in chunks:
        assert count_tokens(chunk.content) <= MAX_TOKENS
