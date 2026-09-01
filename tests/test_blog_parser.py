from datetime import datetime
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.blog_parser import parse_blog_html

SAMPLE_HTML = PROJECT_ROOT / "data" / "raw" / "blog" / "綴夏.html"


@pytest.fixture(scope="module")
def document():
    return parse_blog_html(SAMPLE_HTML)


def test_parse_blog_html_sample(document):
    assert document.document_id == "104653"
    assert document.author == "井上 和"
    assert document.title == "綴夏"
    assert document.published_at == datetime(2026, 6, 13, 15, 4)
    assert document.source_url == "https://www.nogizaka46.com/s/n46/diary/detail/104653"
    assert document.source_type == "blog"

    assert len(document.images) == 3
    for url in document.images:
        assert url.startswith("https://www.nogizaka46.com/")

    assert document.content
    assert "<img" not in document.content
    for url in document.images:
        assert url not in document.content
