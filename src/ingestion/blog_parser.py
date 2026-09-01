from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from src.models.blog_document import BlogDocument

SITE_ORIGIN = "https://www.nogizaka46.com"
PUBLISHED_AT_FORMAT = "%Y.%m.%d %H:%M"
DOCUMENT_ID_PATTERN = re.compile(r"/diary/detail/(\d+)")


def parse_blog_html(html_path: str | Path) -> BlogDocument:
    """Parse one local Nogizaka46 blog HTML file into a BlogDocument."""
    html = Path(html_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    title = _require_text(soup, "h1.bd--hd__ttl", "title")
    author = _require_text(soup, "p.bd--prof__name", "author")
    source_url = _require_source_url(soup)
    document_id = _require_document_id(source_url)
    published_at = _require_published_at(soup)

    body = soup.select_one("div.bd--edit")
    if body is None:
        raise ValueError("Required field 'content' could not be found: missing div.bd--edit")

    images = _extract_images(body)
    content = _extract_content(body)
    if not content:
        raise ValueError("Required field 'content' could not be found: div.bd--edit is empty")

    return BlogDocument(
        document_id=document_id,
        author=author,
        title=title,
        published_at=published_at,
        source_url=source_url,
        content=content,
        images=images,
    )


def _require_text(soup: BeautifulSoup, selector: str, field_name: str) -> str:
    element = soup.select_one(selector)
    text = element.get_text(strip=True) if element else ""
    if not text:
        raise ValueError(f"Required field '{field_name}' could not be found: missing {selector}")
    return text


def _require_source_url(soup: BeautifulSoup) -> str:
    tag = soup.find("meta", property="og:url")
    source_url = tag.get("content", "").strip() if isinstance(tag, Tag) else ""
    if not source_url:
        raise ValueError("Required field 'source_url' could not be found: missing meta property='og:url'")
    return source_url


def _require_document_id(source_url: str) -> str:
    match = DOCUMENT_ID_PATTERN.search(source_url)
    if not match:
        raise ValueError(
            f"Required field 'document_id' could not be found: no diary ID in source_url '{source_url}'"
        )
    return match.group(1)


def _require_published_at(soup: BeautifulSoup) -> datetime:
    raw = _require_text(soup, "p.bd--hd__date", "published_at")
    try:
        return datetime.strptime(raw, PUBLISHED_AT_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"Required field 'published_at' could not be parsed from '{raw}' "
            f"using format '{PUBLISHED_AT_FORMAT}'"
        ) from exc


def _extract_images(body: Tag) -> list[str]:
    images: list[str] = []
    for img in body.find_all("img"):
        src = img.get("src", "").strip()
        if not src:
            continue
        images.append(urljoin(SITE_ORIGIN, src))
    return images


def _extract_content(body: Tag) -> str:
    fragment = BeautifulSoup(str(body), "html.parser")
    root = fragment.select_one("div.bd--edit") or fragment

    for img in root.find_all("img"):
        img.decompose()
    for br in root.find_all("br"):
        br.replace_with("\n")

    text = root.get_text()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
