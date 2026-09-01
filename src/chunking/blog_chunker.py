from src.chunking.token_counter import count_tokens
from src.models.blog_document import BlogDocument
from src.models.chunk import Chunk

DEFAULT_MAX_TOKENS = 300
DEFAULT_OVERLAP_TOKENS = 50


def chunk_blog_document(
    document: BlogDocument,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    paragraphs = [part.strip() for part in document.content.split("\n\n")]
    paragraphs = [part for part in paragraphs if part]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []

    for paragraph in paragraphs:
        if _fits(current, paragraph, max_tokens):
            current.append(paragraph)
            continue

        if current:
            chunks.append(_build_chunk(document, len(chunks), current))
            current = _overlap_paragraphs(current, overlap_tokens)
            if _fits(current, paragraph, max_tokens):
                current.append(paragraph)
            else:
                current = [paragraph]
        else:
            current = [paragraph]

    if current:
        chunks.append(_build_chunk(document, len(chunks), current))

    return chunks


def _join(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs)


def _fits(current: list[str], paragraph: str, max_tokens: int) -> bool:
    return count_tokens(_join(current + [paragraph])) <= max_tokens


def _overlap_paragraphs(paragraphs: list[str], overlap_tokens: int) -> list[str]:
    if not paragraphs or overlap_tokens <= 0:
        return []

    selected: list[str] = []
    for paragraph in reversed(paragraphs):
        selected = [paragraph] + selected
        if count_tokens(_join(selected)) >= overlap_tokens:
            break
    return selected


def _build_chunk(document: BlogDocument, chunk_index: int, paragraphs: list[str]) -> Chunk:
    return Chunk(
        chunk_id=f"{document.document_id}:{chunk_index}",
        document_id=document.document_id,
        content=_join(paragraphs),
        author=document.author,
        title=document.title,
        published_at=document.published_at,
        source_url=document.source_url,
        source_type=document.source_type,
    )
