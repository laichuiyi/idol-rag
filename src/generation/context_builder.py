from src.models.chunk import Chunk


def build_context(results: list[tuple[Chunk, float]]) -> str:
    if not results:
        return ""

    blocks: list[str] = []
    for index, (chunk, _score) in enumerate(results, start=1):
        published = chunk.published_at.strftime("%Y-%m-%d %H:%M")
        blocks.append(
            f"[Source {index}]\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Author: {chunk.author}\n"
            f"Title: {chunk.title}\n"
            f"Published: {published}\n"
            f"URL: {chunk.source_url}\n"
            f"Content:\n"
            f"{chunk.content}"
        )
    return "\n\n".join(blocks)
