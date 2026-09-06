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


def build_answer_prompt(question: str, context: str) -> str:
    evidence = context.strip()
    if evidence:
        evidence_section = evidence
    else:
        evidence_section = (
            "No retrieved evidence is available. "
            "You must not confirm the claim."
        )

    return (
        "You are an evidence-grounded assistant.\n"
        "Answer the user's question using ONLY the provided evidence.\n"
        "Do not use outside knowledge.\n"
        "Do not invent facts that are not supported by the evidence.\n"
        "\n"
        "Evidence rules:\n"
        "- Determine whether the evidence actually supports the claim in the question.\n"
        "- If the evidence is sufficient, answer directly and explain briefly.\n"
        "- Cite supporting evidence using [Source N], where N matches the source blocks below.\n"
        "- Do not invent source numbers, URLs, titles, or other citations.\n"
        "- If the evidence is insufficient or ambiguous, explicitly say that the current archive "
        "does not provide enough evidence to confirm it.\n"
        "- Do not treat retrieval rank or source order as proof.\n"
        "- Do not claim that something was never said. Only say it was not found or cannot be "
        "confirmed from the current archive.\n"
        "- Distinguish clearly between what the source text directly says and your interpretation.\n"
        "- Write the user-facing answer in the same language as the user's question. "
        "Quoted source text may remain in its original language.\n"
        "\n"
        f"Question:\n{question}\n"
        "\n"
        f"Evidence:\n{evidence_section}"
    )
