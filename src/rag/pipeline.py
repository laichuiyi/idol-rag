from src.generation.context_builder import build_context
from src.generation.generator import Generator
from src.generation.prompt_builder import build_answer_prompt
from src.retrieval.retriever import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.retriever.retrieve(question, top_k=top_k)
        context = build_context(results)
        prompt = build_answer_prompt(question, context)
        return self.generator.generate(prompt)
