from pathlib import Path
import os
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.generator import Generator

PROMPT = "請用廣東話一句回答: RAG 入面 Retriever 主要負責做咩？"


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

    generator = Generator(api_key=api_key, model_name=model_name)
    answer = generator.generate(PROMPT)

    print(f"model: {generator.model_name}")
    print(f"answer: {answer}")


if __name__ == "__main__":
    main()
