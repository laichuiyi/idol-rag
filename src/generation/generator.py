from google import genai


class Generator:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text or ""
