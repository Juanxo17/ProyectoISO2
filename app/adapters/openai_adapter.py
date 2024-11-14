# app/adapters/openai_adapter.py
import openai
from app.core.ports import LlmPort


class OpenAIAdapter(LlmPort):
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        openai.api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate_text(self, prompt: str, retrieval_context: str) -> str:
        print(prompt)

        response = openai.ChatCompletion.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": f"The context is: {retrieval_context}, please respond to the following question: ",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        return str(response["choices"][0]["message"]["content"])
