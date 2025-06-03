import openai
import json


class OpenAIModel:
    def __init__(self):
        self.client = openai.OpenAI()

    def embed(self, query: str) -> list[float]:
        query = "Explique o conceito de elasticidade no contexto de sistemas reativos"

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[query],
            dimensions=384,
            encoding_format="float",
        )

        return response.data[0].embedding

    def chat(self, messages: list[dict], model: str = "gpt-4.1-mini") -> dict:
        response = self.client.chat.completions.create(
            model=model, messages=messages, response_format={"type": "json_object"}
        )
        return (
            response.choices[0].message.content
            if isinstance(response.choices[0].message.content, dict)
            else json.loads(response.choices[0].message.content)
        )
