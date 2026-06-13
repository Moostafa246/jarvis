import os
from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def embed_chunks(chunks: list[dict], batch_size: int = 100) -> list[dict]:
    client = _get_client()
    texts = [c["text"] for c in chunks]
    vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        vectors.extend([item.embedding for item in response.data])

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks
