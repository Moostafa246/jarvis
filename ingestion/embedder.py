from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

_ef = None


def _get_ef():
    global _ef
    if _ef is None:
        _ef = DefaultEmbeddingFunction()
    return _ef


def embed_chunks(chunks: list[dict], batch_size: int = 100) -> list[dict]:
    ef = _get_ef()
    texts = [c["text"] for c in chunks]
    vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        vectors.extend(ef(batch))

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks
