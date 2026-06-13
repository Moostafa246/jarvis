import json
import os
import uuid
from pathlib import Path

import chromadb


_collection = None
_log: dict = {"claude_ai": {}, "claude_code": {}}
_log_path: Path | None = None


def init(chroma_path: str, log_path: str):
    global _collection, _log, _log_path

    _log_path = Path(os.path.expanduser(log_path))
    _log_path.parent.mkdir(parents=True, exist_ok=True)

    if _log_path.exists():
        _log = json.loads(_log_path.read_text())
    else:
        _log = {"claude_ai": {}, "claude_code": {}}

    client = chromadb.PersistentClient(path=os.path.expanduser(chroma_path))
    _collection = client.get_or_create_collection(
        name="jarvis_conversations",
        metadata={"hnsw:space": "cosine"},
    )


def _save_log():
    _log_path.write_text(json.dumps(_log, indent=2))


def already_ingested_claude_ai(conversation_id: str, updated_at: str) -> bool:
    stored = _log["claude_ai"].get(conversation_id)
    return stored == updated_at


def already_ingested_claude_code(file_path: str, mtime: str) -> bool:
    stored = _log["claude_code"].get(file_path)
    return stored == mtime


def delete_conversation_chunks(conversation_id: str):
    results = _collection.get(where={"conversation_id": conversation_id})
    if results["ids"]:
        _collection.delete(ids=results["ids"])


def store_chunks(chunks: list[dict]):
    if not chunks:
        return

    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [c["text"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    _collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def mark_claude_ai_ingested(conversation_id: str, updated_at: str):
    _log["claude_ai"][conversation_id] = updated_at
    _save_log()


def mark_claude_code_ingested(file_path: str, mtime: str):
    _log["claude_code"][file_path] = mtime
    _save_log()


def total_chunks() -> int:
    return _collection.count()


def query(text: str, n_results: int = 5, embeddings_fn=None) -> list[dict]:
    if embeddings_fn:
        vector = embeddings_fn([text])[0]
        results = _collection.query(query_embeddings=[vector], n_results=n_results)
    else:
        results = _collection.query(query_texts=[text], n_results=n_results)

    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return output
