#!/usr/bin/env python3
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from ingestion import store, embedder

QUERIES = [
    "what have I been working on related to IdiCafe?",
    "what decisions did I make about the house move?",
    "what is campaign agent v2?",
]


def embed_one(text: str) -> list[float]:
    return embedder.embed_chunks([{"text": text}])[0]["embedding"]


def main():
    chroma_path = os.environ.get("CHROMA_DB_PATH", "~/jarvis/db/chroma_db")
    log_path = os.environ.get("INGESTION_LOG_PATH", "~/jarvis/db/ingestion_log.json")
    store.init(chroma_path, log_path)

    for query in QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("="*60)

        vector = embed_one(query)
        results = store.query(query, n_results=5, embeddings_fn=lambda texts: [embed_one(t) for t in texts])

        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"\n[{i}] source={meta.get('source')}  date={meta.get('date')}  score={1 - r['distance']:.3f}")
            print(f"    title: {meta.get('conversation_title', 'N/A')}")
            print(f"    text:  {r['text'][:300].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
