#!/usr/bin/env python3
"""
Push local ChromaDB chunks to Railway.
Run once to seed, then nightly cron keeps it updated via the same endpoint.
"""
import os
import sys
import json
import math
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from ingestion import store

RAILWAY_URL = os.environ.get("RAILWAY_URL", "https://courageous-mindfulness-production-7174.up.railway.app")
INGEST_SECRET = os.environ.get("INGEST_SECRET", "")
BATCH_SIZE = 100


def push_all():
    # Init local ChromaDB
    store.init(
        os.environ.get("CHROMA_DB_PATH", "~/jarvis/db/chroma_db"),
        os.environ.get("INGESTION_LOG_PATH", "~/jarvis/db/ingestion_log.json"),
    )

    print(f"Reading local ChromaDB...")
    result = store._collection.get(include=["documents", "embeddings", "metadatas"])

    ids = result["ids"]
    documents = result["documents"]
    embeddings = [e.tolist() if hasattr(e, "tolist") else list(e) for e in result["embeddings"]]
    metadatas = result["metadatas"]
    total = len(ids)

    print(f"Found {total} chunks locally.")

    # Check what's already on Railway
    try:
        req = urllib.request.Request(f"{RAILWAY_URL}/ingest/stats")
        with urllib.request.urlopen(req) as resp:
            stats = json.loads(resp.read())
            print(f"Railway currently has {stats['total_chunks']} chunks.")
    except Exception as e:
        print(f"Could not reach Railway: {e}")
        sys.exit(1)

    headers = {"Content-Type": "application/json"}
    if INGEST_SECRET:
        headers["Authorization"] = f"Bearer {INGEST_SECRET}"

    n_batches = math.ceil(total / BATCH_SIZE)
    pushed = 0

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)

        batch = {
            "ids": ids[start:end],
            "documents": documents[start:end],
            "embeddings": embeddings[start:end],
            "metadatas": metadatas[start:end],
        }

        data = json.dumps(batch).encode()
        req = urllib.request.Request(
            f"{RAILWAY_URL}/ingest/chunks",
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                pushed += result["inserted"]
                print(f"  Batch {i+1}/{n_batches} — pushed {result['inserted']} chunks ({pushed}/{total} total)")
        except urllib.error.HTTPError as e:
            print(f"  Batch {i+1} failed: {e.code} {e.read().decode()}")
            sys.exit(1)

    print(f"\nDone. {pushed} chunks now on Railway.")

    # Verify
    req = urllib.request.Request(f"{RAILWAY_URL}/ingest/stats")
    with urllib.request.urlopen(req) as resp:
        stats = json.loads(resp.read())
        print(f"Railway total: {stats['total_chunks']} chunks.")


if __name__ == "__main__":
    push_all()
