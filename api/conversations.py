import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic

DATA_DIR = Path(os.path.expanduser("~/jarvis/data/prive-conversations"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conv_path(topic_slug: str, conv_id: str) -> Path:
    topic_dir = DATA_DIR / topic_slug
    topic_dir.mkdir(parents=True, exist_ok=True)
    return topic_dir / f"{conv_id}.json"


def create_conversation() -> dict:
    conv = {
        "id": str(uuid.uuid4()),
        "title": "New conversation",
        "topic_slug": "uncategorized",
        "topic_label": "Uncategorized",
        "created_at": _now(),
        "updated_at": _now(),
        "messages": [],
        "labeled": False,
    }
    path = _conv_path(conv["topic_slug"], conv["id"])
    path.write_text(json.dumps(conv, indent=2))
    return conv


def load_conversation(conv_id: str) -> dict | None:
    for path in DATA_DIR.rglob(f"{conv_id}.json"):
        return json.loads(path.read_text())
    return None


def save_conversation(conv: dict) -> dict:
    conv["updated_at"] = _now()
    old_path = None
    for p in DATA_DIR.rglob(f"{conv['id']}.json"):
        old_path = p
        break

    new_path = _conv_path(conv["topic_slug"], conv["id"])

    if old_path and old_path != new_path:
        old_path.unlink(missing_ok=True)

    new_path.write_text(json.dumps(conv, indent=2))
    return conv


def list_conversations() -> dict:
    topics: dict[str, dict] = {}
    for path in sorted(DATA_DIR.rglob("*.json")):
        try:
            conv = json.loads(path.read_text())
            if not conv.get("messages"):
                continue
            slug = conv["topic_slug"]
            label = conv["topic_label"]
            if slug not in topics:
                topics[slug] = {"label": label, "conversations": []}
            topics[slug]["conversations"].append({
                "id": conv["id"],
                "title": conv["title"],
                "created_at": conv["created_at"],
                "updated_at": conv["updated_at"],
            })
        except Exception:
            continue

    for slug in topics:
        topics[slug]["conversations"].sort(key=lambda c: c["updated_at"], reverse=True)

    return topics


def auto_label(conv: dict, client: anthropic.Anthropic) -> dict:
    if conv.get("labeled") or len(conv["messages"]) < 2:
        return conv

    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}"
        for m in conv["messages"][:6]
    )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Analyze this conversation and return JSON only (no markdown):
{{
  "topic_slug": "kebab-case-topic (e.g. idicafe, house, work-loblaw, health, finances, personal)",
  "topic_label": "Display name (e.g. IdiCafe, House, Work / Loblaw)",
  "title": "Short specific title for this conversation (5-8 words max)"
}}

Conversation:
{transcript}"""
        }]
    )

    try:
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        conv["topic_slug"] = data.get("topic_slug", "uncategorized").lower().replace(" ", "-")
        conv["topic_label"] = data.get("topic_label", "Uncategorized")
        conv["title"] = data.get("title", "Conversation")
        conv["labeled"] = True
    except Exception:
        pass

    return conv


def ingest_conversation(conv: dict):
    """Re-ingest a Prive conversation into ChromaDB."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion import chunker, embedder, store

    if not conv.get("messages"):
        return

    # Build a conversation object matching the chunker's expected format
    messages = []
    for m in conv["messages"]:
        role = "human" if m["role"] == "user" else "assistant"
        messages.append({
            "role": role,
            "text": m["content"],
            "timestamp": m.get("timestamp", conv["created_at"]),
            "message_uuid": str(uuid.uuid4()),
        })

    conv_obj = {
        "conversation_id": f"prive-{conv['id']}",
        "conversation_title": conv["title"],
        "model": "claude-sonnet-4-6",
        "platform": "PRIVE",
        "created_at": conv["created_at"],
        "updated_at": conv["updated_at"],
        "messages": messages,
    }

    store.delete_conversation_chunks(f"prive-{conv['id']}")
    chunks = chunker.chunk_conversation(conv_obj)
    if not chunks:
        return

    for chunk in chunks:
        chunk["metadata"]["source"] = "prive"
        chunk["metadata"]["topic"] = conv.get("topic_slug", "")

    chunks = embedder.embed_chunks(chunks)
    store.store_chunks(chunks)
