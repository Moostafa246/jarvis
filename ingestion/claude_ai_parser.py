import json
from pathlib import Path
from typing import Optional


def parse_file(path: Path) -> Optional[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = []
    for msg in data.get("chat_messages", []):
        if msg.get("truncated"):
            print(f"  [SKIP] truncated message {msg.get('uuid')} in {data.get('uuid')}")
            continue

        text_parts = [
            block["text"]
            for block in msg.get("content", [])
            if block.get("type") == "text" and block.get("text")
        ]
        if not text_parts:
            continue

        messages.append({
            "role": msg.get("sender", "unknown"),
            "text": "\n".join(text_parts),
            "timestamp": msg.get("created_at", ""),
            "message_uuid": msg.get("uuid", ""),
        })

    if not messages:
        return None

    return {
        "conversation_id": data.get("uuid", ""),
        "conversation_title": data.get("name", "Untitled"),
        "model": data.get("model", ""),
        "platform": data.get("platform", "CLAUDE_AI"),
        "created_at": data.get("created_at", ""),
        "updated_at": data.get("updated_at", ""),
        "messages": messages,
    }


def parse_directory(directory: Path) -> list[dict]:
    results = []
    for path in sorted(directory.glob("*.json")):
        try:
            parsed = parse_file(path)
            if parsed:
                results.append(parsed)
        except Exception as e:
            print(f"  [ERROR] Failed to parse {path.name}: {e}")
    return results
