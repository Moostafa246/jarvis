import json
from pathlib import Path
from typing import Optional

IGNORE_TYPES = {"queue-operation", "attachment", "system", "mode", "last-prompt"}


def parse_file(path: Path) -> Optional[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    title = path.stem
    messages = []
    cwd = ""
    git_branch = ""
    session_id = ""

    for rec in records:
        msg_type = rec.get("type", "")

        if msg_type in IGNORE_TYPES:
            continue

        if msg_type == "ai-title":
            title = rec.get("aiTitle", title)
            continue

        if msg_type == "user":
            content = rec.get("message", {}).get("content", "")
            if not isinstance(content, str):
                continue
            content = content.strip()
            if not content:
                continue
            cwd = cwd or rec.get("cwd", "")
            git_branch = git_branch or rec.get("gitBranch", "")
            session_id = session_id or rec.get("sessionId", "")
            messages.append({
                "role": "user",
                "text": content,
                "timestamp": rec.get("timestamp", ""),
                "message_uuid": rec.get("uuid", ""),
            })
            continue

        if msg_type == "assistant":
            content_blocks = rec.get("message", {}).get("content", [])
            if isinstance(content_blocks, str):
                text = content_blocks.strip()
            else:
                text_parts = [
                    b.get("text", "")
                    for b in content_blocks
                    if b.get("type") == "text" and b.get("text")
                ]
                text = "\n".join(text_parts).strip()
            if not text:
                continue
            messages.append({
                "role": "assistant",
                "text": text,
                "timestamp": rec.get("timestamp", ""),
                "message_uuid": rec.get("uuid", ""),
            })
            continue

    if not messages:
        return None

    return {
        "conversation_id": session_id or path.stem,
        "conversation_title": title,
        "model": "",
        "platform": "CLAUDE_CODE",
        "created_at": messages[0]["timestamp"] if messages else "",
        "updated_at": messages[-1]["timestamp"] if messages else "",
        "cwd": cwd,
        "git_branch": git_branch,
        "messages": messages,
    }


def parse_directory(directory: Path) -> list[tuple[Path, dict]]:
    results = []
    for path in sorted(directory.rglob("*.jsonl")):
        try:
            parsed = parse_file(path)
            if parsed:
                results.append((path, parsed))
        except Exception as e:
            print(f"  [ERROR] Failed to parse {path}: {e}")
    return results
