from typing import Any


def chunk_conversation(conversation: dict, window: int = 3, overlap: int = 1) -> list[dict]:
    messages = conversation["messages"]

    # Pair up messages into (human/user, assistant) turns
    pairs = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] in ("human", "user"):
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                pairs.append((msg, messages[i + 1]))
                i += 2
            else:
                pairs.append((msg, None))
                i += 1
        elif msg["role"] == "assistant":
            pairs.append((None, msg))
            i += 1
        else:
            i += 1

    chunks = []
    step = window - overlap
    for start in range(0, max(1, len(pairs)), step):
        window_pairs = pairs[start: start + window]
        if not window_pairs:
            break

        parts = []
        for human_msg, assistant_msg in window_pairs:
            if human_msg:
                parts.append(f"Human: {human_msg['text']}")
            if assistant_msg:
                parts.append(f"Assistant: {assistant_msg['text']}")

        text = "\n\n".join(parts)
        first_msg = next(
            m for pair in window_pairs for m in pair if m is not None
        )

        base_meta: dict[str, Any] = {
            "source": "claude_ai" if conversation.get("platform") != "CLAUDE_CODE" else "claude_code",
            "conversation_id": conversation["conversation_id"],
            "conversation_title": conversation["conversation_title"],
            "chunk_index": len(chunks),
            "date": first_msg["timestamp"][:10] if first_msg.get("timestamp") else "",
            "model": conversation.get("model", ""),
            "platform": conversation.get("platform", ""),
            "project": conversation.get("cwd", ""),
            "git_branch": conversation.get("git_branch", ""),
        }

        chunks.append({"text": text, "metadata": base_meta})

        # Stop if we've covered all pairs
        if start + window >= len(pairs):
            break

    return chunks
