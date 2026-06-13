import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.claude_ai_parser import parse_file


SAMPLE = {
    "uuid": "test-uuid-1",
    "name": "Test Conversation",
    "summary": "",
    "model": "claude-sonnet-4-5",
    "created_at": "2025-06-23T12:00:00Z",
    "updated_at": "2025-06-23T12:30:00Z",
    "is_starred": False,
    "platform": "CLAUDE_AI",
    "chat_messages": [
        {
            "uuid": "msg-1",
            "content": [{"type": "text", "text": "Hello world"}],
            "sender": "human",
            "created_at": "2025-06-23T12:00:00Z",
            "truncated": False,
        },
        {
            "uuid": "msg-2",
            "content": [{"type": "text", "text": "Hi there!"}],
            "sender": "assistant",
            "created_at": "2025-06-23T12:01:00Z",
            "truncated": False,
        },
        {
            "uuid": "msg-3",
            "content": [{"type": "text", "text": "truncated content"}],
            "sender": "human",
            "created_at": "2025-06-23T12:02:00Z",
            "truncated": True,
        },
    ],
}


def test_basic_parse():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE, f)
        tmp = Path(f.name)

    result = parse_file(tmp)
    assert result is not None
    assert result["conversation_id"] == "test-uuid-1"
    assert result["conversation_title"] == "Test Conversation"
    assert len(result["messages"]) == 2  # truncated message skipped
    assert result["messages"][0]["role"] == "human"
    assert result["messages"][0]["text"] == "Hello world"
    assert result["messages"][1]["role"] == "assistant"
    tmp.unlink()
    print("test_basic_parse: PASSED")


def test_skip_non_text_blocks():
    data = dict(SAMPLE)
    data["chat_messages"] = [
        {
            "uuid": "msg-tool",
            "content": [{"type": "tool_use", "id": "x", "name": "bash", "input": {}}],
            "sender": "assistant",
            "created_at": "2025-06-23T12:00:00Z",
            "truncated": False,
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp = Path(f.name)

    result = parse_file(tmp)
    assert result is None  # no text messages
    tmp.unlink()
    print("test_skip_non_text_blocks: PASSED")


if __name__ == "__main__":
    test_basic_parse()
    test_skip_non_text_blocks()
    print("All tests passed.")
