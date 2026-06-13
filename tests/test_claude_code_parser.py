import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.claude_code_parser import parse_file

SAMPLE_LINES = [
    '{"type": "queue-operation", "operation": "append", "timestamp": "2026-01-01T00:00:00Z", "sessionId": "sess-1", "content": []}',
    '{"type": "ai-title", "aiTitle": "Test Session Title", "timestamp": "2026-01-01T00:00:01Z", "sessionId": "sess-1"}',
    '{"parentUuid": null, "isSidechain": false, "type": "user", "message": {"content": "Hello from user"}, "uuid": "u1", "timestamp": "2026-01-01T00:00:02Z", "cwd": "/Users/test/project", "sessionId": "sess-1", "gitBranch": "main"}',
    '{"parentUuid": "u1", "isSidechain": false, "type": "assistant", "message": {"content": [{"type": "text", "text": "Hello from assistant"}, {"type": "thinking", "thinking": "internal thoughts"}]}, "uuid": "a1", "timestamp": "2026-01-01T00:00:03Z", "sessionId": "sess-1"}',
]


def test_basic_parse():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(SAMPLE_LINES))
        tmp = Path(f.name)

    result = parse_file(tmp)
    assert result is not None
    assert result["conversation_title"] == "Test Session Title"
    assert result["cwd"] == "/Users/test/project"
    assert result["git_branch"] == "main"
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["text"] == "Hello from user"
    assert result["messages"][1]["role"] == "assistant"
    assert "Hello from assistant" in result["messages"][1]["text"]
    assert "internal thoughts" not in result["messages"][1]["text"]
    tmp.unlink()
    print("test_basic_parse: PASSED")


def test_ignores_queue_operations():
    lines = ['{"type": "queue-operation", "operation": "x", "timestamp": "t", "sessionId": "s", "content": []}']
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines))
        tmp = Path(f.name)

    result = parse_file(tmp)
    assert result is None
    tmp.unlink()
    print("test_ignores_queue_operations: PASSED")


if __name__ == "__main__":
    test_basic_parse()
    test_ignores_queue_operations()
    print("All tests passed.")
