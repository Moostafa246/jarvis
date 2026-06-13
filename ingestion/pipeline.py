import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ingestion import claude_ai_parser, claude_code_parser, chunker, embedder, store


def run():
    chroma_path = os.environ.get("CHROMA_DB_PATH", "~/jarvis/db/chroma_db")
    log_path = os.environ.get("INGESTION_LOG_PATH", "~/jarvis/db/ingestion_log.json")
    ai_backup_dir = Path(os.path.expanduser(os.environ.get("CLAUDE_AI_BACKUP_DIR", "~/Downloads/jarvis/claude-ai-backups")))
    code_dir = Path(os.path.expanduser(os.environ.get("CLAUDE_CODE_DIR", "~/.claude/projects")))

    store.init(chroma_path, log_path)

    # --- Source A: claude.ai ---
    ai_files = sorted(ai_backup_dir.glob("*.json"))
    print(f"[claude_ai]   Found {len(ai_files)} conversation files")

    ai_new = 0
    ai_skipped = 0
    ai_chunks = 0

    for path in ai_files:
        try:
            parsed = claude_ai_parser.parse_file(path)
        except Exception as e:
            print(f"  [ERROR] {path.name}: {e}")
            continue
        if not parsed:
            continue

        cid = parsed["conversation_id"]
        updated_at = parsed["updated_at"]

        if store.already_ingested_claude_ai(cid, updated_at):
            ai_skipped += 1
            continue

        store.delete_conversation_chunks(cid)
        chunks = chunker.chunk_conversation(parsed)
        if not chunks:
            continue

        chunks = embedder.embed_chunks(chunks)
        store.store_chunks(chunks)
        store.mark_claude_ai_ingested(cid, updated_at)

        ai_new += 1
        ai_chunks += len(chunks)

    print(f"[claude_ai]   New/updated: {ai_new}  |  Already ingested: {ai_skipped}")
    print(f"[claude_ai]   Processed {ai_new} conversations → {ai_chunks} chunks → embedded → stored")

    # --- Source B: Claude Code ---
    code_sessions = claude_code_parser.parse_directory(code_dir)
    print(f"[claude_code] Found {len(code_sessions)} session files")

    code_new = 0
    code_skipped = 0
    code_chunks = 0

    for path, parsed in code_sessions:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

        if store.already_ingested_claude_code(str(path), mtime):
            code_skipped += 1
            continue

        store.delete_conversation_chunks(parsed["conversation_id"])
        chunks = chunker.chunk_conversation(parsed)
        if not chunks:
            continue

        chunks = embedder.embed_chunks(chunks)
        store.store_chunks(chunks)
        store.mark_claude_code_ingested(str(path), mtime)

        code_new += 1
        code_chunks += len(chunks)

    print(f"[claude_code] New/updated: {code_new}  |  Already ingested: {code_skipped}")
    print(f"[claude_code] Processed {code_new} sessions → {code_chunks} chunks → embedded → stored")

    total = store.total_chunks()
    print(f"\nTotal chunks in ChromaDB: {total}")
    print("Ingestion complete. Run scripts/query_test.py to validate retrieval.")

    # Push new chunks to Railway if configured
    railway_url = os.environ.get("RAILWAY_URL")
    if railway_url and (ai_new > 0 or code_new > 0):
        print(f"\nPushing to Railway ({railway_url})...")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "scripts/push_to_railway.py"],
                capture_output=True, text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"[WARNING] Railway push failed: {result.stderr}")
        except Exception as e:
            print(f"[WARNING] Railway push error: {e}")
