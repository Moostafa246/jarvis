# Jarvis

Personal AI operating system — Phase 1: conversation ingestion pipeline.

Pulls data from claude.ai and Claude Code sessions, embeds with OpenAI, stores in ChromaDB for semantic search.

## Quick start

```bash
/usr/local/bin/python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY
python scripts/run_ingestion.py
python scripts/query_test.py
```

See [CLAUDE.md](CLAUDE.md) for full architecture notes.
