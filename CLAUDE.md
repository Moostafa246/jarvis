# Jarvis — Personal AI Operating System

## What this is
Jarvis is a personal AI OS for Mustafa Kazim. Phase 1 is the ingestion pipeline — pulling conversation data from claude.ai and Claude Code sessions, chunking and embedding it, storing in ChromaDB for semantic retrieval.

## Repo structure
```
jarvis/
├── ingestion/
│   ├── claude_ai_parser.py      # parses ~/Downloads/jarvis/claude-ai-backups/*.json
│   ├── claude_code_parser.py    # parses ~/.claude/projects/**/*.jsonl
│   ├── chunker.py               # sliding window over message pairs (window=3, overlap=1)
│   ├── embedder.py              # OpenAI text-embedding-3-small, batch=100
│   ├── store.py                 # ChromaDB + ingestion_log.json
│   └── pipeline.py              # end-to-end orchestration
├── scripts/
│   ├── run_ingestion.py         # python scripts/run_ingestion.py
│   └── query_test.py            # semantic search validation
└── tests/
    ├── test_claude_ai_parser.py
    └── test_claude_code_parser.py
```

## Setup
```bash
/usr/local/bin/python3.11 -m venv venv
source venv/bin/activate
pip install chromadb openai python-dotenv
cp .env.example .env  # fill in OPENAI_API_KEY
```

## Run
```bash
python scripts/run_ingestion.py
python scripts/query_test.py
```

## Dedup strategy
- claude.ai: skip if `conversation_id` + `updated_at` match ingestion log
- claude_code: skip if file path + mtime match ingestion log
- Re-ingest: delete old chunks, insert new ones

## What's NOT built yet
- FastAPI backend
- Airflow DAG
- MCP server
- Frontend (Electron / iOS)
- Apple Notes / voice / health data ingestion
- PostgreSQL (ingestion_log.json is Phase 1)

## Data sensitivity
ChromaDB (`db/`) and ingestion log are gitignored. Never commit conversation data.
