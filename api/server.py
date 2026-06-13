import os
import json
import sys
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion import store, embedder
from api.conversations import (
    create_conversation, load_conversation, save_conversation,
    list_conversations, auto_label, ingest_conversation
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_initialized = False

def ensure_init():
    global _initialized
    if not _initialized:
        store.init(
            os.environ.get("CHROMA_DB_PATH", "~/jarvis/db/chroma_db"),
            os.environ.get("INGESTION_LOG_PATH", "~/jarvis/db/ingestion_log.json"),
        )
        _initialized = True

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# --- Conversation management endpoints ---

@app.get("/conversations")
def get_conversations():
    return list_conversations()

@app.post("/conversations")
def new_conversation():
    return create_conversation()

@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: str):
    conv = load_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

@app.put("/conversations/{conv_id}/rename")
def rename_conversation(conv_id: str, body: dict):
    conv = load_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Not found")
    if "title" in body:
        conv["title"] = body["title"]
    if "topic_label" in body:
        conv["topic_label"] = body["topic_label"]
        conv["topic_slug"] = body.get("topic_slug", body["topic_label"].lower().replace(" ", "-").replace("/", "-"))
    save_conversation(conv)
    return conv


# --- Chat endpoint ---

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    conversation_id: str | None = None

def retrieve_context(query: str, n: int = 8) -> list[dict]:
    results = store.query(query, n_results=n, embeddings_fn=lambda texts: [
        embedder.embed_chunks([{"text": t}])[0]["embedding"] for t in texts
    ])
    return results

def build_system_prompt(context_chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(context_chunks):
        meta = chunk["metadata"]
        date = meta.get("date", "unknown date")
        title = meta.get("conversation_title", "Untitled")
        source = meta.get("source", "unknown")
        context_blocks.append(f"[{i+1}] {title} ({date}, {source}):\n{chunk['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    return f"""You are Prive, a personal AI assistant with access to Mustafa's complete conversation history and memory.

You have deep context about Mustafa's life, projects, goals, relationships, work, and thinking. You speak directly, confidently, and personally — you know him well. Never say "based on the context provided" — just speak naturally as someone who knows him.

Be concise unless depth is asked for. Match his energy. If he's casual, be casual. If he wants to go deep, go deep.

RELEVANT CONTEXT FROM MUSTAFA'S HISTORY:
{context_str}

Use this context naturally in your response. If a piece of context is directly relevant, reference it specifically (the project, decision, date). Don't list sources — weave them in naturally."""

async def stream_response(message: str, history: list[dict], conv_id: str | None) -> AsyncIterator[str]:
    ensure_init()

    # Load or create conversation
    if conv_id:
        conv = load_conversation(conv_id)
    if not conv_id or not conv:
        conv = create_conversation()
        conv_id = conv["id"]
        yield f"data: {json.dumps({'type': 'conv_id', 'conversation_id': conv_id})}\n\n"

    # Retrieve context
    chunks = retrieve_context(message)
    sources = [
        {
            "title": c["metadata"].get("conversation_title", "Untitled"),
            "date": c["metadata"].get("date", ""),
            "source": c["metadata"].get("source", ""),
            "score": round(1 - c["distance"], 3),
        }
        for c in chunks
    ]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    system = build_system_prompt(chunks)
    messages = []
    for h in history[-10:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    full_text = ""
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            full_text += text
            yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"

    # Save exchange to conversation
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    conv["messages"].append({"role": "user", "content": message, "timestamp": now})
    conv["messages"].append({"role": "assistant", "content": full_text, "timestamp": now})

    # Auto-label after first exchange
    if not conv.get("labeled"):
        conv = auto_label(conv, client)
        yield f"data: {json.dumps({'type': 'labeled', 'topic_slug': conv['topic_slug'], 'topic_label': conv['topic_label'], 'title': conv['title']})}\n\n"

    save_conversation(conv)

    # Ingest into ChromaDB in background (non-blocking for the stream)
    try:
        ingest_conversation(conv)
    except Exception as e:
        print(f"[ingest] Error ingesting prive conversation: {e}")

    yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id})}\n\n"

@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        stream_response(req.message, req.history, req.conversation_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text()
