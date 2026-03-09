"""
Adaptive Memory Agent — Layered memory with namespace, retrieval, episodic, procedural.

Implements all gaps from agent-memory-development skill: namespace scoping, similarity
retrieval, episodic memory, procedural memory (reflection), memory update/reconciliation,
profile, and short-term session management.

Usage:
    python adaptive_memory_agent.py                  # watch ./inbox, serve on :8889
    python adaptive_memory_agent.py --port 8890

Query:
    curl "http://localhost:8889/query?q=what+do+you+know&user_id=alice"
    curl -X POST http://localhost:8889/ingest -d '{"text": "some info", "user_id": "alice"}'
"""

import argparse
import asyncio
import json
import logging
import math
import mimetypes
import os
import shutil
import signal
import sqlite3
import struct
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# ─── Load .env (API key for Gemini) ───────────────────────────
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))

# ─── Config ────────────────────────────────────────────────────

MODEL = os.getenv("MODEL", "gemini-3.1-flash-lite-preview")
DB_PATH = os.getenv("MEMORY_DB", os.getenv("ADAPTIVE_MEMORY_DB", "adaptive_memory.db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
SESSION_WINDOW = int(os.getenv("SESSION_WINDOW", "20"))

# Gap: Namespace — context vars for request-scoped user_id and context
current_user_id: ContextVar[str] = ContextVar("current_user_id", default="default")
current_context: ContextVar[str] = ContextVar("current_context", default="")

TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".log", ".xml", ".yaml", ".yml"}
MEDIA_EXTENSIONS = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp", ".svg": "image/svg+xml",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".flac": "audio/flac",
    ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
    ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
    ".pdf": "application/pdf",
}
ALL_SUPPORTED = TEXT_EXTENSIONS | set(MEDIA_EXTENSIONS.keys())

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="[%H:%M]")
log = logging.getLogger("adaptive-memory-agent")

# ─── Database ──────────────────────────────────────────────────
# Gap: Schema — memories+ns, profiles, episodic, instructions, sessions


def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'default',
            context TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            raw_text TEXT NOT NULL,
            summary TEXT NOT NULL,
            entities TEXT NOT NULL DEFAULT '[]',
            topics TEXT NOT NULL DEFAULT '[]',
            connections TEXT NOT NULL DEFAULT '[]',
            importance REAL NOT NULL DEFAULT 0.5,
            embedding BLOB,
            created_at TEXT NOT NULL,
            consolidated INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(user_id, context);
        CREATE TABLE IF NOT EXISTS consolidations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'default',
            context TEXT NOT NULL DEFAULT '',
            source_ids TEXT NOT NULL,
            summary TEXT NOT NULL,
            insight TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS processed_files (
            path TEXT PRIMARY KEY,
            processed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT '',
            profile_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, context)
        );
        CREATE TABLE IF NOT EXISTS episodic_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'default',
            context TEXT NOT NULL DEFAULT '',
            input_text TEXT NOT NULL,
            output_text TEXT NOT NULL,
            outcome TEXT NOT NULL DEFAULT '',
            embedding BLOB,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_episodic_namespace ON episodic_memories(user_id, context);
        CREATE TABLE IF NOT EXISTS agent_instructions (
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            instructions_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, key)
        );
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT 'default',
            messages_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (session_id, user_id)
        );
    """)
    return db


# ─── Embedding Helper (Gap: Retrieval) ──────────────────────────


def _embed(text: str) -> list[float] | None:
    """Compute embedding for text using Gemini. Returns None on failure."""
    try:
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        client = genai.Client(api_key=api_key)
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text[:8000])
        if result and result.embeddings:
            emb = result.embeddings[0]
            return list(emb.values) if hasattr(emb, "values") else list(emb)
        return None
    except Exception as e:
        log.warning(f"Embedding failed: {e}")
        return None


def _blob_to_vec(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    try:
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))
    except Exception:
        return None


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ─── ADK Tools (Semantic Memory, Namespace-aware) ────────────────
# Gap: Namespace — scope memories by user_id and context


def store_memory(
    raw_text: str,
    summary: str,
    entities: list[str],
    topics: list[str],
    importance: float,
    source: str = "",
    user_id: str = "",
    context: str = "",
) -> dict:
    """Store a processed memory. Gap: Namespace — user_id and context scoping."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    emb = _embed(summary + " " + raw_text[:500])
    emb_blob = _vec_to_blob(emb) if emb else None
    cursor = db.execute(
        """INSERT INTO memories (user_id, context, source, raw_text, summary, entities, topics, importance, embedding, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (uid, ctx, source, raw_text, summary, json.dumps(entities), json.dumps(topics), importance, emb_blob, now),
    )
    db.commit()
    mid = cursor.lastrowid
    db.close()
    log.info(f"Stored memory #{mid}: {summary[:60]}...")
    return {"memory_id": mid, "status": "stored", "summary": summary}


def read_memory(memory_id: int) -> dict | None:
    """Read a single memory by ID. Gap: Update — lookup before update."""
    db = get_db()
    row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    db.close()
    if not row:
        return None
    return {
        "id": row["id"], "user_id": row["user_id"], "context": row["context"],
        "source": row["source"], "raw_text": row["raw_text"], "summary": row["summary"],
        "entities": json.loads(row["entities"]), "topics": json.loads(row["topics"]),
        "importance": row["importance"], "created_at": row["created_at"],
    }


def update_memory(
    memory_id: int,
    raw_text: str,
    summary: str,
    entities: list[str],
    topics: list[str],
    importance: float,
    user_id: str = "",
    context: str = "",
) -> dict:
    """Update an existing memory. Gap: Update — reconcile new info with existing."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    db = get_db()
    row = db.execute("SELECT id FROM memories WHERE id = ? AND user_id = ? AND context = ?", (memory_id, uid, ctx)).fetchone()
    if not row:
        db.close()
        return {"status": "not_found", "memory_id": memory_id}
    emb = _embed(summary + " " + raw_text[:500])
    emb_blob = _vec_to_blob(emb) if emb else None
    db.execute(
        """UPDATE memories SET raw_text=?, summary=?, entities=?, topics=?, importance=?, embedding=?
           WHERE id=?""",
        (raw_text, summary, json.dumps(entities), json.dumps(topics), importance, emb_blob, memory_id),
    )
    db.commit()
    db.close()
    log.info(f"Updated memory #{memory_id}")
    return {"status": "updated", "memory_id": memory_id}


def search_memories(query: str, user_id: str = "", context: str = "", limit: int = 20) -> dict:
    """Retrieve memories by semantic similarity. Gap: Retrieval — embedding-based search."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    q_emb = _embed(query)
    db = get_db()
    rows = db.execute(
        "SELECT id, source, summary, entities, topics, importance, connections, created_at, embedding FROM memories WHERE user_id=? AND context=?",
        (uid, ctx),
    ).fetchall()
    db.close()
    memories = []
    for r in rows:
        mem = {
            "id": r["id"], "source": r["source"], "summary": r["summary"],
            "entities": json.loads(r["entities"]), "topics": json.loads(r["topics"]),
            "importance": r["importance"], "connections": json.loads(r["connections"]),
            "created_at": r["created_at"],
        }
        if q_emb:
            v = _blob_to_vec(r["embedding"])
            mem["_score"] = _cosine_sim(q_emb, v) if v else 0.0
        else:
            mem["_score"] = 1.0
        memories.append(mem)
    memories.sort(key=lambda m: m.pop("_score", 0), reverse=True)
    top = memories[:limit]
    return {"memories": top, "count": len(top)}


def read_all_memories(user_id: str = "", context: str = "", limit: int = 50) -> dict:
    """Read memories by recency. Gap: Namespace — filter by user_id, context."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    db = get_db()
    rows = db.execute(
        "SELECT * FROM memories WHERE user_id=? AND context=? ORDER BY created_at DESC LIMIT ?",
        (uid, ctx, limit),
    ).fetchall()
    memories = []
    for r in rows:
        memories.append({
            "id": r["id"], "source": r["source"], "summary": r["summary"],
            "entities": json.loads(r["entities"]), "topics": json.loads(r["topics"]),
            "importance": r["importance"], "connections": json.loads(r["connections"]),
            "created_at": r["created_at"], "consolidated": bool(r["consolidated"]),
        })
    db.close()
    return {"memories": memories, "count": len(memories)}


def read_unconsolidated_memories(user_id: str = "", context: str = "", limit: int = 10) -> dict:
    """Read unconsolidated memories. Gap: Namespace."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    db = get_db()
    rows = db.execute(
        "SELECT * FROM memories WHERE user_id=? AND context=? AND consolidated=0 ORDER BY created_at DESC LIMIT ?",
        (uid, ctx, limit),
    ).fetchall()
    memories = []
    for r in rows:
        memories.append({
            "id": r["id"], "summary": r["summary"],
            "entities": json.loads(r["entities"]), "topics": json.loads(r["topics"]),
            "importance": r["importance"], "created_at": r["created_at"],
        })
    db.close()
    return {"memories": memories, "count": len(memories)}


def store_consolidation(
    source_ids: list[int],
    summary: str,
    insight: str,
    connections: list[dict],
    user_id: str = "",
    context: str = "",
) -> dict:
    """Store consolidation. Gap: Namespace."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO consolidations (user_id, context, source_ids, summary, insight, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (uid, ctx, json.dumps(source_ids), summary, insight, now),
    )
    for conn in connections:
        from_id, to_id = conn.get("from_id"), conn.get("to_id")
        rel = conn.get("relationship", "")
        if from_id and to_id:
            for mid in [from_id, to_id]:
                row = db.execute("SELECT connections FROM memories WHERE id = ? AND user_id = ? AND context = ?", (mid, uid, ctx)).fetchone()
                if row:
                    existing = json.loads(row["connections"])
                    existing.append({"linked_to": to_id if mid == from_id else from_id, "relationship": rel})
                    db.execute("UPDATE memories SET connections = ? WHERE id = ?", (json.dumps(existing), mid))
    placeholders = ",".join("?" * len(source_ids))
    db.execute(f"UPDATE memories SET consolidated = 1 WHERE id IN ({placeholders}) AND user_id=? AND context=?", source_ids + [uid, ctx])
    db.commit()
    db.close()
    log.info(f"Consolidated {len(source_ids)} memories.")
    return {"status": "consolidated", "memories_processed": len(source_ids), "insight": insight}


def read_consolidation_history(user_id: str = "", context: str = "", limit: int = 10) -> dict:
    """Read consolidation history. Gap: Namespace."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    db = get_db()
    rows = db.execute(
        "SELECT * FROM consolidations WHERE user_id=? AND context=? ORDER BY created_at DESC LIMIT ?",
        (uid, ctx, limit),
    ).fetchall()
    result = [{"summary": r["summary"], "insight": r["insight"], "source_ids": r["source_ids"]} for r in rows]
    db.close()
    return {"consolidations": result, "count": len(result)}


# ─── Profile Tools (Gap: Profile) ──────────────────────────────


def get_profile(user_id: str = "", context: str = "") -> dict:
    """Get merged profile for user/context. Gap: Profile — unified context."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    db = get_db()
    row = db.execute("SELECT profile_json FROM profiles WHERE user_id=? AND context=?", (uid, ctx)).fetchone()
    db.close()
    if not row:
        return {"profile": {}}
    try:
        return {"profile": json.loads(row["profile_json"])}
    except Exception:
        return {"profile": {}}


def update_profile(user_id: str = "", context: str = "", profile_json: str = "{}") -> dict:
    """Upsert merged profile. Gap: Profile — merge new facts into existing."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    prof = profile_json or {}
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO profiles (user_id, context, profile_json, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(user_id, context) DO UPDATE SET profile_json=?, updated_at=?",
        (uid, ctx, json.dumps(prof), now, json.dumps(prof), now),
    )
    db.commit()
    db.close()
    log.info(f"Updated profile for {uid}/{ctx}")
    return {"status": "updated", "profile": prof}


# ─── Episodic Tools (Gap: Episodic) ─────────────────────────────


def store_episode(
    input_text: str,
    output_text: str,
    outcome: str = "",
    user_id: str = "",
    context: str = "",
) -> dict:
    """Store input/output/outcome as episodic memory. Gap: Episodic — few-shot examples."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    emb = _embed(input_text + " " + output_text)
    emb_blob = _vec_to_blob(emb) if emb else None
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    cursor = db.execute(
        "INSERT INTO episodic_memories (user_id, context, input_text, output_text, outcome, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (uid, ctx, input_text, output_text, outcome, emb_blob, now),
    )
    eid = cursor.lastrowid
    db.commit()
    db.close()
    return {"episode_id": eid, "status": "stored"}


def search_episodes(query: str, user_id: str = "", context: str = "", limit: int = 5) -> dict:
    """Retrieve relevant episodic memories. Gap: Episodic — few-shot retrieval."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    q_emb = _embed(query)
    db = get_db()
    rows = db.execute(
        "SELECT id, input_text, output_text, outcome, embedding, created_at FROM episodic_memories WHERE user_id=? AND context=?",
        (uid, ctx),
    ).fetchall()
    db.close()
    episodes = []
    for r in rows:
        ep = {"id": r["id"], "input": r["input_text"], "output": r["output_text"], "outcome": r["outcome"], "created_at": r["created_at"]}
        if q_emb:
            v = _blob_to_vec(r["embedding"])
            ep["_score"] = _cosine_sim(q_emb, v) if v else 0.0
        else:
            ep["_score"] = 1.0
        episodes.append(ep)
    episodes.sort(key=lambda e: e.pop("_score", 0), reverse=True)
    return {"episodes": episodes[:limit], "count": len(episodes[:limit])}


# ─── Procedural Tools (Gap: Procedural) ─────────────────────────


def get_instructions(user_id: str = "", key: str = "default") -> dict:
    """Get stored agent instructions. Gap: Procedural — rules/prompts."""
    uid = user_id or current_user_id.get()
    db = get_db()
    row = db.execute("SELECT instructions_json FROM agent_instructions WHERE user_id=? AND key=?", (uid, key)).fetchone()
    db.close()
    if not row:
        return {"instructions": ""}
    val = row["instructions_json"]
    try:
        parsed = json.loads(val)
        return {"instructions": parsed if isinstance(parsed, str) else json.dumps(parsed)}
    except Exception:
        return {"instructions": val}


def update_instructions(user_id: str = "", key: str = "default", instructions: str = "") -> dict:
    """Store refined instructions. Gap: Procedural — reflection output."""
    uid = user_id or current_user_id.get()
    instr = json.dumps(instructions) if isinstance(instructions, dict) else str(instructions)
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO agent_instructions (user_id, key, instructions_json, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(user_id, key) DO UPDATE SET instructions_json=?, updated_at=?",
        (uid, key, instr, now, instr, now),
    )
    db.commit()
    db.close()
    log.info(f"Updated instructions for {uid}/{key}")
    return {"status": "updated"}


# ─── Session Tools (Gap: Short-term) ────────────────────────────


def append_session_message(session_id: str, user_id: str, role: str, content: str) -> dict:
    """Append message; trim to sliding window. Gap: Short-term — session history."""
    uid = user_id or current_user_id.get()
    db = get_db()
    row = db.execute("SELECT messages_json FROM sessions WHERE session_id=? AND user_id=?", (session_id, uid)).fetchone()
    messages = json.loads(row["messages_json"]) if row else []
    messages.append({"role": role, "content": content})
    messages = messages[-SESSION_WINDOW:]
    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO sessions (session_id, user_id, messages_json, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(session_id, user_id) DO UPDATE SET messages_json=?, updated_at=?",
        (session_id, uid, json.dumps(messages), now, json.dumps(messages), now),
    )
    db.commit()
    db.close()
    return {"status": "appended", "count": len(messages)}


def get_session_messages(session_id: str, user_id: str = "", limit: int = 10) -> dict:
    """Get last N session messages. Gap: Short-term — sliding window retrieval."""
    uid = user_id or current_user_id.get()
    db = get_db()
    row = db.execute("SELECT messages_json FROM sessions WHERE session_id=? AND user_id=?", (session_id, uid)).fetchone()
    db.close()
    messages = json.loads(row["messages_json"]) if row else []
    return {"messages": messages[-limit:], "count": len(messages[-limit:])}


# ─── Shared Tools ───────────────────────────────────────────────


def get_memory_stats(user_id: str = "", context: str = "") -> dict:
    """Memory stats. Gap: Namespace."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    db = get_db()
    total = db.execute("SELECT COUNT(*) as c FROM memories WHERE user_id=? AND context=?", (uid, ctx)).fetchone()["c"]
    unconsolidated = db.execute("SELECT COUNT(*) as c FROM memories WHERE user_id=? AND context=? AND consolidated=0", (uid, ctx)).fetchone()["c"]
    consolidations = db.execute("SELECT COUNT(*) as c FROM consolidations WHERE user_id=? AND context=?", (uid, ctx)).fetchone()["c"]
    db.close()
    return {"total_memories": total, "unconsolidated": unconsolidated, "consolidations": consolidations}


def delete_memory(memory_id: int, user_id: str = "", context: str = "") -> dict:
    """Delete memory. Gap: Namespace."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    db = get_db()
    row = db.execute("SELECT 1 FROM memories WHERE id=? AND user_id=? AND context=?", (memory_id, uid, ctx)).fetchone()
    if not row:
        db.close()
        return {"status": "not_found", "memory_id": memory_id}
    db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
    db.commit()
    db.close()
    return {"status": "deleted", "memory_id": memory_id}


def clear_all_memories(inbox_path: str | None = None, user_id: str = "", context: str = "") -> dict:
    """Clear memories. Gap: Namespace — scoped by user_id and context."""
    uid = user_id or current_user_id.get()
    ctx = context if context is not None else current_context.get()
    db = get_db()
    mem_count = db.execute("SELECT COUNT(*) as c FROM memories WHERE user_id=? AND context=?", (uid, ctx)).fetchone()["c"]
    db.execute("DELETE FROM memories WHERE user_id=? AND context=?", (uid, ctx))
    db.execute("DELETE FROM consolidations WHERE user_id=? AND context=?", (uid, ctx))
    db.execute("DELETE FROM processed_files")
    db.commit()
    db.close()
    files_deleted = 0
    if inbox_path:
        folder = Path(inbox_path)
        if folder.is_dir():
            for f in folder.iterdir():
                if f.name.startswith("."):
                    continue
                try:
                    if f.is_file():
                        f.unlink()
                        files_deleted += 1
                    elif f.is_dir():
                        shutil.rmtree(f)
                        files_deleted += 1
                except OSError as e:
                    log.error(f"Failed to delete {f.name}: {e}")
    return {"status": "cleared", "memories_deleted": mem_count, "files_deleted": files_deleted}


# ─── ADK Agents ────────────────────────────────────────────────


def build_agents():
    # Gap: Update, Profile — ingest can update and manage profile
    ingest_agent = Agent(
        name="ingest_agent",
        model=MODEL,
        description="Processes raw text or media into structured memory. Call when new information arrives. May update existing memories or profile.",
        instruction=(
            "You are a Memory Ingest Agent. For any input:\n"
            "1. Optionally call search_memories to find overlapping memories\n"
            "2. If new info contradicts or refines existing memory, call update_memory; else call store_memory\n"
            "3. For user-specific facts/preferences, call update_profile to merge into profile\n"
            "4. Extract: summary, entities, topics, importance (0-1)\n"
            "5. Always call at least store_memory or update_memory. Confirm in one sentence."
        ),
        tools=[store_memory, update_memory, search_memories, get_profile, update_profile],
    )

    # Gap: Namespace — consolidate scoped by user_id, context
    consolidate_agent = Agent(
        name="consolidate_agent",
        model=MODEL,
        description="Merges related memories and finds patterns. Call periodically.",
        instruction=(
            "Call read_unconsolidated_memories. If fewer than 2, say nothing to consolidate.\n"
            "Find connections, create synthesized summary and insight. Call store_consolidation with source_ids, summary, insight, connections."
        ),
        tools=[read_unconsolidated_memories, store_consolidation],
    )

    # Gap: Retrieval, Profile, Episodic — query uses search, profile, episodes
    query_agent = Agent(
        name="query_agent",
        model=MODEL,
        description="Answers questions using stored memories.",
        instruction=(
            "1. Call search_memories with the question to get relevant memories\n"
            "2. Call get_profile for user context\n"
            "3. Call search_episodes for relevant past examples (few-shot)\n"
            "4. Call read_consolidation_history for insights\n"
            "5. Synthesize answer from memories, profile, episodes. Cite sources. If no relevant info, say so."
        ),
        tools=[search_memories, get_profile, search_episodes, read_consolidation_history],
    )

    # Gap: Procedural — reflection refines instructions
    reflection_agent = Agent(
        name="reflection_agent",
        model=MODEL,
        description="Refines agent instructions based on feedback.",
        instruction=(
            "You refine the agent's instructions. Given current instructions and user feedback:\n"
            "1. Call get_instructions to get current rules\n"
            "2. Produce refined instructions that incorporate feedback\n"
            "3. Call update_instructions with the new instructions\n"
            "Confirm the refinement in one sentence."
        ),
        tools=[get_instructions, update_instructions],
    )

    # Gap: Episodic — store feedback as episode
    episodic_agent = Agent(
        name="episodic_agent",
        model=MODEL,
        description="Stores successful interactions as episodic memory for few-shot learning.",
        instruction=(
            "When user provides feedback (e.g. 'that worked', 'good response'):\n"
            "1. Extract the input (user request) and output (agent response) from context\n"
            "2. Call store_episode with input_text, output_text, outcome (e.g. 'success')\n"
            "Confirm storage in one sentence."
        ),
        tools=[store_episode],
    )

    orchestrator = Agent(
        name="adaptive_memory_orchestrator",
        model=MODEL,
        description="Routes to ingest, consolidate, query, reflection, or episodic agent.",
        instruction=(
            "Route requests:\n"
            "- New information -> ingest_agent\n"
            "- Consolidation -> consolidate_agent\n"
            "- Questions -> query_agent\n"
            "- Reflect/refine behavior -> reflection_agent\n"
            "- Feedback (worked/didn't work) -> episodic_agent\n"
            "- Status -> get_memory_stats\n"
            "Pass user_id and context from request to tools. Give brief summary after sub-agent completes."
        ),
        sub_agents=[ingest_agent, consolidate_agent, query_agent, reflection_agent, episodic_agent],
        tools=[get_memory_stats],
    )
    return orchestrator


# ─── Agent Runner ──────────────────────────────────────────────


class AdaptiveMemoryAgent:
    def __init__(self):
        self.agent = build_agents()
        self.session_service = InMemorySessionService()
        self.runner = Runner(agent=self.agent, app_name="adaptive_memory", session_service=self.session_service)

    async def run(self, message: str, user_id: str = "default", context: str = "") -> str:
        tok_uid = current_user_id.set(user_id)
        tok_ctx = current_context.set(context)
        try:
            session = await self.session_service.create_session(app_name="adaptive_memory", user_id=user_id)
            content = types.Content(role="user", parts=[types.Part.from_text(text=message)])
            response = ""
            async for event in self.runner.run_async(user_id=user_id, session_id=session.id, new_message=content):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            response += part.text
            return response
        finally:
            current_user_id.reset(tok_uid)
            current_context.reset(tok_ctx)

    async def run_multimodal(self, text: str, file_bytes: bytes, mime_type: str, user_id: str = "default", context: str = "") -> str:
        tok_uid = current_user_id.set(user_id)
        tok_ctx = current_context.set(context)
        try:
            session = await self.session_service.create_session(app_name="adaptive_memory", user_id=user_id)
            parts = [types.Part.from_text(text=text), types.Part.from_bytes(data=file_bytes, mime_type=mime_type)]
            content = types.Content(role="user", parts=parts)
            response = ""
            async for event in self.runner.run_async(user_id=user_id, session_id=session.id, new_message=content):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            response += part.text
            return response
        finally:
            current_user_id.reset(tok_uid)
            current_context.reset(tok_ctx)

    async def ingest(self, text: str, source: str = "", user_id: str = "default", context: str = "") -> str:
        msg = f"Remember this (source: {source}):\n\n{text}" if source else f"Remember this:\n\n{text}"
        return await self.run(msg, user_id=user_id, context=context)

    async def ingest_file(self, file_path: Path, user_id: str = "default", context: str = "") -> str:
        suffix = file_path.suffix.lower()
        mime_type = MEDIA_EXTENSIONS.get(suffix) or (mimetypes.guess_type(str(file_path))[0] or "application/octet-stream")
        file_bytes = file_path.read_bytes()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > 20:
            log.warning(f"Skipping {file_path.name} ({size_mb:.1f}MB)")
            return f"Skipped: file too large ({size_mb:.1f}MB)"
        prompt = f"Remember this file (source: {file_path.name}). Analyze and extract for memory storage."
        return await self.run_multimodal(prompt, file_bytes, mime_type, user_id=user_id, context=context)

    async def consolidate(self, user_id: str = "default", context: str = "") -> str:
        return await self.run("Consolidate unconsolidated memories.", user_id=user_id, context=context)

    async def query(self, question: str, user_id: str = "default", context: str = "") -> str:
        return await self.run(f"Based on my memories, answer: {question}", user_id=user_id, context=context)

    async def reflect(self, feedback: str, user_id: str = "default", context: str = "") -> str:
        return await self.run(f"Reflect and refine your instructions based on this feedback: {feedback}", user_id=user_id, context=context)

    async def store_feedback(self, input_text: str, output_text: str, outcome: str, user_id: str = "default", context: str = "") -> str:
        return await self.run(
            f"Store this as episodic memory. Input: {input_text[:200]}... Output: {output_text[:200]}... Outcome: {outcome}",
            user_id=user_id,
            context=context,
        )

    async def status(self, user_id: str = "default", context: str = "") -> str:
        return await self.run("Give a status report on my memory system.", user_id=user_id, context=context)


# ─── File Watcher ──────────────────────────────────────────────

async def watch_folder(agent: AdaptiveMemoryAgent, folder: Path, poll_interval: int = 5, user_id: str = "default", context: str = ""):
    folder.mkdir(parents=True, exist_ok=True)
    db = get_db()
    log.info(f"Watching: {folder}/")
    while True:
        try:
            for f in sorted(folder.iterdir()):
                if f.name.startswith(".") or f.suffix.lower() not in ALL_SUPPORTED:
                    continue
                if db.execute("SELECT 1 FROM processed_files WHERE path=?", (str(f),)).fetchone():
                    continue
                try:
                    if f.suffix.lower() in TEXT_EXTENSIONS:
                        text = f.read_text(encoding="utf-8", errors="replace")[:10000]
                        if text.strip():
                            await agent.ingest(text, source=f.name, user_id=user_id, context=context)
                    else:
                        await agent.ingest_file(f, user_id=user_id, context=context)
                except Exception as e:
                    log.error(f"Error ingesting {f.name}: {e}")
                db.execute("INSERT OR REPLACE INTO processed_files (path, processed_at) VALUES (?, ?)", (str(f), datetime.now(timezone.utc).isoformat()))
                db.commit()
        except Exception as e:
            log.error(f"Watch error: {e}")
        await asyncio.sleep(poll_interval)


# ─── Consolidation Loop ────────────────────────────────────────


async def consolidation_loop(agent: AdaptiveMemoryAgent, interval_minutes: int = 30):
    log.info(f"Consolidation every {interval_minutes} minutes")
    while True:
        await asyncio.sleep(interval_minutes * 60)
        try:
            db = get_db()
            rows = db.execute("SELECT DISTINCT user_id, context FROM memories WHERE consolidated=0").fetchall()
            db.close()
            for r in rows:
                uid, ctx = r["user_id"], r["context"]
                count = len(read_unconsolidated_memories(user_id=uid, context=ctx)["memories"])
                if count >= 2:
                    log.info(f"Consolidating ({uid}/{ctx})...")
                    await agent.consolidate(user_id=uid, context=ctx)
        except Exception as e:
            log.error(f"Consolidation error: {e}")


# ─── HTTP API ──────────────────────────────────────────────────


def build_http(agent: AdaptiveMemoryAgent, watch_path: str = "./inbox"):
    app = web.Application()

    def _ns(request):
        return request.query.get("user_id", request.headers.get("X-User-Id", "default")), request.query.get("context", request.headers.get("X-Context", ""))

    async def handle_query(request):
        q = request.query.get("q", "").strip()
        if not q:
            return web.json_response({"error": "missing ?q="}, status=400)
        uid, ctx = _ns(request)
        answer = await agent.query(q, user_id=uid, context=ctx)
        return web.json_response({"question": q, "answer": answer})

    async def handle_ingest(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        text = data.get("text", "").strip()
        if not text:
            return web.json_response({"error": "missing 'text'"}, status=400)
        uid = data.get("user_id", "default")
        ctx = data.get("context", "")
        result = await agent.ingest(text, source=data.get("source", "api"), user_id=uid, context=ctx)
        return web.json_response({"status": "ingested", "response": result})

    async def handle_consolidate(request):
        uid, ctx = _ns(request)
        result = await agent.consolidate(user_id=uid, context=ctx)
        return web.json_response({"status": "done", "response": result})

    async def handle_reflect(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        feedback = data.get("feedback", "").strip()
        uid = data.get("user_id", "default")
        ctx = data.get("context", "")
        result = await agent.reflect(feedback, user_id=uid, context=ctx)
        return web.json_response({"status": "done", "response": result})

    async def handle_episode(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        uid = data.get("user_id", "default")
        ctx = data.get("context", "")
        store_episode(
            data.get("input", ""),
            data.get("output", ""),
            data.get("outcome", ""),
            user_id=uid,
            context=ctx,
        )
        return web.json_response({"status": "stored"})

    async def handle_profile(request):
        uid, ctx = _ns(request)
        return web.json_response(get_profile(user_id=uid, context=ctx))

    async def handle_status(request):
        uid, ctx = _ns(request)
        return web.json_response(get_memory_stats(user_id=uid, context=ctx))

    async def handle_memories(request):
        uid, ctx = _ns(request)
        return web.json_response(search_memories(request.query.get("q", ""), user_id=uid, context=ctx))

    async def handle_delete(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        uid = data.get("user_id", "default")
        ctx = data.get("context", "")
        result = delete_memory(int(data.get("memory_id", 0)), user_id=uid, context=ctx)
        return web.json_response(result)

    async def handle_clear(request):
        uid, ctx = _ns(request)
        result = clear_all_memories(inbox_path=watch_path, user_id=uid, context=ctx)
        return web.json_response(result)

    app.router.add_get("/query", handle_query)
    app.router.add_post("/ingest", handle_ingest)
    app.router.add_post("/consolidate", handle_consolidate)
    app.router.add_post("/reflect", handle_reflect)
    app.router.add_post("/episode", handle_episode)
    app.router.add_get("/profile", handle_profile)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/memories", handle_memories)
    app.router.add_post("/delete", handle_delete)
    app.router.add_post("/clear", handle_clear)
    return app


# ─── Main ──────────────────────────────────────────────────────


async def main_async(args):
    agent = AdaptiveMemoryAgent()
    log.info("Adaptive Memory Agent starting")
    log.info(f"   Model: {MODEL}")
    log.info(f"   DB: {DB_PATH}")
    log.info(f"   Port: {args.port}")
    tasks = [
        asyncio.create_task(watch_folder(agent, Path(args.watch))),
        asyncio.create_task(consolidation_loop(agent, args.consolidate_every)),
    ]
    app = build_http(agent, watch_path=args.watch)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", args.port)
    await site.start()
    log.info(f"Agent running: http://127.0.0.1:{args.port}")
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Adaptive Memory Agent")
    parser.add_argument("--watch", default="./inbox")
    parser.add_argument("--port", type=int, default=8889)
    parser.add_argument("--consolidate-every", type=int, default=30)
    args = parser.parse_args()
    loop = asyncio.new_event_loop()

    def shutdown(sig):
        log.info(f"Shutting down (signal {sig})...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown, sig)

    try:
        loop.run_until_complete(main_async(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        loop.close()
        log.info("Agent stopped.")


if __name__ == "__main__":
    main()
