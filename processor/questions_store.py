import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def get_db_path(base_dir: str) -> str:
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "questions.sqlite3")


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                snippet_hash TEXT,
                model TEXT,
                prompt_version TEXT,
                result_json TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS question_decisions (
                interview_id TEXT,
                question_id TEXT,
                question_text TEXT,
                start_time TEXT,
                end_time TEXT,
                status TEXT,
                confidence REAL,
                notes TEXT,
                edited INTEGER,
                last_model TEXT,
                last_prompt_version TEXT,
                updated_at TEXT,
                PRIMARY KEY (interview_id, question_id)
            );

            CREATE TABLE IF NOT EXISTS verify_usage_daily (
                usage_date TEXT PRIMARY KEY,
                call_count INTEGER
            );
            """
        )


def load_decisions(db_path: str, interview_id: str) -> Dict[str, Dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT question_id, question_text, status, confidence, notes, edited,
                   last_model, last_prompt_version
            FROM question_decisions
            WHERE interview_id = ?
            """,
            (interview_id,),
        ).fetchall()

    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        out[row["question_id"]] = {
            "question_text": row["question_text"],
            "status": row["status"],
            "confidence": row["confidence"],
            "notes": row["notes"] or "",
            "edited": bool(row["edited"]),
            "last_model": row["last_model"],
            "last_prompt_version": row["last_prompt_version"],
        }
    return out


def upsert_decision(
    db_path: str,
    interview_id: str,
    row: Dict[str, Any],
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO question_decisions (
                interview_id, question_id, question_text, start_time, end_time,
                status, confidence, notes, edited, last_model, last_prompt_version, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(interview_id, question_id) DO UPDATE SET
                question_text=excluded.question_text,
                start_time=excluded.start_time,
                end_time=excluded.end_time,
                status=excluded.status,
                confidence=excluded.confidence,
                notes=excluded.notes,
                edited=excluded.edited,
                last_model=excluded.last_model,
                last_prompt_version=excluded.last_prompt_version,
                updated_at=excluded.updated_at
            """,
            (
                interview_id,
                row.get("id"),
                row.get("question_text"),
                row.get("start_time"),
                row.get("end_time"),
                row.get("status"),
                float(row.get("confidence", 0.0)),
                row.get("notes", ""),
                1 if row.get("edited") else 0,
                row.get("verification", {}).get("last_model"),
                row.get("verification", {}).get("last_prompt_version"),
                utc_now_iso(),
            ),
        )


def get_cached_llm_result(db_path: str, cache_key: str) -> Optional[Dict[str, Any]]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT result_json FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["result_json"])
    except json.JSONDecodeError:
        return None


def set_cached_llm_result(
    db_path: str,
    cache_key: str,
    snippet_hash: str,
    model: str,
    prompt_version: str,
    result: Dict[str, Any],
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO llm_cache (cache_key, snippet_hash, model, prompt_version, result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                snippet_hash=excluded.snippet_hash,
                model=excluded.model,
                prompt_version=excluded.prompt_version,
                result_json=excluded.result_json,
                created_at=excluded.created_at
            """,
            (
                cache_key,
                snippet_hash,
                model,
                prompt_version,
                json.dumps(result, ensure_ascii=False),
                utc_now_iso(),
            ),
        )


def get_daily_usage(db_path: str, usage_date: Optional[str] = None) -> int:
    day = usage_date or utc_today()
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT call_count FROM verify_usage_daily WHERE usage_date = ?",
            (day,),
        ).fetchone()
    if not row:
        return 0
    return int(row["call_count"])


def increment_daily_usage(db_path: str, delta: int, usage_date: Optional[str] = None) -> int:
    day = usage_date or utc_today()
    current = get_daily_usage(db_path, day)
    updated = current + max(0, int(delta))
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO verify_usage_daily (usage_date, call_count)
            VALUES (?, ?)
            ON CONFLICT(usage_date) DO UPDATE SET call_count = excluded.call_count
            """,
            (day, updated),
        )
    return updated
