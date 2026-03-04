import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

LOW_THRESHOLD = 0.65
HIGH_THRESHOLD = 0.80
VALID_STATUSES = {"unreviewed", "verified", "rejected", "needs_review"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_interview_id(value: str) -> str:
    if not value:
        return ""
    base = os.path.splitext(os.path.basename(value))[0]
    base = re.sub(r"_interview(_transcript)?(_\d+)?$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\s+", "_", base.strip())
    base = re.sub(r"[^A-Za-z0-9_\-]", "", base)
    return base


def resolve_interview_id(
    explicit: Optional[str],
    srt_path: Optional[str],
    using_sample: bool = False,
) -> str:
    if explicit and explicit.strip():
        return to_interview_id(explicit)
    if using_sample:
        return "Amos_C_Brown"
    if srt_path:
        return to_interview_id(srt_path)
    return ""


def parse_time_to_seconds(value: str) -> Optional[float]:
    if not value:
        return None
    cleaned = value.strip().replace(",", ".")
    if " --> " in cleaned:
        cleaned = cleaned.split(" --> ")[0].strip()
    parts = cleaned.split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def format_seconds(value: float) -> str:
    safe = max(0.0, float(value))
    hours = int(safe // 3600)
    minutes = int((safe % 3600) // 60)
    seconds = int(safe % 60)
    millis = int(round((safe - int(safe)) * 1000))
    if millis == 1000:
        millis = 0
        seconds += 1
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def normalize_timestamp(value: str, fallback_seconds: Optional[float] = None) -> str:
    seconds = parse_time_to_seconds(value or "")
    if seconds is None:
        seconds = fallback_seconds if fallback_seconds is not None else 0.0
    return format_seconds(seconds)


def confidence_band(confidence: float) -> str:
    if confidence >= HIGH_THRESHOLD:
        return "high"
    if confidence >= LOW_THRESHOLD:
        return "medium"
    return "low"


def hash_text(value: str) -> str:
    return hashlib.sha1((value or "").encode("utf-8")).hexdigest()


def stable_question_id(interview_id: str, start_time: str, question_text: str) -> str:
    material = f"{interview_id}|{start_time}|{(question_text or '').strip().lower()}"
    return hash_text(material)[:16]


def _artifact_candidates(
    base_dir: str,
    interview_id: str,
) -> List[str]:
    parent = os.path.dirname(base_dir)
    candidates = [
        os.path.join(base_dir, "questions", f"questions_{interview_id}.json"),
        os.path.join(parent, "civil-rights-history-project", "factcheck", "data", "questions", f"questions_{interview_id}.json"),
    ]
    return candidates


def normalize_artifact_questions(interview_id: str, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        text = (item.get("questionText") or item.get("question_text") or "").strip()
        if not text:
            continue

        start_time = normalize_timestamp(str(item.get("startTime") or item.get("start_time") or "00:00:00,000"))
        end_time = normalize_timestamp(str(item.get("endTime") or item.get("end_time") or start_time))

        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        status = str(item.get("status") or "unreviewed").strip().lower()
        if status not in VALID_STATUSES:
            status = "unreviewed"

        qid = str(item.get("id") or "").strip() or stable_question_id(interview_id, start_time, text)

        band = confidence_band(conf)
        verification = item.get("verification") if isinstance(item.get("verification"), dict) else {}

        row = {
            "id": qid,
            "question_text": text,
            "start_time": start_time,
            "end_time": end_time,
            "confidence": round(conf, 3),
            "confidence_band": band,
            "status": status,
            "is_low_confidence": bool(item.get("isLowConfidence", band == "low")),
            "source": item.get("source"),
            "speaker_source": item.get("speakerSource"),
            "flags": item.get("flags") if isinstance(item.get("flags"), list) else [],
            "notes": str(item.get("notes") or ""),
            "edited": bool(item.get("edited", False)),
            "verification": {
                "last_method": verification.get("last_method") or "artifact",
                "last_model": verification.get("last_model"),
                "last_prompt_version": verification.get("last_prompt_version"),
                "last_checked_at": verification.get("last_checked_at"),
                "reason_code": verification.get("reason_code"),
            },
        }
        rows.append(row)

    rows.sort(key=lambda row: parse_time_to_seconds(row["start_time"]) or 0.0)
    return rows


def load_questions_artifact(
    base_dir: str,
    interview_id: str,
    explicit_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    paths: List[str] = []
    if explicit_path:
        paths.append(os.path.abspath(explicit_path))
    paths.extend(_artifact_candidates(base_dir, interview_id))

    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            return [], path, f"Failed to read questions artifact: {exc}"

        questions = payload.get("questions")
        if not isinstance(questions, list):
            return [], path, "Invalid artifact format: missing questions[]"

        artifact_interview_id = payload.get("interviewId") or interview_id
        normalized = normalize_artifact_questions(str(artifact_interview_id), questions)
        return normalized, path, None

    return [], None, f"No questions artifact found for interviewId '{interview_id}'"


def apply_saved_decisions(
    rows: List[Dict[str, Any]],
    decisions_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not decisions_by_id:
        return rows

    updated: List[Dict[str, Any]] = []
    for row in rows:
        decision = decisions_by_id.get(row["id"])
        if decision:
            row["question_text"] = decision.get("question_text", row["question_text"])
            row["status"] = decision.get("status", row["status"])
            row["notes"] = decision.get("notes", row["notes"])
            row["edited"] = bool(decision.get("edited", row["edited"]))
            conf = decision.get("confidence")
            if conf is not None:
                try:
                    row["confidence"] = round(max(0.0, min(1.0, float(conf))), 3)
                except (TypeError, ValueError):
                    pass
            row["confidence_band"] = confidence_band(float(row["confidence"]))
            row["is_low_confidence"] = row["confidence_band"] == "low"
            row.setdefault("verification", {})
            row["verification"]["last_model"] = decision.get("last_model")
            row["verification"]["last_prompt_version"] = decision.get("last_prompt_version")
        updated.append(row)
    return updated


def normalize_rows_from_ui(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            conf = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        status = str(row.get("status") or "unreviewed").strip().lower()
        if status not in VALID_STATUSES:
            status = "unreviewed"

        entry = {
            "id": str(row.get("id") or "").strip(),
            "question_text": str(row.get("question_text") or "").strip(),
            "start_time": normalize_timestamp(str(row.get("start_time") or "00:00:00,000")),
            "end_time": normalize_timestamp(str(row.get("end_time") or row.get("start_time") or "00:00:00,000")),
            "confidence": round(conf, 3),
            "confidence_band": confidence_band(conf),
            "status": status,
            "is_low_confidence": confidence_band(conf) == "low",
            "source": row.get("source"),
            "speaker_source": row.get("speaker_source"),
            "flags": row.get("flags") if isinstance(row.get("flags"), list) else [],
            "notes": str(row.get("notes") or ""),
            "edited": bool(row.get("edited", False)),
            "verification": row.get("verification") if isinstance(row.get("verification"), dict) else {
                "last_method": "artifact",
                "last_model": None,
                "last_prompt_version": None,
                "last_checked_at": None,
                "reason_code": None,
            },
        }
        if entry["id"] and entry["question_text"]:
            normalized.append(entry)

    normalized.sort(key=lambda r: parse_time_to_seconds(r["start_time"]) or 0.0)
    return normalized


def extract_snippet_by_timestamp(
    segments: List[Any],
    start_time: str,
    window_segments: int = 2,
) -> Dict[str, Any]:
    if not segments:
        return {
            "snippet_text": "",
            "start_idx": None,
            "end_idx": None,
            "start_time": "",
            "end_time": "",
            "segment_count": 0,
        }

    target = parse_time_to_seconds(start_time) or 0.0
    chosen_idx = 0
    smallest_gap = float("inf")

    for idx, seg in enumerate(segments):
        seg_start = getattr(seg, "get_start_seconds", lambda: 0.0)()
        seg_end = getattr(seg, "get_end_seconds", lambda: seg_start)()
        if seg_start <= target <= seg_end:
            chosen_idx = idx
            break
        gap = min(abs(target - seg_start), abs(target - seg_end))
        if gap < smallest_gap:
            smallest_gap = gap
            chosen_idx = idx

    start_idx = max(0, chosen_idx - window_segments)
    end_idx = min(len(segments) - 1, chosen_idx + window_segments)
    window = segments[start_idx:end_idx + 1]

    snippet_text = " ".join([(getattr(seg, "text", "") or "").strip() for seg in window]).strip()
    return {
        "snippet_text": snippet_text,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_time": getattr(window[0], "start_time", "") if window else "",
        "end_time": getattr(window[-1], "end_time", "") if window else "",
        "segment_count": len(window),
    }


def compute_question_stats(rows: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    rows = rows or []
    by_status = {"unreviewed": 0, "verified": 0, "rejected": 0, "needs_review": 0}
    by_band = {"high": 0, "medium": 0, "low": 0}

    for row in rows:
        status = row.get("status", "unreviewed")
        band = row.get("confidence_band", "low")
        if status in by_status:
            by_status[status] += 1
        if band in by_band:
            by_band[band] += 1

    return {
        "total": len(rows),
        "status": by_status,
        "confidence": by_band,
        "low_confidence_count": by_band["low"],
        "updated_at": utc_now_iso(),
    }
