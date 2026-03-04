import json
from typing import Any, Dict


def verify_question(
    ctx: Any,
    question_text: str,
    snippet_text: str,
    model: str = "gpt-4o-mini",
    prompt_version: str = "question_verify_v1",
) -> Dict[str, Any]:
    system_prompt = (
        "You are verifying if an utterance is a genuine interviewer question in an oral-history transcript. "
        "Return ONLY strict JSON with keys: is_question(boolean), confidence(number 0..1), reason_code(string), "
        "status_suggestion(string: verified|rejected|needs_review)."
    )

    user_prompt = (
        f"Question candidate: {question_text}\n\n"
        f"Transcript snippet around timestamp:\n{snippet_text}\n\n"
        "Rules:\n"
        "- verified: clear information-seeking question from interviewer\n"
        "- rejected: not a question or obvious filler/tag\n"
        "- needs_review: ambiguous/rhetorical/insufficient context"
    )

    response = ctx.client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=160,
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)

    is_question = bool(payload.get("is_question", False))
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason_code = str(payload.get("reason_code") or "unspecified")
    suggestion = str(payload.get("status_suggestion") or "needs_review").strip().lower()
    if suggestion not in {"verified", "rejected", "needs_review"}:
        if is_question and confidence >= 0.75:
            suggestion = "verified"
        elif not is_question and confidence >= 0.75:
            suggestion = "rejected"
        else:
            suggestion = "needs_review"

    return {
        "is_question": is_question,
        "confidence": round(confidence, 3),
        "reason_code": reason_code,
        "status_suggestion": suggestion,
        "model": model,
        "prompt_version": prompt_version,
    }
