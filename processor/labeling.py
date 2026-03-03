"""
Step 2 — Labeling: Assign topic categories and subtopics to each text block.
"""

import re
from typing import List, Dict, Any, Optional
from .shared import (
    ProcessorContext, MAIN_TOPICS,
    call_openai_json, load_prompt
)


def label_text_blocks(
    ctx: ProcessorContext,
    text_blocks: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Send text blocks to GPT and get topic labels per block.

    If system_prompt / user_prompt are provided, they override the defaults.
    The user_prompt should contain {analysis_text} and the system_prompt
    should contain {topics_list} as placeholders.
    """
    if not text_blocks:
        return []

    topics_list = '", "'.join(MAIN_TOPICS)

    # ── Build the analysis text from blocks ────────────────────────
    max_total_chars = 14000
    n = len(text_blocks)

    def _header(i: int, block: Dict[str, Any]) -> str:
        return f"\n\n--- B{i + 1} ({block['start_time']} - {block['end_time']}) ---\n"

    headers = [_header(i, b) for i, b in enumerate(text_blocks)]
    header_total = sum(len(h) for h in headers)
    remaining = max_total_chars - header_total

    min_per_block = 80
    max_per_block = 260

    if remaining <= n * min_per_block:
        per_block = min_per_block
    else:
        per_block = min(max_per_block, remaining // n)

    print(f"toc prompt budget: max_total_chars={max_total_chars}, blocks={n}, per_block={per_block}")

    def _trim_text(text: str, limit: int) -> str:
        text = (text or "").strip()
        if limit <= 0 or len(text) <= limit:
            return text
        clipped = text[:limit].rsplit(' ', 1)[0].strip()
        return (clipped or text[:limit].strip()) + " …"

    parts = []
    for i, block in enumerate(text_blocks):
        txt = _trim_text(block.get("text") or "", per_block)
        parts.append(headers[i] + txt)

    analysis_text = "".join(parts)

    # ── Resolve prompts ────────────────────────────────────────────
    if system_prompt is None:
        system_prompt = load_prompt(ctx, 'label_text_blocks_for_toc_system.txt')
    system_prompt = system_prompt.replace('{topics_list}', topics_list)

    if user_prompt is None:
        user_prompt = load_prompt(ctx, 'label_text_blocks_for_toc_user.txt')
    user_prompt = user_prompt.replace('{analysis_text}', analysis_text)

    # ── Call OpenAI ────────────────────────────────────────────────
    toc_max_tokens = min(9000, 800 + 140 * len(text_blocks))

    resp = call_openai_json(
        ctx, system_prompt, user_prompt,
        model=ctx.toc_model, max_tokens=toc_max_tokens
    )

    if isinstance(resp, dict) and "error" in resp:
        raise RuntimeError(f"TOC labeling OpenAI error: {resp['error']}")

    # ── Validate response ──────────────────────────────────────────
    blocks = resp.get("blocks", []) if isinstance(resp, dict) else []
    expected = len(text_blocks)
    got = len(blocks)

    if got < expected:
        returned = set()
        for it in blocks:
            bn = it.get("block_number")
            if isinstance(bn, int):
                returned.add(bn)
            elif isinstance(bn, str):
                m = re.search(r"\d+", bn)
                if m:
                    returned.add(int(m.group()))
        missing = [i for i in range(1, expected + 1) if i not in returned]
        raise RuntimeError(
            f"TOC labeling incomplete: got {got}/{expected}. "
            f"missing blocks: {missing[:12]}{'...' if len(missing) > 12 else ''}"
        )

    # ── Normalize output ───────────────────────────────────────────
    out = []
    for item in blocks:
        if not isinstance(item, dict):
            continue

        bn = item.get("block_number")
        if isinstance(bn, str):
            m = re.search(r"\d+", bn)
            bn = int(m.group()) if m else None
        if not isinstance(bn, int) or bn < 1 or bn > len(text_blocks):
            continue

        cat = item.get("main_topic_category", "")
        if cat not in MAIN_TOPICS:
            cat = MAIN_TOPICS[0]

        subs = item.get("subtopics", [])
        if not isinstance(subs, list):
            subs = []
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()][:5]
        if len(subs) < 3:
            subs += ["misc"] * (3 - len(subs))

        try:
            conf = float(item.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        out.append({
            "block_number": bn,
            "main_topic_category": cat,
            "subtopics": subs,
            "confidence": conf
        })

    by_bn = {x["block_number"]: x for x in out}

    if len(by_bn) == 0:
        raise RuntimeError("TOC labeling returned 0 valid blocks (model output likely invalid).")

    # Fill missing blocks with fallback
    filled = []
    for i in range(1, len(text_blocks) + 1):
        if i in by_bn:
            filled.append(by_bn[i])
        else:
            filled.append({
                "block_number": i,
                "main_topic_category": MAIN_TOPICS[0],
                "subtopics": [],
                "confidence": 0.0
            })

    return filled
