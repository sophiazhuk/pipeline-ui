"""
Civil Rights History Project - Demo UI
Flask app that breaks the interview processing pipeline into
individually controllable steps.
"""

import json
import os
import shutil
from io import BytesIO
from threading import Lock
from uuid import uuid4

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, session
from werkzeug.local import LocalProxy
from werkzeug.utils import secure_filename

# ── app setup ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY') or os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── pipeline state (per browser session, resets on restart) ───────────
# Each visitor gets an isolated in-memory pipeline state.
_STATE_LOCK = Lock()
_SESSION_STATES = {}


def _new_state():
    return {
        # step 1 - upload / blocking
        "api_key": None,
        "using_sample": False,
        "srt_path": None,
        "block_size": 23,
        "segments": None,            # List[SRTSegment]
        "plaintext_transcript": None,
        "text_blocks": None,         # List[Dict]

        # step 2 - labeling
        "labeling_sys_prompt": "",
        "labeling_user_prompt": "",
        "block_topics": None,        # List[Dict]

        # step 3 - toc
        "toc_bundle": None,          # {"toc": [...], "topic_index": {...}}

        # step 4 - chapterization
        "chapterization_sys_prompt": "",
        "chapterization_user_prompt": "",
        "chapter_breaks": None,      # List[Tuple[int, int]]
        "chapter_breaks_preview": None,

        # step 5 - summarization
        "main_summary_sys_prompt": "",
        "main_summary_user_prompt": "",
        "chapter_sys_prompt": "",
        "chapter_user_prompt": "",
        "main_summary": None,        # Dict
        "chapters": None,            # List[Dict]

        # step 6 - tuning
        "eval_sys_prompt": "",
        "eval_user_prompt": "",
        "revision_sys_prompt": "",
        "revision_user_prompt": "",
        "quality_threshold": 80,
        "accuracy_threshold": 80,
        "max_retries": 3,
        "tuning_results": None,

        # processor instance
        "processor": None,
    }


def _get_session_id():
    sid = session.get('sid')
    if not sid:
        sid = uuid4().hex
        session['sid'] = sid
    return sid


def _get_state():
    sid = _get_session_id()
    with _STATE_LOCK:
        if sid not in _SESSION_STATES:
            _SESSION_STATES[sid] = _new_state()
    return _SESSION_STATES[sid]


state = LocalProxy(_get_state)


def current_api_key():
    return (state.get("api_key") or "").strip()


def has_api_key():
    """Return True when an API key is available for the current browser session."""
    return bool(current_api_key())


def mask_api_key(api_key):
    """Return a safe, partially masked preview of the current API key."""
    if not api_key:
        return ''
    if len(api_key) <= 8:
        return '•' * len(api_key)
    return f"{api_key[:4]}…{api_key[-4:]}"


def _session_upload_dir(reset=False):
    path = os.path.join(app.config['UPLOAD_FOLDER'], _get_session_id())
    if reset:
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def _render_upload(api_key_error=None):
    session_api_key = current_api_key()
    return render_template(
        'upload.html',
        state=state,
        api_key_present=bool(session_api_key),
        api_key_masked=mask_api_key(session_api_key),
        api_key_error=api_key_error,
    )


def _reset_downstream():
    """Reset all downstream state when a new file is uploaded or blocking re-runs."""
    state["using_sample"] = False
    state["block_topics"] = None
    state["toc_bundle"] = None
    state["chapter_breaks"] = None
    state["chapter_breaks_preview"] = None
    state["main_summary"] = None
    state["chapters"] = None
    state["tuning_results"] = None
    # Reset prompts so they reload from files
    state["labeling_sys_prompt"] = ""
    state["labeling_user_prompt"] = ""
    state["chapterization_sys_prompt"] = ""
    state["chapterization_user_prompt"] = ""
    state["main_summary_sys_prompt"] = ""
    state["main_summary_user_prompt"] = ""
    state["chapter_sys_prompt"] = ""
    state["chapter_user_prompt"] = ""
    state["eval_sys_prompt"] = ""
    state["eval_user_prompt"] = ""
    state["revision_sys_prompt"] = ""
    state["revision_user_prompt"] = ""
    # Reset processor so it reinits with the current API key and block size
    state["processor"] = None


def get_ctx():
    """Lazy-init the ProcessorContext so we only create it once per browser session."""
    if state["processor"] is None:
        from processor import ProcessorContext

        prompts_dir = _find_path('processor_prompts')
        facts_path = _find_path('civil_rights_facts.json')
        rubric_path = _find_path('StandardizedRubric_1.md')

        state["processor"] = ProcessorContext(
            api_key=current_api_key(),
            chapter_block_size=state["block_size"],
            prompts_dir=prompts_dir or 'processor_prompts',
            facts_path=facts_path or 'civil_rights_facts.json',
            rubric_path=rubric_path or 'StandardizedRubric_1.md',
        )
    return state["processor"]


def _find_path(name):
    """Search for a file/dir in the app dir and parent dir."""
    for base in [BASE_DIR, os.path.dirname(BASE_DIR)]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


def load_prompt_file(filename):
    """Load a prompt file from processor_prompts/."""
    for base in [BASE_DIR, os.path.dirname(BASE_DIR)]:
        path = os.path.join(base, 'processor_prompts', filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
    return f"[prompt file not found: {filename}]"


# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — UPLOAD / BLOCKING
# ══════════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def upload_page():
    return _render_upload()


@app.route('/upload', methods=['POST'])
def upload_run():
    """Parse an uploaded SRT file or the bundled sample and build text blocks."""
    submitted_api_key = (request.form.get('api_key') or '').strip()
    if submitted_api_key and submitted_api_key != current_api_key():
        state["api_key"] = submitted_api_key
        state["processor"] = None
    elif not has_api_key():
        return _render_upload('Enter an API key before running the pipeline.')

    use_sample = request.form.get('use_sample') == 'on'
    uploaded_file = request.files.get('srt_file')
    if not use_sample and (not uploaded_file or not uploaded_file.filename):
        return _render_upload('Select an .srt file or use the bundled sample interview.')

    block_size = int(request.form.get('block_size', 23))

    # Reset all downstream state from previous runs
    _reset_downstream()
    session_dir = _session_upload_dir(reset=True)

    if use_sample:
        filepath = _find_path('interview.srt')
        if not filepath:
            return _render_upload('The bundled sample interview file was not found.')
        state["using_sample"] = True
    else:
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(session_dir, filename)
        uploaded_file.save(filepath)
        state["using_sample"] = False

    from srt_parser import parse_srt_file
    segments = parse_srt_file(filepath)
    plaintext = ' '.join([s.text for s in segments])

    # Update state
    state["srt_path"] = filepath
    state["block_size"] = block_size
    state["segments"] = segments
    state["plaintext_transcript"] = plaintext

    # Build text blocks
    ctx = get_ctx()
    ctx.chapter_block_size = block_size

    from processor.blocking import build_text_blocks
    text_blocks = build_text_blocks(ctx, segments, plaintext)
    state["text_blocks"] = text_blocks

    return redirect(url_for('blocking_output'))


@app.route('/blocking/output', methods=['GET'])
def blocking_output():
    return render_template('blocking_output.html', state=state)


# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — LABELING
# ══════════════════════════════════════════════════════════════════════

@app.route('/labeling', methods=['GET'])
def labeling_page():
    if not state["labeling_sys_prompt"]:
        state["labeling_sys_prompt"] = load_prompt_file('label_text_blocks_for_toc_system.txt')
    if not state["labeling_user_prompt"]:
        state["labeling_user_prompt"] = load_prompt_file('label_text_blocks_for_toc_user.txt')

    return render_template('labeling.html', state=state)


@app.route('/labeling/run', methods=['POST'])
def labeling_run():
    """Run labeling with user-edited prompts."""
    state["labeling_sys_prompt"] = request.form.get('sys_prompt', '')
    state["labeling_user_prompt"] = request.form.get('user_prompt', '')

    text_blocks = state["text_blocks"]
    if not text_blocks:
        return redirect(url_for('upload_page'))

    try:
        ctx = get_ctx()
        from processor.labeling import label_text_blocks
        block_topics = label_text_blocks(
            ctx, text_blocks,
            system_prompt=state["labeling_sys_prompt"],
            user_prompt=state["labeling_user_prompt"]
        )
    except Exception as e:
        state["block_topics"] = None
        return render_template(
            'labeling.html',
            state=state,
            labeling_error=(
                "Labeling failed. This app currently uses the OpenAI API endpoint, so non-OpenAI keys "
                "from the provider links will not work here unless the backend is adapted for that provider. "
                f"Details: {e}"
            )
        )

    state["block_topics"] = block_topics
    return render_template('labeling.html', state=state, just_ran=True)


@app.route('/labeling/update_output', methods=['POST'])
def labeling_update_output():
    """User manually edited the labeling output."""
    edited = request.form.get('edited_output', '')
    try:
        state["block_topics"] = json.loads(edited)
    except json.JSONDecodeError:
        pass  # keep old state if bad JSON
    return redirect(url_for('toc_page'))


# ══════════════════════════════════════════════════════════════════════
#  STEP 3 — TOC (pure logic, no API call)
# ══════════════════════════════════════════════════════════════════════

@app.route('/toc', methods=['GET'])
def toc_page():
    if state["text_blocks"] and state["block_topics"] and not state["toc_bundle"]:
        from processor.toc import build_hierarchical_toc
        toc_bundle = build_hierarchical_toc(state["text_blocks"], state["block_topics"])
        state["toc_bundle"] = toc_bundle

    return render_template('toc.html', state=state)


@app.route('/toc/update_output', methods=['POST'])
def toc_update_output():
    """User manually edited the TOC output."""
    edited = request.form.get('edited_output', '')
    try:
        state["toc_bundle"] = json.loads(edited)
    except json.JSONDecodeError:
        pass
    return redirect(url_for('chapterization_page'))


# ══════════════════════════════════════════════════════════════════════
#  STEP 4 — CHAPTERIZATION
# ══════════════════════════════════════════════════════════════════════

@app.route('/chapterization', methods=['GET'])
def chapterization_page():
    if not state["chapterization_sys_prompt"]:
        state["chapterization_sys_prompt"] = load_prompt_file('detect_topic_transitions_system.txt')
    if not state["chapterization_user_prompt"]:
        state["chapterization_user_prompt"] = load_prompt_file('detect_topic_transitions_user.txt')

    return render_template('chapterization.html', state=state)


@app.route('/chapterization/run', methods=['POST'])
def chapterization_run():
    """Run chapterization with user-edited prompts."""
    state["chapterization_sys_prompt"] = request.form.get('sys_prompt', '')
    state["chapterization_user_prompt"] = request.form.get('user_prompt', '')

    ctx = get_ctx()
    text_blocks = state["text_blocks"]
    block_topics = state["block_topics"]

    if not text_blocks:
        return redirect(url_for('upload_page'))

    from processor.chapterization import detect_topic_transitions, build_chapter_preview
    chapter_breaks = detect_topic_transitions(
        ctx, text_blocks, block_topics,
        system_prompt=state["chapterization_sys_prompt"],
        user_prompt=state["chapterization_user_prompt"]
    )
    state["chapter_breaks"] = chapter_breaks

    preview = build_chapter_preview(
        chapter_breaks, state["segments"], state["plaintext_transcript"]
    )
    state["chapter_breaks_preview"] = preview

    return render_template('chapterization.html', state=state, just_ran=True)


# ══════════════════════════════════════════════════════════════════════
#  STEP 5 — SUMMARIZATION
# ══════════════════════════════════════════════════════════════════════

@app.route('/summarization', methods=['GET'])
def summarization_page():
    if not state["main_summary_sys_prompt"]:
        state["main_summary_sys_prompt"] = load_prompt_file('generate_main_summary_system.txt')
    if not state["main_summary_user_prompt"]:
        state["main_summary_user_prompt"] = load_prompt_file('generate_main_summary_user.txt')
    if not state["chapter_sys_prompt"]:
        state["chapter_sys_prompt"] = load_prompt_file('generate_chapter_system.txt')
    if not state["chapter_user_prompt"]:
        state["chapter_user_prompt"] = load_prompt_file('generate_chapter_user.txt')

    return render_template('summarization.html', state=state)


@app.route('/summarization/run_main', methods=['POST'])
def summarization_run_main():
    """Generate main summary."""
    state["main_summary_sys_prompt"] = request.form.get('main_sys_prompt', '')
    state["main_summary_user_prompt"] = request.form.get('main_user_prompt', '')

    ctx = get_ctx()
    transcript = state["plaintext_transcript"]
    interview_name = os.path.basename(state["srt_path"] or "unknown")

    from processor.summarization import generate_main_summary
    main_summary = generate_main_summary(
        ctx, transcript, interview_name,
        system_prompt=state["main_summary_sys_prompt"],
        user_prompt=state["main_summary_user_prompt"]
    )
    state["main_summary"] = main_summary

    return render_template('summarization.html', state=state, ran_main=True)


@app.route('/summarization/run_chapters', methods=['POST'])
def summarization_run_chapters():
    """Generate chapter summaries."""
    state["chapter_sys_prompt"] = request.form.get('chapter_sys_prompt', '')
    state["chapter_user_prompt"] = request.form.get('chapter_user_prompt', '')

    ctx = get_ctx()
    segments = state["segments"]
    interview_name = os.path.basename(state["srt_path"] or "unknown")
    plaintext = state["plaintext_transcript"]
    chapter_breaks = state["chapter_breaks"]

    from processor.summarization import generate_chapters
    chapters = generate_chapters(
        ctx, segments, interview_name, plaintext, chapter_breaks,
        system_prompt=state["chapter_sys_prompt"],
        user_prompt=state["chapter_user_prompt"]
    )
    state["chapters"] = chapters

    return render_template('summarization.html', state=state, ran_chapters=True)


# ══════════════════════════════════════════════════════════════════════
#  STEP 6 — TUNING (scoring / regeneration)
# ══════════════════════════════════════════════════════════════════════

@app.route('/tuning', methods=['GET'])
def tuning_page():
    if not state["eval_sys_prompt"]:
        state["eval_sys_prompt"] = load_prompt_file('score_summary_system.txt')
    if not state["eval_user_prompt"]:
        state["eval_user_prompt"] = load_prompt_file('score_summary_user.txt')
    if not state["revision_sys_prompt"]:
        state["revision_sys_prompt"] = load_prompt_file('regenerate_main_summary_system.txt')
    if not state["revision_user_prompt"]:
        state["revision_user_prompt"] = load_prompt_file('regenerate_main_summary_user.txt')

    return render_template('tuning.html', state=state)


@app.route('/tuning/run', methods=['POST'])
def tuning_run():
    """Run scoring and regeneration loop with user-set thresholds."""
    state["quality_threshold"] = int(request.form.get('quality_threshold', 80))
    state["accuracy_threshold"] = int(request.form.get('accuracy_threshold', 80))
    state["max_retries"] = int(request.form.get('max_retries', 3))
    state["eval_sys_prompt"] = request.form.get('eval_sys_prompt', '')
    state["eval_user_prompt"] = request.form.get('eval_user_prompt', '')
    state["revision_sys_prompt"] = request.form.get('revision_sys_prompt', '')
    state["revision_user_prompt"] = request.form.get('revision_user_prompt', '')

    ctx = get_ctx()
    transcript = state["plaintext_transcript"]

    from processor.tuning import run_tuning_loop, score_chapter
    from processor.blocking import extract_plaintext_section

    tuning_results = {"main_summary": None, "chapters": []}

    # Score and regenerate main summary
    if state["main_summary"]:
        result = run_tuning_loop(
            ctx,
            summary=state["main_summary"],
            transcript=transcript,
            content_type="main_summary",
            quality_threshold=state["quality_threshold"],
            accuracy_threshold=state["accuracy_threshold"],
            max_retries=state["max_retries"],
            eval_sys_prompt=state["eval_sys_prompt"],
            eval_user_prompt=state["eval_user_prompt"],
            revision_sys_prompt=state["revision_sys_prompt"],
            revision_user_prompt=state["revision_user_prompt"],
        )
        tuning_results["main_summary"] = result
        state["main_summary"] = result["summary"]

    # Score chapters with actual chapter text
    if state["chapters"] and state["chapter_breaks"]:
        for i, chapter in enumerate(state["chapters"]):
            # Extract real chapter text from the transcript using break indices
            if i < len(state["chapter_breaks"]):
                start_idx, end_idx = state["chapter_breaks"][i]
                chapter_text = extract_plaintext_section(
                    state["plaintext_transcript"],
                    state["segments"],
                    start_idx,
                    end_idx
                )
            else:
                chapter_text = ""

            scores = score_chapter(ctx, chapter, chapter_text)
            tuning_results["chapters"].append({
                "chapter": chapter,
                "scores": scores
            })

    state["tuning_results"] = tuning_results
    return render_template('tuning.html', state=state, just_ran=True)


# ══════════════════════════════════════════════════════════════════════
#  RESULTS — final output + download
# ══════════════════════════════════════════════════════════════════════

@app.route('/results', methods=['GET'])
def results_page():
    return render_template('results.html', state=state)


@app.route('/results/download', methods=['GET'])
def results_download():
    """Download full results as JSON."""
    result = {
        "interview_name": os.path.basename(state["srt_path"] or "unknown"),
        "block_size": state["block_size"],
        "text_blocks": state["text_blocks"],
        "block_topics": state["block_topics"],
        "toc": state["toc_bundle"],
        "chapter_breaks": state["chapter_breaks"],
        "chapter_breaks_preview": state["chapter_breaks_preview"],
        "main_summary": state["main_summary"],
        "chapters": state["chapters"],
        "tuning_results": state["tuning_results"],
    }

    payload = json.dumps(result, indent=2, ensure_ascii=False, default=str).encode('utf-8')
    return send_file(
        BytesIO(payload),
        mimetype='application/json',
        as_attachment=True,
        download_name='results.json',
    )


# ══════════════════════════════════════════════════════════════════════
#  API ENDPOINTS (for async JS calls if needed later)
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/state', methods=['GET'])
def api_state():
    """Return current browser-session pipeline state as JSON (for debugging)."""
    safe_state = {}
    for k, v in state.items():
        if k == "processor":
            safe_state[k] = "initialized" if v else None
        elif k == "api_key":
            safe_state[k] = mask_api_key(v) if v else None
        elif k == "segments":
            safe_state[k] = len(v) if v else None
        else:
            try:
                json.dumps(v)
                safe_state[k] = v
            except (TypeError, ValueError):
                safe_state[k] = str(v)
    return jsonify(safe_state)


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    port = int(os.getenv('PORT', '5001'))
    debug = (os.getenv('FLASK_DEBUG', '').strip().lower() in {'1', 'true', 'yes', 'on'})
    app.run(host='0.0.0.0', port=port, debug=debug)



