"""FastAPI backend that wraps the Finance ChatBot's multi-agent pipelines.

The web app exposes two workflows behind a small job system:

  * ``analysis``  - answer a question about a (cleaned) dataset + produce charts
  * ``cleaning``  - run the multi-agent cleaning workflow and save the output

Jobs are serialized (only one runs at a time) because generated charts share a
single output directory, and the job dict doubles as a progress/status record
that the frontend polls.
"""

import os
import re
import sys
import time
import uuid
import threading
import traceback
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "Analysis_Phase"))
sys.path.insert(0, str(ROOT / "Cleaning_Phase"))

# rag_env ships both libomp.dll (conda-forge LLVM) and libiomp5md.dll (MKL).
# Importing langchain + matplotlib in one process kills the kernel with
# OMP Error #15; this documented workaround must be set before imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

import multi_agent_analysis as analysis_local
import multi_agent_analysis_openrouter as analysis_cloud
import multi_agent_cleaning as cleaning_local
import multi_agent_cleaning_openrouter as cleaning_cloud

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = ROOT / "webapp"
FRONTEND = BASE / "frontend"
RAW_DIR = ROOT / "Handling_Data"
CLEAN_DIR = ROOT / "Cleaning_Phase"
CHART_DIR = ROOT / "Analysis_Phase" / "charts"

app = FastAPI(title="Finance ChatBot · Web Console", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Module / dataset helpers
# ---------------------------------------------------------------------------
def _analysis_module(backend: str):
    return analysis_cloud if backend == "openrouter" else analysis_local


def _cleaning_module(backend: str):
    return cleaning_cloud if backend == "openrouter" else cleaning_local


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_serializable(v) for v in obj]
    if hasattr(obj, "item") and hasattr(obj, "dtype"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


_DATASET_CACHE: Dict[str, tuple] = {}


def _scan_datasets() -> List[dict]:
    out: List[dict] = []
    for folder in (CLEAN_DIR, RAW_DIR):
        for f in sorted(folder.glob("*.csv")):
            key = str(f)
            try:
                mtime = f.stat().st_mtime_ns
                size = f.stat().st_size
            except OSError:
                continue
            cached = _DATASET_CACHE.get(key)
            if cached and cached[0] == mtime and cached[1] == size:
                out.append(cached[2])
                continue
            entry = {
                "id": f.relative_to(ROOT).as_posix(),
                "name": f.name,
                "folder": folder.name,
                "rows": None,
                "columns": [],
                "size_kb": round(size / 1024, 1),
            }
            try:
                entry["columns"] = list(pd.read_csv(f, nrows=0).columns)
            except Exception:
                pass
            try:
                entry["rows"] = len(pd.read_csv(f, usecols=[0]))
            except Exception:
                pass
            _DATASET_CACHE[key] = (mtime, size, entry)
            out.append(entry)
    return out


def _lookup_dataset(dataset_id: str) -> Optional[dict]:
    for d in _scan_datasets():
        if d["id"] == dataset_id:
            return d
    return None


# ---------------------------------------------------------------------------
# Job system (single-flight)
# ---------------------------------------------------------------------------
JOBS: Dict[str, dict] = {}
_JOB_SEQ: List[str] = []
_SEQ_LOCK = threading.Lock()
_RUN_SEM = threading.Semaphore(1)

# ---------------------------------------------------------------------------
# Chat history (per-session memory for follow-up questions)
# ---------------------------------------------------------------------------
CHAT_HISTORIES: Dict[str, List[dict]] = {}
CHAT_HISTORY_MAX = 5  # keep last N Q&A pairs per session


def _get_chat_history(session_id: str) -> str:
    """Format chat history for injection into the planner prompt."""
    msgs = CHAT_HISTORIES.get(session_id, [])
    if not msgs:
        return ""
    lines = []
    for m in msgs[-CHAT_HISTORY_MAX:]:
        lines.append(f"Q: {m['question']}")
        ans = m.get("answer", "")
        if len(ans) > 300:
            ans = ans[:300] + "..."
        lines.append(f"A: {ans}")
        lines.append("")
    return "\n".join(lines)


def _store_chat_message(session_id: str, question: str, answer: str) -> None:
    """Store a Q&A pair in the session's chat history."""
    if not session_id:
        return
    if session_id not in CHAT_HISTORIES:
        CHAT_HISTORIES[session_id] = []
    CHAT_HISTORIES[session_id].append({
        "question": question,
        "answer": answer,
        "timestamp": time.time(),
    })
    # keep only last N
    CHAT_HISTORIES[session_id] = CHAT_HISTORIES[session_id][-CHAT_HISTORY_MAX:]


def _new_job(job_type: str, backend: str, payload: dict) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job: dict = {
        "job_id": job_id,
        "type": job_type,
        "backend": backend,
        "status": "queued",
        "queue_position": 0,
        "progress": {},
        "result": None,
        "error": None,
        "charts": [],
        "payload": payload,
        "created_at": time.time(),
        "started_at": None,
        "ended_at": None,
    }
    with _SEQ_LOCK:
        JOBS[job_id] = job
        _JOB_SEQ.append(job_id)
    worker = threading.Thread(target=_worker, args=(job_id,), daemon=True)
    worker.start()
    return job


def _worker(job_id: str) -> None:
    job = JOBS[job_id]
    with _RUN_SEM:  # only one job at a time -> generated chart files never collide
        try:
            job["started_at"] = time.time()
            job["status"] = "running"
            if job["type"] == "analysis":
                _run_analysis(job)
            else:
                _run_cleaning(job)
            job["status"] = "done"
        except Exception:
            job["status"] = "error"
            job["error"] = traceback.format_exc(limit=6)
        finally:
            job["ended_at"] = time.time()
            with _SEQ_LOCK:
                if job_id in _JOB_SEQ:
                    _JOB_SEQ.remove(job_id)


def _queue_position(job_id: str) -> int:
    with _SEQ_LOCK:
        return _JOB_SEQ.index(job_id) if job_id in _JOB_SEQ else 0


def _public_job(job: dict) -> dict:
    job_id = job["job_id"]
    error = job["error"]
    return {
        "job_id": job_id,
        "type": job["type"],
        "backend": job["backend"],
        "status": job["status"],
        "queue_position": _queue_position(job_id) if job["status"] == "queued" else 0,
        "progress": {k: _make_serializable(v) for k, v in job["progress"].items()},
        "result": _make_serializable(job["result"]),
        "charts": [
            {"filename": c["filename"], "kind": c["kind"], "url": c["url"]}
            for c in job["charts"]
        ],
        "error": _short_error(error),
    }


def _short_error(tb: Optional[str]) -> Optional[str]:
    """Last meaningful line of a Python traceback for the UI card."""
    if not tb:
        return tb
    lines = [ln for ln in tb.splitlines() if ln.strip()]
    if not lines:
        return tb
    last = lines[-1].strip()
    if last.lower().startswith("during task"):
        return lines[-2].strip() if len(lines) > 1 else last
    return last


# ---------------------------------------------------------------------------
# Workflow runners
# ---------------------------------------------------------------------------
def _stream_job(job: dict, graph, initial_state: dict, config: dict = None):
    """Run the compiled graph via stream(), forwarding per-node progress."""
    if config is None:
        config = {"configurable": {"thread_id": f"web_{job['job_id']}"}}
    final_state: dict = {}
    for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
        # Some langgraph versions yield (step, updates) tuples, others yield the
        # updates dict directly - accept both.
        updates = chunk[1] if isinstance(chunk, tuple) and len(chunk) == 2 else chunk
        if not isinstance(updates, dict):
            continue
        for node, snapshot in updates.items():
            job["progress"] = {
                "node": node,
                "revision_count": int(snapshot.get("revision_count", 0)),
                "max_revisions": int(snapshot.get("max_revisions", 3)),
                "current_step": str(snapshot.get("current_step", "")),
                "label": _STEP_LABELS.get(node, node),
            }
            final_state = snapshot
    return final_state


_STEP_LABELS = {
    "load_and_profile": "Load & Profile",
    "planner": "Plan",
    "coder": "Code",
    "reviser": "Critique",
    "synthesizer": "Synthesize",
}


def _run_analysis(job: dict) -> None:
    mod = _analysis_module(job["backend"])
    session_id = job["payload"].get("session_id", "")
    thread_id = f"session_{session_id}" if session_id else f"web_{job['job_id']}"
    chat_history = _get_chat_history(session_id) if session_id else ""
    graph = mod.build_analysis_graph()
    initial = {
        "csv_path": job["payload"]["csv_path"],
        "question": job["payload"]["question"],
        "schema_context": mod.build_schema_context(job["payload"]["csv_path"]),
        "analysis_plan": "",
        "generated_code": "",
        "execution_result": {},
        "critique_result": {},
        "suggested_fix": "",
        "final_answer": "",
        "chart_paths": [],
        "revision_count": 0,
        "max_revisions": job["payload"].get("max_revisions", 3),
        "current_step": "start",
        "errors": [],
        "chart_kind": job["payload"].get("chart_kind", "auto"),
        "chat_history": chat_history,
    }
    config = {"configurable": {"thread_id": thread_id}}
    state = _stream_job(job, graph, initial, config=config)
    if not state:
        raise RuntimeError("Workflow finished without producing any state.")

    chart_paths = state.get("chart_paths") or []
    job["charts"] = [_chart_record(job["job_id"], p) for p in chart_paths]

    critique = state.get("critique_result") or {}
    job["result"] = {
        "question": state.get("question"),
        "answer": state.get("final_answer", ""),
        "validated": bool(critique.get("passed")),
        "revision_count": state.get("revision_count", 0),
        "max_revisions": state.get("max_revisions", 3),
        "current_step": state.get("current_step"),
        "errors": state.get("errors") or [],
        "result_text": critique.get("result_text", ""),
        "plan": state.get("analysis_plan", ""),
        "code": state.get("generated_code", ""),
    }

    # Store in chat history for follow-up questions
    if session_id:
        _store_chat_message(
            session_id,
            job["payload"]["question"],
            state.get("final_answer", ""),
        )


def _run_cleaning(job: dict) -> None:
    mod = _cleaning_module(job["backend"])
    graph = mod.build_cleaning_graph()
    initial = {
        "csv_path": job["payload"]["csv_path"],
        "llm_context": "",
        "cleaning_plan": "",
        "generated_code": "",
        "execution_result": {},
        "validation_result": {},
        "revision_count": 0,
        "max_revisions": job["payload"].get("max_revisions", 3),
        "current_step": "start",
        "errors": [],
    }
    state = _stream_job(job, graph, initial)
    if not state:
        raise RuntimeError("Cleaning workflow finished without producing any state.")

    out_name = f"{Path(job['payload']['csv_path']).stem}_cleaned.csv"
    out_path = CLEAN_DIR / out_name
    df = mod.run_and_save_cleaned_data(
        job["payload"]["csv_path"],
        str(out_path),
        result=state,
        max_revisions=job["payload"].get("max_revisions", 3),
    )
    rows = len(df)
    try:
        rows_before = len(pd.read_csv(job["payload"]["csv_path"], usecols=[0]))
    except Exception:
        rows_before = rows
    exec_result = state.get("execution_result") or {}
    val_result = state.get("validation_result") or {}
    job["result"] = {
        "output_name": out_name,
        "output_rel": (Path("Cleaning_Phase") / out_name).as_posix(),
        "output_csv": str(out_path),
        "rows_before": rows_before,
        "rows_after": rows,
        "lost_ratio": round(100 * (1 - rows / rows_before) if rows_before else 0, 2),
        "columns": list(df.columns),
        "preview": _make_serializable(df.head(5).to_dict("records")),
        "cleaning_plan": state.get("cleaning_plan", ""),
        "generated_code": state.get("generated_code", ""),
        "column_changes": exec_result.get("column_changes") or {},
        "detected_issues": exec_result.get("detected_issues")
        or exec_result.get("issues")
        or [],
        "columns_before": exec_result.get("columns_before") or [],
        "columns_after": exec_result.get("columns_after") or [],
        "missing_before": exec_result.get("missing_before") or {},
        "missing_after": exec_result.get("missing_after") or {},
        "revision_count": state.get("revision_count", 0),
        "checks": val_result.get("checks") or [],
        "warnings": val_result.get("warnings") or [],
        "failures": val_result.get("failures") or [],
    }
    _DATASET_CACHE.pop(str(out_path), None)


def _chart_record(job_id: str, path: str) -> dict:
    p = Path(path)
    return {
        "filename": p.name,
        "kind": p.suffix.lstrip(".").lower(),
        "url": f"/api/charts/{job_id}/{urllib.parse.quote(p.name)}",
        "path": str(p),
    }


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
class AnalysisRequest(BaseModel):
    dataset: str
    question: str = Field(min_length=1)
    backend: str = "openrouter"
    max_revisions: int = Field(default=3, ge=1, le=6)
    chart_kind: str = Field(default="auto", pattern="^(auto|interactive)$")
    session_id: str = ""  # for chat history continuity


class CleaningRequest(BaseModel):
    dataset: str
    backend: str = "openrouter"
    max_revisions: int = Field(default=3, ge=1, le=6)


@app.get("/api/health")
def health():
    ok = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=1.2):
            ok = True
    except Exception:
        ok = False
    return {
        "status": "ok",
        "ollama": ok,
        "ts": time.time(),
    }


@app.get("/api/datasets")
def datasets():
    return {"datasets": _scan_datasets()}


@app.get("/api/datasets/preview")
def dataset_preview(path: str):
    entry = _lookup_dataset(path)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    full = ROOT / Path(path)
    try:
        df = pd.read_csv(full, nrows=8)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read dataset: {exc}")
    return {
        "id": entry["id"],
        "columns": entry["columns"],
        "rows": entry["rows"],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "preview": _make_serializable(df.to_dict("records")),
    }


@app.post("/api/upload", status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    """Accept a CSV from the user's computer and add it to the dataset pool."""
    original = file.filename or "upload.csv"
    if not original.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only .csv files are accepted.")
    safe = re.sub(r"[^\w .-]", "_", Path(original).name).strip()
    if not safe or safe == ".":
        safe = "upload.csv"

    dest = RAW_DIR / safe
    counter = 1
    while dest.exists():
        dest = RAW_DIR / f"{Path(safe).stem} ({counter}){Path(safe).suffix}"
        counter += 1

    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read upload: {exc}")
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    dest.write_bytes(content)
    _DATASET_CACHE.pop(str(dest), None)
    entry = _lookup_dataset(dest.relative_to(ROOT).as_posix())
    return {"dataset": entry}


@app.get("/api/datasets/download")
def download_dataset(path: str):
    """Download a CSV (usually a cleaned one) to the user's device."""
    entry = _lookup_dataset(path)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    full = (ROOT / Path(path)).resolve()
    if not full.is_file():
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    media = "text/csv; charset=utf-8" if entry["folder"] == "Cleaning_Phase" else None
    return FileResponse(
        path=str(full),
        filename=full.name,
        media_type=media,
    )


@app.post("/api/analysis", status_code=202)
def start_analysis(req: AnalysisRequest):
    entry = _lookup_dataset(req.dataset)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if req.backend not in {"local", "openrouter"}:
        raise HTTPException(status_code=422, detail="backend must be 'local' or 'openrouter'.")
    job = _new_job("analysis", req.backend, {
        "csv_path": str(ROOT / Path(req.dataset)),
        "question": req.question.strip(),
        "max_revisions": req.max_revisions,
        "chart_kind": req.chart_kind,
        "session_id": req.session_id,
    })
    return {"job_id": job["job_id"], "status": "queued"}


@app.get("/api/analysis/{job_id}")
def analysis_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _public_job(job)


@app.get("/api/sessions/{session_id}/history")
def get_chat_history(session_id: str):
    """Return chat history for a session (for debugging / UI display)."""
    msgs = CHAT_HISTORIES.get(session_id, [])
    return {
        "session_id": session_id,
        "messages": msgs,
        "count": len(msgs),
    }


@app.post("/api/cleaning", status_code=202)
def start_cleaning(req: CleaningRequest):
    entry = _lookup_dataset(req.dataset)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if req.backend not in {"local", "openrouter"}:
        raise HTTPException(status_code=422, detail="backend must be 'local' or 'openrouter'.")
    job = _new_job("cleaning", req.backend, {
        "csv_path": str(ROOT / Path(req.dataset)),
        "max_revisions": req.max_revisions,
    })
    return {"job_id": job["job_id"], "status": "queued"}


@app.get("/api/cleaning/{job_id}")
def cleaning_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _public_job(job)


@app.get("/api/charts/{job_id}/{filename}")
def serve_chart(job_id: str, filename: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    allowed = {c["filename"]: c["path"] for c in job["charts"]}
    path = allowed.get(urllib.parse.unquote(filename))
    if not path:
        raise HTTPException(status_code=404, detail="Chart not found for this job.")
    media = {
        ".html": "text/html",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
    }.get(Path(path).suffix.lower())
    return FileResponse(path, media_type=media)


# Static frontend (mounted last so /api/* wins)
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)