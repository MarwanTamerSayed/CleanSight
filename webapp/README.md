# FinChat · Web Console

A web UI around the Finance ChatBot's multi-agent pipelines (FastAPI + vanilla JS/CSS/HTML).

## Features
- **Ask Assistant** — pick a dataset + backend, ask a question, and watch the Planner → Coder → Critique → Synthesizer pipeline live while it validates and revises until it produces an answer with charts (matplotlib PNGs + interactive plotly HTML).
- **Data Prep** — run the multi-agent cleaning workflow on any raw CSV, save the validated clean file, and see before/after row counts, the cleaning plan, per-column changes, and validation checks. Download the cleaned CSV to your device.
- **CSV upload** — drop a file into the Data Prep drop zone or use the "Upload CSV" button in the top bar; it is saved to `Handling_Data/` and immediately selectable for cleaning or analysis (max 60 MB).
- **Charts in the chat** — each analysis answer renders its charts inline right under the question, each with "open" and "download" buttons. Toggle **interactive charts** above the send button to prefer hoverable Plotly charts.
- **Table rendering** — whitespace-delimited and markdown pipe-delimited tables are auto-detected and rendered as styled HTML tables with right-aligned numeric columns.
- **Collapsible pipeline** — the Analysis Pipeline and Analysis Details cards collapse/expand to keep the chat area spacious.
- **Number formatting** — large numbers are displayed with thousands separators (e.g., 144,630).
- **Two backends** — OpenRouter (`minimax-minimax-m3:free`) or local Ollama (`qwen2.5:7b`).
- Jobs run single-flight so generated charts never collide.

## Project layout
```
webapp/
├── run.py                  # one-command launcher
├── requirements.txt
├── backend/
│   └── main.py             # FastAPI app + job system
└── frontend/
    ├── index.html
    ├── css/style.css
    └── js/app.js
```

## How to run

1. Use the conda environment that already has the pipeline deps (python 3.11, `C:\Users\LOQ\anaconda3\envs\rag_env`):
   ```powershell
   conda activate rag_env
   cd "D:\Finance ChatBot\webapp"
   ```

2. If anything is missing, install requirements:
   ```powershell
   pip install -r requirements.txt
   ```

3. Start the server (default http://127.0.0.1:8000):
   ```powershell
   python run.py
   ```
   (or `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` from `webapp/`)

4. Open http://127.0.0.1:8000 in your browser.

## Backends
- **OpenRouter** — needs `D:\Finance ChatBot\.env` with `open_router_API=sk-or-v1-...` and `model=...` (already present).
- **Local** — needs Ollama running (`ollama serve`) with `qwen2.5:7b` pulled.

## Notes
- The `.env` with the API key stays in the project root; the backend reads it via the existing pipeline modules.
- If the server kernel ever dies with `OMP: Error #15`, it is the known libomp/libiomp5md conflict — the modules set `KMP_DUPLICATE_LIB_OK=TRUE` before importing to work around it.