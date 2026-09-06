"""Launch the FinChat web console.

Usage (from this webapp/ folder or project root):
    python webapp/run.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)