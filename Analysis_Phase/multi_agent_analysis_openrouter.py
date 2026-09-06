import sys
import os

# rag_env ships both libomp.dll (conda-forge LLVM) and libiomp5md.dll (MKL);
# importing the langchain stack then matplotlib trips OMP Error #15 and kills
# the process with no Python traceable error. This is the documented workaround
# for Jupyter kernels (see http://openmp.llvm.org). Must be set before imports.
if "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import time
import traceback
import ast as ast_module
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal, TypedDict
from datetime import datetime

import pandas as pd
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class AnalysisState(TypedDict):
    csv_path: str                 # cleaned dataset (Analysis phase input)
    question: str                 # natural-language question
    schema_context: str           # profiled cleaned schema for the LLM
    analysis_plan: str            # Agent 1 output: English plan
    generated_code: str           # Agent 2 output: pandas + plotting code
    execution_result: Dict[str, Any]
    critique_result: Dict[str, Any]   # {passed, reasons, suggested_fix}
    suggested_fix: str
    final_answer: str
    chart_paths: List[str]
    revision_count: int
    max_revisions: int
    current_step: str
    errors: List[str]
    chart_kind: str               # "auto" | "interactive" - chart render preference
    chat_history: str             # prior Q&A context for follow-up questions


def create_openrouter_llm(
    temperature: float = 0.3,
    env_path: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> ChatOpenAI:
    """Create an LLM backed by the OpenRouter API.

    Reads credentials from the project's ``.env`` file (keys ``open_router_API``
    and ``model``) unless overridden via the ``api_key`` / ``model`` arguments.
    """
    if env_path is None:
        env_path = str(Path(__file__).resolve().parent.parent / ".env")

    if api_key is None or model is None:
        from dotenv import load_dotenv

        load_dotenv(env_path)

        if api_key is None:
            api_key = os.getenv("open_router_API")
        if model is None:
            model = os.getenv("model")

    if not api_key:
        raise ValueError("OpenRouter API key not found (set 'open_router_API' in the .env).")
    if not model:
        raise ValueError("OpenRouter model not found (set 'model' in the .env).")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )


def _make_serializable(obj: Any) -> Any:
    """Convert numpy/pandas/native types to JSON-serializable values."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if pd.isna(obj):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def build_schema_context(csv_path: str) -> str:
    """Build a compact schema + preview context for the planner from the
    ALREADY-CLEANED dataset. Dates are parsed so the context reports true
    dtypes instead of reading 'Date' back as object."""
    df = pd.read_csv(csv_path, parse_dates=["Date"] if "Date" in pd.read_csv(csv_path, nrows=1).columns else None)
    lines = [f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns", "", "Columns:"]
    for col in df.columns:
        s = df[col]
        line = f"  - {col} ({s.dtype}): missing={s.isna().mean() * 100:.1f}%, unique={s.nunique()}"
        if pd.api.types.is_numeric_dtype(s):
            if s.notna().any():
                line += f", min={s.min():.2f} max={s.max():.2f} mean={s.mean():.2f}"
        elif pd.api.types.is_datetime64_any_dtype(s):
            if s.notna().any():
                line += f", range={s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}"
        elif pd.api.types.is_object_dtype(s):
            line += f", sample={s.dropna().astype(str).unique()[:5].tolist()}"
        lines.append(line)
    lines.append(f"\nSample rows (first 5):")
    for i, row in enumerate(df.head(5).to_dict(orient="records")):
        lines.append(f"  Row {i}: {_make_serializable(row)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent 1: Analysis Planner
# ---------------------------------------------------------------------------
PLANNER_PROMPT = PromptTemplate.from_template("""
You are an expert data analyst. Given a CLEANED dataset schema and a user question,
produce a precise ANALYSIS PLAN that another LLM will turn into executable pandas +
plotting code.

{chat_history_section}
====================
CLEANED DATASET SCHEMA
====================
{schema_context}

====================
USER QUESTION
====================
{question}

====================
YOUR JOB
====================
1. Restate exactly what the question is asking (what metric / comparison / trend).
2. List the concrete steps to answer it: which columns to use, what aggregation,
   grouping, filtering, and calculations are needed.
3. Specify the ANSWER SHAPE expected from the question, chosen from:
   - single_value  (e.g. "how many / total / average" -> one number)
   - top_list      (e.g. "top / which are the best / compare" -> a small ranked table)
   - distribution  (e.g. "distribution / spread / histogram")
   - trend_over_time (e.g. "over time / by month / growth" -> requires a Date/group-by)
   - relationship  (e.g. "relationship / correlation" between numeric columns)
   - summary_table (otherwise, a grouped summary DataFrame)
4. Decide whether a CHART is useful, and if so which type (bar, line, scatter,
   histogram, box) and with what x/y/series. Only request charts that genuinely
   help answer the question. For matplotlib save the figure with
   fig.savefig(f"{{output_dir}}/chart_1.png", dpi=120); for plotly use
   fig.write_html(f"{{output_dir}}/chart_1.html").
5. List VALIDATION_CHECKs the reviewer should verify (e.g. correct aggregation,
   sensible axis labels, non-empty result).

RULES:
- Use ONLY the column names that exist in the schema. Never invent columns.
- Do not write code. Output the plan in clean structured text.
- Be specific and unambiguous so the coder can act without guessing.

OUTPUT FORMAT:
QUESTION_UNDERSTANDING:
...

ANSWER_SHAPE: <one of the shapes above>

ANALYSIS_STEPS:
1. ...
2. ...

CHART_RECOMMENDATIONS:
- If a chart helps: describe type + fields + purpose. If no chart is needed, say NONE.

VALIDATION_CHECKS:
- ...
""")

# ---------------------------------------------------------------------------
# Agent 2: Coder
# ---------------------------------------------------------------------------
CODER_PROMPT = PromptTemplate.from_template("""
You are writing Python code to answer a data question about a CLEANED dataframe `df`.

AVAILABLE COLUMNS (use ONLY these, exact spelling from the schema):
{schema_context}

USER QUESTION:
{question}

ANALYSIS PLAN:
{analysis_plan}

CONSTRAINTS:
- The dataframe is already clean. Do NOT modify/clean it; only read and analyze it.
- Available environment already imported: pd, np, matplotlib.pyplot (as plt),
  seaborn (as sns), plotly.express (as px), plotly.graph_objects (as go),
  and a string variable `output_dir` (a path where you may save charts).
- Store the ANSWER in a variable named `result`. It can be a number, a string, a
  Series, or a small DataFrame - matching the ANSWER_SHAPE in the plan.
- If you create MATPLOTLIB/SEABORN figures, save each one WITHIN the analysis script
  using:  fig.savefig(f"{{output_dir}}/chart_1.png", dpi=120, bbox_inches="tight")
  then plt.close(fig). Number charts sequentially.
- If you create PLOTLY figures, save with:
  fig.write_html(f"{{output_dir}}/chart_1.html")
- A chart is only useful if it directly supports answering the question. Do not add
  decorative or redundant figures.
- Set a clear title and labeled axes on every figure.
- Return ONLY raw Python code. NO markdown fences, NO backticks, NO ```python, NO ```.
- No comments or explanations. No print statements (they are ignored).
- Do not re-read the CSV - `df` already exists.
- Import extra libraries ONLY if truly needed; the standard analysis libs are preloaded.
- If a needed column seems missing, do not guess - assign `result = "MISSING_COLUMN: <name>"`.

{chart_preference}

{feedback_block}
""")

REVISER_PROMPT = PromptTemplate.from_template("""
You are a strict data-analysis reviewer. Review the generated analysis code, its
execution result, and the layered critique. Fix any problem by rewriting the code.

USER QUESTION:
{question}

ANALYSIS PLAN:
{analysis_plan}

GENERATED CODE:
{code}

EXECUTION RESULT:
{execution_result}

CRITIQUE RETURNED:
{critique_result}

CRITIQUE GUIDANCE TO APPLY:
{critique_guidance}

RULES:
- Re-use the same environment and `result`/`output_dir` conventions as the original.
- Fix the exact issue(s) raised in the critique. Common fixes:
  - KeyError / wrong column -> use the correct column name from the schema.
  - Wrong answer shape -> produce the shape the question implies.
  - Empty result -> the filter/group is wrong; fix it.
  - Math/logic error (e.g. growth rate, percentage) -> recompute correctly.
  - Chart issues -> label axes, set a title, save via the output_dir convention.
- Keep the analysis correct and the chart(s) genuinely useful.
- Return ONLY the complete corrected raw Python code. NO markdown fences, NO backticks,
  NO explanation. If the code is already fully correct, return exactly: ANALYSIS_OK
""")

SYNTHESIZER_PROMPT = PromptTemplate.from_template("""
You are a data analyst writing the final answer to a user's question.

USER QUESTION:
{question}

ANSWER (raw result from the executed analysis):
{result_value}

ANALYSIS PLAN SUMMARY:
{analysis_plan}

CHARTS PRODUCED:
{chart_paths}

Write a concise, correct natural-language answer (2-5 sentences). If the answer is
uncertain, say so honestly. Mention any chart(s) that were produced so the user knows
what to look at. Do not fabricate numbers - use only the result provided.

If the analysis FAILED to produce an answer (result is empty/missing), instead say
clearly that the question could not be confidently answered and why.
""")

MAX_FALLBACK_LEN = 4000


def planner_agent(state: AnalysisState) -> AnalysisState:
    llm = create_openrouter_llm(temperature=0.0)
    chain = PLANNER_PROMPT | llm | StrOutputParser()
    chat_history = state.get("chat_history", "")
    chat_history_section = ""
    if chat_history:
        chat_history_section = (
            "====================\n"
            "CONVERSATION HISTORY (prior Q&A on this dataset)\n"
            "====================\n"
            f"{chat_history}\n"
            "\nUse the history above to understand follow-up context. "
            "If the user's question references something from earlier (e.g. 'what about last month', "
            "'compare with that'), use the history to resolve the reference."
        )
    response = chain.invoke({
        "schema_context": state["schema_context"],
        "question": state["question"],
        "chat_history_section": chat_history_section,
    })
    state["analysis_plan"] = response.strip()
    state["current_step"] = "planned"
    return state


def _strip_markdown_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def _feedback_block(state: AnalysisState) -> str:
    """Injected extra context for the coder on a revision attempt."""
    if not state.get("suggested_fix") and not state.get("execution_result"):
        return ""
    parts = ["A PREVIOUS ATTEMPT FAILED OR NEEDED REVIEW. Fix these issues:"]
    fb = state.get("suggested_fix", "")
    if fb:
        parts.append(f"REVIEWER FEEDBACK: {fb}")
    ex = state.get("execution_result") or {}
    if ex.get("error"):
        parts.append(f"EXECUTION ERROR: {ex['error']}")
    cr = state.get("critique_result") or {}
    reasons = cr.get("reasons") or []
    if reasons:
        parts.append("CRITIQUE ITEMS: " + " | ".join(str(r) for r in reasons[:5]))
    prev_code = state.get("generated_code", "")
    if prev_code:
        parts.append(f"PREVIOUS CODE WAS:\n{prev_code[:MAX_FALLBACK_LEN]}")
    return "\n\n".join(parts)


def _chart_preference(state: AnalysisState) -> str:
    """Prefer interactive Plotly HTML charts when the user requests them."""
    kind = (state.get("chart_kind") or "auto").lower()
    if kind == "interactive":
        return (
            "CHART PREFERENCE: interactive charts. Build figures with Plotly "
            "(px / go) and save them via fig.write_html(f\"{output_dir}/chart_N.html\"). "
            "If a matplotlib chart is unavoidable, ALSO write the same chart "
            "interactively with Plotly and save it as .html.\n"
        )
    return "CHART PREFERENCE: either matplotlib PNG or interactive Plotly HTML is fine; "
    "the best chart for the question wins."


def coder_agent(state: AnalysisState) -> AnalysisState:
    llm = create_openrouter_llm(temperature=0.0)
    chain = CODER_PROMPT | llm | StrOutputParser()
    response = chain.invoke({
        "schema_context": state["schema_context"],
        "question": state["question"],
        "analysis_plan": state["analysis_plan"],
        "chart_preference": _chart_preference(state),
        "feedback_block": _feedback_block(state),
    })
    state["generated_code"] = _strip_markdown_fences(response)
    state["current_step"] = "coded"
    return state


# ---------------------------------------------------------------------------
# Execution (sandboxed) with chart collection
# ---------------------------------------------------------------------------
def _analysis_globals():
    """Lazily provide the sandbox namespace for generated analysis code.

    Plotting libraries are imported only when a generated script actually runs,
    and only ``Agg`` is forced when no interactive/inline backend is already
    active. This avoids disturbing Jupyter's ``%matplotlib inline`` backend and
    keeps notebook kernels stable.
    """
    g = _analysis_globals._cache
    if g is not None:
        return g
    import matplotlib

    try:
        backend = matplotlib.get_backend().lower()
        if "inline" not in backend and "module://" not in backend:
            matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go

    g = {
        "pd": pd,
        "np": np,
        "os": os,
        "plt": plt,
        "sns": sns,
        "px": px,
        "go": go,
    }
    _analysis_globals._cache = g
    return g


_analysis_globals._cache = None


_KNOWN_DATAFRAME_ATTRS = set(dir(pd.DataFrame)) | set(dir(pd.Series)) | {
    "plot", "hist", "box", "value_counts", "nlargest", "nsmallest", "groupby", "sort_values",
    "reset_index", "rename", "describe", "agg", "aggregate", "sum", "mean", "min", "max",
    "median", "count", "std", "var", "to_numpy", "to_frame", "to_dict", "to_list", "columns",
    "index", "shape", "dtypes", "isnull", "notnull", "unique", "dropna", "fillna", "apply",
    "merge", "pivot", "pivot_table", "melt", "head", "tail", "sample", "copy", "astype",
    "drop", "loc", "iloc", "isin", "between", "clip", "round", "abs", "cumsum", "diff",
    "shift", "rank", "quantile",
}


def _chart_files(output_dir: str) -> List[str]:
    out = Path(output_dir)
    if not out.exists():
        return []
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.html"):
        for f in out.glob(ext):
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            if size > 0:
                files.append({"path": str(f), "size": size, "kind": f.suffix.lstrip(".")})
    return files


def _chart_mtimes(output_dir: str) -> Dict[str, float]:
    out = Path(output_dir)
    if not out.exists():
        return {}
    mtimes = {}
    for c in _chart_files(output_dir):
        try:
            mtimes[os.path.normpath(c["path"])] = os.path.getmtime(c["path"])
        except OSError:
            pass
    return mtimes


def _collect_new_charts(before: Dict[str, float], output_dir: str, started: float) -> List[str]:
    """Return chart files created or re-saved during this execution.

    Filenames can repeat across runs (e.g. chart_1.png), so a purely
    name-based diff is wrong; we use modification time >= the execution start.
    """
    after = _chart_mtimes(output_dir)
    return sorted(
        p for p, mt in after.items()
        if p not in before or mt >= started - 0.5
    )


def _safe_exec(code: str, df: pd.DataFrame, output_dir: str, seed_files: set):
    """Execute the analysis code with plotting libs available and trap errors.

    Returns (result_holder, error, error_line, statements_executed).
    No ``sys.settrace`` is used: global tracing destabilises IPython/Jupyter
    kernels, so the failing line is located from the traceback instead.
    """
    g = _analysis_globals()
    local_vars = {"df": df.copy(), "output_dir": output_dir, "result": None}
    try:
        exec(compile(code, "<generated>", "exec"), g, local_vars)
    except Exception as e:
        error_line = None
        tb = sys.exc_info()[2]
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == "<generated>":
                error_line = tb.tb_lineno
                break
            tb = tb.tb_next
        return None, f"{type(e).__name__}: {e}", error_line, 0
    finally:
        # Never close figures here - in a notebook that would close the user's
        # own inline figures. Generated figures are saved to output_dir by the
        # code itself, so there is nothing to clean up.
        pass
    return local_vars.get("result"), None, None, 0


def _refresh_charts(state: AnalysisState, output_dir: str, before: Dict[str, float], started: float) -> None:
    state["chart_paths"] = _collect_new_charts(before, output_dir, started)


# ---------------------------------------------------------------------------
# Layered critique (L1 static, L2 exec/empty, L3 type-consistency, L4 LLM judge)
# ---------------------------------------------------------------------------
def _question_shape(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ("how many", "how much", "count", "total of", "number of", "sum of")):
        return "single_value"
    if any(k in q for k in ("trend", "over time", "by month", "by year", "growth", "monthly", "daily", "seasonal")):
        return "trend_over_time"
    if any(k in q for k in ("distribution", "histogram", "spread")):
        return "distribution"
    if any(k in q for k in ("relationship", "correlation")):
        return "relationship"
    if any(k in q for k in ("top", "best", "which", "compare", "highest", "lowest", "rank")):
        return "top_list"
    if any(k in q for k in ("avg", "average", "mean", "ratio", "percentage", "percent")):
        return "single_value"
    return "summary_table"


_EXPECTED_RESULT_TYPES = {
    "single_value": (int, float, np.integer, np.floating),
    "top_list": (list, pd.Series, pd.DataFrame),
    "distribution": (pd.Series, pd.DataFrame, np.ndarray),
    "trend_over_time": (pd.Series, pd.DataFrame, list),
    "relationship": (pd.Series, pd.DataFrame, np.ndarray),
    "summary_table": (pd.DataFrame, pd.Series, dict),
}


def _expected_type_label(shape: str) -> str:
    labels = {
        "single_value": "a single number",
        "top_list": "a list/series/small table of top items",
        "distribution": "a series/dataframe (distribution values)",
        "trend_over_time": "a time-series series/dataframe",
        "relationship": "a series/dataframe of paired values",
        "summary_table": "a summary DataFrame",
    }
    return labels.get(shape, "an appropriate result")


def critique_result_value(result: Any, state: AnalysisState) -> Dict[str, Any]:
    """Layered rule-based critique of the analysis result + charts.

    Returns a dict with: passed, reasons (list), suggestions (list), result_text.
    """
    out = {
        "passed": True,
        "reasons": [],
        "suggestions": [],
        "result_text": "",
        "result_type": None,
    }
    code = state["generated_code"]

    # --- L1: static AST checks ---
    try:
        tree = ast_module.parse(code)
    except SyntaxError as e:
        out["passed"] = False
        out["reasons"].append(f"Syntax error in generated code: {e}")
        out["suggestions"].append("Rewrite the code so it parses as valid Python.")
        return out

    # result must be assigned
    result_assigned = False
    for node in ast_module.walk(tree):
        if isinstance(node, (ast_module.Assign, ast_module.AnnAssign)):
            for tgt in (node.targets if isinstance(node, ast_module.Assign) else [node.target]):
                if isinstance(tgt, ast_module.Name) and tgt.id == "result":
                    result_assigned = True
    if not result_assigned:
        out["passed"] = False
        out["reasons"].append("Code never assigns `result` (required answer variable).")
        out["suggestions"].append("Store the final answer in a variable named `result`.")

    # column-name validation against actual df columns
    df = pd.read_csv(state["csv_path"], parse_dates=["Date"] if "Date" in pd.read_csv(state["csv_path"], nrows=1).columns else None)
    cols = set(df.columns)
    referenced = set()
    call_func_attrs = set()
    for node in ast_module.walk(tree):
        if isinstance(node, ast_module.Call) and isinstance(node.func, ast_module.Attribute):
            call_func_attrs.add(node.func.attr)
    for node in ast_module.walk(tree):
        # df["col"] / df[...] subscript pattern
        if isinstance(node, ast_module.Subscript):
            try:
                if isinstance(node.value, ast_module.Name) and node.value.id == "df":
                    sl = node.slice if hasattr(node, "slice") else node
                    sl = getattr(sl, "value", sl)
                    key = ast_module.literal_eval(sl)
                    if isinstance(key, str):
                        referenced.add(key)
            except Exception:
                pass
        # df.col dot-access - flag only when NOT a method call and NOT a known attr
        elif isinstance(node, ast_module.Attribute) and isinstance(node.value, ast_module.Name) and node.value.id == "df":
            if node.attr not in call_func_attrs and node.attr not in _KNOWN_DATAFRAME_ATTRS and not node.attr.startswith("_"):
                referenced.add(node.attr)
    unknown = {c for c in referenced if c not in cols}
    if unknown:
        out["passed"] = False
        out["reasons"].append(f"Referenced non-existent column(s): {sorted(unknown)}")
        out["suggestions"].append(
            f"Use only these existing columns: {sorted(cols)}. Replace/remove references to {sorted(unknown)}."
        )

    # --- L2: exec / empty checks ---
    output_dir = os.path.join(Path(state["csv_path"]).parent.parent, "Analysis_Phase", "charts")
    os.makedirs(output_dir, exist_ok=True)
    before_charts = _chart_mtimes(output_dir)
    started = time.time()
    result, error, error_line, _ = _safe_exec(code, df, output_dir, before_charts)
    _refresh_charts(state, output_dir, before_charts, started)

    if error is not None:
        out["passed"] = False
        out["reasons"].append(f"Execution error{(' on line ' + str(error_line)) if error_line else ''}: {error}")
        out["suggestions"].append("Fix the error. If it is a KeyError, use a real column name from the schema.")

    empty = False
    if result is None:
        empty = True
    elif isinstance(result, (pd.DataFrame, pd.Series)) and len(result) == 0:
        empty = True
    elif isinstance(result, (list, tuple, dict)) and len(result) == 0:
        empty = True
    if empty:
        out["passed"] = False
        out["reasons"].append("Analysis produced an EMPTY result (result is None/empty).")
        out["suggestions"].append("The computation returned nothing - check the filter/group logic and ensure it returns a value.")

    # --- L3: type-consistency with the question shape ---
    shape = _question_shape(state["question"])
    expected = _EXPECTED_RESULT_TYPES.get(shape)
    if result is not None and expected is not None:
        ok_type = isinstance(result, expected)
        if not ok_type:
            out["passed"] = False
            out["reasons"].append(
                f"Result type {type(result).__name__} does not match a question expecting "
                f"{_expected_type_label(shape)}."
            )
            out["suggestions"].append(
                f"Recompute so `result` is {_expected_type_label(shape)}. "
                f"(e.g. use .sum()/.mean() for a single number, .groupby() for a table)."
            )

    # --- Chart usefulness check ---
    chart_count = len(state["chart_paths"])
    wants_chart = "CHART_RECOMMENDATIONS" in state["analysis_plan"] and "NONE" not in (state["analysis_plan"].partition("CHART_RECOMMENDATIONS")[2])[:200]
    if wants_chart and chart_count == 0:
        out["passed"] = False
        out["reasons"].append("The plan asked for a chart but no chart file was created.")
        out["suggestions"].append("Add a figure and save it via fig.savefig(f\"{output_dir}/chart_N.png\", dpi=120) (matplotlib) or fig.write_html(f\"{output_dir}/chart_N.html\") (plotly).")
    if chart_count > 0:
        out["reasons"].append(f"Produced {chart_count} chart file(s): {state['chart_paths']}")

    # result_text for the synthesizer
    out["result_text"] = _result_to_text(result)

    return out


def _result_to_text(result: Any) -> str:
    if result is None:
        return "NO RESULT"
    try:
        if isinstance(result, pd.DataFrame):
            return result.to_string(index=False)[:MAX_FALLBACK_LEN]
        if isinstance(result, pd.Series):
            return result.to_string()[:MAX_FALLBACK_LEN]
        if isinstance(result, (int, float, np.integer, np.floating)):
            try:
                return f"{float(result):,.4f}"
            except Exception:
                return str(result)
        if isinstance(result, (list, tuple)):
            return str(result)[:MAX_FALLBACK_LEN]
        return str(result)[:MAX_FALLBACK_LEN]
    except Exception:
        return str(result)[:MAX_FALLBACK_LEN]


def llm_judge(state: AnalysisState) -> Dict[str, Any]:
    """L4: LLM-as-judge - returns dict with passes, reason, suggested_fix."""
    judgement = {
        "passes": True,
        "reason": "Rule-based critique passed; LLM judge not required.",
        "suggested_fix": "",
    }
    cr = state.get("critique_result") or {}
    if cr.get("passed", True):
        return judgement
    llm = create_openrouter_llm(temperature=0.0)
    prompt = PromptTemplate.from_template("""
You are a senior data-science code reviewer. Decide whether the generated analysis code
correctly and completely answers the question, and if not, how to fix it.

QUESTION:
{question}

ANALYSIS PLAN:
{analysis_plan}

GENERATED CODE:
{code}

RULE-BASED CRITIQUE:
{critique_reasons}

RESULT:
{result_text}

Return a judgment in EXACTLY this format (no markdown):
passes: true|false
reason: <one short sentence>
suggested_fix: <concrete corrected instruction the coder can apply>
""")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "question": state["question"],
        "analysis_plan": state["analysis_plan"],
        "code": state["generated_code"][:MAX_FALLBACK_LEN],
        "critique_reasons": "\n".join("- " + str(r) for r in (cr.get("reasons") or [])),
        "result_text": (cr.get("result_text") or "")[:MAX_FALLBACK_LEN],
    })
    passes = "passes: true" in response.lower() or "passes:true" in response.lower().replace(" ", "")
    reason = ""
    suggested_fix = ""
    m = re.search(r"reason\s*:\s*(.+)", response, re.IGNORECASE | re.MULTILINE)
    if m:
        reason = m.group(1).strip()
    m = re.search(r"suggested_fix\s*:\s*(.+)", response, re.IGNORECASE | re.MULTILINE)
    if m:
        suggested_fix = m.group(1).strip()
    judgement = {
        "passes": bool(passes),
        "reason": reason or response.strip()[:300],
        "suggested_fix": suggested_fix,
    }
    return judgement


def reviser_agent(state: AnalysisState) -> AnalysisState:
    """Agent 3: run code, critique (rules + LLM judge), revise if needed."""
    try:
        crit = critique_result_value(None, state)
    except Exception as e:
        crit = {
            "passed": False,
            "reasons": [f"Critique machinery failed: {e}"],
            "suggestions": [],
            "result_text": "",
        }
    state["critique_result"] = _make_serializable(crit)

    # if rules passed, optionally run the LLM judge for logical correctness
    if crit["passed"]:
        try:
            judge = llm_judge(state)
        except Exception as e:
            judge = {"passes": True, "reason": f"Judge unavailable ({e}); rules passed.", "suggested_fix": ""}
        if not judge["passes"]:
            crit["passed"] = False
            crit["reasons"].append(f"LLM judge: {judge['reason']}")
            state["critique_result"] = _make_serializable(crit)
        state["suggested_fix"] = judge.get("suggested_fix", "")

    if crit["passed"]:
        state["current_step"] = "validated"
        return state

    if state["revision_count"] >= state["max_revisions"]:
        state["errors"].append(f"Max revisions ({state['max_revisions']}) reached")
        state["current_step"] = "max_revisions_reached"
        return state

    state["revision_count"] += 1
    state["suggested_fix"] = _build_suggested_fix(crit, state)

    llm = create_openrouter_llm(temperature=0.1)
    chain = REVISER_PROMPT | llm | StrOutputParser()
    response = chain.invoke({
        "question": state["question"],
        "analysis_plan": state["analysis_plan"],
        "code": state["generated_code"],
        "execution_result": str(state.get("execution_result") or {}),
        "critique_result": str(_make_serializable(crit)),
        "critique_guidance": _build_suggested_fix(crit, state),
    })
    response = response.strip()
    if response == "ANALYSIS_OK":
        state["current_step"] = "validated"
    else:
        revised = _strip_markdown_fences(response)
        if revised and len(revised) > 10:
            state["generated_code"] = revised
            state["current_step"] = f"revised_{state['revision_count']}"
        else:
            state["errors"].append("Revision failed to produce valid code")
            state["current_step"] = "revision_failed"
    return state


def _build_suggested_fix(crit: Dict[str, Any], state: AnalysisState) -> str:
    parts = []
    for s in (crit.get("suggestions") or []):
        parts.append(s)
    judge_fix = state.get("suggested_fix") or ""
    if judge_fix:
        parts.append(judge_fix)
    return " | ".join(parts) if parts else "Review the code for correctness and fix it."


def synthesizer_agent(state: AnalysisState) -> AnalysisState:
    if state.get("current_step") == "validated":
        cr = state.get("critique_result") or {}
        result_value = cr.get("result_text", "")
        charts = "\n".join(state.get("chart_paths") or []) or "none"
    else:
        result_value = "No validated result available."
        charts = "\n".join(state.get("chart_paths") or []) or "none"
    llm = create_openrouter_llm(temperature=0.3)
    chain = SYNTHESIZER_PROMPT | llm | StrOutputParser()
    state["final_answer"] = chain.invoke({
        "question": state["question"],
        "result_value": result_value[:MAX_FALLBACK_LEN],
        "analysis_plan": state["analysis_plan"],
        "chart_paths": charts,
    }).strip()
    state["current_step"] = "synthesized"
    return state


def should_continue_revision(state: AnalysisState) -> Literal["reviser", "end"]:
    if state["current_step"] in ("validated", "max_revisions_reached", "revision_failed"):
        return "end"
    return "reviser"


def build_analysis_graph(checkpointer=None):
    workflow = StateGraph(AnalysisState)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("coder", coder_agent)
    workflow.add_node("reviser", reviser_agent)
    workflow.add_node("synthesizer", synthesizer_agent)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "coder")
    workflow.add_edge("coder", "reviser")
    workflow.add_conditional_edges("reviser", should_continue_revision, {"reviser": "reviser", "end": "synthesizer"})
    workflow.add_edge("synthesizer", END)
    return workflow.compile(checkpointer=checkpointer)


def run_analysis_workflow(
    csv_path: str,
    question: str,
    max_revisions: int = 3,
    thread_id: Optional[str] = None,
    chat_history: str = "",
) -> AnalysisState:
    checkpointer = MemorySaver()
    graph = build_analysis_graph(checkpointer=checkpointer)
    schema_context = build_schema_context(csv_path)
    if not thread_id:
        thread_id = f"analysis_{datetime.now().timestamp()}"
    initial_state: AnalysisState = {
        "csv_path": csv_path,
        "question": question,
        "schema_context": schema_context,
        "analysis_plan": "",
        "generated_code": "",
        "execution_result": {},
        "critique_result": {},
        "suggested_fix": "",
        "final_answer": "",
        "chart_paths": [],
        "revision_count": 0,
        "max_revisions": max_revisions,
        "current_step": "start",
        "errors": [],
        "chart_kind": "auto",
        "chat_history": chat_history,
    }
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(initial_state, config=config)


def run_and_get_answer(result: AnalysisState) -> Dict[str, Any]:
    """Return a user-facing summary dict from a finished workflow state."""
    validated = result.get("current_step") == "validated"
    return {
        "question": result.get("question"),
        "answer": result.get("final_answer"),
        "validated": validated,
        "revision_count": result.get("revision_count"),
        "current_step": result.get("current_step"),
        "errors": result.get("errors"),
        "charts": result.get("chart_paths"),
        "result_text": (result.get("critique_result") or {}).get("result_text"),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-agent data analysis (cleaned CSV -> answer + charts).")
    parser.add_argument("--csv", default=r"D:\Finance ChatBot\Cleaning_Phase\restaurant_data_cleaned_openrouter.csv")
    parser.add_argument("--question", default="What are the top 5 best-selling dishes by total customers?")
    parser.add_argument("--max-revisions", type=int, default=3)
    # parse_known_args() so this also works when run from a Jupyter notebook via
    # %run, where ipykernel appends its own --f=<connection file> argument.
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("MULTI-AGENT DATA ANALYSIS WORKFLOW (OpenRouter backend)")
    print("=" * 60)

    result = run_analysis_workflow(args.csv, args.question, args.max_revisions)

    print(f"\nFinal Step: {result['current_step']}")
    print(f"Revisions: {result['revision_count']}")
    if result["errors"]:
        print(f"Errors: {result['errors']}")

    print("\n" + "=" * 60)
    print("ANALYSIS PLAN")
    print("=" * 60)
    print(result["analysis_plan"])

    print("\n" + "=" * 60)
    print("GENERATED CODE")
    print("=" * 60)
    print(result["generated_code"])

    print("\n" + "=" * 60)
    print("CRITIQUE RESULT")
    print("=" * 60)
    cr = result.get("critique_result") or {}
    print(f"Passed: {cr.get('passed')}")
    for r in cr.get("reasons") or []:
        print(f"  - {r}")
    if result.get("chart_paths"):
        print("\nCharts produced:")
        for c in result["chart_paths"]:
            print(f"  - {c}")

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result["final_answer"])
