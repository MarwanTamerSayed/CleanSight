import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal, TypedDict
from datetime import datetime

import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

sys.path.append(str(Path(__file__).parent.parent))
from Handling_Data.data_profiler import get_data_profile, profile_to_llm_context


class CleaningState(TypedDict):
    csv_path: str
    llm_context: str
    cleaning_plan: str
    generated_code: str
    execution_result: Dict[str, Any]
    validation_result: Dict[str, Any]
    revision_count: int
    max_revisions: int
    current_step: str
    errors: List[str]


def create_llm(temperature: float = 0.3) -> ChatOllama:
    return ChatOllama(model="qwen2.5:7b", temperature=temperature)


def _make_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to native Python types for serialization."""
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


PLANNER_PROMPT = PromptTemplate.from_template("""
You are an expert data quality and data preprocessing analyst.

Your task is to analyze the provided CSV dataset profile and create a
TARGETED DATA CLEANING STRATEGY.

The cleaning strategy will NOT be executed by you.
A second LLM will use your strategy to generate Pandas code and execute it
on the original DataFrame.

========================
DATASET PROFILE
========================

{llm_context}

========================
YOUR OBJECTIVE
========================

Identify the actual data quality problems present in the dataset and decide
how each problem should be handled.

Do NOT suggest generic cleaning steps unless the dataset profile provides
evidence that they are needed.

Your strategy should consider:

1. Missing values
   - Identify columns with missing values.
   - Decide whether to drop rows, drop columns, or impute values.
   - Choose an appropriate imputation method based on the column type and
     available information.

2. Duplicate records
   - Determine whether duplicate rows exist.
   - Recommend whether they should be removed.

3. Data types
   - Identify columns with incorrect or inconsistent data types.
   - Recommend the correct type and conversion approach.

4. Invalid or inconsistent values
   - Detect values that violate the expected meaning or format of a column.
   - Recommend how they should be corrected, replaced, or removed.

5. Categorical data
   - Identify inconsistent category representations such as different
     capitalization, spacing, abbreviations, or spelling.
   - Recommend normalization only when evidence exists.

6. Numerical data
   - Identify suspicious values, impossible values, or potential outliers.
   - Do NOT automatically remove outliers.
   - Only recommend treatment when the profile provides evidence that they
     are problematic.

7. String/text columns
   - Identify unnecessary whitespace, inconsistent formatting, or malformed
     values when present.

8. Date/time columns
   - Identify columns that should be converted to datetime.
   - Identify inconsistent or invalid date formats.

9. Data integrity
   - Consider relationships between columns when possible.
   - Identify contradictions or logically impossible combinations.

10. Preservation
    - Avoid removing useful information unnecessarily.
    - Preserve the original meaning of the dataset.
    - Prefer the least destructive cleaning operation that solves the problem.

========================
IMPORTANT RULES
========================

- Base every recommendation on evidence from the dataset profile.
- Do not invent problems that are not indicated by the profile.
- Do not generate Pandas code.
- Do not execute anything.
- Do not perform the cleaning yourself.
- Your output is a PLAN for another LLM.
- Be precise about which columns are affected.
- Explain WHY each cleaning operation is required.
- Specify what the second LLM should do, but leave the actual implementation
  to the second LLM.
- If no cleaning is required for a particular category, explicitly say so.
- Be conservative: never recommend dropping data without a clear reason.

========================
OUTPUT FORMAT
========================

Return the following structure:

DATASET_SUMMARY:
- Brief description of the dataset based only on the provided profile.

CLEANING_ACTIONS:

For every required cleaning action, provide:

Action:
Affected Columns:
Problem:
Evidence:
Recommended Treatment:
Reason:
Priority: [HIGH / MEDIUM / LOW]

ORDER_OF_OPERATIONS:
Provide the recommended order in which the cleaning operations should be
performed.

VALIDATION_REQUIREMENTS:
List the checks that the second LLM should perform after cleaning to verify
that the operations were successful and that the dataset was not damaged.

UNNECESSARY_OPERATIONS:
List common cleaning operations that should NOT be performed on this dataset
because there is no evidence that they are necessary.

FINAL_STRATEGY:
Provide a concise step-by-step cleaning strategy that another LLM can directly
translate into Pandas operations.
""")


CODER_PROMPT = PromptTemplate.from_template("""
Generate executable Pandas cleaning code based on the cleaning strategy below.

Rules:
- Use the existing DataFrame named `df`.
- Return ONLY raw Python code. NO markdown fences, NO backticks, NO ```python, NO ```.
- Do not include any comments or explanations.
- Do not invent cleaning operations unsupported by the plan.
- Preserve valid data and avoid unnecessary outlier removal.
- Handle edge cases gracefully (try/except where appropriate)
- Do not re-read the CSV; assume `df` already exists
- Import statements allowed at top if needed
- Use ONLY the concrete column names named by the cleaning plan. Never operate on a
  column the plan does not mention.
- Apply a transformation ONLY to the specific columns it targets. NEVER blanket-loop
  over ALL columns (e.g. `for col in df.columns`) unless the plan explicitly says so.
  In particular:
  - Only convert a column to datetime if the plan identifies it as a date column.
    NEVER run `pd.to_datetime` on text/description columns.
  - Only run `.str.strip()` / `.str.lower()` on columns the plan says to normalize as
    text. If a column was converted to datetime, do not use `.str` on it afterwards.
- CRITICAL: Write GENERIC logic (no hardcoded values). Bounds/stats must be computed
  from the data, but the target COLUMNS come from the plan.
  - DO NOT hardcode specific values (like clip(0, 100), replace({{100: 95}}), etc.)
  - DO NOT assume a specific date format - use pd.to_datetime(col, errors='coerce')
  - Calculate outlier bounds DYNAMICALLY from the data (IQR method)
  - Only drop rows if the plan explicitly says to drop duplicates or missing values
- CRITICAL ROW PRESERVATION: The cleaning must KEEP at least 90% of the original rows.
  - NEVER drop rows to treat outliers. Treat outliers with winsorizing/clipping
    (e.g. df[col] = df[col].clip(lower_bound, upper_bound)), NOT by filtering them out.
  - NEVER call df.dropna() on a datetime column just because some values fail to parse;
    use errors='coerce' and leave the unparsed values as NaT instead of dropping rows.
  - Only drop rows for exactly two reasons and nothing else:
    (a) removing exact duplicate rows, and (b) removing rows missing a required key column
    as explicitly requested by the plan.
  - If a filter would remove more than 10% of rows, do NOT apply it.

CLEANING PLAN:
{Plan}
""")


REVISER_PROMPT = PromptTemplate.from_template("""
You are a code reviewer and data validation expert.

Review the generated Pandas cleaning code and its execution results.
Identify any errors, logical issues, or data quality problems.

ORIGINAL CLEANING PLAN:
{plan}

GENERATED CODE:
{code}

EXECUTION RESULT:
{execution_result}

VALIDATION CHECKS PERFORMED:
{validation_result}

DATASET PROFILE (for reference):
{llm_context}

Check for:
1. Syntax/execution errors
2. Logical errors (e.g., wrong column names, incorrect operations)
3. Data integrity issues (e.g., unintended data loss, NaN proliferation)
4. Whether the code follows the cleaning plan correctly
5. Whether validation requirements are met
6. CRITICAL: Code must be GENERIC - no hardcoded values like clip(0, 100), replace({{100: 95}}), specific date formats. All bounds must be calculated from data.
7. CRITICAL: Code must NOT drop the dataset. Reject any code that:
   - Filters out whole rows to treat outliers (must clip/winsorize instead)
   - Drops every row of a column (df.dropna on dates after errors='coerce')
   - Removes more than 10% of the original rows. If the execution result shows a large
     row decrease or the resulting DataFrame is empty, this is a hard failure.
   - Example of forbidden destructive code: `df = df[(df['col'] > lo) & (df['col'] < hi)]`
   Replace it with clipping: `df['col'] = df['col'].clip(lo, hi)`.
8. CRITICAL: Do not blanket-convert every object column to datetime, and never apply
   `.str` accessors to a column after it was converted to datetime (this raises
   `Can only use .str accessor with string values!`). Only touch the columns the
   cleaning plan names, with the correct operation for each.

If there are issues, provide ONLY the corrected Python code (no markdown, no explanations).
If the code is correct and validation passes, respond with exactly: VALIDATION_PASSED

Output format (choose ONE):
- If validation passes: VALIDATION_PASSED
- If needs revision: provide ONLY the complete corrected Python code
""")


def load_and_profile(state: CleaningState) -> CleaningState:
    """Load CSV and create profile for LLM context."""
    profile = get_data_profile(state["csv_path"])
    state["llm_context"] = profile_to_llm_context(profile)
    state["current_step"] = "profiled"
    return state


def planner_agent(state: CleaningState) -> CleaningState:
    """Agent 1: Create cleaning plan from data profile."""
    llm = create_llm(temperature=0.0)
    chain = PLANNER_PROMPT | llm | StrOutputParser()

    response = chain.invoke({"llm_context": state["llm_context"]})
    state["cleaning_plan"] = response
    state["current_step"] = "planned"
    return state


def _strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences from LLM output."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def coder_agent(state: CleaningState) -> CleaningState:
    """Agent 2: Generate pandas cleaning code from plan."""
    llm = create_llm(temperature=0.0)
    chain = CODER_PROMPT | llm | StrOutputParser()

    response = chain.invoke({"Plan": state["cleaning_plan"]})
    state["generated_code"] = _strip_markdown_fences(response)
    state["current_step"] = "coded"
    return state


def _column_change_summary(before: pd.DataFrame, after: pd.DataFrame) -> Dict[str, Any]:
    """Summarize what changed per column (dtype, missing, numeric range) so the
    user can verify each planned cleaning step actually took effect."""
    summary: Dict[str, Any] = {}
    for col in before.columns:
        entry: Dict[str, Any] = {}
        b, a = before[col], after[col]
        if pd.api.types.is_datetime64_any_dtype(a) and not pd.api.types.is_datetime64_any_dtype(b):
            entry["converted_to"] = "datetime"
        if str(b.dtype) != str(a.dtype):
            entry["dtype"] = f"{b.dtype} -> {a.dtype}"
        if pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(a):
            bn, an = b.dropna(), a.dropna()
            if len(bn) and len(an):
                b_min, b_max = float(bn.min()), float(bn.max())
                a_min, a_max = float(an.min()), float(an.max())
                if (b_min, b_max) != (a_min, a_max):
                    entry["range"] = f"[{b_min:.2f}, {b_max:.2f}] -> [{a_min:.2f}, {a_max:.2f}]"
        if pd.api.types.is_object_dtype(b):
            b_stripped = b.astype(str).str.strip().str.lower()
            if (b.astype(str) != b_stripped).any():
                entry["normalized"] = "string stripped/lowercased"
        summary[col] = entry
    return {k: v for k, v in summary.items() if v}


def _safe_exec(code: str, df: pd.DataFrame):
    """Execute all lines of the generated code and trap mid-block errors.

    ``exec`` runs the whole block top-to-bottom, but it STOPS at the first
    line that raises an exception, so lines below the failing one never run.
    This helper detects exactly which line failed (reported as "error_line",
    1-based, relative to the generated code) and how many statements were
    reached, so partial execution is never mistaken for success.

    Returns ``(result_df, error, error_line, statements_executed)``.
    ``result_df`` is ``None`` on failure. ``statements_executed`` counts every
    statement that was entered (including inside loops), which is used to
    detect top-of-block-only execution.
    """
    local_vars = {"df": df.copy(), "pd": pd, "np": np}
    sys.settrace(_CountingTracer())
    try:
        exec(compile(code, "<generated>", "exec"), {"pd": pd, "np": np}, local_vars)
    except Exception as e:
        # Locate the frame inside the generated code to get the failing line.
        error_line = None
        tb = sys.exc_info()[2]
        while tb is not None:
            if getattr(tb.tb_frame, "f_code", None) is not None and tb.tb_frame.f_code.co_filename == "<generated>":
                error_line = tb.tb_lineno
                break
            tb = tb.tb_next
        sys.settrace(None)
        return None, f"{type(e).__name__}: {e}", error_line, _CountingTracer.count
    sys.settrace(None)
    return local_vars["df"], None, None, _CountingTracer.count


class _CountingTracer:
    """Counts how many statements of the executed code were actually run."""

    count = 0

    def __call__(self, frame, event, arg):
        if event == "line" and frame.f_code.co_filename == "<generated>":
            _CountingTracer.count += 1
        return self


def execute_code(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Execute ALL the generated code on a copy of the dataframe.

    If any line fails, the failing line number is recorded so the Reviser can
    fix it. A low ``statements_executed`` relative to the number of lines is
    reported as a warning so code that only runs its first lines (e.g. an early
    exit or an exception part-way) is flagged instead of silently passing.
    """
    df_copy = df.copy()
    result_df, error, error_line, statements = _safe_exec(code, df)

    if error is not None:
        return {
            "success": False,
            "error": error,
            "error_line": error_line,
            "statements_executed": statements,
            "rows_before": int(len(df)),
            "rows_after": None,
            "columns_before": list(df.columns),
            "columns_after": None,
            "dtypes_before": _make_serializable(df.dtypes.to_dict()),
            "dtypes_after": None,
            "missing_before": _make_serializable(df.isna().sum().to_dict()),
            "missing_after": None,
            "column_changes": {},
            "sample_after": None,
        }

    if result_df is None:
        result_df = df_copy
    return {
        "success": True,
        "error": None,
        "error_line": None,
        "statements_executed": statements,
        "rows_before": int(len(df)),
        "rows_after": int(len(result_df)),
        "columns_before": list(df.columns),
        "columns_after": list(result_df.columns),
        "dtypes_before": _make_serializable(df.dtypes.to_dict()),
        "dtypes_after": _make_serializable(result_df.dtypes.to_dict()),
        "missing_before": _make_serializable(df.isna().sum().to_dict()),
        "missing_after": _make_serializable(result_df.isna().sum().to_dict()),
        "column_changes": _make_serializable(_column_change_summary(df, result_df)),
        "sample_after": _make_serializable(result_df.head(5).to_dict(orient="records")),
    }


def validate_results(execution_result: Dict[str, Any], plan: str, csv_path: str, generated_code: str) -> Dict[str, Any]:
    """Validate the cleaning results against requirements."""
    validation = {
        "passed": True,
        "checks": [],
        "warnings": [],
        "failures": []
    }

    if not execution_result["success"]:
        validation["passed"] = False
        err_line = execution_result.get("error_line")
        loc = f" (line {err_line} of generated code)" if err_line else ""
        validation["failures"].append(f"Execution failed{loc}: {execution_result['error']}")
        return _make_serializable(validation)

    df = pd.read_csv(csv_path)
    # Re-execute to get result_df for validation, trapping mid-block errors.
    result_df, exec_error, error_line, _ = _safe_exec(generated_code, df)
    if exec_error is not None:
        line_txt = f" on line {error_line}" if error_line else ""
        text = "\n".join(
            f"{i + 1:>3} | {l}" for i, l in enumerate(generated_code.splitlines())
        )
        validation["passed"] = False
        validation["failures"].append(
            f"Generated code stopped{line_txt}: {exec_error}\nGenerated code:\n{text}"
        )
        return _make_serializable(validation)

    check_duplicates = int(result_df.duplicated().sum())
    validation["checks"].append(f"Duplicate rows remaining: {check_duplicates}")
    if check_duplicates > 0:
        validation["warnings"].append(f"Still has {check_duplicates} duplicate rows")

    missing_after = result_df.isna().sum()
    missing_before = df.isna().sum()
    total_missing = int(missing_after.sum())
    validation["checks"].append(f"Total missing values after cleaning: {total_missing}")

    # Data-loss guard: if any column gained a large number of NEW missing values
    # (e.g. dates parsed to NaT, or a column wrongly coerced), treat it as a failure.
    newly_missing = missing_after.subtract(missing_before).clip(lower=0)
    bad_cols = newly_missing[newly_missing > len(result_df) * 0.05]
    if not bad_cols.empty:
        detail = _make_serializable(bad_cols.to_dict())
        validation["failures"].append(
            f"Data loss: {len(bad_cols)} column(s) gained >5% new missing values: {detail}. "
            f"Likely a bad date parse (errors='coerce' to NaT) or a faulty conversion."
        )
        validation["passed"] = False

    if "Date" in result_df.columns:
        is_datetime = pd.api.types.is_datetime64_any_dtype(result_df["Date"])
        validation["checks"].append(f"Date column is datetime: {is_datetime}")
        if not is_datetime:
            validation["failures"].append("Date column not converted to datetime")
            validation["passed"] = False
        else:
            # Confirm the conversion actually took effect and did not destroy dates.
            nat_count = int(result_df["Date"].isna().sum())
            validation["checks"].append(f"Date column NaT values: {nat_count}")
            validation["warnings"].append(f"Date column has {nat_count} unparsed (NaT) values")

    row_loss = execution_result["rows_before"] - execution_result["rows_after"]
    validation["checks"].append(f"Rows removed: {row_loss}")
    if execution_result["rows_after"] == 0:
        validation["failures"].append("Cleaning produced an EMPTY DataFrame - all rows were dropped")
        validation["passed"] = False
    elif row_loss > execution_result["rows_before"] * 0.1:
        validation["failures"].append(
            f"Excessive row loss: {row_loss} rows "
            f"({row_loss / execution_result['rows_before'] * 100:.1f}%) - "
            f"expected to keep at least 90% of rows"
        )
        validation["passed"] = False

    if "Price" in result_df.columns:
        price_missing = int(result_df["Price"].isna().sum())
        validation["checks"].append(f"Price missing values: {price_missing}")

    return _make_serializable(validation)


def reviser_agent(state: CleaningState) -> CleaningState:
    """Agent 3: Review code, execute, validate, and revise if needed."""
    df = pd.read_csv(state["csv_path"])

    execution_result = execute_code(state["generated_code"], df)
    state["execution_result"] = _make_serializable(execution_result)

    validation_result = validate_results(execution_result, state["cleaning_plan"], state["csv_path"], state["generated_code"])
    state["validation_result"] = validation_result

    if validation_result["passed"] and execution_result["success"]:
        state["current_step"] = "validated"
        return state

    if state["revision_count"] >= state["max_revisions"]:
        state["errors"].append(f"Max revisions ({state['max_revisions']}) reached")
        state["current_step"] = "max_revisions_reached"
        return state

    state["revision_count"] += 1

    llm = create_llm(temperature=0.1)
    chain = REVISER_PROMPT | llm | StrOutputParser()

    response = chain.invoke({
        "plan": state["cleaning_plan"],
        "code": state["generated_code"],
        "execution_result": str(execution_result),
        "validation_result": str(validation_result),
        "llm_context": state["llm_context"]
    })

    response = response.strip()
    
    if response == "VALIDATION_PASSED":
        state["current_step"] = "validated"
    else:
        revised_code = _strip_markdown_fences(response)
        if revised_code and len(revised_code) > 10:
            state["generated_code"] = revised_code
            state["current_step"] = f"revised_{state['revision_count']}"
        else:
            state["errors"].append("Revision failed to produce valid code")
            state["current_step"] = "revision_failed"

    return state


def should_continue_revision(state: CleaningState) -> Literal["reviser", "end"]:
    """Route to reviser if validation failed and revisions remain."""
    if state["current_step"] == "validated":
        return "end"
    if state["current_step"] == "max_revisions_reached":
        return "end"
    if state["current_step"] == "revision_failed":
        return "end"
    if state["current_step"].startswith("revised"):
        return "reviser"
    return "reviser"


def build_cleaning_graph():
    """Build the LangGraph multi-agent workflow."""
    workflow = StateGraph(CleaningState)

    workflow.add_node("load_and_profile", load_and_profile)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("coder", coder_agent)
    workflow.add_node("reviser", reviser_agent)

    workflow.set_entry_point("load_and_profile")

    workflow.add_edge("load_and_profile", "planner")
    workflow.add_edge("planner", "coder")
    workflow.add_edge("coder", "reviser")

    workflow.add_conditional_edges(
        "reviser",
        should_continue_revision,
        {
            "reviser": "reviser",
            "end": END
        }
    )

    return workflow.compile()


def run_cleaning_workflow(csv_path: str, max_revisions: int = 3) -> CleaningState:
    """Run the complete multi-agent cleaning workflow."""
    graph = build_cleaning_graph()

    initial_state: CleaningState = {
        "csv_path": csv_path,
        "llm_context": "",
        "cleaning_plan": "",
        "generated_code": "",
        "execution_result": {},
        "validation_result": {},
        "revision_count": 0,
        "max_revisions": max_revisions,
        "current_step": "start",
        "errors": []
    }

    config = {"configurable": {"thread_id": f"cleaning_{datetime.now().timestamp()}"}}

    final_state = graph.invoke(initial_state, config=config)
    return final_state


def run_and_save_cleaned_data(
    csv_path: str,
    output_path: str,
    result: Optional[CleaningState] = None,
    max_revisions: int = 3,
    min_row_preserve_ratio: float = 0.8,
) -> pd.DataFrame:
    """Save cleaned data to CSV from an already-validated workflow result.

    Never re-runs the workflow internally: it reuses the validated
    ``generated_code`` from ``result`` (or runs the workflow once if a result
    is not supplied). Before writing, it enforces a row-preservation guard so
    that a destructive cleaning step can never silently wipe the dataset.
    """
    if result is None:
        result = run_cleaning_workflow(csv_path, max_revisions)

    code = result.get("generated_code", "")
    if not code:
        raise RuntimeError("Workflow produced no cleaning code; aborting save.")

    df = pd.read_csv(csv_path)
    rows_before = len(df)
    cleaned_df, exec_error, error_line, _ = _safe_exec(code, df)
    if exec_error is not None:
        loc = f" on line {error_line} of generated code" if error_line else ""
        raise RuntimeError(
            f"Generated code did not run to completion{loc}: {exec_error}. "
            f"Refusing to save a partially-cleaned file."
        )

    rows_after = len(cleaned_df)

    # Hard safety guard: never silently drop most of the data.
    if rows_after < rows_before * min_row_preserve_ratio:
        raise RuntimeError(
            f"Cleaning would drop too many rows ({rows_before} -> {rows_after}, "
            f"{100 * (1 - rows_after / rows_before if rows_before else 0):.1f}% lost). "
            f"Aborting save to protect the dataset and avoiding writing an empty file."
        )

    cleaned_df.to_csv(output_path, index=False)
    return cleaned_df


if __name__ == "__main__":
    csv_path = r"D:\Finance ChatBot\Handling_Data\restaurant_data.csv"
    output_path = r"D:\Finance ChatBot\Cleaning_Phase\restaurant_data_cleaned.csv"

    print("=" * 60)
    print("MULTI-AGENT DATA CLEANING WORKFLOW")
    print("=" * 60)

    result = run_cleaning_workflow(csv_path, max_revisions=3)

    print(f"\nFinal Step: {result['current_step']}")
    print(f"Revisions: {result['revision_count']}")

    if result["errors"]:
        print(f"\nErrors: {result['errors']}")

    print("\n" + "=" * 60)
    print("CLEANING PLAN")
    print("=" * 60)
    print(result["cleaning_plan"])

    print("\n" + "=" * 60)
    print("GENERATED CODE")
    print("=" * 60)
    print(result["generated_code"])

    print("\n" + "=" * 60)
    print("EXECUTION RESULT")
    print("=" * 60)
    exec_result = result["execution_result"]
    print(f"Success: {exec_result.get('success')}")
    print(f"Rows: {exec_result.get('rows_before')} -> {exec_result.get('rows_after')}")
    print(f"Error: {exec_result.get('error')}")
    print("\nPer-column changes applied (verifies datetime/outlier/text steps):")
    changes = exec_result.get("column_changes") or {}
    if changes:
        for col, detail in changes.items():
            print(f"  - {col}: {detail}")
    else:
        print("  (none detected)")

    print("\n" + "=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)
    val_result = result["validation_result"]
    print(f"Passed: {val_result.get('passed')}")
    print(f"Checks: {val_result.get('checks')}")
    print(f"Warnings: {val_result.get('warnings')}")
    print(f"Failures: {val_result.get('failures')}")

    # Save cleaned data (reuses the validated result above, does NOT re-run)
    print("\n" + "=" * 60)
    print("SAVING CLEANED DATA")
    print("=" * 60)
    cleaned_df = run_and_save_cleaned_data(csv_path, output_path, result=result, min_row_preserve_ratio=0.8)
    print(f"Saved to: {output_path}")
    print(f"Shape: {cleaned_df.shape}")
    print(f"Dtypes:\n{cleaned_df.dtypes}")
    print(f"Missing:\n{cleaned_df.isna().sum()}")