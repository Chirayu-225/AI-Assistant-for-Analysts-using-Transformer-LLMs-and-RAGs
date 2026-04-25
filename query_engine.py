"""
query_engine.py — Query routing, prompt construction, and LLM calls.

Architecture:
  1. Shortcut layer  — handles simple queries with direct pandas (no LLM).
  2. Prompt builder  — constructs a hardened, context-aware prompt.
  3. LLM call        — cached wrapper around ollama.chat.
  4. Code cleanup    — strips imports, plt.show(), adds plt.gcf().
"""

import re
import pandas as pd
import ollama
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# 1. SHORTCUT LAYER
# ─────────────────────────────────────────────────────────────────────────────

def detect_shortcut(query: str, df: pd.DataFrame) -> str | None:
    """
    Return a pandas one-liner for simple, unambiguous queries.
    Returns None if the query should go to the LLM.
    """
    q    = query.lower().strip()
    cols = df.columns.tolist()

    def _find_col(*keywords) -> str | None:
        for kw in keywords:
            for col in cols:
                if kw in col.lower():
                    return col
        return None

    # Row / column counts
    if any(p in q for p in ["how many rows", "count rows", "number of rows", "row count"]):
        return "result = len(df)"
    if any(p in q for p in ["how many columns", "count columns", "number of columns", "column count"]):
        return "result = len(df.columns)"

    # Column listing
    if any(p in q for p in ["show columns", "list columns", "column names", "what columns", "show all columns"]):
        return "result = pd.DataFrame({'column': df.columns.tolist(), 'dtype': [str(d) for d in df.dtypes.values]})"

    # Null / missing analysis
    if any(p in q for p in ["null", "missing", "nan", "empty values"]):
        return "result = df.isnull().sum().rename('null_count').to_frame()"

    # Descriptive statistics
    if any(p in q for p in ["describe", "summary stats", "statistical summary", "statistics"]):
        return "result = df.describe(include='all')"

    # Head / tail
    if any(p in q for p in ["first 5", "first five", "show first", "head"]):
        n = 10 if ("10" in q or "ten" in q) else 5
        return f"result = df.head({n})"
    if any(p in q for p in ["last 5", "last five", "show last", "tail"]):
        n = 10 if ("10" in q or "ten" in q) else 5
        return f"result = df.tail({n})"

    # Duplicates
    if any(p in q for p in ["duplicate", "duplicates", "duplicate rows"]):
        return "result = df.duplicated().sum()"

    # Shape
    if any(p in q for p in ["shape", "dimensions", "size of"]):
        return "result = pd.DataFrame({'rows': [len(df)], 'columns': [len(df.columns)]})"

    # Data types
    if any(p in q for p in ["data types", "dtypes", "column types"]):
        return "result = df.dtypes.rename('dtype').to_frame()"

    # Mean / average of a specific column
    if any(p in q for p in ["mean of", "average of", "avg of"]):
        col = _find_col(*q.split())
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return f"result = df['{col}'].mean()"

    # Sum of a specific column
    if "sum of" in q or ("total" in q and "total rows" not in q):
        col = _find_col(*q.split())
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return f"result = df['{col}'].sum()"

    # Max / min
    if any(p in q for p in ["maximum", "max of", "highest"]):
        col = _find_col(*q.split())
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return f"result = df['{col}'].max()"
    if any(p in q for p in ["minimum", "min of", "lowest"]):
        col = _find_col(*q.split())
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return f"result = df['{col}'].min()"

    # Unique / value counts
    if any(p in q for p in ["unique values", "distinct values", "value counts"]):
        col = _find_col(*q.split())
        if col:
            return f"result = df['{col}'].value_counts().to_frame('count')"

    # Sort
    if any(p in q for p in ["sort by", "order by", "sorted by"]):
        col = _find_col(*q.split())
        ascending = "ascending" in q or "asc" in q
        if col:
            return f"result = df.sort_values('{col}', ascending={ascending}).head(20)"

    # Correlation
    if any(p in q for p in ["correlation", "corr matrix", "correlations"]):
        return "result = df.select_dtypes(include='number').corr().round(3)"

    return None  # → send to LLM


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_prompt(schema_text: str, user_query: str, history_text: str = "") -> str:
    return f"""You are an expert Python data analyst.

Dataset information:
DataFrame name: df
Exact column names (use ONLY these, do not guess or invent):
{schema_text}
{history_text}
User request:
{user_query}

Available libraries (already imported, do NOT import anything):
- pandas as pd
- numpy as np
- plotly.express as px
- plotly.graph_objects as go
- matplotlib.pyplot as plt (fallback only)

Rules:
- Generate ONLY valid Python code. No explanations, no markdown, no imports.
- CRITICAL: ONLY use column names from the exact list above. Do NOT guess column names.
- CRITICAL: Before using a column, verify it exists: if 'col_name' not in df.columns: raise ValueError('Column not found')
- IF calculating a number or table, assign it to a variable named 'result'.
- IF drawing ANY chart or visualisation:
    * PREFER plotly.express (px) or plotly.graph_objects (go) for interactive charts.
    * For bar charts use: result = px.bar(df, x='col', y='col')
    * For line charts use: result = px.line(df, x='col', y='col')
    * For scatter plots use: result = px.scatter(df, x='col', y='col')
    * For pie charts use: result = px.pie(df, names='col', values='col')
    * For histograms use: result = px.histogram(df, x='col')
    * For heatmaps use: result = px.imshow(df.select_dtypes(include='number').corr())
    * Assign the figure directly to result — do NOT call .show()
    * If using matplotlib, the LAST line MUST be: result = plt.gcf() and NEVER use plt.show()
"""


def build_retry_prompt(code: str, error: str) -> str:
    return f"""The following Python code failed with error: {error}

Failing code:
{code}

Fix ONLY the error. Do not rewrite everything. Keep the same logic.
Output ONLY the corrected Python code block. No explanations.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3. CACHED LLM CALL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def cached_llm_call(model: str, prompt_text: str, temperature: float) -> str:
    """Cached wrapper — identical (model, prompt, temp) returns instantly."""
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        options={"temperature": temperature},
    )
    return resp["message"]["content"]


def llm_with_fallback(prompt_text: str, temperature: float = 0.1,
                      primary: str = "qwen2.5-coder:3b") -> str:
    """Try primary model; on any exception returns empty string."""
    try:
        return cached_llm_call(primary, prompt_text, temperature)
    except Exception:
        return cached_llm_call("qwen2.5-coder:3b", prompt_text, temperature)


def insight_llm(prompt_text: str, temperature: float = 0.3) -> str:
    """Insight calls prefer the reasoning model, fallback to coder."""
    try:
        resp = ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt_text}],
            options={"temperature": temperature},
        )
        return resp["message"]["content"]
    except Exception:
        resp = ollama.chat(
            model="qwen2.5-coder:3b",
            messages=[{"role": "user", "content": prompt_text}],
            options={"temperature": temperature},
        )
        return resp["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. CODE POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_clean_code(raw_output: str) -> str:
    """
    Extract code block from LLM output.
    - Strips import statements (libraries are pre-injected into local_vars).
    - Strips plt.show() calls.
    - For matplotlib plots: auto-appends result = plt.gcf() if missing.
    - For Plotly figures: leaves result assignment untouched.
    """
    match = re.search(r"\x60\x60\x60(?:python)?\s*(.*?)\x60\x60\x60", raw_output, re.DOTALL)
    code  = match.group(1).strip() if match else raw_output.strip()
    code  = "\n".join(
        line for line in code.splitlines()
        if not line.strip().startswith("import ")
        and not line.strip().startswith("from ")
    )
    code = code.replace("plt.show()", "")
    code = code.replace(".show()", "")   # catch fig.show() from Plotly

    # Only auto-append plt.gcf() for matplotlib — not for Plotly
    uses_plotly = "px." in code or "go." in code or "plotly" in code.lower()
    uses_mpl    = "plt." in code
    has_result  = "result = plt.gcf()" in code or "result=plt.gcf()" in code

    if uses_mpl and not uses_plotly and not has_result:
        code += "\nresult = plt.gcf()"

    return code
