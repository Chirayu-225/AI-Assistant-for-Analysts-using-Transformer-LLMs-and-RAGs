"""
profiler.py — Dataset profiling and insight prompt generation.

build_dataset_profile()     → structured dict (cached)
format_profile_for_llm()    → LLM-readable string
build_insight_prompt()      → context-aware prompt: result + full dataset
build_auto_analyze_prompt() → proactive dataset intelligence prompt
build_explain_prompt()      → drill-down explanation prompt
"""

import pandas as pd
import numpy as np
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PROFILER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_dataset_profile(dataframe: pd.DataFrame, _cache_key: str = "") -> dict:
    """
    Compute a structured statistical profile. Cached by Streamlit.
    Pass _cache_key="cleaned" after cleaning to force a fresh profile
    and prevent stale raw-data stats being used for insights.
    Expensive ops use a 5000-row sample for large datasets.
    """
    profile: dict = {
        "rows":    dataframe.shape[0],
        "columns": dataframe.shape[1],
    }

    sample_df = (
        dataframe.sample(min(len(dataframe), 5000), random_state=42)
        if len(dataframe) > 5000 else dataframe
    )

    # Nulls — full dataset (cheap)
    null_counts = dataframe.isnull().sum()
    profile["nulls"] = {col: int(n) for col, n in null_counts.items() if n > 0}
    profile["null_pct"] = {
        col: round(n / len(dataframe) * 100, 1)
        for col, n in profile["nulls"].items()
    }

    # Numeric stats — sample
    num_df = sample_df.select_dtypes(include="number")
    if not num_df.empty:
        profile["numeric_stats"] = num_df.describe().round(2).to_dict()

    # Top categories — sample, cap at 6 cols
    cat_cols = sample_df.select_dtypes(include=["object", "category"]).columns
    profile["top_categories"] = {
        col: sample_df[col].value_counts().head(5).to_dict()
        for col in cat_cols[:6]
    }

    # Strong correlations — sample
    if len(num_df.columns) >= 2:
        corr  = num_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        strong = (
            upper.stack()
            .reset_index()
            .rename(columns={"level_0": "col_a", "level_1": "col_b", 0: "corr"})
            .query("corr > 0.5")
            .sort_values("corr", ascending=False)
            .head(5)
        )
        profile["strong_correlations"] = [
            {"col_a": r.col_a, "col_b": r.col_b, "corr": round(r.corr, 3)}
            for r in strong.itertuples()
        ]

    # Outliers via IQR — sample
    outliers: dict = {}
    for col in num_df.columns:
        q1  = sample_df[col].quantile(0.25)
        q3  = sample_df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            n_out = sample_df[
                (sample_df[col] < q1 - 1.5 * iqr) |
                (sample_df[col] > q3 + 1.5 * iqr)
            ].shape[0]
            if n_out > 0:
                outliers[col] = n_out
    profile["outliers"] = outliers

    return profile


def format_profile_for_llm(profile: dict) -> str:
    lines = [f"Dataset: {profile['rows']:,} rows × {profile['columns']} columns"]

    if profile.get("nulls"):
        null_str = ", ".join(
            f"{col} ({pct}%)" for col, pct in profile.get("null_pct", {}).items()
        )
        lines.append(f"Missing values: {null_str}")
    else:
        lines.append("Missing values: none")

    if profile.get("numeric_stats"):
        lines.append("\nNumeric column stats (mean / std / min / max):")
        for col, stats in profile["numeric_stats"].items():
            lines.append(
                f"  {col}: mean={stats.get('mean','?')}  "
                f"std={stats.get('std','?')}  "
                f"min={stats.get('min','?')}  "
                f"max={stats.get('max','?')}"
            )

    if profile.get("top_categories"):
        lines.append("\nTop category values:")
        for col, vals in profile["top_categories"].items():
            top = ", ".join(f"{k}({v})" for k, v in list(vals.items())[:3])
            lines.append(f"  {col}: {top}")

    if profile.get("strong_correlations"):
        lines.append("\nStrong correlations (>0.5):")
        for c in profile["strong_correlations"]:
            lines.append(f"  {c['col_a']} ↔ {c['col_b']}: {c['corr']}")

    if profile.get("outliers"):
        lines.append("\nOutliers detected (IQR method):")
        for col, n in profile["outliers"].items():
            lines.append(f"  {col}: {n} outlier(s)")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# INSIGHT PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def build_insight_prompt(user_query: str, result_profile_str: str,
                         dataset_profile_str: str) -> str:
    """
    Context-aware: passes BOTH the result profile AND the full dataset
    profile so the model can compare the result against broader context.
    Strict actionability rules — no data quality complaints, no raw descriptions.
    """
    return f"""You are a senior business analyst presenting findings to a CEO.
The data has already been cleaned. Do NOT mention nulls, missing values, data types,
or data quality. The audience does not care about the data — they care about the business.

A user asked: "{user_query}"

ANALYSIS RESULT:
{result_profile_str}

FULL DATASET CONTEXT:
{dataset_profile_str}

Write 3 to 5 BUSINESS insights ranked by impact. Each insight must:
1. State a specific, quantified business finding (revenue, volume, rank, gap, trend)
2. Immediately follow it with a concrete business action

CORRECT examples:
1• Classic Cars generates 39% of total revenue but only 18% of order volume — increase unit price or upsell accessories to this segment
2• Motorcycles have the lowest average order value ($950) vs the dataset mean ($3,200) — investigate pricing strategy or target higher-value customers
3• Q4 sales are 2.3× higher than Q1 — align inventory procurement and staffing to seasonal demand pattern

WRONG examples (do NOT write these):
✗ The dataset contains 2,823 rows and 25 columns
✗ The sales column has 81 outliers detected using IQR
✗ There are missing values in addressline2
✗ The data shows that Classic Cars is the most common category

SELF-CHECK before writing each insight: ask yourself —
"Would a CEO find this useful for making a decision?" If no, rewrite it.
"Does it tell them what to DO, not just what IS?" If no, add the action.
"Does it mention the data structure rather than the business?" If yes, delete it.

Format: 1• [quantified business finding] — [specific action to take]
"""


def build_auto_analyze_prompt(dataset_profile_str: str) -> str:
    return f"""You are a senior business analyst presenting to a CEO for the first time.
The data has already been cleaned and standardised. Do NOT mention nulls, missing values,
data types, or anything about data quality. Focus entirely on business findings.

DATASET PROFILE:
{dataset_profile_str}

Write the 5 most impactful BUSINESS insights from this dataset, ranked by business value.

Each insight must answer: "What does this mean for the business, and what should we do?"

CORRECT examples:
1• Classic Cars(967 orders) outsells Motorcycles(331) by 3× — concentrate sales and marketing resources on this category
2• The top 2 product lines contribute over 65% of order volume — diversification risk is high; develop the bottom 3 lines
3• Sales std deviation ($1.2M) is 40% of the mean — highly inconsistent performance; identify what drives the top-quartile transactions

WRONG examples (never write these):
✗ The dataset has 2,823 rows
✗ The status column shows a skewed distribution
✗ There are missing values in the territory column
✗ The quantityordered column has 8 outliers

SELF-CHECK: Every insight must name a specific business implication and a specific action.
If an insight only describes the data without saying what to do about it — rewrite it.

Format: 1• [quantified business finding] — [specific action]
"""


def build_explain_prompt(insight: str, user_query: str) -> str:
    return f"""You are a senior business analyst explaining a finding to a decision-maker.

Insight to explain: {insight}

This came from a dataset analysis where the user asked: "{user_query}"

Explain this in business terms. Cover exactly these four points:

1. Why this pattern likely exists — what business dynamics or behaviours cause it
2. What the business risk or opportunity is — quantify if possible
3. What a decision-maker should do — be specific about the action, team, or budget involved
4. What could make this insight wrong — data limitations, seasonality, confounders

Be direct. No filler. Plain English. No mentions of data cleaning or data structure.
"""
