"""
cleaning.py — Deterministic rule-based data cleaning pipeline.

Runs before the LLM pass. Handles all known, pattern-matchable
data quality issues without any LLM involvement.
"""

import pandas as pd
import numpy as np


def rule_based_clean(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply deterministic cleaning steps in sequence.
    Returns (cleaned_df, log_of_actions).
    """
    df_c = raw_df.copy()
    log  = []

    # ── 1. Promote row 0 to header if columns look auto-generated ─────────────
    first_row = df_c.iloc[0]
    cols_are_integers = all(str(c).strip().lstrip("-").isdigit() for c in df_c.columns)
    cols_are_unnamed  = all(str(c).startswith("Unnamed") for c in df_c.columns)
    row0_looks_like_header = (
        first_row.apply(
            lambda v: isinstance(v, str)
            and not str(v).replace(".", "", 1).replace("-", "", 1).isdigit()
        ).sum()
        >= len(df_c.columns) * 0.6
    )
    if (cols_are_integers or cols_are_unnamed) and row0_looks_like_header:
        df_c.columns = [str(v).strip() for v in df_c.iloc[0]]
        df_c = df_c.iloc[1:].reset_index(drop=True)
        log.append("✓ Promoted row 0 to column headers")

    # ── 2. Normalise column names ──────────────────────────────────────────────
    original_cols = df_c.columns.tolist()
    df_c.columns = [str(c).strip().lower().replace(" ", "_") for c in df_c.columns]
    if df_c.columns.tolist() != original_cols:
        log.append("✓ Normalised column names (lowercase + underscores)")

    # ── 3. Drop fully empty columns and rows ───────────────────────────────────
    before_cols = df_c.shape[1]
    df_c.dropna(axis=1, how="all", inplace=True)
    if df_c.shape[1] < before_cols:
        log.append(f"✓ Dropped {before_cols - df_c.shape[1]} fully-empty column(s)")

    before_rows = df_c.shape[0]
    df_c.dropna(axis=0, how="all", inplace=True)
    if df_c.shape[0] < before_rows:
        log.append(f"✓ Dropped {before_rows - df_c.shape[0]} fully-empty row(s)")

    # ── 4. Drop exact duplicate rows ──────────────────────────────────────────
    before_rows = df_c.shape[0]
    df_c.drop_duplicates(inplace=True)
    if df_c.shape[0] < before_rows:
        log.append(f"✓ Removed {before_rows - df_c.shape[0]} duplicate row(s)")

    # ── 5. Per-column type cleaning ───────────────────────────────────────────
    currency_re = r"[\$£€₹¥₩₺₽]"

    def _is_string_col(series):
        return (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        )

    for col in df_c.columns:
        if not _is_string_col(df_c[col]):
            continue

        series = df_c[col].astype(str).str.strip()

        # 5a. Parenthetical negatives: (1000) → -1000
        paren_mask = series.str.match(r"^\([\d,\.]+\)$", na=False)
        if paren_mask.any():
            series = series.str.replace(r"\((.*?)\)", r"-\1", regex=True)
            log.append(f"✓ Converted parenthetical negatives in '{col}'")

        # 5b. Currency
        currency_mask = series.str.contains(currency_re, regex=True, na=False)
        if currency_mask.sum() > len(series) * 0.4:
            cleaned = (
                series.str.replace(currency_re, "", regex=True)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            parsed = pd.to_numeric(cleaned, errors="coerce")
            if parsed.notna().sum() > currency_mask.sum() * 0.5:
                df_c[col] = parsed
                log.append(f"✓ Parsed currency values in '{col}'")
            continue

        # 5c. Percentages
        pct_mask = series.str.endswith("%", na=False)
        if pct_mask.sum() > len(series) * 0.4:
            cleaned = series.str.replace("%", "", regex=False).str.strip()
            parsed  = pd.to_numeric(cleaned, errors="coerce")
            if parsed.notna().sum() > pct_mask.sum() * 0.5:
                df_c[col] = parsed / 100.0
                log.append(f"✓ Converted percentage values in '{col}' (÷100)")
            continue

        # 5d. Thousands separators
        comma_mask = series.str.match(r"^-?[\d,]+\.?\d*$", na=False)
        if comma_mask.sum() > len(series) * 0.4:
            cleaned = series.str.replace(",", "", regex=False)
            parsed  = pd.to_numeric(cleaned, errors="coerce")
            if parsed.notna().sum() > comma_mask.sum() * 0.5:
                df_c[col] = parsed
                log.append(f"✓ Removed thousands separators in '{col}'")
            continue

        # 5e. Datetime
        parsed_date = pd.to_datetime(series, errors="coerce")
        if parsed_date.notna().sum() > len(series) * 0.6:
            df_c[col] = parsed_date
            log.append(f"✓ Parsed datetime in '{col}'")
            continue

        # 5f. Plain numeric strings
        parsed = pd.to_numeric(series, errors="coerce")
        non_null = series.replace("nan", pd.NA).dropna()
        if len(non_null) > 0 and parsed.notna().sum() / len(non_null) > 0.7:
            df_c[col] = parsed
            log.append(f"✓ Converted '{col}' to numeric")

    # ── 6. Strip whitespace from remaining strings (preserve real NaN) ─────────
    for col in df_c.columns:
        if _is_string_col(df_c[col]):
            mask = df_c[col].notna()
            df_c.loc[mask, col] = df_c.loc[mask, col].astype(str).str.strip()

    # ── 7. Replace sentinel nulls ─────────────────────────────────────────────
    sentinel_strings = {"nan", "none", "null", "n/a", "na", "--", "-", ""}
    sentinel_numbers = [-999, -9999, 999999, 9999999]

    for col in df_c.columns:
        if _is_string_col(df_c[col]):
            lower = df_c[col].str.lower().str.strip()
            mask  = lower.isin(sentinel_strings)
            if mask.any():
                df_c.loc[mask, col] = np.nan
                log.append(f"✓ Replaced {mask.sum()} sentinel string null(s) in '{col}'")
        elif pd.api.types.is_numeric_dtype(df_c[col]):
            mask = df_c[col].isin(sentinel_numbers)
            if mask.any():
                df_c.loc[mask, col] = np.nan
                log.append(f"✓ Replaced {mask.sum()} sentinel numeric null(s) in '{col}'")

    df_c = df_c.reset_index(drop=True)
    if not log:
        log.append("✓ No issues detected — dataset looks clean")
    return df_c, log
