"""
sandbox.py — Hardened code execution layer.

Two-layer defence:
  1. AST analysis  — catches imports, attribute access, dangerous calls
                     regardless of encoding tricks or obfuscation.
  2. Pattern guard — fast string scan as secondary check.
  3. exec() with __builtins__:{} — empties the built-in namespace.
"""

import ast

# ── Layer 2: string-level blocklist ──────────────────────────────────────────
BLOCKED_PATTERNS = [
    "while True", "while 1", "for(;;)",
    "open(", "eval(", "exec(",
    "os.", "sys.", "subprocess", "shutil",
    "socket", "requests", "urllib", "pathlib",
    "globals()", "locals()",
    "getattr", "setattr", "delattr",
]

BLOCKED_NAMES = {
    "open", "eval", "exec", "compile", "input",
    "__import__", "importlib", "breakpoint",
    "memoryview", "vars", "dir",
}

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket",
    "requests", "urllib", "pathlib", "importlib",
    "ctypes", "pickle", "shelve", "multiprocessing",
    "threading", "signal", "pty", "fcntl",
}


# ── Layer 1: AST analysis ─────────────────────────────────────────────────────
def is_code_safe_ast(code: str) -> tuple[bool, str]:
    """
    Parse the code into an AST and walk every node.
    Catches imports, dangerous calls, and blocked module attribute access.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""  # let exec() surface the syntax error

    for node in ast.walk(tree):
        # Block ALL import statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                return False, f"import '{name}' blocked"

        # Block calls to dangerous built-in names
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_NAMES:
                    return False, f"call to '{node.func.id}' blocked"
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in BLOCKED_MODULES:
                        return False, f"module access '{node.func.value.id}.{node.func.attr}' blocked"

        # Block attribute access on blocked modules
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in BLOCKED_MODULES:
                    return False, f"attribute access '{node.value.id}.{node.attr}' blocked"

        # Block Name references to dangerous identifiers
        if isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES:
                return False, f"name '{node.id}' blocked"

    return True, ""


# ── Layer 2: string-level pattern guard ───────────────────────────────────────
def is_code_safe_patterns(code: str) -> tuple[bool, str]:
    if "__" in code:
        return False, "__ (dunder) usage blocked"
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return False, f"pattern '{pattern}' blocked"
    return True, ""


# ── Combined guard ────────────────────────────────────────────────────────────
def is_code_safe(code: str) -> tuple[bool, str]:
    """Run both layers. Both must pass."""
    safe, reason = is_code_safe_ast(code)
    if not safe:
        return False, reason
    return is_code_safe_patterns(code)


# ── Hardened exec ─────────────────────────────────────────────────────────────
def safe_exec(code: str, local_vars: dict) -> tuple[bool, str]:
    """
    Execute code in a hardened sandbox.
    - Runs both safety layers before exec.
    - Passes __builtins__:{} to empty the built-in namespace.
    """
    safe, reason = is_code_safe(code)
    if not safe:
        return False, f"Security block: {reason}"
    try:
        exec(code, {"__builtins__": {}}, local_vars)
        return True, ""
    except Exception as e:
        return False, str(e)
