"""
Prompt and code transforms for evaluation experiments.

Provides deterministic, pluggable transforms that can be applied to:
- Input prompts (prompt_transform): applied to the user prompt before sending to the LLM
- Generated code (code_transform): applied to failed code before embedding in error feedback

All transforms are pure functions with signature (str) -> str.
"""

from __future__ import annotations

import re

# Optional dependency — gracefully degrade to regex fallback
try:
    import python_minifier
    _HAS_PYTHON_MINIFIER = True
except ImportError:
    _HAS_PYTHON_MINIFIER = False


# ---------------------------------------------------------------------------
# Prompt transforms
# ---------------------------------------------------------------------------

def strip_whitespace(text: str) -> str:
    """
    Remove blank lines, collapse runs of internal whitespace, strip trailing spaces.

    Safe for mixed natural-language + code content (preserves leading indentation).
    This is a lossless transformation: no semantic content is removed.
    """
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        if stripped:
            match = re.match(r'^(\s*)', stripped)
            indent = match.group(1) if match else ""
            body = stripped[len(indent):]
            body = re.sub(r'  +', ' ', body)
            out.append(indent + body)
    return "\n".join(out)


def minify_prompt(text: str) -> str:
    """
    Minify a Python prompt while preserving docstrings.

    Designed for HumanEval-style prompts (function signature + docstring).
    Removes blank lines, comments, and collapses whitespace in code lines,
    but leaves the docstring intact since it contains the problem specification
    (description, examples, constraints).

    Args:
        text: Python code prompt containing a function signature with docstring.

    Returns:
        Minified prompt with docstring preserved.
    """
    if not text or not text.strip():
        return text

    lines = text.splitlines()
    out: list[str] = []
    in_docstring = False
    docstring_delim: str | None = None

    for line in lines:
        stripped = line.strip()

        # Track docstring boundaries
        if not in_docstring:
            # Check for docstring opening
            for delim in ('"""', "'''"):
                if delim in stripped:
                    idx = stripped.index(delim)
                    # Check if docstring opens and closes on the same line
                    rest = stripped[idx + 3:]
                    if delim in rest:
                        # Single-line docstring — preserve entire line
                        out.append(line.rstrip())
                        break
                    else:
                        in_docstring = True
                        docstring_delim = delim
                        out.append(line.rstrip())
                        break
            else:
                # Not a docstring line — apply minification
                if not stripped:
                    continue  # Drop blank lines
                # Strip comments (but not inside strings)
                if '#' in line:
                    line = _strip_comment(line)
                    if not line.strip():
                        continue
                # Collapse internal whitespace runs (preserve indentation)
                match = re.match(r'^(\s*)', line)
                indent = match.group(1) if match else ""
                body = line[len(indent):].rstrip()
                body = re.sub(r'  +', ' ', body)
                out.append(indent + body)
        else:
            # Inside docstring — preserve verbatim
            out.append(line.rstrip())
            if docstring_delim and docstring_delim in stripped:
                # Check it's actually closing (not the opening line again)
                if stripped.endswith(docstring_delim):
                    in_docstring = False
                    docstring_delim = None

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Code transforms
# ---------------------------------------------------------------------------

def minify_python(code: str) -> str:
    """
    Minify Python code: remove comments, docstrings, collapse whitespace.

    Uses python-minifier if available (production-grade AST-based minification),
    otherwise falls back to conservative regex-based minification.

    Safe options: no identifier renaming, no literal hoisting. The minified
    code is semantically equivalent to the original.

    Intended use: minify LLM-generated code before embedding it in error
    feedback prompts, reducing token growth on retries.

    Args:
        code: Python source code to minify.

    Returns:
        Minified Python code. Returns original code if minification fails.
    """
    if not code or not code.strip():
        return code

    if _HAS_PYTHON_MINIFIER:
        try:
            return python_minifier.minify(
                code,
                rename_locals=False,
                rename_globals=False,
                remove_literal_statements=True,
                hoist_literals=False,
                remove_annotations=True,
                combine_imports=True,
                remove_pass=True,
                remove_object_base=True,
            )
        except Exception:
            pass

    # Fallback: regex-based minification (less aggressive but safer)
    return _minify_python_regex(code)


def _minify_python_regex(code: str) -> str:
    """Regex-based Python minification fallback."""
    try:
        result = code

        # Remove single-line comments (preserve shebang and encoding)
        lines = result.split('\n')
        processed: list[str] = []
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if i < 2 and (stripped.startswith('#!') or 'coding' in stripped[:30]):
                processed.append(line)
            elif '#' in line:
                processed.append(_strip_comment(line))
            else:
                processed.append(line)

        result = '\n'.join(processed)

        # Remove module-level docstrings
        result = re.sub(r'^("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')\s*\n', '', result)

        # Collapse multiple blank lines
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)

        # Remove trailing whitespace per line
        result = '\n'.join(line.rstrip() for line in result.split('\n'))

        return result.strip()
    except Exception:
        return code


def _strip_comment(line: str) -> str:
    """Strip trailing comment from a Python line, respecting string literals."""
    in_string = False
    string_char = None
    j = 0
    while j < len(line):
        char = line[j]
        if not in_string:
            if char in ('"', "'"):
                triple = line[j:j+3]
                if triple in ('"""', "'''"):
                    in_string = True
                    string_char = triple
                    j += 3
                    continue
                else:
                    in_string = True
                    string_char = char
            elif char == '#':
                return line[:j].rstrip()
        else:
            if char == '\\' and j + 1 < len(line):
                j += 2
                continue
            elif len(string_char) == 3 and line[j:j+3] == string_char:
                j += 3
                in_string = False
                continue
            elif len(string_char) == 1 and char == string_char:
                in_string = False
        j += 1
    return line.rstrip()
