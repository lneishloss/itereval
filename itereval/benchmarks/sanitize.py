"""
Code sanitization for LLM-generated output.

Strips markdown fences, trailing explanation text, and trailing test code
from LLM responses to extract clean Python function bodies.

Used by the pass@1 runner to clean generated code before execution.
"""

from __future__ import annotations

import re


def sanitize_code(response: str, entry_point: str) -> str:
    """
    Extract and sanitize Python code from an LLM response.

    Processing steps:
    1. Extract code from markdown fences (```python ... ``` or ``` ... ```)
    2. Strip trailing natural-language explanation after the function
    3. Strip trailing test/check code that may conflict with the harness
    4. Preserve leading imports if present

    Args:
        response: Raw LLM response text
        entry_point: Name of the function being implemented

    Returns:
        Cleaned Python code string
    """
    code = _extract_from_fences(response)
    code = _strip_trailing_explanation(code)
    code = _strip_trailing_tests(code, entry_point)
    return code.rstrip() + "\n"


def _extract_from_fences(response: str) -> str:
    """Extract code from markdown fences, or return raw response."""
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
    return response.strip()


def _strip_trailing_explanation(code: str) -> str:
    """Remove trailing natural-language explanation after code ends.

    Looks for lines that don't start with valid Python syntax and
    aren't blank, after the last function/class definition ends.
    """
    lines = code.split("\n")
    # Find last line that looks like code (indented, or starts with keyword/symbol)
    code_pattern = re.compile(
        r"^(\s+\S|def |class |import |from |@|#|if |elif |else:|"
        r"for |while |try:|except|finally:|with |return |raise |"
        r"yield |assert |pass|break|continue|\)|\]|\}|$)"
    )
    last_code_idx = len(lines) - 1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "":
            continue
        if code_pattern.match(lines[i]):
            last_code_idx = i
            break
        # Non-code line: check if it's a comment or string continuation
        stripped = lines[i].strip()
        if stripped.startswith(("#", '"', "'", ")")):
            last_code_idx = i
            break
    return "\n".join(lines[: last_code_idx + 1])


def _strip_trailing_tests(code: str, entry_point: str) -> str:
    """Remove trailing test/check code that may conflict with test harness.

    Strips patterns like:
    - check(entry_point)
    - assert entry_point(...)
    - if __name__ == "__main__":
    - print(entry_point(...))
    """
    lines = code.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        # Skip standalone test invocations at module level
        if not line.startswith((" ", "\t")):
            if stripped.startswith(f"check({entry_point}"):
                continue
            if stripped.startswith(f"assert {entry_point}("):
                continue
            if stripped.startswith(f"print({entry_point}("):
                continue
            if stripped == 'if __name__ == "__main__":':
                # Drop everything from here on
                break
            if stripped == "if __name__ == '__main__':":
                break
        result.append(line)
    return "\n".join(result)
