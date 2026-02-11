"""Tests for itereval.benchmarks.sanitize — LLM output cleanup."""

import pytest

from itereval.benchmarks.sanitize import sanitize_code


class TestSanitizeCode:
    """Tests for extracting clean Python from LLM responses."""

    def test_plain_code(self):
        """Raw Python code should pass through with trailing newline."""
        code = "def foo():\n    return 1"
        result = sanitize_code(code, "foo")
        assert result.strip() == code.strip()
        assert result.endswith("\n")

    def test_markdown_fences(self):
        """Code in ```python ... ``` fences should be extracted."""
        response = (
            "Here's the solution:\n\n"
            "```python\n"
            "def foo():\n"
            "    return 42\n"
            "```\n\n"
            "This works because..."
        )
        result = sanitize_code(response, "foo")
        assert "def foo():" in result
        assert "return 42" in result
        assert "```" not in result
        assert "Here's the solution" not in result

    def test_plain_fences(self):
        """Code in ``` ... ``` (no language tag) should be extracted."""
        response = "```\ndef foo():\n    return 1\n```"
        result = sanitize_code(response, "foo")
        assert "def foo():" in result
        assert "```" not in result

    def test_strips_trailing_check(self):
        """Trailing check(entry_point) should be removed."""
        code = "def foo():\n    return 1\n\ncheck(foo)"
        result = sanitize_code(code, "foo")
        assert "check(foo)" not in result
        assert "def foo():" in result

    def test_strips_trailing_assert(self):
        """Trailing assert entry_point(...) should be removed."""
        code = "def foo(x):\n    return x + 1\n\nassert foo(1) == 2"
        result = sanitize_code(code, "foo")
        assert "assert foo(1)" not in result

    def test_strips_trailing_print(self):
        """Trailing print(entry_point(...)) should be removed."""
        code = "def foo():\n    return 42\n\nprint(foo())"
        result = sanitize_code(code, "foo")
        assert "print(foo())" not in result

    def test_strips_main_guard(self):
        """if __name__ == '__main__' block should be removed."""
        code = (
            "def foo():\n"
            "    return 1\n"
            '\n'
            'if __name__ == "__main__":\n'
            "    print(foo())\n"
        )
        result = sanitize_code(code, "foo")
        assert "__main__" not in result
        assert "def foo():" in result

    def test_preserves_imports(self):
        """Leading imports should be preserved."""
        code = "import math\n\ndef foo():\n    return math.pi"
        result = sanitize_code(code, "foo")
        assert "import math" in result
        assert "math.pi" in result

    def test_multiple_fenced_blocks(self):
        """Should extract the longest fenced block."""
        response = (
            "```python\nx = 1\n```\n\n"
            "Actually, here's the correct one:\n\n"
            "```python\ndef foo():\n    return 42\n```"
        )
        result = sanitize_code(response, "foo")
        assert "def foo():" in result

    def test_empty_response(self):
        """Empty response should return just a newline."""
        result = sanitize_code("", "foo")
        assert result == "\n"
