"""Tests for itereval.transforms — prompt/code minification and whitespace stripping."""

import pytest

from itereval.transforms import strip_whitespace, minify_prompt, minify_python


# =========================================================================
# strip_whitespace
# =========================================================================

class TestStripWhitespace:
    """Tests for lossless whitespace stripping."""

    def test_removes_blank_lines(self):
        text = "line one\n\n\nline two"
        result = strip_whitespace(text)
        assert "\n\n" not in result
        assert "line one" in result
        assert "line two" in result

    def test_collapses_internal_whitespace(self):
        text = "hello    world    test"
        result = strip_whitespace(text)
        assert result == "hello world test"

    def test_preserves_leading_indentation(self):
        text = "def foo():\n    x = 1\n    return x"
        result = strip_whitespace(text)
        assert "    x = 1" in result
        assert "    return x" in result

    def test_strips_trailing_spaces(self):
        text = "hello   \nworld   "
        result = strip_whitespace(text)
        for line in result.splitlines():
            assert line == line.rstrip()

    def test_empty_input(self):
        assert strip_whitespace("") == ""

    def test_single_line(self):
        assert strip_whitespace("hello") == "hello"

    def test_preserves_single_spaces(self):
        text = "a b c d"
        result = strip_whitespace(text)
        assert result == "a b c d"

    def test_realistic_python(self):
        text = (
            "from typing import List\n"
            "\n"
            "\n"
            "def foo(x:  int) ->  bool:\n"
            "    return  x > 0\n"
        )
        result = strip_whitespace(text)
        assert "from typing import List" in result
        assert "def foo(x: int) -> bool:" in result
        assert " return x > 0" in result
        # Blank lines should be gone
        assert "\n\n" not in result


# =========================================================================
# minify_prompt
# =========================================================================

class TestMinifyPrompt:
    """Tests for docstring-preserving prompt minification."""

    def test_preserves_triple_quote_docstring(self):
        text = (
            'def foo(x: int) -> bool:\n'
            '    """Check if x is positive.\n'
            '    >>> foo(1)\n'
            '    True\n'
            '    >>> foo(-1)\n'
            '    False\n'
            '    """\n'
        )
        result = minify_prompt(text)
        assert '"""Check if x is positive.' in result
        assert ">>> foo(1)" in result
        assert ">>> foo(-1)" in result
        assert '"""' in result

    def test_removes_blank_lines(self):
        text = "from typing import List\n\n\ndef foo():\n    pass"
        result = minify_prompt(text)
        assert "\n\n" not in result

    def test_removes_comments(self):
        text = (
            "# This is a comment\n"
            "def foo():  # inline comment\n"
            "    pass\n"
        )
        result = minify_prompt(text)
        assert "# This is a comment" not in result
        assert "# inline comment" not in result
        assert "def foo():" in result

    def test_collapses_whitespace_outside_docstring(self):
        text = "from  typing  import  List\ndef foo():\n    pass"
        result = minify_prompt(text)
        assert "from typing import List" in result

    def test_preserves_docstring_whitespace(self):
        """Whitespace inside docstrings should not be collapsed."""
        text = (
            'def foo():\n'
            '    """This  has  extra  spaces.\n'
            '    And  here  too.\n'
            '    """\n'
        )
        result = minify_prompt(text)
        assert "This  has  extra  spaces." in result
        assert "And  here  too." in result

    def test_empty_input(self):
        assert minify_prompt("") == ""
        assert minify_prompt("   ") == "   "

    def test_no_docstring(self):
        """Code without docstrings should just be minified."""
        text = (
            "# comment\n"
            "\n"
            "def foo():\n"
            "    return 1\n"
        )
        result = minify_prompt(text)
        assert "# comment" not in result
        assert "\n\n" not in result
        assert "def foo():" in result

    def test_humaneval_prompt(self):
        """Realistic HumanEval prompt should preserve docstring."""
        text = (
            'from typing import List\n'
            '\n'
            '\n'
            'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n'
            '    """ Check if in given list of numbers, are any two numbers closer\n'
            '    to each other than given threshold.\n'
            '    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n'
            '    False\n'
            '    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n'
            '    True\n'
            '    """\n'
        )
        result = minify_prompt(text)
        # Docstring content preserved
        assert "Check if in given list" in result
        assert "has_close_elements([1.0, 2.0, 3.0], 0.5)" in result
        assert "True" in result
        # Blank lines between import and def removed
        assert "\n\n\n" not in result
        # Import and def still present
        assert "from typing import List" in result
        assert "def has_close_elements" in result

    def test_single_line_docstring(self):
        text = (
            'def foo():\n'
            '    """Return 1."""\n'
            '    return 1\n'
        )
        result = minify_prompt(text)
        assert '"""Return 1."""' in result

    def test_preserves_indentation(self):
        text = (
            "def foo():\n"
            "    x = 1\n"
            "    if x:\n"
            "        return True\n"
        )
        result = minify_prompt(text)
        assert "    x = 1" in result
        assert "        return True" in result


# =========================================================================
# minify_python
# =========================================================================

class TestMinifyPython:
    """Tests for Python code minification (used in error feedback)."""

    def test_removes_comments(self):
        code = "x = 1  # assign x\n# standalone comment\ny = 2\n"
        result = minify_python(code)
        assert "# assign x" not in result
        assert "# standalone" not in result
        # python-minifier may collapse "x = 1" to "x=1" — check assignment exists
        assert "x" in result and "1" in result
        assert "y" in result and "2" in result

    def test_collapses_blank_lines(self):
        code = "x = 1\n\n\n\ny = 2\n"
        result = minify_python(code)
        assert "\n\n\n" not in result

    def test_empty_input(self):
        assert minify_python("") == ""
        assert minify_python("   ") == "   "

    def test_preserves_string_hashes(self):
        """Hash inside strings should not be treated as comments."""
        code = 's = "hello # world"\n'
        result = minify_python(code)
        assert "hello # world" in result

    def test_returns_valid_python(self):
        """Minified code should still be valid Python."""
        code = (
            "def add(a, b):\n"
            "    # Add two numbers\n"
            "    return a + b\n"
        )
        result = minify_python(code)
        # Should be parseable
        compile(result, "<test>", "exec")

    def test_preserves_semantics(self):
        """Minified code should produce same result when executed."""
        code = (
            "def factorial(n):\n"
            "    # Compute factorial\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        )
        result = minify_python(code)
        ns = {}
        exec(result, ns)
        assert ns["factorial"](5) == 120

    def test_shebang_preserved(self):
        """Shebang line should be preserved."""
        code = "#!/usr/bin/env python3\nx = 1\n"
        result = minify_python(code)
        assert "#!/usr/bin/env python3" in result
