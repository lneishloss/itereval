"""Tests for benchmark loaders and base dataclasses."""

import pytest

from itereval.benchmarks.base import BenchmarkProblem
from itereval.benchmarks.humaneval import HumanEvalLoader
from itereval.benchmarks.mbpp import MBPPLoader


# =========================================================================
# BenchmarkProblem
# =========================================================================

class TestBenchmarkProblem:
    """Tests for the BenchmarkProblem dataclass."""

    def test_creation(self):
        p = BenchmarkProblem(
            task_id="HumanEval/0",
            prompt="def foo():\n    pass",
            test_code="assert foo() is None",
            entry_point="foo",
            benchmark="humaneval",
        )
        assert p.task_id == "HumanEval/0"
        assert p.benchmark == "humaneval"
        assert p.canonical_solution == ""

    def test_to_dict(self):
        p = BenchmarkProblem(
            task_id="test/1",
            prompt="def bar(): pass",
            test_code="assert True",
            entry_point="bar",
            benchmark="humaneval",
            canonical_solution="pass",
        )
        d = p.to_dict()
        assert d["task_id"] == "test/1"
        assert d["canonical_solution"] == "pass"
        assert set(d.keys()) == {
            "task_id", "prompt", "test_code", "entry_point",
            "benchmark", "canonical_solution",
        }

    def test_from_dict(self):
        d = {
            "task_id": "test/2",
            "prompt": "def baz(): return 1",
            "test_code": "assert baz() == 1",
            "entry_point": "baz",
            "benchmark": "humaneval",
        }
        p = BenchmarkProblem.from_dict(d)
        assert p.task_id == "test/2"
        assert p.entry_point == "baz"
        assert p.canonical_solution == ""

    def test_roundtrip(self):
        """to_dict -> from_dict should preserve all fields."""
        original = BenchmarkProblem(
            task_id="HumanEval/42",
            prompt="def answer(): return 42",
            test_code="assert answer() == 42",
            entry_point="answer",
            benchmark="humaneval",
            canonical_solution="return 42",
        )
        d = original.to_dict()
        restored = BenchmarkProblem.from_dict(d)
        assert restored.task_id == original.task_id
        assert restored.canonical_solution == original.canonical_solution


# =========================================================================
# HumanEval Loader
# =========================================================================

class TestHumanEvalLoader:
    """Tests for HumanEval benchmark loading."""

    def test_load_bundled(self):
        """Bundled subset should have 20 problems."""
        loader = HumanEvalLoader()
        problems = loader.load_bundled()
        assert len(problems) == 20

    def test_bundled_problem_structure(self):
        """Each bundled problem should have all required fields."""
        loader = HumanEvalLoader()
        problems = loader.load_bundled()
        for p in problems:
            assert p.task_id.startswith("HumanEval/")
            assert p.prompt.strip() != ""
            assert p.test_code.strip() != ""
            assert p.entry_point.strip() != ""
            assert p.benchmark == "humaneval"

    def test_load_with_n(self):
        """Loading with n should return at most n problems."""
        loader = HumanEvalLoader()
        problems = loader.load(n=5)
        assert len(problems) <= 5

    def test_bundled_problems_are_distinct(self):
        """All task_ids should be unique."""
        loader = HumanEvalLoader()
        problems = loader.load_bundled()
        ids = [p.task_id for p in problems]
        assert len(ids) == len(set(ids))

    def test_test_harness_format(self):
        """Test harness should produce executable Python."""
        loader = HumanEvalLoader()
        problems = loader.load_bundled()
        p = problems[0]  # has_close_elements
        code = "def has_close_elements(numbers, threshold):\n    return False\n"
        harness = HumanEvalLoader.format_test_harness(code, p)
        assert "check(has_close_elements)" in harness
        assert "from typing import" in harness
        # Should be parseable Python
        compile(harness, "<test>", "exec")

    def test_test_harness_with_correct_solution(self):
        """Correct solution should pass the test harness."""
        loader = HumanEvalLoader()
        problems = loader.load_bundled()
        # HumanEval/2: truncate_number
        p = next(pp for pp in problems if pp.task_id == "HumanEval/2")
        code = "def truncate_number(number: float) -> float:\n    return number % 1.0\n"
        harness = HumanEvalLoader.format_test_harness(code, p)
        exec(harness, {})


# =========================================================================
# MBPP Loader
# =========================================================================

class TestMBPPLoader:
    """Tests for MBPP benchmark loading."""

    def test_load_bundled(self):
        """Bundled subset should have 20 problems."""
        loader = MBPPLoader()
        problems = loader.load_bundled()
        assert len(problems) == 20

    def test_bundled_problem_structure(self):
        """Each bundled problem should have all required fields."""
        loader = MBPPLoader()
        problems = loader.load_bundled()
        for p in problems:
            assert p.task_id.startswith("mbpp/")
            assert p.prompt.strip() != ""
            assert p.test_code.strip() != ""
            assert p.entry_point.strip() != ""
            assert p.benchmark == "mbpp"

    def test_load_with_n(self):
        loader = MBPPLoader()
        problems = loader.load(n=3)
        assert len(problems) <= 3

    def test_test_harness_format(self):
        """Test harness should produce executable Python."""
        loader = MBPPLoader()
        problems = loader.load_bundled()
        p = problems[0]
        code = f"def {p.entry_point}(*args): pass\n"
        harness = MBPPLoader.format_test_harness(code, p)
        # Should be parseable
        compile(harness, "<test>", "exec")

    def test_test_harness_with_correct_solution(self):
        """Correct solution should pass the test harness."""
        loader = MBPPLoader()
        problems = loader.load_bundled()
        # mbpp/8: square_nums
        p = next(pp for pp in problems if pp.task_id == "mbpp/8")
        code = "def square_nums(nums):\n    return [i**2 for i in nums]\n"
        harness = MBPPLoader.format_test_harness(code, p)
        exec(harness, {})
