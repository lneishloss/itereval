"""
Base dataclasses for benchmark problems.

Defines the core data structure used by HumanEval and MBPP loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class BenchmarkProblem:
    """
    A single benchmark problem from HumanEval or MBPP.

    Attributes:
        task_id: Unique identifier (e.g., "HumanEval/0" or "mbpp/1")
        prompt: The problem prompt (function signature + docstring for HumanEval,
                problem description for MBPP)
        canonical_solution: Reference solution (optional, for analysis)
        test_code: Executable test code that validates the solution
        entry_point: Name of the function to be implemented
        benchmark: Which benchmark this comes from ("humaneval" or "mbpp")
    """
    task_id: str
    prompt: str
    test_code: str
    entry_point: str
    benchmark: Literal["humaneval", "mbpp"]
    canonical_solution: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "test_code": self.test_code,
            "entry_point": self.entry_point,
            "benchmark": self.benchmark,
            "canonical_solution": self.canonical_solution,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkProblem":
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            test_code=data["test_code"],
            entry_point=data["entry_point"],
            benchmark=data.get("benchmark", "humaneval"),
            canonical_solution=data.get("canonical_solution", ""),
        )
