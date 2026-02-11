"""
HumanEval benchmark loader.

HumanEval is OpenAI's benchmark of 164 hand-written Python programming
problems designed to evaluate functional correctness of code generation.

Each problem consists of:
- A function signature with docstring (the prompt)
- A canonical solution
- Test cases in the form of `def check(candidate): ...`

Reference:
    Chen et al. (2021) "Evaluating Large Language Models Trained on Code"
    https://arxiv.org/abs/2107.03374
    https://github.com/openai/human-eval

Dataset:
    https://huggingface.co/datasets/openai/openai_humaneval
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import BenchmarkProblem

logger = logging.getLogger(__name__)

# Lazy import check for datasets library
_datasets_available: Optional[bool] = None


def _check_datasets() -> bool:
    """Check if the datasets library is available."""
    global _datasets_available
    if _datasets_available is None:
        try:
            import datasets  # noqa: F401
            _datasets_available = True
        except ImportError:
            _datasets_available = False
            logger.info(
                "datasets library not installed. Using bundled subset. "
                "Install with: pip install datasets"
            )
    return _datasets_available


class HumanEvalLoader:
    """
    Loader for the HumanEval benchmark.

    Supports loading from:
    1. Hugging Face datasets library (full 164 problems)
    2. Bundled YAML fallback (20 representative problems)

    Usage:
        loader = HumanEvalLoader()
        problems = loader.load()  # Uses HF if available, else fallback
        problems = loader.load(n=20)  # First 20 problems
        problems = loader.load_bundled()  # Always use bundled subset
    """

    BENCHMARK_NAME = "humaneval"
    TOTAL_PROBLEMS = 164
    HF_DATASET_NAME = "openai/openai_humaneval"

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the loader.

        Args:
            data_dir: Directory containing bundled data files.
                     Defaults to ../data/ relative to this file.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = data_dir

    def load(self, n: Optional[int] = None) -> list[BenchmarkProblem]:
        """
        Load HumanEval problems.

        Tries to load from Hugging Face datasets first, falls back to
        bundled subset if datasets library is not installed.

        Args:
            n: Number of problems to load. None = all available.
               When using bundled fallback, max is 20.

        Returns:
            List of BenchmarkProblem instances.
        """
        if _check_datasets():
            return self._load_from_hf(n)
        else:
            problems = self.load_bundled()
            if n is not None and n < len(problems):
                problems = problems[:n]
            return problems

    def _load_from_hf(self, n: Optional[int] = None) -> list[BenchmarkProblem]:
        """Load problems from Hugging Face datasets."""
        from datasets import load_dataset

        logger.info(f"Loading HumanEval from Hugging Face ({self.HF_DATASET_NAME})")

        try:
            ds = load_dataset(self.HF_DATASET_NAME, split="test")
        except Exception as e:
            logger.warning(f"Failed to load from HF: {e}. Using bundled fallback.")
            return self.load_bundled()

        problems = []
        for item in ds:
            problem = BenchmarkProblem(
                task_id=item["task_id"],
                prompt=item["prompt"],
                canonical_solution=item.get("canonical_solution", ""),
                test_code=item["test"],
                entry_point=item["entry_point"],
                benchmark=self.BENCHMARK_NAME,
            )
            problems.append(problem)

        logger.info(f"Loaded {len(problems)} HumanEval problems from Hugging Face")

        if n is not None and n < len(problems):
            problems = problems[:n]

        return problems

    def load_bundled(self) -> list[BenchmarkProblem]:
        """
        Load bundled subset of HumanEval problems.

        This subset contains 20 representative problems that can be used
        when the datasets library is not installed. The problems are
        selected to cover a range of difficulties and programming concepts.

        Returns:
            List of BenchmarkProblem instances (up to 20).
        """
        yaml_path = self.data_dir / "humaneval_subset.yaml"

        if not yaml_path.exists():
            logger.warning(f"Bundled data not found at {yaml_path}")
            return []

        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed. Cannot load bundled data.")
            return []

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load bundled data: {e}")
            return []

        problems = []
        for item in data.get("problems", []):
            problem = BenchmarkProblem(
                task_id=item["task_id"],
                prompt=item["prompt"],
                canonical_solution=item.get("canonical_solution", ""),
                test_code=item["test_code"],
                entry_point=item["entry_point"],
                benchmark=self.BENCHMARK_NAME,
            )
            problems.append(problem)

        logger.info(f"Loaded {len(problems)} HumanEval problems from bundled subset")
        return problems

    @staticmethod
    def format_test_harness(
        generated_code: str,
        problem: BenchmarkProblem,
    ) -> str:
        """
        Format the test harness for HumanEval.

        HumanEval tests use the pattern:
            def check(candidate):
                assert candidate(...) == expected
                ...
            check(entry_point)

        We need to wrap the generated code and call check with the
        generated function as the candidate.

        Args:
            generated_code: The generated Python code
            problem: The benchmark problem with test code

        Returns:
            Complete executable test script
        """
        # HumanEval test format: check(candidate) where candidate is the function
        # Standard imports needed by HumanEval prompts (typing, math, etc.)
        return f'''from typing import *
import math
import re
import sys
import copy
import datetime
import itertools
import collections
import heapq
import statistics
import functools
import hashlib
import string
from io import StringIO
from collections import *
from functools import *
from heapq import *
from itertools import *

# Generated code
{generated_code}

# Test harness
{problem.test_code}

# Execute tests
check({problem.entry_point})
print("PASSED: all assertions")
'''
