"""
MBPP (Mostly Basic Python Problems) benchmark loader.

MBPP is Google's benchmark of crowd-sourced Python programming problems
designed to evaluate the ability of models to synthesize short Python
programs from natural language descriptions.

The sanitized subset contains 427 problems that have been manually
verified for quality and correctness.

Each problem consists of:
- A text description of the task
- Reference code
- Test cases as a list of assertion strings

Reference:
    Austin et al. (2021) "Program Synthesis with Large Language Models"
    https://arxiv.org/abs/2108.07732

Dataset:
    https://huggingface.co/datasets/google-research-datasets/mbpp
"""

from __future__ import annotations

import logging
import re
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


def _extract_function_name(code: str) -> str:
    """
    Extract the main function name from Python code.

    MBPP doesn't provide an entry_point field, so we need to extract
    the function name from the reference code.

    Args:
        code: Python source code

    Returns:
        Function name, or "solution" as fallback
    """
    # Look for function definitions
    match = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    if match:
        return match.group(1)
    return "solution"


class MBPPLoader:
    """
    Loader for the MBPP (Mostly Basic Python Problems) benchmark.

    Supports loading from:
    1. Hugging Face datasets library (full 427 sanitized problems)
    2. Bundled YAML fallback (20 representative problems)

    Usage:
        loader = MBPPLoader()
        problems = loader.load()  # Uses HF if available, else fallback
        problems = loader.load(n=20)  # First 20 problems
        problems = loader.load_bundled()  # Always use bundled subset
    """

    BENCHMARK_NAME = "mbpp"
    TOTAL_PROBLEMS = 427  # Sanitized subset
    HF_DATASET_NAME = "google-research-datasets/mbpp"

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
        Load MBPP problems.

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

        logger.info(f"Loading MBPP from Hugging Face ({self.HF_DATASET_NAME})")

        try:
            # Load the sanitized subset
            ds = load_dataset(
                self.HF_DATASET_NAME,
                "sanitized",
                split="test",
            )
        except Exception as e:
            logger.warning(f"Failed to load from HF: {e}. Using bundled fallback.")
            return self.load_bundled()

        problems = []
        for item in ds:
            prompt = item["prompt"]
            code = item.get("code", "")
            test_list = item.get("test_list", [])
            task_id = f"mbpp/{item['task_id']}"

            # Extract function name from reference code
            entry_point = _extract_function_name(code)

            # Join test assertions into executable code
            # MBPP test_list contains strings like "assert func(1) == 2"
            test_code = "\n".join(test_list)

            problem = BenchmarkProblem(
                task_id=task_id,
                prompt=prompt,
                canonical_solution=code,
                test_code=test_code,
                entry_point=entry_point,
                benchmark=self.BENCHMARK_NAME,
            )
            problems.append(problem)

        logger.info(f"Loaded {len(problems)} MBPP problems from Hugging Face")

        if n is not None and n < len(problems):
            problems = problems[:n]

        return problems

    def load_bundled(self) -> list[BenchmarkProblem]:
        """
        Load bundled subset of MBPP problems.

        This subset contains 20 representative problems that can be used
        when the datasets library is not installed. The problems are
        selected to cover a range of difficulties and programming concepts.

        Returns:
            List of BenchmarkProblem instances (up to 20).
        """
        yaml_path = self.data_dir / "mbpp_subset.yaml"

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
            # Handle both test_code (string) and test_list (array) formats
            test_code = item.get("test_code", "")
            if not test_code and "test_list" in item:
                test_code = "\n".join(item["test_list"])

            problem = BenchmarkProblem(
                task_id=item["task_id"],
                prompt=item["prompt"],
                canonical_solution=item.get("canonical_solution", ""),
                test_code=test_code,
                entry_point=item["entry_point"],
                benchmark=self.BENCHMARK_NAME,
            )
            problems.append(problem)

        logger.info(f"Loaded {len(problems)} MBPP problems from bundled subset")
        return problems

    @staticmethod
    def format_test_harness(
        generated_code: str,
        problem: BenchmarkProblem,
    ) -> str:
        """
        Format the test harness for MBPP.

        MBPP tests are direct assertions that can be executed after
        the generated code.

        Args:
            generated_code: The generated Python code
            problem: The benchmark problem with test code

        Returns:
            Complete executable test script
        """
        return f'''
# Generated code
{generated_code}

# Tests
{problem.test_code}

print("PASSED: all assertions")
'''
