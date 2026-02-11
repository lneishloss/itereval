# itereval — Iterative Code Evaluation with Cost-Per-Solve

An evaluation framework for LLM code generation that models real-world usage: iterative retry loops with error feedback, measured by **cost-per-solve (CPS)** — the total API cost to get working code.

## Why?

Standard pass@k generates k independent attempts and checks if any pass. But that's not how anyone uses AI for coding — when code fails, you paste the error back and say "fix this." Each retry builds on the last.

This framework adds error feedback between attempts and tracks the cost of each retry, giving you a single metric that captures both accuracy and efficiency: **how much does it cost to get a correct answer?**

## Quick Start

```bash
pip install -e .

# Set your API key
export ANTHROPIC_API_KEY=sk-...

# Run 20 HumanEval problems, 3 attempts each
python scripts/run_iterative.py -n 20

# Full HumanEval (164 problems, 3 attempts)
python scripts/run_iterative.py --all

# Dry run (show config, no API calls)
python scripts/run_iterative.py --dry-run
```

## What It Measures

The default A/B sweep (`run_ab_sweep.py`) runs two experiments sharing a single control arm:

**Experiment 1 — Prompt engineering (conciseness instruction):**

| | Control | Treatment |
|---|---|---|
| System prompt | "Write correct, clean Python code..." | "Write correct Python code in a compact, minified style..." |
| User prompt | Original HumanEval prompt | Same |
| Error feedback | Raw generated code | Raw generated code |

**Experiment 2 — Python minification:**

| | Control | Treatment |
|---|---|---|
| System prompt | Control (same for both arms) | Control (same for both arms) |
| User prompt | Original HumanEval prompt | Minified (comments/whitespace removed, docstrings preserved) |
| Error feedback | Raw generated code | Minified generated code |

For each problem, both arms get up to k attempts. When an attempt fails, the error output and generated code are appended to the prompt for the next attempt. The treatment arm can optionally transform the generated code before embedding it in error feedback (via `code_transform`), reducing token growth across retries.

### Key Metrics

- **Cost-per-solve (CPS)**: `total_cost / problems_solved` — accuracy-adjusted by construction
- **Solve rate**: Fraction of problems solved within k attempts (with Wilson CIs)
- **Token breakdown**: 4 buckets — control/treatment × input/output
- **Error feedback efficiency**: Tokens added per retry
- **Improvement rate**: % of retries that fix a previously-failing problem
- **Convergence curve**: Cumulative solve rate by attempt number

### Statistical Methods

- **Wilson score intervals** for solve rates — preferred over normal approximation (Wald) intervals for small-sample binary outcomes (per Bowyer et al., ICML 2025)
- **McNemar's test** for paired binary outcomes — the correct test for comparing two conditions on the same dataset (per Dietterich, 1998), using exact binomial test when discordant pairs < 25
- **Bootstrap CIs** on cost-per-solve savings — resamples all N problems with replacement, capturing both cost and solve-rate variation
- **Paired permutation test** on per-problem total costs — assumption-free significance test for right-skewed cost distributions where a paired t-test's normality assumption would be violated

## CLI Options

```bash
# Shared-control A/B sweep (recommended — runs both experiments efficiently)
python scripts/run_ab_sweep.py --preset isolate-ivs -n 20 --iterative --max-attempts 3

# Full HumanEval sweep
python scripts/run_ab_sweep.py --preset isolate-ivs --all --iterative --max-attempts 3

# Dry run (show config, no API calls)
python scripts/run_ab_sweep.py --preset isolate-ivs --dry-run

# Single iterative experiment (system prompt only, no minification)
python scripts/run_iterative.py -n 20 --max-attempts 3

# Pass@1 (single attempt, no retries)
python scripts/run_pass1.py -n 20

# Use same prompt for both arms (isolate prompt_transform effect only)
python scripts/run_iterative.py --same-prompt -n 20

# Verbose output (full prompts, code, errors per problem)
python scripts/run_iterative.py -v -n 10

# Different model
python scripts/run_iterative.py --model claude-opus-4-20250514 -n 10

# MBPP benchmark instead of HumanEval
python scripts/run_iterative.py --benchmark mbpp -n 20
```

## Pluggable Transforms

The framework supports two types of pluggable transforms:

- **`prompt_transform`**: Applied to the user prompt (and accumulated error feedback) before each API call. Reduces input tokens.
- **`code_transform`**: Applied to generated code before embedding it in error feedback for retries. Reduces token growth across iterations.

Both are pure functions with signature `(str) -> str`. By default, only the system prompt differs between arms.

```python
from itereval.benchmarks.iterative_runner import IterativeRunner
from itereval.transforms import minify_prompt

runner = IterativeRunner(
    prompt_transform=minify_prompt,
    transform_name="minify_prompt",
    max_attempts=3,
)
summary = runner.run_sync(n_problems=20)
print(summary.format_report())
```

The `code_transform` is available via the A/B sweep (`run_ab_sweep.py`), which supports both transform types per experiment.

## Dependencies

| Package | Required? | Fallback |
|---|---|---|
| `anthropic` | Yes | — |
| `scipy` | Yes | — |
| `tiktoken` | Optional | `len(text) // 4` heuristic |
| `datasets` | Optional | Bundled 20-problem YAML subsets |
| `pyyaml` | Optional | Only if using bundled fallback |
| `python-minifier` | Optional | Regex-based minification fallback |
| `matplotlib` | Optional | Only needed for figure generation |

```bash
# Minimal
pip install -e .

# With all optional deps
pip install -e ".[all]"
```

## Project Structure

```
itereval/
├── __init__.py
├── utils.py                  # Token estimation, model pricing
├── statistics.py             # Wilson CIs, McNemar, bootstrap, permutation
├── transforms.py             # Prompt/code transforms (minify, strip whitespace)
├── figures.py                # Figure generation (matplotlib)
├── benchmarks/
│   ├── base.py               # BenchmarkProblem dataclass
│   ├── sanitize.py           # LLM output cleanup
│   ├── humaneval.py          # HumanEval loader (164 problems)
│   ├── mbpp.py               # MBPP loader (427 problems)
│   ├── pass1_runner.py       # Pass@1 A/B runner
│   └── iterative_runner.py   # Iterative CPS runner
└── data/
    ├── humaneval_subset.yaml # Bundled 20-problem fallback
    └── mbpp_subset.yaml      # Bundled 20-problem fallback

scripts/
├── run_pass1.py              # Pass@1 CLI
├── run_iterative.py          # Iterative CPS CLI
├── run_ab_sweep.py           # Shared-control A/B sweep CLI
└── generate_figures.py       # Generate figures from results
```

## License

MIT

## Acknowledgments

This framework grew out of research conducted with Victor de la Peña on prompt compression for LLM tool use. The evaluation methodology and cost-per-solve metric were developed independently, but the broader project context informed the research questions.

## Author

Logan Neishloss — lneishloss@gmail.com
