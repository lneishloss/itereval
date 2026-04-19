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
- **Paired permutation test** on per-problem total costs — distribution-free significance test for right-skewed cost distributions where a paired t-test's normality assumption would be violated

## Results

Full HumanEval (164 problems), Claude Sonnet 4, 3 iterative attempts, shared control arm. See [the writeup](https://lneishloss.github.io/itereval/) for full analysis.

### Experiment 1: Conciseness System Prompt

| Metric | Control | Treatment | Delta |
|---|---|---|---|
| Solve rate | 97.6% | 97.6% | 0pp |
| Cost per solve | $0.00263 | $0.00215 | **-18.3%** |
| Output tokens | 20,244 | 13,359 | **-34.0%** |
| Input tokens | 39,198 | 47,936 | +22.3% |
| Paired permutation p-value | | | **0.0003** |
| Bootstrap 95% CI on CPS savings | | | [8.9%, 27.2%] |

### Experiment 2: Prompt + Code Minification

| Metric | Control | Treatment | Delta |
|---|---|---|---|
| Solve rate | 97.6% | 98.2% | +0.6pp |
| Cost per solve | $0.00263 | $0.00229 | **-12.9%** |
| Output tokens | 20,244 | 17,451 | **-13.8%** |
| Input tokens | 39,198 | 35,878 | -8.5% |
| Paired permutation p-value | | | **0.001** |
| Bootstrap 95% CI on CPS savings | | | [3.9%, 21.9%] |

### Key Takeaways

- **Output tokens are the dominant cost lever.** Output costs 5x more than input across Anthropic's model lineup. In Experiment 1, input tokens *increased* 22% but output tokens *decreased* 34% — net result: 18.3% cost savings.
- **Concise input causes concise output.** A conciseness system prompt produced a 34% output token reduction with zero accuracy loss — the first controlled measurement of this effect on a coding benchmark.
- **Neither intervention degrades accuracy.** Both treatments match or slightly exceed the control solve rate (~97.6%), so CPS savings are driven entirely by cost efficiency.
- **The two interventions are composable.** Prompt engineering and minification use orthogonal mechanisms (system prompt vs. input/feedback transforms) and can be stacked.

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

## References

1. **Wilson, E. B.** (1927). "Probable Inference, the Law of Succession, and Statistical Inference." *Journal of the American Statistical Association*, 22(158), 209–212. [DOI: 10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953)
   — Original derivation of the Wilson score interval used for solve-rate confidence intervals.

2. **Bowyer, S., Aitchison, L., & Ivanova, D. R.** (2025). "Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints." *Proceedings of the 42nd International Conference on Machine Learning (ICML)*, vol. 267, pp. 81143–81184. PMLR. [arXiv: 2503.01747](https://arxiv.org/abs/2503.01747)
   — Recommends Wilson score intervals over Wald (normal-approximation) intervals for LLM evaluation with small sample sizes.

3. **Dietterich, T. G.** (1998). "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms." *Neural Computation*, 10(7), 1895–1923. [DOI: 10.1162/089976698300017197](https://doi.org/10.1162/089976698300017197)
   — Recommends McNemar's test for paired comparisons of classifiers on the same dataset.

4. **Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. de O., Kaplan, J., et al.** (2021). "Evaluating Large Language Models Trained on Code." [arXiv: 2107.03374](https://arxiv.org/abs/2107.03374)
   — Introduces the HumanEval benchmark and the pass@k metric for independent-sample code generation evaluation.

5. **Efron, B. & Tibshirani, R. J.** (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
   — Canonical reference for the bootstrap confidence interval methodology used for cost-per-solve savings estimates.

6. **Agresti, A. & Coull, B. A.** (1998). "Approximate Is Better than 'Exact' for Interval Estimation of Binomial Proportions." *The American Statistician*, 52(2), 119–126. [DOI: 10.1080/00031305.1998.10480550](https://doi.org/10.1080/00031305.1998.10480550)
   — Demonstrates that Wilson intervals outperform Wald intervals for binomial proportion estimation, especially at small sample sizes.

## Author

Logan Neishloss — lneishloss@gmail.com
