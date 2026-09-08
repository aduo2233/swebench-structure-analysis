# SWE-bench task-structure analysis

Why is "auto-fixing a bug" harder than "writing one function"? This repo quantifies the structural gap with the public SWE-bench data.

## What it computes

- files touched per gold fix (Verified 500 vs full 2,294)
- FAIL_TO_PASS vs PASS_TO_PASS test gates
- problem-statement length vs HumanEval prompt length
- human-rated difficulty distribution of Verified tasks
- gold patch size in lines

## Data sources

Parquet files (read directly by the script from these mirrors):

- SWE-bench Verified: https://hf-mirror.com/datasets/princeton-nlp/SWE-bench_Verified
- SWE-bench (full test): https://hf-mirror.com/datasets/princeton-nlp/SWE-bench
- HumanEval: https://hf-mirror.com/datasets/openai/openai_humaneval

Put the parquet files in a directory and point the script there, e.g.:

```bash
mkdir data
curl -L -o data/swebench_verified.parquet "https://hf-mirror.com/datasets/princeton-nlp/SWE-bench_Verified/resolve/main/data/test-00000-of-00001.parquet"
curl -L -o data/swebench_full_test.parquet "https://hf-mirror.com/datasets/princeton-nlp/SWE-bench/resolve/main/data/test-00000-of-00001.parquet"
curl -L -o data/humaneval.parquet "https://hf-mirror.com/datasets/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet"
uv run --with pandas --with pyarrow --with matplotlib python3 analyze_swebench.py
```

The script expects the parquet files under `../.research_tmp/` relative to itself (workspace layout of the original run); edit the `DATA` path if you keep them elsewhere.

## Runtime

A few minutes on CPU. Figures and `stats.json` are written to `results/`.

## Key numbers (as of 2026-09-08)

- Verified: median fix touches 1 file (85.8ingle-file), median 7 changed lines, median 50.5 PASS_TO_PASS regression tests, median 1 FAIL_TO_PASS.
- full: 24.90f fixes touch 2+ files (up to 21).
- HumanEval prompt median 396 chars, tests given in-prompt.
