# TritonModelBenchmark

Benchmark LLMs on **PyTorch → Triton kernel translation** using
[TritonBench-T](https://github.com/thunlp/TritonBench), orchestrated on
[Modal](https://modal.com).

Given a PyTorch operator description from the TritonBench-T Alpaca dataset,
an LLM is asked to emit a self-contained Triton kernel + wrapper. The
generated module is then evaluated on a single NVIDIA T4 across three
upstream phases:

1. **Call accuracy** — does the generated module import and run?
2. **Execution accuracy** — does it produce the same outputs as PyTorch?
3. **Efficiency** — speedup vs. the golden PyTorch baseline.

The whole pipeline lives in one file: [`modal_app.py`](modal_app.py). The
full experimental configuration (hardware, library versions, model
slugs, prompts, sampling params) is pinned in
[`experiment_config.json`](experiment_config.json).

---

## Setup

```bash
pip install -r requirements-local.txt
modal setup
modal secret create tritonbench-llm OPENROUTER_API_KEY=sk-or-...
```

If your secret has a different name, set
`export TRITONBENCH_LLM_SECRET=<name>`.

---

## Run

```bash
# Full benchmark (default: anthropic/claude-sonnet-4-5, dataset=simp)
modal run modal_app.py

# Different model / complex dataset
modal run modal_app.py --model "anthropic/claude-sonnet-4-5" --dataset comp

# Smoke test on 5 ops
modal run modal_app.py -- --limit 5

# Evaluate an existing predictions.jsonl
modal run modal_app.py::evaluate_only --predictions ./preds.jsonl
```

Key flags: `--model`, `--dataset` (`simp`/`comp`), `--limit`,
`--concurrency`, `--gpu` (`T4`/`L4`/`A10`).

---

## Outputs

Artifacts land in the Modal Volume `tritonbench-t-data` under the chosen
`output_subdir`:

- `summary.json` — phase 1/2/3 pass rates, speedup stats (mean / geomean
  / median / % > 1.0x), generation telemetry, phase timings.
- `call_acc/` — kernels that survived phase 1.
- `perf_results/` — generated per-op perf scripts.
- `*.gen_meta.json` — per-item latency, token counts, failure causes.

Pull locally:

```bash
modal volume get tritonbench-t-data results ./results
```

---

## Reproducing a result

All configuration is captured in
[`experiment_config.json`](experiment_config.json) — hardware, container,
library versions, models, prompt, sampling params, retry policy,
benchmark version.