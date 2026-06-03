# TritonModelBenchmark

Benchmark and probe LLMs on **PyTorch → Triton kernel translation**, built on
[TritonBench-T](https://github.com/thunlp/TritonBench) and orchestrated on
[Modal](https://modal.com). All kernel execution runs on a single NVIDIA T4.

The project provides three capabilities:

1. **Generation & evaluation** — an LLM translates a PyTorch operator into a
   Triton kernel, which is then evaluated for *call accuracy* (does it run),
   *execution accuracy* (does it match PyTorch), and *efficiency* (speedup).
2. **Hardware-aware mutation testing** — targeted mutations of the gold Triton
   kernels deliberately provoke specific T4 limits (unsupported dtypes,
   shared-memory overflow, invalid block sizes); each is run on the GPU and the
   observed failure is verified against the prediction, yielding a labeled
   hardware-failure dataset.
3. **Hardware-awareness evaluation** — LLMs are scored on whether they can
   predict if a kernel will run on a target GPU (and why), graded against the
   mutation dataset across escalating levels of supplied GPU detail.

---

## Setup

```bash
pip install -r requirements-local.txt
modal setup
modal secret create tritonbench-llm OPENROUTER_API_KEY=sk-or-...
```

If the secret uses a different name, set `export TRITONBENCH_LLM_SECRET=<name>`.

---

## Usage

Commands target the `modal_app.py` shim; flags shown are the common ones.

**Generation & evaluation**

```bash
# Full benchmark (default model, simp dataset)
modal run modal_app.py

# Choose model / dataset / size
modal run modal_app.py -- --model "anthropic/claude-sonnet-4-5" --dataset comp --limit 20

# Generate only, or evaluate an existing predictions file
modal run modal_app.py::generate_only --model "<slug>" --limit 10
modal run modal_app.py::evaluate_only --predictions ./preds.jsonl
```

Flags: `--model`, `--dataset` (`simp`/`comp`), `--limit`, `--concurrency`,
`--gpu` (`T4`/`L4`/`A10`).

**Hardware-aware mutation testing**

```bash
modal run --detach modal_app.py::mutate_only           # all gold kernels
modal run modal_app.py::mutate_only --limit 20         # subset
```

**Hardware-awareness evaluation** (requires a prior mutation run)

```bash
modal run modal_app.py::build_awareness_set_only       # build the labeled eval set
modal run modal_app.py::hardware_eval_only --models "slug1,slug2,slug3"
```

Long runs accept `modal run --detach`, which keeps them alive after the client
disconnects; results are written to the Volume regardless.

---

## Outputs

Artifacts are written to the Modal Volume `tritonbench-t-data`:

| Subdir | Contents |
|---|---|
| `results/` | `summary.json` (pass rates, speedup stats), `failure_dataset.jsonl`, surviving kernels, perf scripts |
| `mutations/` | `mutation_dataset.jsonl` (one row per kernel × mutation), `mutation_summary.json` |
| `awareness/` | `eval_set.jsonl`, `hardware_eval_results.json` (per-model leaderboard + metrics) |

```bash
modal volume get tritonbench-t-data <subdir> ./<subdir>
```

---

## Architecture

Code lives in the `tritonbench/` package; `modal_app.py` is a thin shim that
re-exports the app and entrypoints, so every `modal run modal_app.py::<name>`
command works unchanged.

| Module | Role |
|---|---|
| `core` | Modal app/image/volume, config, hardware facts |
| `llm` | OpenRouter calls, code extraction, dataset loading |
| `kernels` | isolated kernel execution + failure classification |
| `generate` / `evaluate` | LLM generation; call/execution/efficiency evaluation |
| `mutate` | hardware-aware kernel mutations |
| `awareness` | hardware-awareness eval set + scoring |
| `entrypoints` | local entrypoints |

The full experimental configuration (hardware, library versions, models,
prompt, sampling, retry policy) is pinned in
[`experiment_config.json`](experiment_config.json).
