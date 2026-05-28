"""
TritonBench-T on Modal — translate PyTorch ops to Triton kernels with an LLM,
then evaluate them on the cheapest available Modal GPU (NVIDIA T4).

Pipeline
--------
1. ``generate_predictions``  — calls a configured LLM on each Alpaca
   instruction in ``data/TritonBench_T_<simp|comp>_alpac_v1.json`` and writes a
   ``predictions.jsonl`` into a persistent Modal Volume.
2. ``evaluate``              — runs the three TritonBench-T phases on a GPU:
       phase 1: call accuracy   (does the generated module run at all?)
       phase 2: execution acc.  (does it produce the same outputs as PyTorch?)
       phase 3: efficiency      (speedup vs. the golden PyTorch baseline)

A single ``main`` local entrypoint chains them end-to-end.

Quick start (see README.md for full instructions):

    pip install modal
    modal setup
    modal secret create tritonbench-llm OPENROUTER_API_KEY=sk-or-...
    modal run modal_app.py                        # generate + evaluate
    modal run modal_app.py -- --limit 5           # smoke test on 5 ops
    modal run modal_app.py -- --predictions ./preds.jsonl   # bring your own
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


import modal

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

APP_NAME = "tritonbench-t"
TRITONBENCH_REPO = "https://github.com/thunlp/TritonBench.git"

# Cheapest Modal GPU (compute capability 7.5 — Triton requires >= 7.0).
# Override at runtime via `--gpu A10` etc. on the local entrypoint.
DEFAULT_GPU = "T4"

VOLUME_NAME = "tritonbench-t-data"
DATA_DIR = "/data"           # mount point of the Modal Volume in the container
REPO_DIR = "/opt/TritonBench"

# Default model in OpenRouter format.  Any $0-cost model slug from
# https://openrouter.ai/models works here, e.g.:
#   "mistralai/mistral-7b-instruct"
#   "nvidia/llama-3.1-nemotron-70b-instruct:free"
#   "microsoft/phi-3-mini-128k-instruct:free"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"

# Name of the Modal Secret that holds OPENROUTER_API_KEY.
# Override with an env var if your existing secret is named differently:
#     export TRITONBENCH_LLM_SECRET=my-other-secret
LLM_SECRET_NAME = os.environ.get("TRITONBENCH_LLM_SECRET", "tritonbench-llm")

# Retry knobs for free-tier rate limits (429 / 503 / empty responses).
MAX_RETRIES = 2
RETRY_BASE_DELAY = 4.0   # seconds; doubles each attempt (exponential backoff)

# Per-kernel subprocess resource limits — shared across Phase 1 / Phase 2
# pre-probing and Phase 3 benchmarking.
KERNEL_TIMEOUT  = 60               # wall-clock seconds before killing a subprocess
VIRT_MEM_BYTES  = 12 * 1024 ** 3  # 12 GiB virtual-address ceiling

# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #

# 0_call_acc.py — wrong dataset filename (.json vs .jsonl), wrong test folder
# (G instead of T), and a hardcoded conda interpreter path.
PATCH_CALL_ACC = (
    f"""sed -i """
    f"""-e 's|^statis_path = .*|statis_path = "{REPO_DIR}/data/TritonBench_T_v1.jsonl"|' """
    f"""-e 's|^py_folder = .*|py_folder = "{REPO_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|py_interpreter = __import__("sys").executable|' """
    f"""{REPO_DIR}/EVAL/eval_T/0_call_acc.py"""
)

# 1_exe_acc.py — same hardcoded conda interpreter; gold_folder anchored to
# absolute path.
PATCH_EXE_ACC = (
    f"""sed -i """
    f"""-e 's|^gold_folder = .*|gold_folder = "{REPO_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|py_interpreter = __import__("sys").executable|' """
    f"""{REPO_DIR}/EVAL/eval_T/1_exe_acc.py"""
)

# multiprocess_gpu_run.py — assumes 8 GPUs; we have one.
PATCH_PERF = (
    f"""sed -i 's|^gpu_count = .*|gpu_count = 1|' """
    f"""{REPO_DIR}/performance_metrics/perf_T/run_bench/multiprocess_gpu_run.py"""
)


image = (
    modal.Image.from_registry(
        # Python 3.12: TritonBench's eval scripts use PEP-701 nested-quote
        # f-strings, which require >= 3.12 to parse.
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "triton==3.1.0",
        "tqdm==4.66.5",
        "numpy<2",
        "openai>=1.50",
        "psutil>=5.9",   # used for memory diagnostics in Phase 3
    )
    .run_commands(f"git clone --depth 1 {TRITONBENCH_REPO} {REPO_DIR}")
    .run_commands(PATCH_CALL_ACC, PATCH_EXE_ACC, PATCH_PERF)
    # ProcessPoolExecutor pickles workers by qualified module name, so the
    # eval scripts must be importable as plain `call_acc` / `exe_acc` from any
    # subprocess. Module names can't start with a digit, so symlink them.
    .run_commands(
        f"ln -s {REPO_DIR}/EVAL/eval_T/0_call_acc.py {REPO_DIR}/EVAL/eval_T/call_acc.py",
        f"ln -s {REPO_DIR}/EVAL/eval_T/1_exe_acc.py {REPO_DIR}/EVAL/eval_T/exe_acc.py",
    )
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# --------------------------------------------------------------------------- #
# Generation — LLM-based PyTorch → Triton translation
# --------------------------------------------------------------------------- #

PROMPT_HEADER = (
    "You are an expert in Triton programming, capable of writing Triton kernels "
    "and wrapper functions based on functional descriptions and function "
    "parameters. The wrapper function must fully match the provided function "
    "signature.\n\n"
    "Output a single, self-contained Python module containing: (a) the necessary "
    "imports (torch, triton, triton.language as tl), (b) the Triton kernel(s), "
    "and (c) the wrapper function that the description specifies. Wrap the "
    "entire module in one ```python ... ``` fenced code block. Do NOT include "
    "any test code or example calls — tests will be appended separately."
)


def _load_alpaca(dataset: str) -> list[dict]:
    assert dataset in ("simp", "comp"), "dataset must be 'simp' or 'comp'"
    path = Path(REPO_DIR) / f"data/TritonBench_T_{dataset}_alpac_v1.json"
    return json.loads(path.read_text())


def _build_messages(item: dict) -> list[dict]:
    instr = item["instruction"]
    inp = item.get("input", "") or ""
    user = instr if not inp else f"{instr}\n\n{inp}"
    return [
        {"role": "system", "content": PROMPT_HEADER},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Kernel failure classification (Phase 1 / Phase 2 observability)
# --------------------------------------------------------------------------- #

# Categories that indicate a hardware constraint rather than a code bug.
# Used to set `is_hardware_failure` in the failure dataset.
HARDWARE_FAILURE_TYPES: frozenset[str] = frozenset({
    "shared_mem_overflow",
    "register_overflow",
    "arch_unsupported",
    "dtype_unsupported",
    "oom",
    "illegal_memory_access",
})


def _classify_kernel_failure(stderr: str) -> str:
    """Classify Triton/CUDA compilation or execution stderr into a failure category.

    Distinct from _classify_failure, which handles LLM API errors.

    Hardware checks run before code/logic checks because PTX error strings
    sometimes contain Python exception names (e.g. RuntimeError) that would
    otherwise match the code_error bucket.

    Patterns validated against Triton 3.1.0 / CUDA 12.4 error output.
    """
    s = stderr.lower()

    # --- hardware-attributable -------------------------------------------
    if "out of resource" in s and "shared memory" in s:
        return "shared_mem_overflow"
    if "out of resource" in s and "regist" in s:
        # "regist" matches both "register" and "registers" in PTX compiler output.
        return "register_overflow"
    if "no kernel image is available" in s:
        # CUDA runtime: binary not compiled for this device's compute capability.
        return "arch_unsupported"
    if ("not supported" in s or "unsupported" in s) and any(
        t in s for t in ("bf16", "fp8", "float8")
    ):
        # Ambiguous: "not supported" appears in many error paths; requiring an
        # explicit dtype token (bf16/fp8/float8) narrows to GPU dtype limits.
        return "dtype_unsupported"
    if "cuda out of memory" in s or "out of memory" in s:
        return "oom"
    if "illegal memory access" in s:
        return "illegal_memory_access"

    # --- code / logic ----------------------------------------------------
    if "block_size" in s or "must be a power of 2" in s:
        return "invalid_block_size"
    if any(t in s for t in ("allclose", "mismatch", "incorrect", "not equal")):
        return "numerical_mismatch"
    if any(t in s for t in ("syntaxerror", "nameerror", "importerror")):
        return "code_error"

    return "other_runtime"


def _set_mem_limit() -> None:
    """Pre-exec hook: cap virtual address space to VIRT_MEM_BYTES.

    Prevents a runaway kernel subprocess from OOM-killing the parent container.
    Best-effort — only effective on Linux; silently no-ops elsewhere.
    """
    import resource as _resource
    try:
        _resource.setrlimit(_resource.RLIMIT_AS, (VIRT_MEM_BYTES, VIRT_MEM_BYTES))
    except Exception:
        pass


def _probe_kernel_file(path: Path) -> tuple[int, str]:
    """Run *path* in an isolated subprocess; return (returncode, stderr).

    Mirrors the Phase 3 isolation pattern: KERNEL_TIMEOUT wall-clock limit and
    _set_mem_limit virtual-memory ceiling.  Called before Phase 1 / Phase 2
    upstream scripts run so stderr is captured before failing files are deleted.
    stdout is discarded — only stderr carries failure information.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=KERNEL_TIMEOUT,
            preexec_fn=_set_mem_limit,
        )
        return proc.returncode, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, f"TimeoutExpired: kernel did not complete within {KERNEL_TIMEOUT}s"
    except Exception as exc:
        return -1, str(exc)


def _parse_reset_delay(exc_str: str, fallback: float) -> float:
    """Extract a wait duration from an X-RateLimit-Reset epoch-ms timestamp
    embedded in the error string, falling back to *fallback* seconds if absent
    or in the past.
    """
    import re as _re

    m = _re.search(r"'X-RateLimit-Reset':\s*'(\d+)'", exc_str)
    if m:
        reset_ms = int(m.group(1))
        wait = (reset_ms / 1000.0) - time.time()
        if 0 < wait < 300:   # sanity: only use if 0–5 min in the future
            return wait + 1.0  # +1 s buffer
    return fallback


@dataclass
class GenResult:
    """Outcome of one successful LLM generation call, with telemetry."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    latency_s: float = 0.0


def _classify_failure(exc: Exception) -> str:
    """Bucket a generation exception into a coarse, comparable cause."""
    s = str(exc).lower()
    if "daily free quota" in s or "free-models-per-day" in s:
        return "quota_exhausted"
    if ("rate" in s and "limit" in s) or "429" in s:
        return "rate_limit"
    if "syntaxerror" in s:
        return "syntax_error"
    if "no choices" in s or "empty content" in s:
        return "empty_response"
    status = getattr(exc, "status_code", None)
    if status in (400, 401, 403, 404, 500, 502, 503, 529) or "503" in s or "529" in s:
        return "api_error"
    return "other"


def _gen(messages: list[dict], model: str) -> GenResult:
    """Call the OpenRouter API with smart retry on rate limits.

    Two kinds of 429 from OpenRouter free tier:
      • free-models-per-min  — transient; back off and retry same key.
      • free-models-per-day  — daily hard cap; no point retrying today,
                               raise immediately so the caller can record
                               a clean failure and move on.

    Also handles:
      • None / empty choices  — model at capacity, retry with backoff.
      • HTTP 503 / 529        — upstream overload, retry with backoff.

    Returns a GenResult carrying the reply text plus token-usage and
    latency telemetry from the successful call.
    """
    from openai import OpenAI, RateLimitError, APIStatusError

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            call_start = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=8192,
                temperature=0,
            )
            latency_s = time.perf_counter() - call_start

            # Guard: some models return a response object with None or empty
            # choices instead of raising — treat as a retryable soft failure.
            choices = resp.choices or []
            if not choices:
                raise ValueError("API returned no choices (model at capacity)")
            choice = choices[0]
            msg = choice.message
            content = (msg.content or "") if msg is not None else ""
            if not content.strip():
                raise ValueError("API returned empty content (model at capacity)")

            usage = getattr(resp, "usage", None)
            return GenResult(
                content=content,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                finish_reason=getattr(choice, "finish_reason", "") or "",
                latency_s=latency_s,
            )

        except RateLimitError as exc:
            last_exc = exc
            exc_str = str(exc)
            if "free-models-per-day" in exc_str:
                # Hard daily cap — retrying won't help, surface immediately.
                raise RuntimeError(
                    f"daily free quota exhausted for this API key: {exc}"
                ) from exc
            # Per-minute throttle — wait for the reset window if we can parse
            # it, otherwise fall back to exponential backoff.
            delay = _parse_reset_delay(exc_str, RETRY_BASE_DELAY * (2 ** attempt))
            print(
                f"    [retry {attempt+1}/{MAX_RETRIES}] rate-limit (per-min) — "
                f"waiting {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)

        except ValueError as exc:
            # Empty / no-choices soft failure — exponential backoff.
            last_exc = exc
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(
                f"    [retry {attempt+1}/{MAX_RETRIES}] {exc} — waiting {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)

        except APIStatusError as exc:
            if exc.status_code in (503, 529):
                last_exc = exc
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(
                    f"    [retry {attempt+1}/{MAX_RETRIES}] HTTP {exc.status_code} — "
                    f"waiting {delay:.0f}s",
                    flush=True,
                )
                time.sleep(delay)
            else:
                raise  # non-retryable (400 bad request, 401 auth, etc.)

    raise RuntimeError(
        f"generation failed after {MAX_RETRIES} retries: {last_exc}"
    )


def _extract_code(text: str) -> str:
    """Strip Markdown code fences from an LLM reply; return raw Python source.

    Tries each strategy in order and returns the first one that yields valid
    Python.  Falls back to the raw text if nothing parses cleanly (the
    SyntaxError check in _do() will catch it and log the raw response).

    Handles:
      - ```python\\n...\\n```  (standard)
      - ```py\\n...\\n```
      - ```\\n...\\n```        (no language tag)
      - Multiple blocks — takes the LAST one (some models emit a short
        explanation block then the real code block)
      - Closing fence missing (truncated reply)
      - Raw Python with no fences at all
      - Models that emit ↵ (U+21B5) or \\r\\n instead of real newlines
    """
    import re

    # Normalise model-emitted newline surrogates BEFORE any regex work.
    # Some models (owl-alpha, etc.) emit U+21B5 ↵ as a literal newline stand-in
    # which ends up written verbatim into the .py file, causing SyntaxError.
    text = text.replace("\u21b5", "\n").replace("\r\n", "\n").replace("\r", "\n")

    s = text.strip()

    # Collect ALL fenced blocks; prefer the last one (real code usually last).
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)\n```", s, re.DOTALL)
    if blocks:
        # Return the last block — it's the actual implementation in models that
        # emit explanation first, then code.
        return blocks[-1].strip() + "\n"

    # No closing fence — truncated reply.  Drop the opening fence if present.
    no_open = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    # Strip any trailing ``` that may appear mid-text before prose.
    no_open = re.sub(r"\n?```[\s\S]*$", "", no_open)
    candidate = no_open.strip()
    if candidate:
        return candidate + "\n"

    # Nothing to strip — return as-is and let _is_valid_python decide.
    return s + "\n"


def _is_valid_python(code: str) -> bool:
    """Return True if *code* parses without a SyntaxError."""
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _gen_meta_path(predictions_path: str | Path) -> Path:
    """Sidecar path holding generation metrics for a predictions jsonl."""
    p = Path(predictions_path)
    stem = p.name[:-6] if p.name.endswith(".jsonl") else p.name
    return p.parent / f"{stem}.gen_meta.json"


@app.function(
    timeout=60 * 60 * 4,
    cpu=4,
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name(LLM_SECRET_NAME)],
)
def generate_predictions(
    model: str = DEFAULT_MODEL,
    dataset: str = "simp",
    output_path: str = "predictions.jsonl",
    limit: int | None = None,
    concurrency: int = 4,
) -> str:
    """Generate Triton translations for every entry in the Alpaca dataset.

    Writes ``output_path`` (the predictions jsonl, schema ``{instruction,
    predict}``) plus a ``.gen_meta.json`` sidecar with latency / token /
    failure-cause telemetry. Returns the volume-relative path of the jsonl.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    items = _load_alpaca(dataset)
    if limit:
        items = items[:limit]

    print(f"generating {len(items)} predictions with model={model}", flush=True)

    # Smoke-test the model with a single cheap call before spawning workers.
    # A bad slug (404) or auth failure (401/403) dooms every call — fail fast
    # instead of burning the whole dataset; transient capacity errors are
    # tolerated since the per-item retry logic may still recover.
    print("  smoke-testing model...", end=" ", flush=True)
    try:
        _gen([{"role": "user", "content": "Reply with one word: ready"}], model)
        print("OK", flush=True)
    except RuntimeError as exc:
        if "daily free quota" in str(exc):
            raise  # hard quota — no point starting workers
        print(f"warning: {exc} — proceeding anyway", flush=True)
    except Exception as exc:  # noqa: BLE001
        status = getattr(exc, "status_code", None)
        if status in (400, 401, 403, 404):
            raise RuntimeError(
                f"model unusable (HTTP {status}) — check the slug and "
                f"credentials before retrying: {exc}"
            ) from exc
        print(f"warning: {exc} — proceeding anyway", flush=True)

    def _do(idx_item: tuple[int, dict]) -> tuple[int, dict, dict]:
        i, item = idx_item
        instruction = item["instruction"]
        meta: dict = {
            "index": i,
            "latency_s": None,
            "completion_tokens": None,
            "prompt_tokens": None,
            "truncated": False,
            "fail_reason": None,
        }
        try:
            gr = _gen(_build_messages(item), model)
            meta["latency_s"] = round(gr.latency_s, 3)
            meta["completion_tokens"] = gr.completion_tokens
            meta["prompt_tokens"] = gr.prompt_tokens
            meta["truncated"] = gr.finish_reason == "length"
            code = _extract_code(gr.content)
            if not _is_valid_python(code):
                # Log the first 300 chars of the raw response so you can see
                # what the model returned and tune _extract_code if needed.
                preview = gr.content[:300].replace("\n", "|")
                raise ValueError(
                    f"SyntaxError after fence-strip — raw preview: {preview!r}"
                )
            print(f"  [OK ] {i:4d} {instruction[:60]}", flush=True)
        except Exception as exc:       # noqa: BLE001
            meta["fail_reason"] = _classify_failure(exc)
            code = f"# generation failed: {exc}\n"
            print(
                f"  [ERR] {i:4d} {instruction[:60]} — "
                f"[{meta['fail_reason']}] {exc}",
                flush=True,
            )
        return i, {"instruction": instruction, "predict": code}, meta

    results: list[dict | None] = [None] * len(items)
    metas: list[dict | None] = [None] * len(items)
    failed = 0
    gen_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_do, (i, it)) for i, it in enumerate(items)]
        done = 0
        for fut in as_completed(futs):
            i, rec, meta = fut.result()
            results[i] = rec
            metas[i] = meta
            if meta["fail_reason"]:
                failed += 1
            done += 1
            if done % 10 == 0 or done == len(items):
                print(f"  progress: {done}/{len(items)}  failures so far: {failed}", flush=True)
    gen_wall_s = time.perf_counter() - gen_start

    out = Path(DATA_DIR) / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep the jsonl schema exactly {instruction, predict} — upstream eval
    # scripts read it and may not tolerate extra keys.
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- generation metrics sidecar -----------------------------------------
    latencies = [m["latency_s"] for m in metas if m and m["latency_s"] is not None]
    comp_tokens = [m["completion_tokens"] for m in metas if m and m["completion_tokens"]]
    prompt_tokens = [m["prompt_tokens"] for m in metas if m and m["prompt_tokens"]]
    by_reason: dict[str, int] = {}
    for m in metas:
        if m and m["fail_reason"]:
            by_reason[m["fail_reason"]] = by_reason.get(m["fail_reason"], 0) + 1
    truncated_count = sum(1 for m in metas if m and m["truncated"])

    def _pct(vals: list[float], p: int) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
        return round(s[k], 3)

    gen_meta = {
        "model": model,
        "dataset": dataset,
        "timestamp": int(time.time()),
        "n_items": len(items),
        "concurrency": concurrency,
        "generation": {
            "total_wall_s": round(gen_wall_s, 1),
            "latency_s": {
                "p50": _pct(latencies, 50),
                "p95": _pct(latencies, 95),
                "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            },
            "failures": {"total": failed, "by_reason": by_reason},
            "tokens": {
                "total_completion": sum(comp_tokens),
                "total_prompt": sum(prompt_tokens),
                "mean_completion": (
                    round(statistics.fmean(comp_tokens), 1) if comp_tokens else None
                ),
            },
            "truncated_count": truncated_count,
        },
        "per_item": metas,
    }
    meta_out = _gen_meta_path(out)
    meta_out.write_text(json.dumps(gen_meta, ensure_ascii=False, indent=2))
    data_volume.commit()

    success = len(items) - failed
    print(
        f"\nwrote {out}  ({success}/{len(items)} successful, {failed} failed)",
        flush=True,
    )
    if truncated_count:
        print(
            f"  warning: {truncated_count} response(s) truncated at max_tokens "
            "— those kernels are likely incomplete",
            flush=True,
        )
    print(f"wrote generation metrics -> {meta_out}", flush=True)
    return output_path


# --------------------------------------------------------------------------- #
# Evaluation — runs all three TritonBench-T phases on one GPU
# --------------------------------------------------------------------------- #


def _parse_per_kernel_speedups(stdout: str) -> list[float]:
    """Extract per-kernel speedup ratios from 2_efficiency.py stdout.

    That script prints one ``{filename}: {ratio}`` line per kernel, then a
    final ``speed up: {mean}`` summary line. The summary line is excluded
    automatically here: "speed up" contains a space, so it is not a single
    ``\\S+`` token.
    """
    import re

    out: list[float] = []
    for line in stdout.splitlines():
        m = re.match(r"^(\S+):\s+([0-9]*\.?[0-9]+)\s*$", line)
        if m:
            out.append(float(m.group(2)))
    return out


def _speedup_stats(values: list[float]) -> dict:
    """Aggregate per-kernel speedups into comparison-friendly stats.

    Geomean (not the arithmetic mean upstream uses) is the correct central
    tendency for ratios. ``n_kernels_measured`` is reported so the
    survivorship bias of any speedup aggregate is visible.
    """
    n = len(values)
    if n == 0:
        return {
            "n_kernels_measured": 0,
            "geomean_speedup": None,
            "median_speedup": None,
            "min_speedup": None,
            "max_speedup": None,
            "pct_faster_than_pytorch": None,
        }
    geomean = math.exp(sum(math.log(v) for v in values) / n)
    faster = sum(1 for v in values if v > 1.0)
    return {
        "n_kernels_measured": n,
        "geomean_speedup": round(geomean, 4),
        "median_speedup": round(statistics.median(values), 4),
        "min_speedup": round(min(values), 4),
        "max_speedup": round(max(values), 4),
        "pct_faster_than_pytorch": round(100 * faster / n, 2),
    }


@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 6,
    volumes={DATA_DIR: data_volume},
)
def evaluate(
    predictions_path: str = "predictions.jsonl",
    output_subdir: str = "results",
    model: str = "",
) -> dict:
    """Run TritonBench-T eval phases against an existing predictions.jsonl."""
    pred_full = Path(DATA_DIR) / predictions_path
    if not pred_full.exists():
        raise FileNotFoundError(f"predictions file not found in volume: {pred_full}")

    # Count total predictions and how many are already failed stubs —
    # so we can report an accurate baseline before Phase 1 even runs.
    total = 0
    gen_failures = 0
    for line in pred_full.open():
        total += 1
        rec = json.loads(line)
        code = rec.get("predict", "")
        if code.strip().startswith("# generation failed"):
            gen_failures += 1

    print(
        f"\npredictions file: {total} entries, "
        f"{gen_failures} generation failures ({total - gen_failures} usable)",
        flush=True,
    )

    out_dir = Path(DATA_DIR) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    call_acc_dir = out_dir / "call_acc"
    perf_results_dir = out_dir / "perf_results"

    if call_acc_dir.exists():
        shutil.rmtree(call_acc_dir)
    if perf_results_dir.exists():
        shutil.rmtree(perf_results_dir)

    # Make the eval modules importable as `call_acc` / `exe_acc` from any
    # subprocess (ProcessPoolExecutor pickles workers by qualified name).
    eval_dir = f"{REPO_DIR}/EVAL/eval_T"
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    os.environ["PYTHONPATH"] = eval_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

    import call_acc  # noqa: E402
    import exe_acc   # noqa: E402

    import tempfile
    import torch as _torch

    _gpu_props   = _torch.cuda.get_device_properties(0)
    _compute_cap = f"{_gpu_props.major}.{_gpu_props.minor}"
    _gpu_name    = _torch.cuda.get_device_name(0)
    failure_records: list[dict] = []   # accumulated across Phase 1 and Phase 2

    timings: dict = {}

    # ---- Phase 1 pre-probe: capture stderr before call_acc deletes failures ----
    #
    # call_acc.get_codes_for_test() returns the exact (code, test, filename)
    # triples that call_4file uses internally.  Writing code+test to a temp file
    # and running it in an isolated subprocess replicates Phase 1's acceptance
    # test, so the filenames in phase1_probe map one-to-one to call_acc_dir.
    phase1_probe:  dict[str, tuple[int, str]] = {}  # filename -> (returncode, stderr)
    _probe_codes:  list[str] = []  # raw generated code (no test appended), for kernel_code field
    _probe_fnames: list[str] = []  # TritonBench filenames, same order as predictions

    try:
        _pcodes, _ptests, _pfiles = call_acc.get_codes_for_test(str(pred_full))
        _probe_codes  = list(_pcodes)
        _probe_fnames = list(_pfiles)
        print(
            f"\nphase1 pre-probe: running {len(_pfiles)} kernels in isolated subprocesses"
            f" (timeout={KERNEL_TIMEOUT}s each) ...",
            flush=True,
        )
        with tempfile.TemporaryDirectory() as _td:
            _td_path = Path(_td)
            for _i, (_code, _test, _fname) in enumerate(zip(_pcodes, _ptests, _pfiles), 1):
                _fpath = _td_path / _fname
                _fpath.write_text(_code + "\n" + "#" * 146 + "\n" + _test)
                _rc, _se = _probe_kernel_file(_fpath)
                phase1_probe[_fname] = (_rc, _se)
                if _i % 20 == 0 or _i == len(_pfiles):
                    _nfail = sum(1 for r, _ in phase1_probe.values() if r != 0)
                    print(f"  pre-probe progress: {_i}/{len(_pfiles)}  failures so far: {_nfail}", flush=True)
        _pre_fail = sum(1 for _rc, _ in phase1_probe.values() if _rc != 0)
        print(f"phase1 pre-probe complete: {_pre_fail}/{len(phase1_probe)} predicted failures", flush=True)
    except Exception as _exc:
        print(f"warning: phase1 pre-probe skipped ({_exc})", flush=True)

    # ---- Phase 1: call accuracy ------------------------------------------------
    print("\n" + "=" * 70 + "\n=== Phase 1: call accuracy ===\n" + "=" * 70, flush=True)
    _t = time.perf_counter()
    call_acc.call_4file(str(pred_full), str(call_acc_dir), gpus=[0])
    call_survivors = sorted(p.name for p in call_acc_dir.glob("*.py"))
    timings["phase1_call_acc_s"] = round(time.perf_counter() - _t, 1)
    print(f"\ncall_acc survivors: {len(call_survivors)} / {total}", flush=True)

    # Record Phase 1 failures.  phase1_probe and call_acc use the same filenames
    # (both from get_codes_for_test), so set-difference gives an exact mapping.
    if phase1_probe:
        _call_survivor_set = set(call_survivors)
        for _fname, _code in zip(_probe_fnames, _probe_codes):
            if _fname not in _call_survivor_set:
                _rc, _se = phase1_probe.get(_fname, (-1, ""))
                _ftype = _classify_kernel_failure(_se)
                failure_records.append({
                    "kernel_id":          _fname,
                    "gpu":                _gpu_name,
                    "compute_cap":        _compute_cap,
                    "phase_failed":       1,
                    "failure_type":       _ftype,
                    "is_hardware_failure": _ftype in HARDWARE_FAILURE_TYPES,
                    "stderr_excerpt":     _se[:500].replace("\n", " "),
                    "kernel_code":        _code,
                })
        _hw = sum(1 for r in failure_records if r["phase_failed"] == 1 and r["is_hardware_failure"])
        print(
            f"phase1 failures recorded: {len(failure_records)} total, {_hw} hardware-attributable",
            flush=True,
        )

    # ---- Phase 2 pre-probe: capture stderr before exe_acc deletes failures -----
    #
    # Files in call_acc_dir already contain code + "#"*146 + test (written by
    # Phase 1).  Running them directly replicates what exe_acc does before the
    # stdout comparison step, so any Triton compilation / runtime error shows up
    # in stderr here.  Files that exit 0 but are later deleted by exe_acc failed
    # because their stdout didn't match the golden reference — numerical_mismatch.
    # filename -> (returncode, stderr, raw_kernel_code)
    # raw_kernel_code is read here because exe_acc will delete failing files.
    phase2_probe: dict[str, tuple[int, str, str]] = {}
    if call_survivors:
        print(
            f"\nphase2 pre-probe: running {len(call_survivors)} kernels in isolated subprocesses"
            f" (timeout={KERNEL_TIMEOUT}s each) ...",
            flush=True,
        )
        for _i, _fname in enumerate(call_survivors, 1):
            _fpath = call_acc_dir / _fname
            # Strip the embedded test section to recover the raw generated code
            # (Phase 1 wrote: code + "\n" + "#"*146 + "\n" + test).
            _raw = _fpath.read_text().split("#" * 146)[0].rstrip("\n")
            _rc, _se = _probe_kernel_file(_fpath)
            phase2_probe[_fname] = (_rc, _se, _raw)
            if _i % 20 == 0 or _i == len(call_survivors):
                _nfail = sum(1 for r, _, __ in phase2_probe.values() if r != 0)
                print(f"  pre-probe progress: {_i}/{len(call_survivors)}  failures so far: {_nfail}", flush=True)
        _pre_fail2 = sum(1 for _rc, _, __ in phase2_probe.values() if _rc != 0)
        print(f"phase2 pre-probe complete: {_pre_fail2}/{len(phase2_probe)} runtime failures", flush=True)

    # ---- Phase 2: execution accuracy -------------------------------------------
    print("\n" + "=" * 70 + "\n=== Phase 2: execution accuracy ===\n" + "=" * 70, flush=True)
    _t = time.perf_counter()
    if call_survivors:
        exe_acc.execute_4folder(str(call_acc_dir), gpus=[0])

    # execute_4folder removes files that fail; re-glob for the survivors.
    exec_survivors = sorted(p.name for p in call_acc_dir.glob("*.py"))
    timings["phase2_exec_acc_s"] = round(time.perf_counter() - _t, 1)
    print(f"\nexe_acc survivors: {len(exec_survivors)} / {total}", flush=True)

    # Record Phase 2 failures: call_acc survivors absent from exec_survivors.
    # Probe exit != 0  → Triton/runtime error; classify its stderr.
    # Probe exit == 0  → ran fine in isolation but stdout differed from golden
    #                    (numerical mismatch); no meaningful stderr to classify.
    if phase2_probe:
        _exec_survivor_set = set(exec_survivors)
        _p2_start = len(failure_records)
        for _fname in call_survivors:
            if _fname not in _exec_survivor_set:
                _rc, _se, _kernel_code = phase2_probe.get(_fname, (-1, "", ""))
                if _rc == 0:
                    # Passed bare execution; stdout differed from golden output.
                    _ftype = "numerical_mismatch"
                    _se    = ""
                else:
                    _ftype = _classify_kernel_failure(_se)
                failure_records.append({
                    "kernel_id":           _fname,
                    "gpu":                 _gpu_name,
                    "compute_cap":         _compute_cap,
                    "phase_failed":        2,
                    "failure_type":        _ftype,
                    "is_hardware_failure": _ftype in HARDWARE_FAILURE_TYPES,
                    "stderr_excerpt":      _se[:500].replace("\n", " "),
                    "kernel_code":         _kernel_code,
                })
        _p2_records = failure_records[_p2_start:]
        _hw2 = sum(1 for r in _p2_records if r["is_hardware_failure"])
        print(
            f"phase2 failures recorded: {len(_p2_records)} total, {_hw2} hardware-attributable",
            flush=True,
        )

    # ---- Phase 3: efficiency ---------------------------------------------------
    print("\n" + "=" * 70 + "\n=== Phase 3: efficiency ===\n" + "=" * 70, flush=True)
    _t = time.perf_counter()
    eff_summary = "skipped (no surviving operators)"
    speedup = None
    per_kernel_speedups: list[float] = []
    if exec_survivors:
        perf_root = f"{REPO_DIR}/performance_metrics/perf_T"

        # 3a — generate per-op perf scripts.
        # capture_output so we can print what write_file.py actually did;
        # no check=True — a non-zero exit is logged but we carry on so the
        # scripts that *were* written still get benchmarked.
        write_proc = subprocess.run(
            [
                sys.executable,
                "run_bench/write_file.py",
                "--input_folder_path",
                str(call_acc_dir),
                "--results_path",
                str(perf_results_dir),
            ],
            cwd=perf_root,
            capture_output=True,
            text=True,
        )
        print(f"write_file.py exit={write_proc.returncode}", flush=True)
        if write_proc.stdout.strip():
            print(write_proc.stdout[:1000], flush=True)
        if write_proc.stderr.strip():
            print("[write_file stderr]", write_proc.stderr[:1000], flush=True)

        # Diagnostic: list what was actually written to perf_results_dir.
        perf_results_dir.mkdir(parents=True, exist_ok=True)
        all_written = list(perf_results_dir.iterdir())
        print(
            f"perf_results_dir contains {len(all_written)} items: "
            f"{[p.name for p in all_written[:10]]}",
            flush=True,
        )

        # write_file.py may write scripts into a subdirectory rather than
        # directly into perf_results_dir.  Walk the whole tree.
        perf_scripts_all = sorted(perf_results_dir.rglob("*.py"))
        print(f"found {len(perf_scripts_all)} .py scripts under perf_results_dir", flush=True)

        # If still zero, print the full directory tree for diagnosis.
        if not perf_scripts_all:
            print("[diag] full tree of perf_results_dir:", flush=True)
            for p in sorted(perf_results_dir.rglob("*")):
                print(f"  {p.relative_to(perf_results_dir)}", flush=True)
            # Also check whether write_file.py wrote into the CWD (perf_root)
            # instead of perf_results_dir — some versions do this.
            cwd_py = sorted(Path(perf_root).glob("tmp/*.py"))
            if cwd_py:
                print(
                    f"[diag] found {len(cwd_py)} .py files under {perf_root}/tmp — "
                    "using those instead",
                    flush=True,
                )
                perf_scripts_all = cwd_py

        # 3b — run each generated perf script in its own isolated subprocess.
        #
        # The upstream multiprocess_gpu_run.py pools all kernels together; if
        # one leaks GPU/CPU memory it OOM-kills the entire pool (exit 137) and
        # we lose every result.  Running one-at-a-time with a hard timeout and
        # a per-process memory ceiling lets bad kernels be skipped cleanly.
        # KERNEL_TIMEOUT, VIRT_MEM_BYTES, and _set_mem_limit are module-level.

        perf_scripts = perf_scripts_all
        print(
            f"\nrunning {len(perf_scripts)} perf scripts "
            f"(timeout={KERNEL_TIMEOUT}s each, mem<=12GiB each)",
            flush=True,
        )
        perf_skipped = 0
        for idx, script in enumerate(perf_scripts, 1):
            print(f"  [{idx:2d}/{len(perf_scripts)}] {script.name}", end=" ", flush=True)
            try:
                proc = subprocess.run(
                    [sys.executable, str(script)],
                    cwd=perf_root,
                    timeout=KERNEL_TIMEOUT,
                    preexec_fn=_set_mem_limit,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode == 0:
                    print("OK", flush=True)
                else:
                    snippet = (proc.stderr or proc.stdout or "")[:120].strip()
                    print(f"exit {proc.returncode} — {snippet}", flush=True)
                    perf_skipped += 1
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT (>{KERNEL_TIMEOUT}s) — skipped", flush=True)
                perf_skipped += 1
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR — {exc} — skipped", flush=True)
                perf_skipped += 1

        print(
            f"\nperf scripts: {len(perf_scripts) - perf_skipped} completed, "
            f"{perf_skipped} skipped",
            flush=True,
        )

        # 3c — compute speedup vs. the golden PyTorch numbers.
        # Only run 2_efficiency.py when at least one perf script finished;
        # an empty results dir causes a ZeroDivisionError inside that script.
        completed_count = len(perf_scripts) - perf_skipped
        if completed_count == 0:
            eff_summary = (
                "skipped: all perf scripts failed or timed out "
                f"({perf_skipped}/{len(perf_scripts)} skipped)"
            )
            print(eff_summary, flush=True)
        else:
            eff = subprocess.run(
                [
                    sys.executable,
                    "2_efficiency.py",
                    "--gen_folder",
                    str(perf_results_dir),
                ],
                cwd=f"{REPO_DIR}/EVAL/eval_T",
                capture_output=True,
                text=True,
            )
            eff_summary = eff.stdout
            if eff.stderr:
                # Filter out the ZeroDivisionError that fires when some ops
                # have no benchmark result — it's harmless if others do.
                filtered = [
                    l for l in eff.stderr.splitlines()
                    if "ZeroDivisionError" not in l and "avg" not in l
                ]
                if filtered:
                    eff_summary += "\n[stderr]\n" + "\n".join(filtered)
            for line in eff.stdout.splitlines():
                if line.startswith("speed up:"):
                    try:
                        speedup = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
            per_kernel_speedups = _parse_per_kernel_speedups(eff.stdout)

    timings["phase3_efficiency_s"] = round(time.perf_counter() - _t, 1)

    # ---- generation telemetry from the sidecar (absent for BYO uploads) ------
    gen_meta_file = _gen_meta_path(pred_full)
    generation = None
    if gen_meta_file.exists():
        try:
            generation = json.loads(gen_meta_file.read_text()).get("generation")
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not read {gen_meta_file}: {exc}", flush=True)

    speedup_stats = _speedup_stats(per_kernel_speedups)

    summary = {
        "model": model,
        "total_predictions": total,
        "generation_failures": gen_failures,
        "usable_predictions": total - gen_failures,
        "phase1_call_acc": {
            "passed": len(call_survivors),
            "rate": round(100 * len(call_survivors) / total, 2) if total else 0,
        },
        "phase2_exec_acc": {
            "passed": len(exec_survivors),
            "rate": round(100 * len(exec_survivors) / total, 2) if total else 0,
            "rate_among_phase1": (
                round(100 * len(exec_survivors) / len(call_survivors), 2)
                if call_survivors else None
            ),
        },
        "phase3_efficiency": {
            "mean_speedup": speedup,          # upstream arithmetic mean
            **speedup_stats,                  # geomean / median / %faster / N
            "raw_output_tail": eff_summary[-2000:],
        },
        "timing_s": timings,
        "generation": generation,
        "artifacts_volume": VOLUME_NAME,
        "artifacts_subdir": output_subdir,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    data_volume.commit()
    print(f"\nwrote summary -> {summary_path}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
# Volume helpers + local entrypoints
# --------------------------------------------------------------------------- #


def _upload_local_predictions(local_path: Path) -> str:
    """Upload a local predictions.jsonl to the volume; return its remote path."""
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    remote = f"uploads/{local_path.name}"
    print(f"uploading {local_path} -> volume://{remote}", flush=True)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), remote)
    return remote


@app.local_entrypoint()
def main(
    predictions: str = "",
    model: str = DEFAULT_MODEL,
    dataset: str = "simp",
    limit: int = 0,
    output_subdir: str = "results",
    concurrency: int = 8,
    gpu: str = DEFAULT_GPU,
):
    """End-to-end: (optionally) generate predictions, then evaluate.

    Args:
        predictions: path to a local predictions.jsonl. If set, generation is
            skipped and this file is uploaded to the volume.
        model:       OpenRouter model slug (e.g. "nvidia/llama-3.1-nemotron-70b-instruct:free").
        dataset:     ``simp`` (simple) or ``comp`` (complex) Alpaca instructions.
        limit:       only generate the first N items (useful for smoke tests).
        output_subdir: where to write per-run artifacts inside the volume.
        concurrency: parallel LLM requests (raise for paid models, lower for
            free-tier RPM limits).
        gpu:         Modal GPU type for the eval phase, e.g. "T4", "L4", "A10".
    """
    if predictions:
        remote = _upload_local_predictions(Path(predictions))
    else:
        run_id = int(time.time())

        tag = (
            f"{model.replace('/', '_').replace(':', '_')}"
            f"_{dataset}"
            f"_limit{limit or 'all'}"
            f"_{run_id}"
        )
        try:
            remote = generate_predictions.remote(
                model=model,
                dataset=dataset,
                output_path=f"predictions/{tag}.jsonl",
                limit=limit if limit > 0 else None,
                concurrency=concurrency,
            )
        except Exception as exc:
            print(f"\ngeneration failed: {exc}", flush=True)
            return

    print(f"\nevaluating: volume://{remote}  (eval GPU: {gpu})\n", flush=True)
    summary = evaluate.with_options(gpu=gpu).remote(
        predictions_path=remote,
        output_subdir=output_subdir,
        model=model,
    )
    print("\n=== Final summary ===")
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def evaluate_only(
    predictions: str,
    output_subdir: str = "results",
    gpu: str = DEFAULT_GPU,
):
    """Evaluate an existing local predictions.jsonl without (re)generating.

    Usage:
        modal run modal_app.py::evaluate_only --predictions ./preds.jsonl
    """
    remote = _upload_local_predictions(Path(predictions))
    summary = evaluate.with_options(gpu=gpu).remote(
        predictions_path=remote, output_subdir=output_subdir
    )
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def generate_only(
    model: str = DEFAULT_MODEL,
    dataset: str = "simp",
    limit: int = 0,
    output_path: str = "predictions/predictions.jsonl",
    concurrency: int = 8,
):
    """Generate predictions only; do not evaluate.

    Usage:
        modal run modal_app.py::generate_only --model "nvidia/llama-3.1-nemotron-70b-instruct:free"
    """
    remote = generate_predictions.remote(
        model=model,
        dataset=dataset,
        output_path=output_path,
        limit=limit if limit > 0 else None,
        concurrency=concurrency,
    )
    print(f"wrote volume://{remote}")