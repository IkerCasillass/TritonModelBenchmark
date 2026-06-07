"""Hardware-in-the-loop generate -> run -> refine loop.

A generator writes a Triton kernel; it is compiled and run on the target GPU and
checked against the PyTorch reference; on failure the error is turned into an
actionable hint and the generator revises. The loop ends when the kernel runs and
its output matches the reference, or after ``max_iters``.
"""
from __future__ import annotations

import functools
import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import modal

from . import llm
from . import llm_local
from . import operators as _operators
from .core import (DATA_DIR, DEFAULT_GPU, DEFAULT_INTERP_MODEL, DEFAULT_MODEL,
                   LLM_SECRET_NAME, T4_FAILURE_HINTS, app, data_volume)
from .generator import backends, generate_kernel
from .kernels import (HARDWARE_FAILURE_TYPES, _classify_kernel_failure,
                      _run_kernel_capture)


class JudgeResult(TypedDict):
    compiled: bool
    ran: bool
    correct: bool
    failure_type: str | None
    raw_stderr: str
    diagnostic: str


FEEDBACK_MODES = ("raw", "category", "interpreted", "grounded")

# Failure types that indicate the kernel never reached execution.
_COMPILE_FAILURE_TYPES = frozenset({
    "code_error",
    "triton_api_misuse",
    "triton_unsupported_construct",
    "triton_compilation_error",
})

# Serialises GPU subprocesses: a single T4 cannot safely run concurrent kernels.
_gpu_lock = threading.Lock()

# Guards against runaway kernels. Candidate kernels often hang (bad loops/launch);
# cap each judge well under the default, and cap total time spent per operator so
# one operator can't burn the whole run on repeated timeouts.
JUDGE_TIMEOUT = 30          # seconds per candidate kernel run
OPERATOR_BUDGET = 240       # seconds total across an operator's iterations


def judge_kernel(generated_code: str, operator_id: str) -> JudgeResult:
    """Compile and run the kernel on the target GPU; check against the cached reference output.

    Writes ``code + sep + test`` to a temp file, runs it under ``_run_kernel_capture``,
    and classifies any failure. On a clean run, compares stdout to the operator's
    golden reference (exact match, trailing whitespace stripped).
    """
    spec = _operators.get_operator(operator_id)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
        tmp = fh.name
        # Append print(test_results) so the harness's collected outputs reach
        # stdout — captured identically for the golden, so the compare is real.
        fh.write(generated_code + "\n" + "#" * 146 + "\n" + spec["test_code"]
                 + "\nprint(test_results)\n")

    try:
        with _gpu_lock:
            rc, stdout, stderr = _run_kernel_capture(Path(tmp), timeout=JUDGE_TIMEOUT)
    finally:
        Path(tmp).unlink(missing_ok=True)

    if rc != 0:
        failure_type = _classify_kernel_failure(stderr)
        return {
            "compiled": failure_type not in _COMPILE_FAILURE_TYPES,
            "ran": False,
            "correct": False,
            "failure_type": failure_type,
            "raw_stderr": stderr,
            "diagnostic": "",
        }

    golden = spec["golden_stdout"].rstrip()
    # Require a non-empty golden: matching an empty reference is a false pass.
    if golden and stdout.rstrip() == golden:
        return {
            "compiled": True, "ran": True, "correct": True,
            "failure_type": None, "raw_stderr": "", "diagnostic": "",
        }

    expected = golden.splitlines()
    actual = stdout.rstrip().splitlines()
    diagnostic = next(
        (f"line {i + 1}: expected {e!r}, got {a!r}"
         for i, (e, a) in enumerate(zip(expected, actual)) if e != a),
        f"length mismatch: {len(expected)} vs {len(actual)} lines",
    )
    return {
        "compiled": True, "ran": True, "correct": False,
        "failure_type": "numerical_mismatch",
        "raw_stderr": "",
        "diagnostic": diagnostic,
    }


def interpret_failure(generated_code: str, jr: JudgeResult, *,
                      interp_model: str = "", mode: str = "interpreted") -> str:
    """Return an actionable fix-hint for a failure.

    ``raw`` and ``category`` modes carry no hint (feedback is built from the
    JudgeResult fields directly). ``interpreted`` uses a deterministic lookup.
    ``grounded`` calls the LLM and falls back to the deterministic hint on any error.
    """
    if jr["correct"] or mode in ("raw", "category"):
        return ""

    base_hint = T4_FAILURE_HINTS.get(jr["failure_type"] or "", "Fix the reported error.")
    if mode == "interpreted":
        return base_hint
    if mode != "grounded":
        return base_hint

    # Build a grounded prompt from the category, its deterministic hint, stderr tail, and the code.
    failure_type = jr["failure_type"] or "other_runtime"
    stderr = (jr["raw_stderr"] or "").strip()
    diagnostic = (jr["diagnostic"] or "").strip()
    stderr_tail = stderr[-2000:] if stderr else ""
    evidence = stderr_tail or diagnostic

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Triton (OpenAI Triton) GPU-kernel debugging assistant. "
                "Given a failure category, the T4-specific hint, stderr/diagnostic, and the kernel code, "
                "suggest one concrete, kernel-specific change that will likely fix the issue on an NVIDIA T4 (sm_75). "
                "Be precise and actionable. Output only the fix/instruction, no preamble."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Failure category: {failure_type}\n"
                f"Category hint: {base_hint}\n"
                f"Stderr/diagnostic (tail):\n{evidence}\n\n"
                f"Kernel code:\n{generated_code}\n\n"
                "Task: Provide one concrete edit or small set of edits to this kernel to address the failure. "
                "Mention exact tl.* ops, masks, types, and constants to change. "
                "Do not restate the category; do not propose rewrites unrelated to the error."
            ),
        },
    ]
    try:
        resp = llm._gen(messages, interp_model)
        text = resp.content.strip()
        return text or base_hint
    except Exception:
        # Network/model errors must not break the refine loop.
        return base_hint


def _build_feedback(jr: JudgeResult, hint: str, feedback_mode: str) -> str:
    if feedback_mode == "raw":
        return f"The kernel failed on the target GPU. Raw error:\n{jr['raw_stderr'] or jr['diagnostic']}"
    if feedback_mode == "category":
        return f"The kernel failed on the target GPU with failure category: {jr['failure_type']}."
    if feedback_mode in ("interpreted", "grounded"):
        return f"The kernel failed on the target GPU ({jr['failure_type']}). {hint}"
    return f"The kernel failed on the target GPU ({jr['failure_type']}). {hint}"


def _progress_tier(jr: JudgeResult) -> int:
    """How far the kernel got: 0 no-compile, 1 compiled, 2 ran, 3 correct."""
    if jr["correct"]:
        return 3
    if jr["ran"]:
        return 2
    if jr["compiled"]:
        return 1
    return 0


def _refine_one(operator_id: str, gen_model: str, interp_model: str,
                max_iters: int = 3, feedback_mode: str = "interpreted") -> dict:
    history: list[dict] = []
    status_per_iter: list[str] = []
    ftype_per_iter: list[str | None] = []
    tier_per_iter: list[int] = []
    hint_per_iter: list[str] = []
    final_code = ""
    gen_error = ""

    deadline = time.monotonic() + OPERATOR_BUDGET
    for _ in range(max_iters):
        if time.monotonic() > deadline:
            break  # operator budget exhausted (e.g. repeated timeouts)
        try:
            code = generate_kernel(operator_id, history)
        except Exception as exc:  # generation produced no usable code — feed back and retry
            gen_error = str(exc)
            status_per_iter.append("fail")
            ftype_per_iter.append("generation_error")
            tier_per_iter.append(0)
            hint_per_iter.append("")
            history.append({
                "code": "",
                "judge": {"compiled": False, "ran": False, "correct": False,
                          "failure_type": "generation_error",
                          "raw_stderr": gen_error, "diagnostic": ""},
                "hint": "",
                "feedback": (
                    f"The previous attempt did not produce a usable kernel: {gen_error}. "
                    "Return one complete, valid Python module — imports, the @triton.jit "
                    "kernel, and the wrapper function."
                ),
            })
            continue
        final_code = code
        jr = judge_kernel(code, operator_id)
        status_per_iter.append("pass" if jr["correct"] else "fail")
        ftype_per_iter.append(jr["failure_type"])
        tier_per_iter.append(_progress_tier(jr))
        if jr["correct"]:
            hint_per_iter.append("")
            break
        hint = interpret_failure(code, jr, interp_model=interp_model, mode=feedback_mode)
        hint_per_iter.append(hint[:1000])
        history.append({
            "code": code,
            "judge": jr,
            "hint": hint,
            "feedback": _build_feedback(jr, hint, feedback_mode),
        })

    n = len(status_per_iter)
    hint_helped_per_iter = [
        tier_per_iter[i + 1] > tier_per_iter[i]
        for i in range(n - 1)
        if status_per_iter[i] == "fail"
    ]

    passed = bool(status_per_iter) and status_per_iter[-1] == "pass"
    return {
        "operator_id": operator_id,
        "gen_model": gen_model,
        "interp_model": interp_model,
        "feedback_mode": feedback_mode,
        "passed": passed,
        "iterations": len(status_per_iter),
        "status_per_iter": status_per_iter,
        "failure_type_per_iter": ftype_per_iter,
        "tier_per_iter": tier_per_iter,
        "hint_per_iter": hint_per_iter,
        "hint_helped_per_iter": hint_helped_per_iter,
        "final_code": final_code,
        "gen_error": gen_error or None,
    }


@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 4,
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name(LLM_SECRET_NAME)],
)
def generate_refine(
    gen_model: str = DEFAULT_MODEL,
    interp_model: str = DEFAULT_INTERP_MODEL,
    dataset: str = "simp",
    limit: int | None = None,
    operators: list[str] | None = None,
    max_iters: int = 3,
    feedback_mode: str = "interpreted",
    output_subdir: str = "refine",
    concurrency: int = 2,
    gen_backend: str = "local",
    gen_constrained: bool = True,
) -> dict:
    """Run the refine loop across operators and write per-operator trajectories and a summary.

    ``operators``: optional list of operator ids to target (with or without
    ``.py``); when set it overrides ``limit``-based selection so a run can focus
    on a specific (e.g. easy) operator regardless of dataset order.
    """
    interp_model = interp_model or DEFAULT_INTERP_MODEL
    backends.set_active_backend(gen_backend)
    backends.set_constrained(gen_constrained)
    if gen_backend == "vllm":
        from . import gen_service
        gen_model = gen_service.GEN_MODEL_ID
    else:
        gen_model = llm_local._MODEL_ID
    # When `operators` is set, build_registry runs only those goldens (not the
    # whole suite), so a single-operator run doesn't pay for all 166.
    _operators.build_registry(dataset, limit, operators=operators)
    op_ids = _operators.select_operators()

    _worker = functools.partial(
        _refine_one,
        gen_model=gen_model,
        interp_model=interp_model,
        max_iters=max_iters,
        feedback_mode=feedback_mode,
    )
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        rows = list(ex.map(_worker, op_ids))

    out_dir = Path(DATA_DIR) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "refine_dataset.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    passed = sum(r["passed"] for r in rows)

    # Aggregate whether hints helped (fail -> pass or moved to less-severe category) across all opportunities.
    hint_events = [b for r in rows for b in r.get("hint_helped_per_iter", [])]
    opportunities = len(hint_events)
    helped = sum(1 for b in hint_events if b)
    help_rate = round(helped / opportunities, 4) if opportunities else None

    def _tiers(r):
        return r.get("tier_per_iter") or [0]

    # Compile = reached tier >= 1 (kernel compiled; failed, if at all, at runtime).
    compile_at_1 = sum(_tiers(r)[0] >= 1 for r in rows)
    compiled_ever = sum(max(_tiers(r)) >= 1 for r in rows)

    ran_at_1 = sum(_tiers(r)[0] >= 2 for r in rows)
    ran_ever = sum(max(_tiers(r)) >= 2 for r in rows)

    # Behavioral hardware-awareness: operators that produced a T4-illegal kernel
    # (bf16/shared-mem/etc.) at any iteration.
    hw_ops = sum(
        1 for r in rows
        if any(ft in HARDWARE_FAILURE_TYPES for ft in r["failure_type_per_iter"])
    )
    by_type: dict[str, int] = {}
    for r in rows:
        if r["passed"]:
            continue
        fts = r["failure_type_per_iter"]
        ft = fts[-1] if fts else None
        if ft:
            by_type[ft] = by_type.get(ft, 0) + 1

    summary = {
        "n_operators": n,
        "gen_model": gen_model,
        "interp_model": interp_model,
        "passed": passed,
        "pass_rate": round(passed / n, 4) if n else None,
        "pass_at_1": round(sum(r["status_per_iter"][:1] == ["pass"] for r in rows) / n, 4) if n else None,
        "compile_at_1": round(compile_at_1 / n, 4) if n else None,
        "compile_rate": round(compiled_ever / n, 4) if n else None,
        # TritonBench-comparable metric (kernel runs without crashing).
        "ran_at_1": round(ran_at_1 / n, 4) if n else None,
        "ran_rate": round(ran_ever / n, 4) if n else None,
        "mean_iterations": round(sum(r["iterations"] for r in rows) / n, 2) if n else None,
        "feedback_mode": feedback_mode,
        "hint_helped": {
            "mode": feedback_mode,
            "opportunities": opportunities,
            "helped": helped,
            "help_rate": help_rate,
        },
        "failure_analysis": {
            "by_type": by_type,
            "hardware_failures": hw_ops,
            "hardware_failure_rate": round(hw_ops / n, 4) if n else None,
        },
    }
    (out_dir / "refine_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    data_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(timeout=60 * 30)
def list_operators(dataset: str = "simp", limit: int | None = None) -> list[tuple[str, bool]]:
    """List operator ids (and whether each has a golden file), no GPU/golden runs.

    Use it to pick a target for ``refine_loop --operator <id>`` — prefer simple
    names (add/mul/relu/gelu) over ``fused_*`` for a first passing smoke test.
    """
    rows = _operators.list_operator_ids(dataset, limit)
    for fname, has_gold in rows:
        flag = "" if has_gold else "  (no gold file)"
        print(f"{fname}{flag}", flush=True)
    print(f"\n{len(rows)} operators", flush=True)
    return rows
