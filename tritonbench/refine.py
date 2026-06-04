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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import modal

from . import operators as _operators
from .core import (DATA_DIR, DEFAULT_GPU, DEFAULT_MODEL, LLM_SECRET_NAME,
                   T4_FAILURE_HINTS, app, data_volume)
from .generator import generate_kernel
from .kernels import _classify_kernel_failure, _run_kernel_capture


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


def judge_kernel(generated_code: str, operator_id: str) -> JudgeResult:
    """Compile and run the kernel on the target GPU; check against the cached reference output.

    Writes ``code + sep + test`` to a temp file, runs it under ``_run_kernel_capture``,
    and classifies any failure. On a clean run, compares stdout to the operator's
    golden reference (exact match, trailing whitespace stripped).
    """
    spec = _operators.get_operator(operator_id)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
        tmp = fh.name
        fh.write(generated_code + "\n" + "#" * 146 + "\n" + spec["test_code"])

    try:
        with _gpu_lock:
            rc, stdout, stderr = _run_kernel_capture(Path(tmp))
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

    if stdout.rstrip() == spec["golden_stdout"].rstrip():
        return {
            "compiled": True, "ran": True, "correct": True,
            "failure_type": None, "raw_stderr": "", "diagnostic": "",
        }

    expected = spec["golden_stdout"].rstrip().splitlines()
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
    ``grounded`` is implemented by the interpreter module.
    """
    if jr["correct"] or mode in ("raw", "category"):
        return ""
    return T4_FAILURE_HINTS.get(jr["failure_type"] or "", "Fix the reported error.")


def _build_feedback(jr: JudgeResult, hint: str, feedback_mode: str) -> str:
    if feedback_mode == "raw":
        return f"The kernel failed on the target GPU. Raw error:\n{jr['raw_stderr'] or jr['diagnostic']}"
    if feedback_mode == "category":
        return f"The kernel failed on the target GPU with failure category: {jr['failure_type']}."
    return f"The kernel failed on the target GPU ({jr['failure_type']}). {hint}"


def _refine_one(operator_id: str, gen_model: str, interp_model: str,
                max_iters: int = 5, feedback_mode: str = "interpreted") -> dict:
    history: list[dict] = []
    status_per_iter: list[str] = []
    ftype_per_iter: list[str | None] = []
    final_code = ""

    for _ in range(max_iters):
        code = generate_kernel(operator_id, history)
        final_code = code
        jr = judge_kernel(code, operator_id)
        status_per_iter.append("pass" if jr["correct"] else "fail")
        ftype_per_iter.append(jr["failure_type"])
        if jr["correct"]:
            break
        hint = interpret_failure(code, jr, interp_model=interp_model, mode=feedback_mode)
        history.append({
            "code": code,
            "judge": jr,
            "hint": hint,
            "feedback": _build_feedback(jr, hint, feedback_mode),
        })

    n = len(status_per_iter)
    hint_helped_per_iter = [
        status_per_iter[i + 1] == "pass" or ftype_per_iter[i + 1] != ftype_per_iter[i]
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
        "hint_helped_per_iter": hint_helped_per_iter,
        "final_code": final_code,
    }


@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 4,
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name(LLM_SECRET_NAME)],
)
def generate_refine(
    gen_model: str = DEFAULT_MODEL,
    interp_model: str = "",
    dataset: str = "simp",
    limit: int | None = None,
    max_iters: int = 5,
    feedback_mode: str = "interpreted",
    output_subdir: str = "refine",
    concurrency: int = 4,
) -> dict:
    """Run the refine loop across operators and write per-operator trajectories and a summary."""
    interp_model = interp_model or gen_model
    _operators.build_registry(dataset, limit)
    operators = _operators.select_operators(limit)

    _worker = functools.partial(
        _refine_one,
        gen_model=gen_model,
        interp_model=interp_model,
        max_iters=max_iters,
        feedback_mode=feedback_mode,
    )
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        rows = list(ex.map(_worker, operators))

    out_dir = Path(DATA_DIR) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "refine_dataset.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    summary = {
        "n_operators": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4) if n else None,
        "pass_at_1": round(sum(r["status_per_iter"][:1] == ["pass"] for r in rows) / n, 4) if n else None,
        "mean_iterations": round(sum(r["iterations"] for r in rows) / n, 2) if n else None,
        "feedback_mode": feedback_mode,
    }
    (out_dir / "refine_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    data_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary
