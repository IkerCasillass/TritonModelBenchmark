"""Hardware-in-the-loop generate -> run -> refine loop.

A generator writes a Triton kernel; it is compiled and run on the target GPU and
checked against the PyTorch reference; on failure the error is turned into an
actionable hint and the generator revises. The loop ends when the kernel runs and
its output matches the reference, or after ``max_iters``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import modal

from .core import (DATA_DIR, DEFAULT_GPU, DEFAULT_MODEL, LLM_SECRET_NAME,
                   T4_FAILURE_HINTS, app, data_volume)
from .generator import generate_kernel


class JudgeResult(TypedDict):
    compiled: bool
    ran: bool
    correct: bool
    failure_type: str | None     # a _classify_kernel_failure label, or None
    raw_stderr: str
    diagnostic: str              # e.g. a stdout diff on numerical mismatch


FEEDBACK_MODES = ("interpreted", "category", "raw")


def judge_kernel(generated_code: str, operator_id: str) -> JudgeResult:
    """Compile and run the kernel on the target GPU and check correctness.

    Write ``code + sep + test`` to a temp file, run via
    ``kernels._probe_kernel_file`` and classify any failure with
    ``kernels._classify_kernel_failure``; on a clean run, compare stdout to the
    cached PyTorch-reference output (mismatch -> ``numerical_mismatch``).
    """
    raise NotImplementedError


def interpret_failure(generated_code: str, jr: JudgeResult) -> str:
    """Return an actionable fix-hint for a failure, keyed on its classified type.

    Deterministic baseline; may be upgraded to a model call grounded by the same
    ``failure_type`` + raw stderr.
    """
    if jr["correct"]:
        return ""
    return T4_FAILURE_HINTS.get(jr["failure_type"] or "", "Fix the reported error.")


def _build_feedback(jr: JudgeResult, hint: str, feedback_mode: str) -> str:
    """Render the feedback shown to the generator for the next attempt."""
    if feedback_mode == "raw":
        return f"The kernel failed on the target GPU. Raw error:\n{jr['raw_stderr'] or jr['diagnostic']}"
    if feedback_mode == "category":
        return f"The kernel failed on the target GPU with failure category: {jr['failure_type']}."
    return f"The kernel failed on the target GPU ({jr['failure_type']}). {hint}"


def _refine_one(operator_id: str, gen_model: str, interp_model: str,
                max_iters: int = 5, feedback_mode: str = "interpreted") -> dict:
    """Run the loop for one operator; return its trajectory."""
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
        hint = interpret_failure(code, jr) if feedback_mode == "interpreted" else ""
        history.append({"code": code, "judge": jr, "hint": hint,
                        "feedback": _build_feedback(jr, hint, feedback_mode)})
    passed = bool(status_per_iter) and status_per_iter[-1] == "pass"
    return {
        "operator_id": operator_id, "gen_model": gen_model, "interp_model": interp_model,
        "feedback_mode": feedback_mode, "passed": passed,
        "iterations": len(status_per_iter),
        "status_per_iter": status_per_iter, "failure_type_per_iter": ftype_per_iter,
        "final_code": final_code,
    }


def _select_operators(limit: int | None) -> list[str]:
    """Operator ids to refine, drawn from the TritonBench-T set (golden refs cached)."""
    raise NotImplementedError


@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 4,
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name(LLM_SECRET_NAME)],
)
def generate_refine(
    gen_model: str = DEFAULT_MODEL,
    interp_model: str = "",
    limit: int | None = None,
    max_iters: int = 5,
    feedback_mode: str = "interpreted",
    output_subdir: str = "refine",
) -> dict:
    """Run the refine loop across operators and write per-operator trajectories + a summary."""
    interp_model = interp_model or gen_model
    operators = _select_operators(limit)
    rows = [_refine_one(op, gen_model, interp_model, max_iters, feedback_mode)
            for op in operators]

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
