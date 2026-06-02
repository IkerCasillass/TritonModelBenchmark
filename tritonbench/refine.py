"""Phase C — hardware-in-the-loop generate->run->refine loop.

OWNER A (you): judge_kernel (real hardware judge), interpret_failure (small
interpreter grounded by the classifier), the loop controller `_refine_one`, the
`generate_refine` Modal entrypoint, and metrics.

Section 0 scaffold: the FROZEN CONTRACT + runnable STUBs. Each owner replaces the
STUB body of their function; the seam (signatures + JudgeResult) does not change.
See docs/PHASE_C_TASKS.md. The stubs are content-reactive so the whole loop runs
end-to-end today (no GPU, no real model) to prove the wiring.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import modal

from .core import *  # noqa: F401,F403
from .core import T4_FAILURE_HINTS
from .generator import generate_kernel   # owned by B/C/D


class JudgeResult(TypedDict):
    compiled: bool
    ran: bool
    correct: bool
    failure_type: str | None     # one of kernels._classify_kernel_failure's labels, or None
    raw_stderr: str
    diagnostic: str              # e.g. a stdout diff on numerical_mismatch


FEEDBACK_MODES = ("interpreted", "category", "raw")


# --- CONTRACT (Owner A fills these two) ------------------------------------- #

def judge_kernel(generated_code: str, operator_id: str) -> JudgeResult:
    """[OWNER A — STUB] Hardware-in-the-loop judge: compile+run on the T4 and check
    correctness vs the PyTorch reference.

    Real implementation (reuse, no rebuild):
      - write `code + "\\n" + "#"*146 + "\\n" + test` to a tmp .py
      - run via `kernels._probe_kernel_file(path, timeout=...)`; non-zero -> classify
        with `kernels._classify_kernel_failure(stderr)`
      - on clean run, compare stdout to the cached golden (PyTorch ref) -> set
        correct / failure_type="numerical_mismatch" + a short diff in `diagnostic`.

    Stub: content-reactive so the loop demonstrably iterates — 'bf16' in the code
    'fails' as a dtype error, otherwise it 'passes'.
    """
    if "bf16" in generated_code or "bfloat16" in generated_code:
        return JudgeResult(
            compiled=False, ran=False, correct=False,
            failure_type="dtype_unsupported",
            raw_stderr="Feature '.bf16' requires .target sm_80 or higher",
            diagnostic="",
        )
    return JudgeResult(compiled=True, ran=True, correct=True,
                       failure_type=None, raw_stderr="", diagnostic="")


def interpret_failure(generated_code: str, jr: JudgeResult) -> str:
    """[OWNER A — STUB] Turn the hardware failure into an actionable, code-level fix.

    Real implementation: a small-model `llm._gen` call, GROUNDED by the pinned
    `failure_type` (-> `core.T4_FAILURE_HINTS`) + raw stderr, so it phrases the fix
    without re-diagnosing. Stub returns the curated category hint directly.
    """
    if jr["correct"]:
        return ""
    return T4_FAILURE_HINTS.get(jr["failure_type"] or "", "Fix the reported error.")


# --- Loop controller (Owner A) ---------------------------------------------- #

def _build_feedback(jr: JudgeResult, hint: str, feedback_mode: str) -> str:
    """Shape what the generator sees next, per the ablation mode."""
    if feedback_mode == "raw":
        return f"The kernel failed on the T4. Raw error:\n{jr['raw_stderr'] or jr['diagnostic']}"
    if feedback_mode == "category":
        return f"The kernel failed on the T4 with failure category: {jr['failure_type']}."
    return f"The kernel failed on the T4 ({jr['failure_type']}). {hint}"   # interpreted


def _refine_one(operator_id: str, gen_model: str, interp_model: str,
                max_iters: int = 5, feedback_mode: str = "interpreted") -> dict:
    """generate -> judge -> interpret, until correct or max_iters. Returns trajectory."""
    history: list[dict] = []
    status_per_iter: list[str] = []
    ftype_per_iter: list[str | None] = []
    final_code = ""
    for it in range(max_iters):
        code = generate_kernel(operator_id, history)        # B/C/D
        final_code = code
        jr = judge_kernel(code, operator_id)                # A (hardware)
        status_per_iter.append("pass" if jr["correct"] else "fail")
        ftype_per_iter.append(jr["failure_type"])
        if jr["correct"]:
            break
        hint = interpret_failure(code, jr) if feedback_mode == "interpreted" else ""  # A
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


@app.function(  # noqa: F405
    gpu=DEFAULT_GPU,  # noqa: F405
    timeout=60 * 60 * 4,
    volumes={DATA_DIR: data_volume},  # noqa: F405
    secrets=[modal.Secret.from_name(LLM_SECRET_NAME)],  # noqa: F405
)
def generate_refine(
    gen_model: str = DEFAULT_MODEL,  # noqa: F405
    interp_model: str = "",
    limit: int | None = None,
    max_iters: int = 5,
    feedback_mode: str = "interpreted",
    output_subdir: str = "refine",
) -> dict:
    """Phase C orchestrator. SCAFFOLD: runs `_refine_one` across operators on stubs.

    Real version (Owner A): operator list via `llm._load_alpaca` +
    `call_acc.get_corresponding_files`, precompute & cache golden stdout per
    operator, run `_refine_one` across them with a ThreadPoolExecutor.
    """
    interp_model = interp_model or gen_model
    # SCAFFOLD operator ids (the real version loads the alpaca/TritonBench-T set).
    operators = [f"stub_op_{i}.py" for i in range(limit or 3)]
    rows = [_refine_one(op, gen_model, interp_model, max_iters, feedback_mode)
            for op in operators]

    out_dir = Path(DATA_DIR) / output_subdir  # noqa: F405
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
        "scaffold": True,   # remove when judge_kernel/generate_kernel are real
    }
    (out_dir / "refine_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    data_volume.commit()  # noqa: F405
    print(json.dumps(summary, indent=2), flush=True)
    return summary
