"""Task D — generation engine + model bench + stub evaluator.  OWNER: <teammate D>.

Deliverables:
  1. `generate_kernel(operator_id, history) -> code` — the integration seam: build
     messages (prompting.py + refinement.py), call the small model (`llm._gen`),
     extract + validate code (`llm._extract_code` / `_is_valid_python`).
  2. `stub_evaluator` — a scriptable JudgeResult source so B & C can develop the
     generator before A's real `judge_kernel` exists.
  3. (stretch) a multi-small-model bench: run the loop across candidate slugs.

The STUB below is content-reactive so the whole loop is demonstrable end-to-end
with no GPU and no real model: first attempt emits bf16 (→ a dtype "failure"),
then switches to fp32 once feedback arrives.
"""
from __future__ import annotations


def generate_kernel(operator_id: str, history: list[dict]) -> str:
    """[OWNER D — STUB] Return Triton source for operator_id, revised from history.

    Real version: messages = refinement.build_messages(..., prompting.build_system_prompt(),
    prompting.build_user_prompt(...)); code = llm._extract_code(llm._gen(messages, model)).
    """
    if not history:
        return (
            "import torch\nimport triton\nimport triton.language as tl\n"
            "# STUB first attempt (uses bf16 -> should fail on a T4)\n"
            "def kernel():\n    return tl.zeros([16], dtype=tl.bfloat16)\n"
        )
    return (
        "import torch\nimport triton\nimport triton.language as tl\n"
        "# STUB revised attempt (fp32 after feedback)\n"
        "def kernel():\n    return tl.zeros([16], dtype=tl.float32)\n"
    )


def stub_evaluator(generated_code: str, operator_id: str) -> dict:
    """Scriptable stand-in for A's `judge_kernel` so B/C can develop in isolation.

    Mirror of the JudgeResult contract; 'bf16' in the code → dtype failure, else pass.
    """
    if "bf16" in generated_code or "bfloat16" in generated_code:
        return {"compiled": False, "ran": False, "correct": False,
                "failure_type": "dtype_unsupported",
                "raw_stderr": "Feature '.bf16' requires .target sm_80 or higher",
                "diagnostic": ""}
    return {"compiled": True, "ran": True, "correct": True,
            "failure_type": None, "raw_stderr": "", "diagnostic": ""}
