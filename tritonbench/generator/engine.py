"""Kernel generation engine: assemble prompts, call the model, extract code."""
from __future__ import annotations


def generate_kernel(operator_id: str, history: list[dict]) -> str:
    """Generate (or revise) Triton source for ``operator_id``.

    Build messages from ``prompting`` + ``refinement``, call the model via
    ``llm._gen``, and return validated source via ``llm._extract_code`` /
    ``_is_valid_python``. ``history`` carries prior attempts and their feedback.
    """
    raise NotImplementedError


def stub_evaluator(generated_code: str, operator_id: str) -> dict:
    """Deterministic ``judge_kernel`` test double (no GPU): bf16 -> dtype failure, else pass."""
    if "bf16" in generated_code or "bfloat16" in generated_code:
        return {"compiled": False, "ran": False, "correct": False,
                "failure_type": "dtype_unsupported",
                "raw_stderr": "Feature '.bf16' requires .target sm_80 or higher",
                "diagnostic": ""}
    return {"compiled": True, "ran": True, "correct": True,
            "failure_type": None, "raw_stderr": "", "diagnostic": ""}
