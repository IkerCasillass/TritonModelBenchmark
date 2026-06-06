"""Kernel generation engine: assemble prompts, call the model, extract code."""
from __future__ import annotations

from ..llm import _extract_code, _is_valid_python
from .llm_local import _gen_constrained  # ← NEW: grammar-constrained Qwen path
from .prompting import build_system_prompt, build_user_prompt
from .refinement import build_messages


def _resolve_instruction(operator_id: str, history: list[dict]) -> str:
    """Best-effort task text lookup without hardcoding prompt content."""
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        instruction = entry.get("instruction") or entry.get("prompt") or entry.get("task")
        if isinstance(instruction, str) and instruction.strip():
            return instruction
    return operator_id


def generate_kernel(operator_id: str, history: list[dict]) -> str:
    """Generate (or revise) Triton source for ``operator_id``.

    Build messages from ``prompting`` + ``refinement``, call the model via
    grammar-constrained Qwen, and return validated source via
    ``llm._extract_code`` / ``_is_valid_python``.
    ``history`` carries prior attempts and their feedback.
    """
    instruction = _resolve_instruction(operator_id, history)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(operator_id, instruction)
    messages = build_messages(operator_id, history, system_prompt, user_prompt)

    try:
        raw = _gen_constrained(messages)  # ← CHANGED: was _gen(messages, DEFAULT_MODEL)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"kernel generation failed for {operator_id}: {exc}"
        ) from exc

    code = _extract_code(raw)  # ← CHANGED: was _extract_code(result.content)
    if not code.strip():
        raise ValueError(
            f"LLM returned an empty code payload for {operator_id}"
        )
    if not _is_valid_python(code):
        raise ValueError(
            f"LLM returned invalid Python for {operator_id}"
        )
    return code


def stub_evaluator(generated_code: str, operator_id: str) -> dict:
    """Deterministic ``judge_kernel`` test double (no GPU): bf16 -> dtype failure, else pass."""
    if "bf16" in generated_code or "bfloat16" in generated_code:
        return {"compiled": False, "ran": False, "correct": False,
                "failure_type": "dtype_unsupported",
                "raw_stderr": "Feature '.bf16' requires .target sm_80 or higher",
                "diagnostic": ""}
    return {"compiled": True, "ran": True, "correct": True,
            "failure_type": None, "raw_stderr": "", "diagnostic": ""}