"""Kernel generation engine: assemble prompts, call the model, extract code."""
from __future__ import annotations

from ..llm import _extract_code, _is_valid_python
from ..llm_local import _gen_constrained
from ..operators import get_instruction
from .prompting import build_system_prompt, build_user_prompt
from .refinement import build_messages

# Fixed boilerplate every generated module needs. The grammar emits only kernel
# logic (no imports) — these are identical for every operator, so they are
# injected deterministically here rather than spent on grammar tokens. The judge
# runs exactly what generate_kernel returns; it does not (and must not) patch code.
_IMPORTS = "import torch\nimport triton\nimport triton.language as tl\n\n\n"


def generate_kernel(operator_id: str, history: list[dict]) -> str:
    """Generate (or revise) Triton source for ``operator_id``.

    Build messages from ``prompting`` + ``refinement``, call the model via
    grammar-constrained Qwen, validate the generated logic, then prepend the
    fixed import header so the return value is a *complete, runnable module*.
    ``history`` carries prior attempts and their feedback.
    """
    try:
        instruction = get_instruction(operator_id)
    except KeyError:
        instruction = operator_id
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(operator_id, instruction)
    messages = build_messages(operator_id, history, system_prompt, user_prompt)

    try:
        raw = _gen_constrained(messages)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"kernel generation failed for {operator_id}: {exc}"
        ) from exc

    code = _extract_code(raw)
    if not code.strip():
        raise ValueError(
            f"LLM returned an empty code payload for {operator_id}: {raw[:800]!r}"
        )
    # Validate the generated logic itself — a truncated kernel fails here and is
    # surfaced as a generation error for the refine loop to retry.
    if not _is_valid_python(code):
        raise ValueError(
            f"LLM returned invalid Python for {operator_id}: {code[:800]!r}"
        )
    return _IMPORTS + code.lstrip()


def stub_evaluator(generated_code: str, operator_id: str) -> dict:
    """Deterministic ``judge_kernel`` test double (no GPU): bf16 -> dtype failure, else pass."""
    if "bf16" in generated_code or "bfloat16" in generated_code:
        return {"compiled": False, "ran": False, "correct": False,
                "failure_type": "dtype_unsupported",
                "raw_stderr": "Feature '.bf16' requires .target sm_80 or higher",
                "diagnostic": ""}
    return {"compiled": True, "ran": True, "correct": True,
            "failure_type": None, "raw_stderr": "", "diagnostic": ""}