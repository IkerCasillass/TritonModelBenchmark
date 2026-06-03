"""Prompt construction for hardware-aware kernel generation."""
from __future__ import annotations

from ..core import T4_HARDWARE
from ..llm import PROMPT_HEADER


def build_system_prompt() -> str:
    """Base generation instructions plus the target GPU's hardware constraints."""
    constraints = "\n".join(f"- {c}" for c in T4_HARDWARE["constraints"])
    return (
        f"{PROMPT_HEADER}\n\n"
        f"Target GPU: {T4_HARDWARE['name']} "
        f"(compute capability {T4_HARDWARE['compute_capability']}, "
        f"{T4_HARDWARE['arch']}). Hardware constraints:\n{constraints}"
    )


def build_user_prompt(operator_id: str, instruction: str) -> str:
    """Format an operator's task instruction for the user turn."""
    return instruction
