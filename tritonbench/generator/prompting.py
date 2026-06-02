"""Task B — hardware-aware prompting.  OWNER: <teammate B>.

Deliverable: build the prompt that makes the *first* generation attempt
hardware-aware, from `T4_HARDWARE` (shared facts) + the operator's instruction.
This is the "does it get it right one-shot" lever.

Tests (against the stub evaluator in engine.py, no A needed): generate one-shot
over N operators, measure pass@1 and which hardware mistakes recur.
"""
from __future__ import annotations

from ..core import T4_HARDWARE
from ..llm import PROMPT_HEADER


def build_system_prompt() -> str:
    """[OWNER B — STUB] System prompt: base instructions + the T4 constraints.

    Replace with the tuned version. Below is a minimal grounding from T4_HARDWARE.
    """
    constraints = "\n".join(f"- {c}" for c in T4_HARDWARE["constraints"])
    return (
        f"{PROMPT_HEADER}\n\n"
        f"TARGET GPU: {T4_HARDWARE['name']} "
        f"(compute capability {T4_HARDWARE['compute_capability']}, "
        f"{T4_HARDWARE['arch']}). Respect these hardware constraints:\n{constraints}"
    )


def build_user_prompt(operator_id: str, instruction: str) -> str:
    """[OWNER B — STUB] Format the operator's task instruction for the user turn."""
    return instruction
