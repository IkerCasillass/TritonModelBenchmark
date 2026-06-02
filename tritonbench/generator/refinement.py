"""Task C — refinement strategy + the feedback-mode ablation.  OWNER: <teammate C>.

Deliverable: turn the loop `history` (prior code + JudgeResult + hint) into the
next-turn messages — conversation management, anti-repetition / give-up logic,
and the three feedback_mode variants ("interpreted" | "category" | "raw") that
drive the ablation (does the interpreter's hint beat raw stderr?).

Tests (against engine.py's scriptable stub evaluator, no A needed): assert the
kernel *changes appropriately* after a given hint (e.g. drops bf16 after a dtype
hint) and that the loop terminates.
"""
from __future__ import annotations


def build_messages(operator_id: str, history: list[dict], system_prompt: str,
                   user_prompt: str) -> list[dict]:
    """[OWNER C — STUB] Build the chat messages, folding in prior attempts + feedback.

    `history[i]` = {"code", "judge": JudgeResult, "hint", "feedback"} (see refine._refine_one).
    Replace with the real transcript shaping. Stub: system + user only (no refinement),
    plus the most recent feedback appended if present.
    """
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    if history:
        last = history[-1]
        msgs.append({"role": "assistant", "content": last["code"]})
        msgs.append({"role": "user", "content": last["feedback"]})
    return msgs
