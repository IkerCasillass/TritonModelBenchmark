"""Conversation/feedback shaping for the refinement loop."""
from __future__ import annotations


def build_messages(operator_id: str, history: list[dict], system_prompt: str,
                   user_prompt: str) -> list[dict]:
    """Build the chat messages, folding in the most recent attempt and its feedback.

    ``history[i]`` = ``{"code", "judge": JudgeResult, "hint", "feedback"}``.
    """
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    if history:
        last = history[-1]
        msgs.append({"role": "assistant", "content": last["code"]})
        msgs.append({"role": "user", "content": last["feedback"]})
    return msgs
