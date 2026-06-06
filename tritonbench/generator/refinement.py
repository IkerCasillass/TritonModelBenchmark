"""Conversation/feedback shaping for the refinement loop."""
from __future__ import annotations

def _judge_field(judge, name: str):
    """Safely read a field from either a dict-like or object-like JudgeResult."""
    if judge is None:
        return None
    
    if isinstance(judge, dict):
        return judge.get(name)
    
    return getattr(judge, name, None)

def build_messages(operator_id: str, history: list[dict], system_prompt: str, user_prompt: str) -> list[dict]:
    """Build the chat messages, folding in previous attempts and feedback.

    history[i] = {"code", "judge": JudgeResult, "hint", "feedback"}.
    """
    _ = operator_id  # can be used to customize the prompt per operator if desired

    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    # initial request the model should see

    recent_history = history[-2:]  # last 2 attempts only — keep the prompt small under the 4096 ctx cap

    start_attempt = len(history) - len(recent_history) + 1

    for attempt, turn in enumerate(recent_history, start=start_attempt):
        code = turn.get("code") or "# No code was captured for this attempt."

        msgs.append({
            "role": "assistant",
            "content": f"```python\n{code}\n```"
        })

        parts: list[str] = [
            f"The previous attempt failed. This is failed attempt {attempt} of {len(history)}."
        ]

        feedback = turn.get("feedback")
        if feedback:
            parts.append(f"Feedback:\n{feedback}")
        
        judge = turn.get("judge")
        failure_type = _judge_field(judge, "failure_type")
        diagnostic = _judge_field(judge, "diagnostic")
        raw_stderr = _judge_field(judge, "raw_stderr")

        if failure_type:
            parts.append(f"Failure type: {failure_type}")
        if diagnostic:
            parts.append(f"Diagnostic message:\n{diagnostic}")
        if raw_stderr:
            parts.append(f"Raw stderr:\n{str(raw_stderr)[:400]}")  # keep small: tight context budget
        
        hint = turn.get("hint")
        if hint:
            parts.append(f"Hint:\n{hint}")
        
        parts.append(
            "Revise the implementation. Do not repeat the same mistake. "
            "Preserve the required function signatures and imports expected by the benchmark. "
            "Return only one complete valid Python source file containing the Triton kernel and wrapper. "
            "Do not include explanations."
        )

        msgs.append({
            "role": "user",
            "content": "\n\n".join(parts),
        })


    return msgs

