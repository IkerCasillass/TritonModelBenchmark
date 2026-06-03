"""LLM generation: OpenRouter calls, code extraction, alpaca loading."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .core import *  # noqa: F401,F403

PROMPT_HEADER = (
    "You are an expert in Triton programming, capable of writing Triton kernels "
    "and wrapper functions based on functional descriptions and function "
    "parameters. The wrapper function must fully match the provided function "
    "signature.\n\n"
    "Output a single, self-contained Python module containing: (a) the necessary "
    "imports (torch, triton, triton.language as tl), (b) the Triton kernel(s), "
    "and (c) the wrapper function that the description specifies. Wrap the "
    "entire module in one ```python ... ``` fenced code block. Do NOT include "
    "any test code or example calls — tests will be appended separately."
)


def _load_alpaca(dataset: str) -> list[dict]:
    assert dataset in ("simp", "comp"), "dataset must be 'simp' or 'comp'"
    path = Path(REPO_DIR) / f"data/TritonBench_T_{dataset}_alpac_v1.json"
    return json.loads(path.read_text())


def _build_messages(item: dict) -> list[dict]:
    instr = item["instruction"]
    inp = item.get("input", "") or ""
    user = instr if not inp else f"{instr}\n\n{inp}"
    return [
        {"role": "system", "content": PROMPT_HEADER},
        {"role": "user", "content": user},
    ]

def _parse_reset_delay(exc_str: str, fallback: float) -> float:
    """Extract a wait duration from an X-RateLimit-Reset epoch-ms timestamp
    embedded in the error string, falling back to *fallback* seconds if absent
    or in the past.
    """
    import re as _re

    m = _re.search(r"'X-RateLimit-Reset':\s*'(\d+)'", exc_str)
    if m:
        reset_ms = int(m.group(1))
        wait = (reset_ms / 1000.0) - time.time()
        if 0 < wait < 300:   # sanity: only use if 0–5 min in the future
            return wait + 1.0  # +1 s buffer
    return fallback


@dataclass
class GenResult:
    """Outcome of one successful LLM generation call, with telemetry."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    latency_s: float = 0.0


def _classify_failure(exc: Exception) -> str:
    """Bucket a generation exception into a coarse, comparable cause."""
    s = str(exc).lower()
    if "daily free quota" in s or "free-models-per-day" in s:
        return "quota_exhausted"
    if ("rate" in s and "limit" in s) or "429" in s:
        return "rate_limit"
    if "syntaxerror" in s:
        return "syntax_error"
    if "no choices" in s or "empty content" in s:
        return "empty_response"
    status = getattr(exc, "status_code", None)
    if status in (400, 401, 403, 404, 500, 502, 503, 529) or "503" in s or "529" in s:
        return "api_error"
    return "other"


def _gen(messages: list[dict], model: str) -> GenResult:
    """Call the OpenRouter API with smart retry on rate limits.

    Two kinds of 429 from OpenRouter free tier:
      • free-models-per-min  — transient; back off and retry same key.
      • free-models-per-day  — daily hard cap; no point retrying today,
                               raise immediately so the caller can record
                               a clean failure and move on.

    Also handles:
      • None / empty choices  — model at capacity, retry with backoff.
      • HTTP 503 / 529        — upstream overload, retry with backoff.

    Returns a GenResult carrying the reply text plus token-usage and
    latency telemetry from the successful call.
    """
    from openai import OpenAI, RateLimitError, APIStatusError

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            call_start = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=8192,
                temperature=0,
            )
            latency_s = time.perf_counter() - call_start

            # Guard: some models return a response object with None or empty
            # choices instead of raising — treat as a retryable soft failure.
            choices = resp.choices or []
            if not choices:
                raise ValueError("API returned no choices (model at capacity)")
            choice = choices[0]
            msg = choice.message
            content = (msg.content or "") if msg is not None else ""
            if not content.strip():
                raise ValueError("API returned empty content (model at capacity)")

            usage = getattr(resp, "usage", None)
            return GenResult(
                content=content,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                finish_reason=getattr(choice, "finish_reason", "") or "",
                latency_s=latency_s,
            )

        except RateLimitError as exc:
            last_exc = exc
            exc_str = str(exc)
            if "free-models-per-day" in exc_str:
                # Hard daily cap — retrying won't help, surface immediately.
                raise RuntimeError(
                    f"daily free quota exhausted for this API key: {exc}"
                ) from exc
            # Per-minute throttle — wait for the reset window if we can parse
            # it, otherwise fall back to exponential backoff.
            delay = _parse_reset_delay(exc_str, RETRY_BASE_DELAY * (2 ** attempt))
            print(
                f"    [retry {attempt+1}/{MAX_RETRIES}] rate-limit (per-min) — "
                f"waiting {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)

        except ValueError as exc:
            # Empty / no-choices soft failure — exponential backoff.
            last_exc = exc
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(
                f"    [retry {attempt+1}/{MAX_RETRIES}] {exc} — waiting {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)

        except APIStatusError as exc:
            if exc.status_code in (503, 529):
                last_exc = exc
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(
                    f"    [retry {attempt+1}/{MAX_RETRIES}] HTTP {exc.status_code} — "
                    f"waiting {delay:.0f}s",
                    flush=True,
                )
                time.sleep(delay)
            else:
                raise  # non-retryable (400 bad request, 401 auth, etc.)

    raise RuntimeError(
        f"generation failed after {MAX_RETRIES} retries: {last_exc}"
    )


def _extract_code(text: str) -> str:
    """Strip Markdown code fences from an LLM reply; return raw Python source.

    Tries each strategy in order and returns the first one that yields valid
    Python.  Falls back to the raw text if nothing parses cleanly (the
    SyntaxError check in _do() will catch it and log the raw response).

    Handles:
      - ```python\\n...\\n```  (standard)
      - ```py\\n...\\n```
      - ```\\n...\\n```        (no language tag)
      - Multiple blocks — takes the LAST one (some models emit a short
        explanation block then the real code block)
      - Closing fence missing (truncated reply)
      - Raw Python with no fences at all
      - Models that emit ↵ (U+21B5) or \\r\\n instead of real newlines
    """
    import re

    # Normalise model-emitted newline surrogates BEFORE any regex work.
    # Some models (owl-alpha, etc.) emit U+21B5 ↵ as a literal newline stand-in
    # which ends up written verbatim into the .py file, causing SyntaxError.
    text = text.replace("\u21b5", "\n").replace("\r\n", "\n").replace("\r", "\n")

    s = text.strip()

    # Collect ALL fenced blocks; prefer the last one (real code usually last).
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)\n```", s, re.DOTALL)
    if blocks:
        # Return the last block — it's the actual implementation in models that
        # emit explanation first, then code.
        return blocks[-1].strip() + "\n"

    # No closing fence — truncated reply.  Drop the opening fence if present.
    no_open = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    # Strip any trailing ``` that may appear mid-text before prose.
    no_open = re.sub(r"\n?```[\s\S]*$", "", no_open)
    candidate = no_open.strip()
    if candidate:
        return candidate + "\n"

    # Nothing to strip — return as-is and let _is_valid_python decide.
    return s + "\n"


def _is_valid_python(code: str) -> bool:
    """Return True if *code* parses without a SyntaxError."""
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _gen_meta_path(predictions_path: str | Path) -> Path:
    """Sidecar path holding generation metrics for a predictions jsonl."""
    p = Path(predictions_path)
    stem = p.name[:-6] if p.name.endswith(".jsonl") else p.name
    return p.parent / f"{stem}.gen_meta.json"

