"""LLM PyTorch->Triton kernel generation."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import modal

from .core import *  # noqa: F401,F403
from .llm import (GenResult, _build_messages, _classify_failure, _extract_code,
                  _gen, _gen_meta_path, _is_valid_python, _load_alpaca)

@app.function(
    timeout=60 * 60 * 4,
    cpu=4,
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name(LLM_SECRET_NAME)],
)
def generate_predictions(
    model: str = DEFAULT_MODEL,
    dataset: str = "simp",
    output_path: str = "predictions.jsonl",
    limit: int | None = None,
    concurrency: int = 4,
) -> str:
    """Generate Triton translations for every entry in the Alpaca dataset.

    Writes ``output_path`` (the predictions jsonl, schema ``{instruction,
    predict}``) plus a ``.gen_meta.json`` sidecar with latency / token /
    failure-cause telemetry. Returns the volume-relative path of the jsonl.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    items = _load_alpaca(dataset)
    if limit:
        items = items[:limit]

    print(f"generating {len(items)} predictions with model={model}", flush=True)

    # Smoke-test the model with a single cheap call before spawning workers.
    # A bad slug (404) or auth failure (401/403) dooms every call — fail fast
    # instead of burning the whole dataset; transient capacity errors are
    # tolerated since the per-item retry logic may still recover.
    print("  smoke-testing model...", end=" ", flush=True)
    try:
        _gen([{"role": "user", "content": "Reply with one word: ready"}], model)
        print("OK", flush=True)
    except RuntimeError as exc:
        if "daily free quota" in str(exc):
            raise  # hard quota — no point starting workers
        print(f"warning: {exc} — proceeding anyway", flush=True)
    except Exception as exc:  # noqa: BLE001
        status = getattr(exc, "status_code", None)
        if status in (400, 401, 403, 404):
            raise RuntimeError(
                f"model unusable (HTTP {status}) — check the slug and "
                f"credentials before retrying: {exc}"
            ) from exc
        print(f"warning: {exc} — proceeding anyway", flush=True)

    def _do(idx_item: tuple[int, dict]) -> tuple[int, dict, dict]:
        i, item = idx_item
        instruction = item["instruction"]
        meta: dict = {
            "index": i,
            "latency_s": None,
            "completion_tokens": None,
            "prompt_tokens": None,
            "truncated": False,
            "fail_reason": None,
        }
        try:
            gr = _gen(_build_messages(item), model)
            meta["latency_s"] = round(gr.latency_s, 3)
            meta["completion_tokens"] = gr.completion_tokens
            meta["prompt_tokens"] = gr.prompt_tokens
            meta["truncated"] = gr.finish_reason == "length"
            code = _extract_code(gr.content)
            if not _is_valid_python(code):
                # Log the first 300 chars of the raw response so you can see
                # what the model returned and tune _extract_code if needed.
                preview = gr.content[:300].replace("\n", "|")
                raise ValueError(
                    f"SyntaxError after fence-strip — raw preview: {preview!r}"
                )
            print(f"  [OK ] {i:4d} {instruction[:60]}", flush=True)
        except Exception as exc:       # noqa: BLE001
            meta["fail_reason"] = _classify_failure(exc)
            code = f"# generation failed: {exc}\n"
            print(
                f"  [ERR] {i:4d} {instruction[:60]} — "
                f"[{meta['fail_reason']}] {exc}",
                flush=True,
            )
        return i, {"instruction": instruction, "predict": code}, meta

    results: list[dict | None] = [None] * len(items)
    metas: list[dict | None] = [None] * len(items)
    failed = 0
    gen_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_do, (i, it)) for i, it in enumerate(items)]
        done = 0
        for fut in as_completed(futs):
            i, rec, meta = fut.result()
            results[i] = rec
            metas[i] = meta
            if meta["fail_reason"]:
                failed += 1
            done += 1
            if done % 10 == 0 or done == len(items):
                print(f"  progress: {done}/{len(items)}  failures so far: {failed}", flush=True)
    gen_wall_s = time.perf_counter() - gen_start

    out = Path(DATA_DIR) / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep the jsonl schema exactly {instruction, predict} — upstream eval
    # scripts read it and may not tolerate extra keys.
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- generation metrics sidecar -----------------------------------------
    latencies = [m["latency_s"] for m in metas if m and m["latency_s"] is not None]
    comp_tokens = [m["completion_tokens"] for m in metas if m and m["completion_tokens"]]
    prompt_tokens = [m["prompt_tokens"] for m in metas if m and m["prompt_tokens"]]
    by_reason: dict[str, int] = {}
    for m in metas:
        if m and m["fail_reason"]:
            by_reason[m["fail_reason"]] = by_reason.get(m["fail_reason"], 0) + 1
    truncated_count = sum(1 for m in metas if m and m["truncated"])

    def _pct(vals: list[float], p: int) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
        return round(s[k], 3)

    gen_meta = {
        "model": model,
        "dataset": dataset,
        "timestamp": int(time.time()),
        "n_items": len(items),
        "concurrency": concurrency,
        "generation": {
            "total_wall_s": round(gen_wall_s, 1),
            "latency_s": {
                "p50": _pct(latencies, 50),
                "p95": _pct(latencies, 95),
                "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            },
            "failures": {"total": failed, "by_reason": by_reason},
            "tokens": {
                "total_completion": sum(comp_tokens),
                "total_prompt": sum(prompt_tokens),
                "mean_completion": (
                    round(statistics.fmean(comp_tokens), 1) if comp_tokens else None
                ),
            },
            "truncated_count": truncated_count,
        },
        "per_item": metas,
    }
    meta_out = _gen_meta_path(out)
    meta_out.write_text(json.dumps(gen_meta, ensure_ascii=False, indent=2))
    data_volume.commit()

    success = len(items) - failed
    print(
        f"\nwrote {out}  ({success}/{len(items)} successful, {failed} failed)",
        flush=True,
    )
    if truncated_count:
        print(
            f"  warning: {truncated_count} response(s) truncated at max_tokens "
            "— those kernels are likely incomplete",
            flush=True,
        )
    print(f"wrote generation metrics -> {meta_out}", flush=True)
    return output_path

