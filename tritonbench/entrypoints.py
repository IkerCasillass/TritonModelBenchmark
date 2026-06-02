"""All local entrypoints. Importing this registers every @app.function."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .core import DEFAULT_GPU, DEFAULT_MODEL, VOLUME_NAME, app, data_volume
from .generate import generate_predictions
from .evaluate import evaluate
from .mutate import generate_mutations
from .awareness import build_awareness_set, hardware_eval
from .refine import generate_refine

def _upload_local_predictions(local_path: Path) -> str:
    """Upload a local predictions.jsonl to the volume; return its remote path."""
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    remote = f"uploads/{local_path.name}"
    print(f"uploading {local_path} -> volume://{remote}", flush=True)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), remote)
    return remote


@app.local_entrypoint()
def main(
    predictions: str = "",
    model: str = DEFAULT_MODEL,
    dataset: str = "simp",
    limit: int = 0,
    output_subdir: str = "results",
    concurrency: int = 8,
    gpu: str = DEFAULT_GPU,
):
    """End-to-end: (optionally) generate predictions, then evaluate.

    Args:
        predictions: path to a local predictions.jsonl. If set, generation is
            skipped and this file is uploaded to the volume.
        model:       OpenRouter model slug (e.g. "nvidia/llama-3.1-nemotron-70b-instruct:free").
        dataset:     ``simp`` (simple) or ``comp`` (complex) Alpaca instructions.
        limit:       only generate the first N items (useful for smoke tests).
        output_subdir: where to write per-run artifacts inside the volume.
        concurrency: parallel LLM requests (raise for paid models, lower for
            free-tier RPM limits).
        gpu:         Modal GPU type for the eval phase, e.g. "T4", "L4", "A10".
    """
    if predictions:
        remote = _upload_local_predictions(Path(predictions))
    else:
        run_id = int(time.time())

        tag = (
            f"{model.replace('/', '_').replace(':', '_')}"
            f"_{dataset}"
            f"_limit{limit or 'all'}"
            f"_{run_id}"
        )
        try:
            remote = generate_predictions.remote(
                model=model,
                dataset=dataset,
                output_path=f"predictions/{tag}.jsonl",
                limit=limit if limit > 0 else None,
                concurrency=concurrency,
            )
        except Exception as exc:
            print(f"\ngeneration failed: {exc}", flush=True)
            return

    print(f"\nevaluating: volume://{remote}  (eval GPU: {gpu})\n", flush=True)
    summary = evaluate.with_options(gpu=gpu).remote(
        predictions_path=remote,
        output_subdir=output_subdir,
        model=model,
    )
    print("\n=== Final summary ===")
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def evaluate_only(
    predictions: str,
    output_subdir: str = "results",
    gpu: str = DEFAULT_GPU,
):
    """Evaluate an existing local predictions.jsonl without (re)generating.

    Usage:
        modal run modal_app.py::evaluate_only --predictions ./preds.jsonl
    """
    remote = _upload_local_predictions(Path(predictions))
    summary = evaluate.with_options(gpu=gpu).remote(
        predictions_path=remote, output_subdir=output_subdir
    )
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def mutate_only(
    limit: int = 0,
    output_subdir: str = "mutations",
    gpu: str = DEFAULT_GPU,
):
    """Phase 2: mutate gold Triton kernels and validate T4 hardware behaviour.

    Usage:
        modal run modal_app.py::mutate_only --limit 10        # watch it
        modal run --detach modal_app.py::mutate_only          # fire-and-forget

    Uses .spawn() rather than .remote() so the run is an independent server-side
    call that Modal does NOT cancel when the local client disconnects.  Combined
    with `modal run --detach`, you can launch the full pass and close your laptop;
    results are committed to the Volume regardless of whether .get() below ever
    returns (disconnecting just abandons the result print, not the computation).
    """
    call = generate_mutations.with_options(gpu=gpu).spawn(
        output_subdir=output_subdir,
        limit=limit if limit > 0 else None,
    )
    print(
        f"spawned generate_mutations (call id: {call.object_id})\n"
        f"results -> volume '{VOLUME_NAME}':/{output_subdir}/  "
        f"(mutation_dataset.jsonl, mutation_summary.json)\n"
        f"pull later with:  modal volume get {VOLUME_NAME} {output_subdir} ./{output_subdir}",
        flush=True,
    )
    # Block for the summary when watching interactively; on disconnect the
    # spawned call keeps running and commits to the Volume on its own.
    summary = call.get()
    print(json.dumps({k: v for k, v in summary.items() if k != "skipped"}, indent=2))


@app.local_entrypoint()
def build_awareness_set_only(
    mutations_subdir: str = "mutations",
    output_path: str = "awareness/eval_set.jsonl",
    limit: int = 0,
    probe_timeout: int = 20,
    gpu: str = DEFAULT_GPU,
):
    """Phase B1: build the hardware-awareness eval set from a mutation dataset.

    Usage:
        modal run modal_app.py::build_awareness_set_only
    """
    meta = build_awareness_set.with_options(gpu=gpu).remote(
        mutations_subdir=mutations_subdir,
        output_path=output_path,
        limit=limit if limit > 0 else None,
        probe_timeout=probe_timeout,
    )
    print(json.dumps(meta, indent=2))


@app.local_entrypoint()
def hardware_eval_only(
    models: str,
    eval_set_path: str = "awareness/eval_set.jsonl",
    conditions: str = "name,cap,full",
    concurrency: int = 8,
    output_subdir: str = "awareness",
):
    """Phase B2: quiz models on the eval set and score hardware-awareness.

    Usage:
        modal run modal_app.py::hardware_eval_only --models "anthropic/claude-sonnet-4-5,openai/gpt-4o-mini"
    """
    # .spawn() (not .remote()) so the eval survives a client disconnect — a long
    # multi-model run shouldn't be canceled (and its OpenRouter spend wasted) if
    # the laptop closes.  Pair with `modal run --detach` for true fire-and-forget;
    # results land in the Volume regardless of whether .get() below returns.
    call = hardware_eval.spawn(
        eval_set_path=eval_set_path,
        models=models,
        conditions=conditions,
        concurrency=concurrency,
        output_subdir=output_subdir,
    )
    print(
        f"spawned hardware_eval (call id: {call.object_id})\n"
        f"results -> volume '{VOLUME_NAME}':/{output_subdir}/hardware_eval_results.json",
        flush=True,
    )
    results = call.get()
    print("\n=== leaderboard (by balanced accuracy, knowledge condition) ===")
    print(json.dumps(results.get("leaderboard", []), indent=2))
    print("\n=== per-condition metrics ===")
    print(json.dumps(results.get("metrics", {}), indent=2))


@app.local_entrypoint()
def generate_only(
    model: str = DEFAULT_MODEL,
    dataset: str = "simp",
    limit: int = 0,
    output_path: str = "predictions/predictions.jsonl",
    concurrency: int = 8,
):
    """Generate predictions only; do not evaluate.

    Usage:
        modal run modal_app.py::generate_only --model "nvidia/llama-3.1-nemotron-70b-instruct:free"
    """
    remote = generate_predictions.remote(
        model=model,
        dataset=dataset,
        output_path=output_path,
        limit=limit if limit > 0 else None,
        concurrency=concurrency,
    )
    print(f"wrote volume://{remote}")


@app.local_entrypoint()
def refine_loop(
    gen_model: str = DEFAULT_MODEL,
    interp_model: str = "",
    limit: int = 0,
    max_iters: int = 5,
    feedback_mode: str = "interpreted",
    output_subdir: str = "refine",
    gpu: str = DEFAULT_GPU,
):
    """Phase C: hardware-in-the-loop generate->run->refine loop.

    SCAFFOLD — currently runs on the Section 0 stubs (no real model/GPU work) to
    prove the seam. Owners replace the stub bodies (see docs/PHASE_C_TASKS.md).

    Usage:
        modal run modal_app.py::refine_loop --gen-model "<small-slug>" --limit 3
    """
    summary = generate_refine.with_options(gpu=gpu).remote(
        gen_model=gen_model,
        interp_model=interp_model,
        limit=limit if limit > 0 else None,
        max_iters=max_iters,
        feedback_mode=feedback_mode,
        output_subdir=output_subdir,
    )
    print(json.dumps(summary, indent=2))
