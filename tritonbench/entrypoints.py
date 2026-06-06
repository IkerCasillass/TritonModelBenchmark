"""All local entrypoints. Importing this registers every @app.function."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .core import DEFAULT_GPU, DEFAULT_INTERP_MODEL, DEFAULT_MODEL, VOLUME_NAME, app, data_volume
from .generate import generate_predictions
from .evaluate import evaluate
from .mutate import generate_mutations
from .awareness import build_awareness_set, hardware_eval
from .refine import generate_refine, judge_kernel, list_operators
from .operators import build_registry, select_operators, get_operator
from . import gen_service  # noqa: F401  — registers ConstrainedGenerator with the app

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
    """Mutate gold Triton kernels and validate the predicted hardware behaviour.

    Usage:
        modal run modal_app.py::mutate_only --limit 10        # watch it
        modal run --detach modal_app.py::mutate_only          # fire-and-forget

    Uses .spawn() so the run is an independent call that survives client
    disconnect; with `modal run --detach`, results commit to the Volume whether
    or not .get() below returns.
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
    """Build the hardware-awareness eval set from a mutation dataset.

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
    """Quiz models on the eval set and score hardware-awareness.

    Usage:
        modal run modal_app.py::hardware_eval_only --models "anthropic/claude-sonnet-4-5,openai/gpt-4o-mini"
    """
    # .spawn() so a long multi-model run survives client disconnect; pair with
    # `modal run --detach`. Results land in the Volume regardless of .get().
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
    interp_model: str = DEFAULT_INTERP_MODEL,
    limit: int = 0,
    max_iters: int = 3,
    feedback_mode: str = "interpreted",
    output_subdir: str = "refine",
    gpu: str = DEFAULT_GPU,
    gen_backend: str = "local",
    gen_constrained: bool = True,
    operator: str = "",
    concurrency: int = 2,
):
    """Hardware-in-the-loop generate -> run -> refine loop.

    Usage:
        modal run modal_app.py::refine_loop --gen-model "<small-slug>" --limit 3
        modal run modal_app.py::refine_loop --operator add          # target one op
        modal run modal_app.py::refine_loop --operator "add,relu"   # target several

    ``operator`` (comma-separated ids, with or without ``.py``) targets specific
    operators regardless of dataset order, overriding ``limit``. Discover ids with
    ``modal run modal_app.py::list_operators_only``.
    """
    operators = [o.strip() for o in operator.split(",") if o.strip()] or None
    summary = generate_refine.with_options(gpu=gpu).remote(
        gen_model=gen_model,
        interp_model=interp_model,
        limit=limit if limit > 0 else None,
        operators=operators,
        max_iters=max_iters,
        feedback_mode=feedback_mode,
        output_subdir=output_subdir,
        gen_backend=gen_backend,
        gen_constrained=gen_constrained,
        concurrency=concurrency,
    )
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def list_operators_only(dataset: str = "simp", limit: int = 0):
    """Print operator ids (and whether each has a golden file). No GPU.

    Usage:
        modal run modal_app.py::list_operators_only
        modal run modal_app.py::list_operators_only --limit 40
    """
    list_operators.remote(dataset=dataset, limit=limit if limit > 0 else None)


_BF16_KERNEL = """\
import torch
import triton
import triton.language as tl

@triton.jit
def _k(x_ptr, o_ptr, N: tl.constexpr):
    idx = tl.arange(0, N)
    x = tl.load(x_ptr + idx).to(tl.bfloat16)
    tl.store(o_ptr + idx, x)

def run():
    x = torch.ones(16, dtype=torch.float32, device="cuda")
    o = torch.empty(16, dtype=torch.bfloat16, device="cuda")
    _k[(1,)](x, o, 16)

run()
"""


@app.function(gpu=DEFAULT_GPU, timeout=600)
def _inspect_golden(operator: str, dataset: str = "simp") -> dict:
    from . import operators as _ops
    oid = operator if operator.endswith(".py") else f"{operator}.py"
    _ops.build_registry(dataset, None, operators=[operator])
    try:
        g = _ops.get_operator(oid)["golden_stdout"]
        return {"oid": oid, "dropped": False, "golden_len": len(g), "golden_head": g[:300]}
    except KeyError:
        return {"oid": oid, "dropped": True}


@app.local_entrypoint()
def inspect_golden(operator: str, dataset: str = "simp", gpu: str = DEFAULT_GPU):
    """Print an operator's cached golden stdout — to check it isn't empty
    (empty golden => any non-crashing kernel trivially 'passes')."""
    print(json.dumps(_inspect_golden.with_options(gpu=gpu).remote(
        operator=operator, dataset=dataset), indent=2))


@app.function(gpu=DEFAULT_GPU, timeout=600)
def _judge_smoke(dataset: str = "simp", limit: int = 5) -> dict:
    import tempfile as _tmp
    from pathlib import Path as _Path
    from .core import REPO_DIR
    from .kernels import _probe_kernel_file, _classify_kernel_failure

    build_registry(dataset, limit)
    ops = select_operators()
    gold_base = _Path(REPO_DIR) / "data" / "TritonBench_T_v1"

    passed, failed = [], []

    for op_id in ops:
        # Gold files contain kernel + "#"*146 + test; strip test before passing
        # to judge_kernel, which appends test itself.
        gold_src = (gold_base / op_id).read_text().split("#" * 146)[0].rstrip("\n")
        jr = judge_kernel(gold_src, op_id)
        ok = jr["correct"] is True
        (passed if ok else failed).append(
            {"test": f"{op_id}:gold", "failure_type": jr["failure_type"], "correct": jr["correct"]}
        )

    # Hardware fixture: a kernel that uses bfloat16, which T4 (sm_75) cannot run.
    # Tests the dtype_unsupported classification path independently of the mutators.
    with _tmp.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
        fh.write(_BF16_KERNEL)
        bf16_path = _Path(fh.name)
    try:
        rc, stderr = _probe_kernel_file(bf16_path)
        ftype = _classify_kernel_failure(stderr)
        ok = rc != 0 and ftype == "dtype_unsupported"
        entry = {"test": "fixture:bf16_dtype_unsupported", "failure_type": ftype, "rc": rc}
        if not ok:
            entry["stderr_tail"] = stderr[-500:]
        (passed if ok else failed).append(entry)
    finally:
        bf16_path.unlink(missing_ok=True)

    return {
        "registry_size": len(ops),
        "passed": len(passed),
        "failed": failed,
    }


@app.local_entrypoint()
def test_evaluator(dataset: str = "simp", limit: int = 5, gpu: str = DEFAULT_GPU):
    """Smoke-test the judge: gold kernels -> correct, bf16 fixture -> dtype_unsupported.

    Usage:
        modal run modal_app.py::test_evaluator --limit 5
    """
    r = _judge_smoke.with_options(gpu=gpu).remote(dataset=dataset, limit=limit)
    total = r["passed"] + len(r["failed"])
    print(json.dumps({
        "registry_size": r["registry_size"],
        "passed": f"{r['passed']}/{total}",
        "all_passed": len(r["failed"]) == 0,
    }, indent=2))
    if r["failed"]:
        print("\nfailed:")
        for f in r["failed"]:
            print(f"  {f['test']}: {f}")
