"""Phase B — hardware-awareness eval (build_awareness_set, hardware_eval)."""
from __future__ import annotations

import json
from pathlib import Path

import modal

from .core import *  # noqa: F401,F403
from .kernels import _probe_kernel_file
from .llm import _classify_failure, _gen
from .mutate import GOLD_TRITON_DIR

# Phase 2 produced ground truth: (kernel_code, T4) -> {runs | fails-with-
# hardware-reason}.  Phase B uses it as an answer key to test whether an LLM
# *knows the hardware*: shown a Triton kernel and a target GPU, can it predict
# whether the kernel will run, and if not, name the hardware reason?
#
# Two steps:
#   B1  build_awareness_set (GPU) — verify each gold kernel runs clean on T4
#       (this gates BOTH classes: a mutation-induced FAIL is only attributable
#       if its gold baseline ran), then build a labeled set whose RUN sample is
#       stratified by operator type to match the FAIL distribution, so a model
#       can't shortcut on "looks like a matmul -> fail".
#   B2  hardware_eval (CPU) — query each model under three escalating GPU-info
#       conditions (knowledge -> reasoning), parse its JSON verdict, score.

# Closed set of hardware failure categories the model is asked to choose from.
# fp8 and bf16 both surface as dtype/arch limits; shared-memory is its own.
HW_EVAL_CATEGORIES: tuple[str, ...] = ("fail_dtype_unsupported", "fail_compile_shared_mem")


def _operator_type(source: str) -> str:
    """Coarse operator class for stratification + shortcut auditing.

    Priority: a matmul that also reduces is still 'matmul'.
    """
    import re

    if "tl.dot" in source:
        return "matmul"
    if re.search(r"tl\.(sum|max|min|argmax|argmin|cumsum|softmax|xor_sum)\s*\(", source):
        return "reduction"
    return "elementwise"


def _build_gpu_conditions(gpu_name: str, compute_cap: str) -> dict[str, str]:
    """Three escalating target-GPU descriptions (knowledge -> reasoning).

    'full' gives architectural *facts* (arch, SRAM, Tensor-Core dtypes) but never
    states the verdict ("no bf16") — inferring that is the reasoning we test.
    Specs below are T4-specific (the eval set is T4); generalise per-GPU later.
    """
    name = f"The target GPU is an NVIDIA {gpu_name}."
    cap = (
        f"The target GPU is an NVIDIA {gpu_name} "
        f"(CUDA compute capability {compute_cap})."
    )
    full = (
        f"The target GPU is an NVIDIA {gpu_name} (CUDA compute capability "
        f"{compute_cap}, Turing architecture). Per-SM shared memory is 64 KB "
        "(48 KB usable by a single block by default). It has fp16 and int8 "
        "Tensor Cores. The register file is 256 KB per SM (max 255 registers "
        "per thread)."
    )
    return {"name": name, "cap": cap, "full": full}


@app.function(gpu=DEFAULT_GPU, timeout=60 * 60 * 2, volumes={DATA_DIR: data_volume})
def build_awareness_set(
    mutations_subdir: str = "mutations",
    output_path: str = "awareness/eval_set.jsonl",
    limit: int | None = None,
    probe_timeout: int = 20,
) -> dict:
    """B1: verify gold kernels on T4 and build the balanced labeled eval set.

    probe_timeout caps each gold-kernel verification run (default 20s — a clean
    gold kernel runs fast; the cap just bounds cost on a pathological hang).
    """
    import random
    import tempfile
    from collections import Counter, defaultdict

    import torch as _torch

    ds_path = Path(DATA_DIR) / mutations_subdir / "mutation_dataset.jsonl"
    if not ds_path.exists():
        raise FileNotFoundError(f"run generate_mutations first — missing {ds_path}")

    props       = _torch.cuda.get_device_properties(0)
    compute_cap = f"{props.major}.{props.minor}"
    gpu_name    = _torch.cuda.get_device_name(0)

    # Validated, in-vocabulary hardware failures, grouped by kernel.
    fails_by_kernel: dict[str, list[dict]] = defaultdict(list)
    for line in ds_path.open():
        r = json.loads(line)
        if (
            r.get("is_hardware_failure")
            and r.get("validated")
            and r.get("actual_behavior") in HW_EVAL_CATEGORIES
        ):
            fails_by_kernel[r["kernel_id"]].append(r)

    gold_paths = sorted(Path(GOLD_TRITON_DIR).glob("*.py"))
    if limit:
        gold_paths = gold_paths[:limit]
    print(
        f"B1: verifying {len(gold_paths)} gold kernels run clean on {gpu_name} "
        f"(cc {compute_cap}) ...",
        flush=True,
    )

    clean_run_pool: list[dict] = []   # verified-clean gold kernels (RUN candidates)
    fail_items:     list[dict] = []   # attributable validated hardware failures
    gold_clean = 0
    with tempfile.TemporaryDirectory() as td:
        for i, kp in enumerate(gold_paths, 1):
            src = kp.read_text()
            fpath = Path(td) / kp.name
            fpath.write_text(src)
            rc, _se = _probe_kernel_file(fpath, timeout=probe_timeout)
            if rc != 0:
                continue  # gold does not run clean on T4 -> drop kernel (both classes)
            gold_clean += 1
            op = _operator_type(src)
            clean_run_pool.append({
                "kernel_id": kp.name, "operator_type": op, "label": "run",
                "expected_failure_category": None, "mutation_type": None,
                "code": src, "gpu": gpu_name, "compute_cap": compute_cap,
            })
            for fr in fails_by_kernel.get(kp.name, []):
                fail_items.append({
                    "kernel_id": kp.name, "operator_type": op, "label": "fail",
                    "expected_failure_category": fr["actual_behavior"],
                    "mutation_type": fr["mutation_type"],
                    "code": fr["mutated_kernel_code"],
                    "gpu": gpu_name, "compute_cap": compute_cap,
                })
            if i % 25 == 0 or i == len(gold_paths):
                print(f"  verified {i}/{len(gold_paths)}  (clean so far: {gold_clean})", flush=True)

    # Stratify the RUN sample to match the FAIL operator-type distribution, sized
    # to the FAIL count, so operator type carries no run/fail signal.  Paired
    # kernels (clean gold that also has a fail) are preferred first so matched
    # pairs exist, then topped up from unpaired clean kernels.  Deterministic.
    fail_ops   = Counter(f["operator_type"] for f in fail_items)
    total_fail = sum(fail_ops.values()) or 1
    paired_ids = set(fails_by_kernel)
    rng = random.Random(0)
    pool_by_op: dict[str, list[dict]] = defaultdict(list)
    for r in clean_run_pool:
        pool_by_op[r["operator_type"]].append(r)
    for op, lst in pool_by_op.items():
        # paired kernels first (stable), then shuffled unpaired
        lst.sort(key=lambda r: (r["kernel_id"] not in paired_ids, r["kernel_id"]))
        head = [r for r in lst if r["kernel_id"] in paired_ids]
        tail = [r for r in lst if r["kernel_id"] not in paired_ids]
        rng.shuffle(tail)
        pool_by_op[op] = head + tail

    run_items: list[dict] = []
    for op, cnt in fail_ops.items():
        target = round(len(fail_items) * cnt / total_fail)
        run_items.extend(pool_by_op.get(op, [])[:target])

    eval_items = fail_items + run_items
    rng.shuffle(eval_items)
    for idx, it in enumerate(eval_items):
        it["item_id"] = idx

    out = Path(DATA_DIR) / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for it in eval_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    meta = {
        "gpu": gpu_name,
        "compute_cap": compute_cap,
        "gold_kernels_probed": len(gold_paths),
        "gold_runs_clean": gold_clean,
        "n_items": len(eval_items),
        "n_fail": len(fail_items),
        "n_run": len(run_items),
        "fail_operator_dist": dict(fail_ops),
        "run_operator_dist": dict(Counter(r["operator_type"] for r in run_items)),
        "fail_category_dist": dict(Counter(f["expected_failure_category"] for f in fail_items)),
        "eval_set_path": output_path,
    }
    (out.parent / "eval_set_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    data_volume.commit()
    print(f"\nB1 wrote {len(eval_items)} items -> {out}", flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    return meta


HW_EVAL_SYSTEM = (
    "You are a GPU systems engineer with deep knowledge of NVIDIA GPU "
    "architectures and the Triton compiler. You are given a Triton kernel "
    "(kernel + Python launch wrapper that runs it) and a description of a target "
    "GPU. Decide whether the kernel will SUCCESSFULLY COMPILE AND RUN on that "
    "exact GPU, or whether it will FAIL for a hardware reason — a datatype the "
    "GPU has no support for, or a resource limit such as shared memory.\n\n"
    "Respond with ONLY a JSON object and nothing else:\n"
    '{"will_run": true|false, "failure_category": null|'
    '"fail_dtype_unsupported"|"fail_compile_shared_mem", '
    '"reason": "<one short sentence>"}\n'
    "Use failure_category null when will_run is true; otherwise pick the single "
    "best-matching category."
)


def _hw_eval_user(gpu_desc: str, code: str) -> str:
    return f"{gpu_desc}\n\nKernel:\n```python\n{code}\n```"


def _parse_verdict(text: str) -> dict | None:
    """Extract the model's JSON verdict; tolerant of code fences / surrounding prose."""
    import re

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
    except Exception:  # noqa: BLE001
        return None
    return d if isinstance(d, dict) and "will_run" in d else None


@app.function(
    timeout=60 * 60 * 4,
    cpu=1,   # I/O-bound (waiting on OpenRouter); 1 core keeps Modal cost minimal
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name(LLM_SECRET_NAME)],
)
def hardware_eval(
    eval_set_path: str = "awareness/eval_set.jsonl",
    models: str = "",
    conditions: str = "name,cap,full",
    concurrency: int = 8,
    output_subdir: str = "awareness",
) -> dict:
    """B2: quiz each model on the labeled set under each GPU-info condition; score."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from collections import defaultdict

    items = [json.loads(l) for l in (Path(DATA_DIR) / eval_set_path).open()]
    if not items:
        raise ValueError(f"empty eval set: {eval_set_path}")
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    cond_list  = [c.strip() for c in conditions.split(",") if c.strip()]
    if not model_list:
        raise ValueError("pass --models 'slug1,slug2,slug3'")

    conds = _build_gpu_conditions(items[0]["gpu"], items[0]["compute_cap"])
    cond_list = [c for c in cond_list if c in conds]

    print(
        f"B2: {len(items)} items x {len(cond_list)} conditions x {len(model_list)} "
        f"models = {len(items)*len(cond_list)*len(model_list)} queries",
        flush=True,
    )

    tasks = [
        (m, c, it)
        for m in model_list
        for c in cond_list
        for it in items
    ]

    def _do(task):
        m, c, it = task
        msgs = [
            {"role": "system", "content": HW_EVAL_SYSTEM},
            {"role": "user", "content": _hw_eval_user(conds[c], it["code"])},
        ]
        try:
            gr = _gen(msgs, m)
            return m, c, it, _parse_verdict(gr.content), None
        except Exception as exc:  # noqa: BLE001
            return m, c, it, None, _classify_failure(exc)

    raw: list[dict] = []
    # metrics[model][cond] accumulators
    acc: dict = defaultdict(lambda: defaultdict(lambda: {
        "run_total": 0, "run_correct": 0,        # specificity
        "fail_total": 0, "fail_correct": 0,      # recall
        "cat_eligible": 0, "cat_correct": 0,     # category accuracy (on caught fails)
        "unparseable": 0,
        "by_op": defaultdict(lambda: {"total": 0, "correct": 0}),
    }))

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_do, t) for t in tasks]
        for fut in as_completed(futs):
            m, c, it, v, err = fut.result()
            gt_run = it["label"] == "run"
            a = acc[m][c]
            rec = {"model": m, "condition": c, "item_id": it["item_id"],
                   "kernel_id": it["kernel_id"], "operator_type": it["operator_type"],
                   "label": it["label"], "expected_category": it["expected_failure_category"],
                   "verdict": v, "error": err}
            raw.append(rec)
            if v is None:
                a["unparseable"] += 1
            else:
                pred_run = bool(v.get("will_run"))
                correct = pred_run == gt_run
                op = a["by_op"][it["operator_type"]]
                op["total"] += 1
                op["correct"] += int(correct)
                if gt_run:
                    a["run_total"] += 1
                    a["run_correct"] += int(correct)
                else:
                    a["fail_total"] += 1
                    a["fail_correct"] += int(correct)
                    if not pred_run:  # model caught the failure -> grade its reason
                        a["cat_eligible"] += 1
                        a["cat_correct"] += int(
                            v.get("failure_category") == it["expected_failure_category"]
                        )
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} queries done", flush=True)

    def _summarize(a: dict) -> dict:
        rt, rc = a["run_total"], a["run_correct"]
        ft, fc = a["fail_total"], a["fail_correct"]
        spec   = rc / rt if rt else None
        recall = fc / ft if ft else None
        bal    = (
            (spec + recall) / 2 if spec is not None and recall is not None else None
        )
        n = rt + ft
        return {
            "n_scored": n,
            "accuracy": round((rc + fc) / n, 4) if n else None,
            "balanced_accuracy": round(bal, 4) if bal is not None else None,
            "fail_recall": round(recall, 4) if recall is not None else None,
            "run_specificity": round(spec, 4) if spec is not None else None,
            "failure_category_accuracy": (
                round(a["cat_correct"] / a["cat_eligible"], 4) if a["cat_eligible"] else None
            ),
            "unparseable": a["unparseable"],
            "by_operator": {
                op: round(d["correct"] / d["total"], 4) if d["total"] else None
                for op, d in a["by_op"].items()
            },
        }

    metrics = {m: {c: _summarize(acc[m][c]) for c in cond_list} for m in model_list}

    # Leaderboard keyed on the strictest knowledge condition available.
    head_cond = "name" if "name" in cond_list else cond_list[0]
    leaderboard = sorted(
        ({"model": m, "condition": head_cond, **metrics[m][head_cond]} for m in model_list),
        key=lambda d: (d["balanced_accuracy"] is not None, d["balanced_accuracy"] or 0),
        reverse=True,
    )

    out_dir = Path(DATA_DIR) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "gpu": items[0]["gpu"],
        "compute_cap": items[0]["compute_cap"],
        "models": model_list,
        "conditions": cond_list,
        "n_items": len(items),
        "metrics": metrics,
        "leaderboard": leaderboard,
    }
    (out_dir / "hardware_eval_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    with (out_dir / "hardware_eval_raw.jsonl").open("w") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    data_volume.commit()
    print(f"\nB2 wrote results -> {out_dir}/hardware_eval_results.json", flush=True)
    return results

