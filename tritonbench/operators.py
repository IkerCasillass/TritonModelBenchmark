"""TritonBench-T operator registry: loads operator specs and caches golden stdout."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from .core import REPO_DIR
from .kernels import _run_kernel_capture
from .llm import _load_alpaca


class OperatorSpec(TypedDict):
    operator_id: str
    instruction: str
    test_code: str
    golden_stdout: str


_OPERATOR_REGISTRY: dict[str, OperatorSpec] = {}


def _instruction(item: dict) -> str:
    inp = item.get("input", "") or ""
    return item["instruction"] if not inp else f"{item['instruction']}\n\n{inp}"


def _normalize_id(name: str) -> str:
    """Accept an operator id with or without the trailing ``.py``."""
    name = name.strip()
    return name if name.endswith(".py") else f"{name}.py"


def _extract_codes(dataset: str, limit: int | None) -> tuple[list[dict], list, list]:
    """Run TritonBench's code/test extraction; return aligned (items, tests, files).

    CPU-only and cheap — no golden execution. Importable only inside the Modal
    image (call_acc lives in the cloned TritonBench repo). Shared by
    build_registry (which then runs goldens) and list_operator_ids (which doesn't).
    """
    eval_dir = f"{REPO_DIR}/EVAL/eval_T"
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    os.environ["PYTHONPATH"] = eval_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

    import call_acc  # noqa: E402  — importable only inside the Modal image

    items = _load_alpaca(dataset)
    if limit:
        items = items[:limit]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
        tmp_path = fh.name
        for item in items:
            fh.write(json.dumps({"instruction": _instruction(item), "predict": "pass\n"}) + "\n")

    try:
        _, tests, files = call_acc.get_codes_for_test(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return items, tests, files


def list_operator_ids(dataset: str = "simp", limit: int | None = None) -> list[tuple[str, bool]]:
    """List operator ids without running goldens (no GPU needed).

    Returns ``[(operator_id, has_gold_file), ...]`` for every dataset item that
    yields a test. Use it to pick a target for ``build_registry(operators=...)``
    — e.g. a simple elementwise op rather than a ``fused_*`` one.
    """
    items, tests, files = _extract_codes(dataset, limit)
    gold_base = Path(REPO_DIR) / "data" / "TritonBench_T_v1"
    out: list[tuple[str, bool]] = []
    for test, fname in zip(tests, files):
        if not fname or not test:
            continue
        out.append((fname, (gold_base / fname).exists()))
    return out


def build_registry(dataset: str = "simp", limit: int | None = None,
                   operators: list[str] | None = None) -> None:
    """Populate the module-level registry from the TritonBench-T suite.

    Runs each operator's golden file to cache its reference stdout. Operators
    whose golden file is missing, errors, or times out are silently dropped
    because they cannot be judged. Must be called once inside the Modal GPU
    function before select_operators or get_operator.

    ``operators``: when given, scan the *full* dataset and build only these ids
    (with or without ``.py``), ignoring ``limit``. This lets a run target a
    specific easy operator no matter where it sits in dataset order. When
    ``operators`` is None, the first ``limit`` items are used (original behavior).
    """
    global _OPERATOR_REGISTRY

    want = {_normalize_id(o) for o in operators} if operators else None
    # When targeting specific ids, load the whole dataset so position doesn't matter.
    items, tests, files = _extract_codes(dataset, None if want else limit)

    _OPERATOR_REGISTRY.clear()
    gold_base = Path(REPO_DIR) / "data" / "TritonBench_T_v1"
    dropped = 0
    considered = 0

    for item, test, fname in zip(items, tests, files):
        if want is not None and fname not in want:
            continue
        considered += 1
        if not fname or not test:
            dropped += 1
            continue
        gold_path = gold_base / fname
        if not gold_path.exists():
            dropped += 1
            continue
        rc, stdout, _ = _run_kernel_capture(gold_path)
        if rc != 0:
            dropped += 1
            continue
        _OPERATOR_REGISTRY[fname] = {
            "operator_id": fname,
            "instruction": _instruction(item),
            "test_code": test,
            "golden_stdout": stdout,
        }

    if want:
        missing = want - set(_OPERATOR_REGISTRY)
        if missing:
            print(f"build_registry: requested operators not built: {sorted(missing)}", flush=True)
    if dropped:
        print(f"build_registry: dropped {dropped}/{considered} operators (golden errors)", flush=True)


def select_operators(limit: int | None = None) -> list[str]:
    keys = list(_OPERATOR_REGISTRY.keys())
    return keys[:limit] if limit else keys


def get_operator(operator_id: str) -> OperatorSpec:
    try:
        return _OPERATOR_REGISTRY[operator_id]
    except KeyError:
        raise KeyError(f"operator '{operator_id}' not in registry; call build_registry first")


def get_instruction(operator_id: str) -> str:
    return get_operator(operator_id)["instruction"]
