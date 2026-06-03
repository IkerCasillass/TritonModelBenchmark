"""Hardware-aware kernel mutations: provoke specific GPU limits on known-good kernels."""
from __future__ import annotations

import json
from pathlib import Path

from .core import *  # noqa: F401,F403
from .kernels import (HARDWARE_FAILURE_TYPES, _classify_kernel_failure,
                      _probe_kernel_file, _run_kernel_capture)
from .llm import _is_valid_python

# Targeted single-variable mutations of the known-good gold Triton kernels
# (TritonBench_G_v1) provoke specific T4 (sm_75) hardware limits, which are then
# verified on the GPU. Each gold file is self-contained — kernel + wrapper, a
# "#"*146 separator, then a module-level test that runs on import — so it runs
# directly in an isolated subprocess.

GOLD_TRITON_DIR = f"{REPO_DIR}/data/TritonBench_G_v1"

# The separator the gold files place between the kernel/wrapper and the tests.
_GOLD_SEP = "#" * 146

# predicted_behavior / actual_behavior vocabulary, shared by the mutators and
# the outcome classifier below.
BEHAVIOR_FAIL_SHARED_MEM = "fail_compile_shared_mem"
BEHAVIOR_FAIL_DTYPE      = "fail_dtype_unsupported"
BEHAVIOR_FAIL_BLOCK_SIZE = "fail_invalid_block_size"
BEHAVIOR_RUNS_NO_BF16    = "runs_no_bf16_speedup"

# Hardware-attributable behaviours — set is_hardware_failure even when
# _classify_kernel_failure labels the raw error generically (e.g. a bf16
# "requires sm_80" ptxas error classifies as other_runtime, yet the observed
# behaviour is a real dtype/arch limit).
HARDWARE_BEHAVIORS: frozenset[str] = frozenset({
    BEHAVIOR_FAIL_SHARED_MEM,
    BEHAVIOR_FAIL_DTYPE,
})

# Compute capability (major version) at/above which bf16 is a real hardware
# feature.  Ampere (sm_80) introduced bf16; Turing T4 (sm_75) has no bf16 PTX
# path, so bf16 there is a dtype failure rather than a slow-but-running kernel.
_BF16_MIN_CC_MAJOR = 8


def _split_gold(source: str) -> tuple[str, str, str]:
    """Split a gold file into (kernel_part, separator, test_part).

    If the "#"*146 separator is absent, everything is treated as kernel_part.
    """
    head, sep, tail = source.partition(_GOLD_SEP)
    return head, sep, tail


def _force_block_value(code: str, names: tuple[str, ...], value: int) -> tuple[str, int]:
    """Force every BLOCK_* constant in *names* to *value*; return (code, n_subs).

    Four narrow, syntax-safe substitution forms, none of which mangle a computed
    right-hand side (e.g. ``BLOCK_SIZE = min(next_pow2(n), 1024)``):

      (1) integer-literal assign / kwarg — ``BLOCK_SIZE = 128`` / ``BLOCK_SIZE=128``
          RHS is ``\\d+`` only, so it can never swallow a comma or paren.
      (2) bare-identifier kwarg — ``kernel[grid](..., BLOCK_M=BLOCK)`` and the
          ``BLOCK_SIZE=BLOCK_SIZE`` passthrough idiom.  The RHS is a single
          identifier never followed by ``(`` / ``.`` / ``[`` (negative lookahead),
          so a computed call RHS like ``= min(...)`` is left untouched.
      (3) constexpr default — ``BLOCK_SIZE: tl.constexpr = 128``.
      (4) autotune Config dict key — ``triton.Config({'BLOCK_SIZE': 128})``.

    The launch kwarg / Config value is what the Triton compiler specialises on,
    so forcing these overrides any computed assignment left untouched upstream.
    """
    import re

    total = 0
    out = code
    for name in names:
        # (1) integer-literal RHS
        out, n1 = re.subn(rf"\b{name}\s*=\s*\d+\b", f"{name}={value}", out)
        # (2) bare-identifier RHS (not a call/attr/index), and not '=='
        out, n2 = re.subn(
            rf"\b{name}\s*=\s*(?!=)[A-Za-z_]\w*\b(?!\s*[(.\[])",
            f"{name}={value}",
            out,
        )
        # (3) constexpr default value
        out, n3 = re.subn(
            rf"(\b{name}\s*:\s*tl\.constexpr\s*=\s*)\d+", rf"\g<1>{value}", out
        )
        # (4) quoted dict key in a triton.Config(...)
        out, n4 = re.subn(rf"(['\"]{name}['\"]\s*:\s*)\d+", rf"\g<1>{value}", out)
        total += n1 + n2 + n3 + n4
    return out, total


def mutate_shared_mem_overflow(source: str) -> str | None:
    """Bump the block tile past T4's 48 KB shared-memory budget.

    2D tiles (BLOCK_M/BLOCK_N) → 256×256 fp32 = 256 KB; otherwise a 1D
    BLOCK_SIZE → 16384 elements × 4 B = 64 KB.  Both exceed T4's default SRAM.
    """
    code, sep, test = _split_gold(source)
    new, n = _force_block_value(code, ("BLOCK_M", "BLOCK_N"), 256)
    if n == 0:
        new, n = _force_block_value(code, ("BLOCK_SIZE",), 16384)
    if n == 0:
        return None
    return new + sep + test


def mutate_force_fp8(source: str) -> str | None:
    """Swap every Triton/Torch float dtype to fp8 (e4m3).  T4 has no fp8 path."""
    new = (
        source.replace("tl.float32", "tl.float8e4nv")
        .replace("tl.float16", "tl.float8e4nv")
        .replace("tl.bfloat16", "tl.float8e4nv")
        .replace("dtype=torch.float32", "dtype=torch.float8_e4m3fn")
    )
    return new if new != source else None


def mutate_force_bf16(source: str) -> str | None:
    """Swap fp32 -> bf16 for storage only (input tensors), leaving in-kernel math.

    Changing only ``dtype=torch.float32`` keeps fp32 accumulators intact (the
    realistic bf16 usage pattern) while still emitting ``.bf16`` PTX, so on
    pre-Ampere (sm_75) the kernel fails at ptxas ("requires .target sm_80") and on
    Ampere+ it runs. Replacing in-kernel ``tl.float32`` too would instead bf16 the
    accumulators against an fp32 ``tl.dot``, an arch-independent type error.

    Returns None when there is no ``dtype=torch.float32`` allocation to change.
    """
    new = source.replace("dtype=torch.float32", "dtype=torch.bfloat16")
    return new if new != source else None


def mutate_non_pow2_block(source: str) -> str | None:
    """Set a block constant to 100 (not a power of 2) — Triton rejects this."""
    code, sep, test = _split_gold(source)
    new, n = _force_block_value(code, ("BLOCK_SIZE",), 100)
    if n == 0:
        new, n = _force_block_value(code, ("BLOCK_M", "BLOCK_N"), 100)
    if n == 0:
        return None
    return new + sep + test


# Registry: (name, fn, predicted_behavior, is_perf_mutation). Failure mutations
# probe stderr; performance mutations time the kernel against the fp32 baseline.
# bf16's default is the Ampere+ perf prediction; generate_mutations overrides it
# to a dtype failure on pre-Ampere GPUs (see _BF16_MIN_CC_MAJOR).
MUTATORS: list[tuple] = [
    ("shared_mem_overflow",  mutate_shared_mem_overflow, BEHAVIOR_FAIL_SHARED_MEM, False),
    ("force_fp8",            mutate_force_fp8,           BEHAVIOR_FAIL_DTYPE,      False),
    ("force_bf16",           mutate_force_bf16,          BEHAVIOR_RUNS_NO_BF16,    True),
    ("non_pow2_block",       mutate_non_pow2_block,      BEHAVIOR_FAIL_BLOCK_SIZE, False),
]


def _failure_to_behavior(failure_type: str, stderr: str) -> str:
    """Map an observed kernel failure to the predicted_behavior vocabulary.

    Direct classifier labels map first; otherwise Triton frequently funnels
    these limits into a generic CompilationError, so disambiguate by keyword.
    Falls back to the raw classifier label when nothing matches (the row stays
    `validated: false`).
    """
    if failure_type == "shared_mem_overflow":
        return BEHAVIOR_FAIL_SHARED_MEM
    if failure_type == "dtype_unsupported":
        return BEHAVIOR_FAIL_DTYPE
    if failure_type == "invalid_block_size":
        return BEHAVIOR_FAIL_BLOCK_SIZE
    s = stderr.lower()
    if any(k in s for k in ("shared memory", "sram", "out of resource", "resource")):
        return BEHAVIOR_FAIL_SHARED_MEM
    if any(k in s for k in ("float8", "fp8", "e4m3", "e5m2")):
        return BEHAVIOR_FAIL_DTYPE
    if "requires .target sm_" in s or "requires sm_" in s:
        # ptxas rejects a dtype/feature unavailable on this compute capability
        # (e.g. "Feature '.bf16' requires .target sm_80 or higher") — an
        # arch/dtype hardware limit, not a code bug.
        return BEHAVIOR_FAIL_DTYPE
    if "power of 2" in s or "power-of-2" in s:
        return BEHAVIOR_FAIL_BLOCK_SIZE
    return failure_type


_TIMING_HARNESS = '''

if __name__ == "__main__":
    import time as _mt, torch as _mtorch
    try:
        for _ in range(3):
            {fn}()
        if _mtorch.cuda.is_available():
            _mtorch.cuda.synchronize()
        _samples = []
        for _ in range(10):
            if _mtorch.cuda.is_available():
                _mtorch.cuda.synchronize()
            _s = _mt.perf_counter()
            {fn}()
            if _mtorch.cuda.is_available():
                _mtorch.cuda.synchronize()
            _samples.append((_mt.perf_counter() - _s) * 1000.0)
        _samples.sort()
        print("MUT_RUNTIME_MS=" + repr(_samples[len(_samples) // 2]))
    except Exception as _e:
        print("MUT_RUNTIME_ERR=" + repr(str(_e)))
'''


def _median_runtime(source: str, workdir: str) -> tuple[float | None, int, str]:
    """Append the timing harness, run isolated; return (median_ms|None, rc, stderr).

    median_ms is None when there is no test_* function to call, the run fails,
    or the harness raised before printing a timing token.
    """
    import re

    m = re.search(r"\bdef (test_\w+)\s*\(", source)
    if not m:
        return None, -1, "no test_* function found for timing"
    timed = source + "\n" + _TIMING_HARNESS.format(fn=m.group(1))
    p = Path(workdir) / "_timed.py"
    p.write_text(timed)
    rc, out, err = _run_kernel_capture(p)
    if rc != 0:
        return None, rc, err
    mm = re.search(r"MUT_RUNTIME_MS=([0-9.eE+\-]+)", out)
    if mm:
        try:
            return float(mm.group(1)), 0, err
        except ValueError:
            pass
    return None, 0, err   # ran clean but produced no timing token

@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 6,
    volumes={DATA_DIR: data_volume},
)
def generate_mutations(
    output_subdir: str = "mutations",
    limit: int | None = None,
) -> dict:
    """Apply each mutation to every gold Triton kernel, verify the predicted
    hardware behaviour on the GPU, and write ``mutation_dataset.jsonl`` +
    ``mutation_summary.json`` to the Volume.
    """
    import tempfile
    import torch as _torch

    gold_paths = sorted(Path(GOLD_TRITON_DIR).glob("*.py"))
    if not gold_paths:
        raise FileNotFoundError(f"no gold Triton kernels found in {GOLD_TRITON_DIR}")
    if limit:
        gold_paths = gold_paths[:limit]

    props       = _torch.cuda.get_device_properties(0)
    compute_cap = f"{props.major}.{props.minor}"
    gpu_name    = _torch.cuda.get_device_name(0)

    print(
        f"\nmutating {len(gold_paths)} gold kernels on {gpu_name} "
        f"(cc {compute_cap}), {len(MUTATORS)} mutations each, "
        f"timeout={KERNEL_TIMEOUT}s per run",
        flush=True,
    )

    rows:  list[dict] = []
    skips: list[dict] = []
    attempted = 0
    applicable = 0
    by_mut = {
        name: {"attempted": 0, "applicable": 0, "validated": 0}
        for name, _, _, _ in MUTATORS
    }

    for ki, kpath in enumerate(gold_paths, 1):
        source = kpath.read_text()
        baseline_ms: float | None = None
        baseline_done = False   # baseline timed lazily, once per kernel

        with tempfile.TemporaryDirectory() as td:
            for name, fn, predicted, is_perf in MUTATORS:
                attempted += 1
                by_mut[name]["attempted"] += 1

                # bf16 is compute-capability-dependent: pre-Ampere (cc major < 8,
                # e.g. T4 sm_75) has no bf16 PTX path, so it's a dtype failure
                # there rather than a slow-but-running kernel.  Ampere+ keeps the
                # performance prediction (runs but without Tensor-Core speedup).
                if name == "force_bf16" and props.major < _BF16_MIN_CC_MAJOR:
                    predicted, is_perf = BEHAVIOR_FAIL_DTYPE, False

                try:
                    mutated = fn(source)
                except Exception as exc:  # noqa: BLE001 — a mutator bug shouldn't abort the run
                    skips.append({"kernel_id": kpath.name, "mutation_type": name,
                                  "reason": f"mutator error: {exc}"})
                    continue
                if mutated is None:
                    skips.append({"kernel_id": kpath.name, "mutation_type": name,
                                  "reason": "not applicable"})
                    continue
                if not _is_valid_python(mutated):
                    # Spec: a mutation that breaks ast.parse is skipped, no row.
                    skips.append({"kernel_id": kpath.name, "mutation_type": name,
                                  "reason": "ast.parse failed after mutation"})
                    continue

                applicable += 1
                by_mut[name]["applicable"] += 1

                failure_type: str | None = None
                mutated_ms:  float | None = None
                row_baseline: float | None = None

                if is_perf:
                    if not baseline_done:
                        baseline_ms, _, _ = _median_runtime(source, td)
                        baseline_done = True
                    mutated_ms, rc, err = _median_runtime(mutated, td)
                    row_baseline = baseline_ms
                    if rc != 0:
                        # Mutation broke the kernel instead of running slower.
                        failure_type = _classify_kernel_failure(err)
                        actual = _failure_to_behavior(failure_type, err)
                        mutated_ms = None
                    elif mutated_ms is not None and baseline_ms is not None:
                        # "validated" = the mutated kernel is NOT faster than fp32
                        # (>= 95% of baseline absorbs measurement noise).
                        actual = (
                            predicted if mutated_ms >= 0.95 * baseline_ms
                            else "runs_faster"
                        )
                    else:
                        actual = "runs_unmeasured"   # ran but timing unavailable
                else:
                    fpath = Path(td) / f"{name}__{kpath.name}"
                    fpath.write_text(mutated)
                    rc, err = _probe_kernel_file(fpath)
                    if rc == 0:
                        # Expected a hardware failure but the kernel ran clean.
                        actual = "runs_clean"
                    else:
                        failure_type = _classify_kernel_failure(err)
                        actual = _failure_to_behavior(failure_type, err)

                validated = (actual == predicted)
                if validated:
                    by_mut[name]["validated"] += 1

                rows.append({
                    "kernel_id":           kpath.name,
                    "original_kernel_path": str(kpath),
                    "mutation_type":       name,
                    "gpu":                 gpu_name,
                    "compute_cap":         compute_cap,
                    "predicted_behavior":  predicted,
                    "actual_behavior":     actual,
                    "validated":           validated,
                    "failure_type":        failure_type,
                    "is_hardware_failure": (
                        (failure_type in HARDWARE_FAILURE_TYPES if failure_type else False)
                        # Also flag arch/dtype limits the raw classifier misses
                        # (e.g. bf16 "requires sm_80" → other_runtime) via the
                        # observed hardware behaviour.
                        or actual in HARDWARE_BEHAVIORS
                    ),
                    "stderr_excerpt":      (err or "")[-2000:].replace("\n", " "),
                    "baseline_runtime_ms": round(row_baseline, 4) if row_baseline is not None else None,
                    "mutated_runtime_ms":  round(mutated_ms, 4) if mutated_ms is not None else None,
                    "mutated_kernel_code": mutated,
                })

        if ki % 5 == 0 or ki == len(gold_paths):
            _val = sum(1 for r in rows if r["validated"])
            print(
                f"  [{ki}/{len(gold_paths)}] kernels done — "
                f"{len(rows)} rows, {_val} validated, {len(skips)} skipped",
                flush=True,
            )

    # ---- write outputs --------------------------------------------------------
    out_dir = Path(DATA_DIR) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "mutation_dataset.jsonl"
    with dataset_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "gpu":                        gpu_name,
        "compute_cap":                compute_cap,
        "total_kernels_processed":    len(gold_paths),
        "total_mutations_attempted":  attempted,
        "total_mutations_applicable": applicable,
        "by_mutation": {
            name: {
                "attempted":       s["attempted"],
                "applicable":      s["applicable"],
                "validated":       s["validated"],
                "validation_rate": (round(s["validated"] / s["applicable"], 4)
                                    if s["applicable"] else None),
                "applicability_rate": (round(s["applicable"] / s["attempted"], 4)
                                       if s["attempted"] else None),
            }
            for name, s in by_mut.items()
        },
        "skipped": skips,
    }
    summary_path = out_dir / "mutation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    data_volume.commit()

    print(f"\nwrote mutation dataset -> {dataset_path} ({len(rows)} rows)", flush=True)
    print(f"wrote mutation summary -> {summary_path}", flush=True)
    # Flag any mutation whose applicability fell below 50% so it can be tuned.
    for name, s in summary["by_mutation"].items():
        ar = s["applicability_rate"]
        if ar is not None and ar < 0.5:
            _reasons = [k["reason"] for k in skips if k["mutation_type"] == name]
            print(
                f"  [low applicability] {name}: {ar:.0%} "
                f"({s['applicable']}/{s['attempted']}) — sample skip reasons: "
                f"{_reasons[:3]}",
                flush=True,
            )
    return summary

