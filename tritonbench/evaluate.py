"""Kernel evaluation: call-accuracy, execution-accuracy, and efficiency stages."""
from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .core import *  # noqa: F401,F403
from .kernels import (HARDWARE_FAILURE_TYPES, _classify_kernel_failure,
                      _probe_kernel_file, _set_mem_limit)
from .llm import _gen_meta_path

def _parse_per_kernel_speedups(stdout: str) -> list[float]:
    """Extract per-kernel speedup ratios from 2_efficiency.py stdout.

    That script prints one ``{filename}: {ratio}`` line per kernel, then a
    final ``speed up: {mean}`` summary line. The summary line is excluded
    automatically here: "speed up" contains a space, so it is not a single
    ``\\S+`` token.
    """
    import re

    out: list[float] = []
    for line in stdout.splitlines():
        m = re.match(r"^(\S+):\s+([0-9]*\.?[0-9]+)\s*$", line)
        if m:
            out.append(float(m.group(2)))
    return out


def _speedup_stats(values: list[float]) -> dict:
    """Aggregate per-kernel speedups into comparison-friendly stats.

    Geomean (not the arithmetic mean upstream uses) is the correct central
    tendency for ratios. ``n_kernels_measured`` is reported so the
    survivorship bias of any speedup aggregate is visible.
    """
    n = len(values)
    if n == 0:
        return {
            "n_kernels_measured": 0,
            "geomean_speedup": None,
            "median_speedup": None,
            "min_speedup": None,
            "max_speedup": None,
            "pct_faster_than_pytorch": None,
        }
    geomean = math.exp(sum(math.log(v) for v in values) / n)
    faster = sum(1 for v in values if v > 1.0)
    return {
        "n_kernels_measured": n,
        "geomean_speedup": round(geomean, 4),
        "median_speedup": round(statistics.median(values), 4),
        "min_speedup": round(min(values), 4),
        "max_speedup": round(max(values), 4),
        "pct_faster_than_pytorch": round(100 * faster / n, 2),
    }


@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 6,
    volumes={DATA_DIR: data_volume},
)
def evaluate(
    predictions_path: str = "predictions.jsonl",
    output_subdir: str = "results",
    model: str = "",
) -> dict:
    """Run TritonBench-T eval phases against an existing predictions.jsonl."""
    pred_full = Path(DATA_DIR) / predictions_path
    if not pred_full.exists():
        raise FileNotFoundError(f"predictions file not found in volume: {pred_full}")

    # Count total predictions and how many are already failed stubs —
    # so we can report an accurate baseline before Phase 1 even runs.
    total = 0
    gen_failures = 0
    for line in pred_full.open():
        total += 1
        rec = json.loads(line)
        code = rec.get("predict", "")
        if code.strip().startswith("# generation failed"):
            gen_failures += 1

    print(
        f"\npredictions file: {total} entries, "
        f"{gen_failures} generation failures ({total - gen_failures} usable)",
        flush=True,
    )

    out_dir = Path(DATA_DIR) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    call_acc_dir = out_dir / "call_acc"
    perf_results_dir = out_dir / "perf_results"

    if call_acc_dir.exists():
        shutil.rmtree(call_acc_dir)
    if perf_results_dir.exists():
        shutil.rmtree(perf_results_dir)

    # Make the eval modules importable as `call_acc` / `exe_acc` from any
    # subprocess (ProcessPoolExecutor pickles workers by qualified name).
    eval_dir = f"{REPO_DIR}/EVAL/eval_T"
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    os.environ["PYTHONPATH"] = eval_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

    import call_acc  # noqa: E402
    import exe_acc   # noqa: E402

    import tempfile
    import torch as _torch

    _gpu_props   = _torch.cuda.get_device_properties(0)
    _compute_cap = f"{_gpu_props.major}.{_gpu_props.minor}"
    _gpu_name    = _torch.cuda.get_device_name(0)
    failure_records: list[dict] = []   # accumulated across Phase 1 and Phase 2

    timings: dict = {}

    # ---- Phase 1 pre-probe: capture stderr before call_acc deletes failures ----
    #
    # call_acc.get_codes_for_test() returns the exact (code, test, filename)
    # triples that call_4file uses internally.  Writing code+test to a temp file
    # and running it in an isolated subprocess replicates Phase 1's acceptance
    # test, so the filenames in phase1_probe map one-to-one to call_acc_dir.
    phase1_probe:  dict[str, tuple[int, str]] = {}  # filename -> (returncode, stderr)
    _probe_codes:  list[str] = []  # raw generated code (no test appended), for kernel_code field
    _probe_fnames: list[str] = []  # TritonBench filenames, same order as predictions

    try:
        _pcodes, _ptests, _pfiles = call_acc.get_codes_for_test(str(pred_full))
        _probe_codes  = list(_pcodes)
        _probe_fnames = list(_pfiles)
        print(
            f"\nphase1 pre-probe: running {len(_pfiles)} kernels in isolated subprocesses"
            f" (timeout={KERNEL_TIMEOUT}s each) ...",
            flush=True,
        )
        with tempfile.TemporaryDirectory() as _td:
            _td_path = Path(_td)
            for _i, (_code, _test, _fname) in enumerate(zip(_pcodes, _ptests, _pfiles), 1):
                _fpath = _td_path / _fname
                _fpath.write_text(_code + "\n" + "#" * 146 + "\n" + _test)
                _rc, _se = _probe_kernel_file(_fpath)
                phase1_probe[_fname] = (_rc, _se)
                if _i % 20 == 0 or _i == len(_pfiles):
                    _nfail = sum(1 for r, _ in phase1_probe.values() if r != 0)
                    print(f"  pre-probe progress: {_i}/{len(_pfiles)}  failures so far: {_nfail}", flush=True)
        _pre_fail = sum(1 for _rc, _ in phase1_probe.values() if _rc != 0)
        print(f"phase1 pre-probe complete: {_pre_fail}/{len(phase1_probe)} predicted failures", flush=True)
    except Exception as _exc:
        print(f"warning: phase1 pre-probe skipped ({_exc})", flush=True)

    # ---- Phase 1: call accuracy ------------------------------------------------
    print("\n" + "=" * 70 + "\n=== Phase 1: call accuracy ===\n" + "=" * 70, flush=True)
    _t = time.perf_counter()
    call_acc.call_4file(str(pred_full), str(call_acc_dir), gpus=[0])
    call_survivors = sorted(p.name for p in call_acc_dir.glob("*.py"))
    timings["phase1_call_acc_s"] = round(time.perf_counter() - _t, 1)
    print(f"\ncall_acc survivors: {len(call_survivors)} / {total}", flush=True)

    # Record Phase 1 failures.  phase1_probe and call_acc use the same filenames
    # (both from get_codes_for_test), so set-difference gives an exact mapping.
    if phase1_probe:
        _call_survivor_set = set(call_survivors)
        for _fname, _code in zip(_probe_fnames, _probe_codes):
            if _fname not in _call_survivor_set:
                _rc, _se = phase1_probe.get(_fname, (-1, ""))
                _ftype = _classify_kernel_failure(_se)
                failure_records.append({
                    "kernel_id":          _fname,
                    "gpu":                _gpu_name,
                    "compute_cap":        _compute_cap,
                    "phase_failed":       1,
                    "failure_type":       _ftype,
                    "is_hardware_failure": _ftype in HARDWARE_FAILURE_TYPES,
                    "stderr_excerpt":     _se[-2000:].replace("\n", " "),
                    "kernel_code":        _code,
                })
        _hw = sum(1 for r in failure_records if r["phase_failed"] == 1 and r["is_hardware_failure"])
        print(
            f"phase1 failures recorded: {len(failure_records)} total, {_hw} hardware-attributable",
            flush=True,
        )

    # ---- Phase 2 pre-probe: capture stderr before exe_acc deletes failures -----
    #
    # Files in call_acc_dir already contain code + "#"*146 + test (written by
    # Phase 1).  Running them directly replicates what exe_acc does before the
    # stdout comparison step, so any Triton compilation / runtime error shows up
    # in stderr here.  Files that exit 0 but are later deleted by exe_acc failed
    # because their stdout didn't match the golden reference — numerical_mismatch.
    # filename -> (returncode, stderr, raw_kernel_code)
    # raw_kernel_code is read here because exe_acc will delete failing files.
    phase2_probe: dict[str, tuple[int, str, str]] = {}
    if call_survivors:
        print(
            f"\nphase2 pre-probe: running {len(call_survivors)} kernels in isolated subprocesses"
            f" (timeout={KERNEL_TIMEOUT}s each) ...",
            flush=True,
        )
        for _i, _fname in enumerate(call_survivors, 1):
            _fpath = call_acc_dir / _fname
            # Strip the embedded test section to recover the raw generated code
            # (Phase 1 wrote: code + "\n" + "#"*146 + "\n" + test).
            _raw = _fpath.read_text().split("#" * 146)[0].rstrip("\n")
            _rc, _se = _probe_kernel_file(_fpath)
            phase2_probe[_fname] = (_rc, _se, _raw)
            if _i % 20 == 0 or _i == len(call_survivors):
                _nfail = sum(1 for r, _, __ in phase2_probe.values() if r != 0)
                print(f"  pre-probe progress: {_i}/{len(call_survivors)}  failures so far: {_nfail}", flush=True)
        _pre_fail2 = sum(1 for _rc, _, __ in phase2_probe.values() if _rc != 0)
        print(f"phase2 pre-probe complete: {_pre_fail2}/{len(phase2_probe)} runtime failures", flush=True)

    # ---- Phase 2: execution accuracy -------------------------------------------
    print("\n" + "=" * 70 + "\n=== Phase 2: execution accuracy ===\n" + "=" * 70, flush=True)
    _t = time.perf_counter()
    if call_survivors:
        exe_acc.execute_4folder(str(call_acc_dir), gpus=[0])

    # execute_4folder removes files that fail; re-glob for the survivors.
    exec_survivors = sorted(p.name for p in call_acc_dir.glob("*.py"))
    timings["phase2_exec_acc_s"] = round(time.perf_counter() - _t, 1)
    print(f"\nexe_acc survivors: {len(exec_survivors)} / {total}", flush=True)

    # Record Phase 2 failures: call_acc survivors absent from exec_survivors.
    # Probe exit != 0  → Triton/runtime error; classify its stderr.
    # Probe exit == 0  → ran fine in isolation but stdout differed from golden
    #                    (numerical mismatch); no meaningful stderr to classify.
    if phase2_probe:
        _exec_survivor_set = set(exec_survivors)
        _p2_start = len(failure_records)
        for _fname in call_survivors:
            if _fname not in _exec_survivor_set:
                _rc, _se, _kernel_code = phase2_probe.get(_fname, (-1, "", ""))
                if _rc == 0:
                    # Passed bare execution; stdout differed from golden output.
                    _ftype = "numerical_mismatch"
                    _se    = ""
                else:
                    _ftype = _classify_kernel_failure(_se)
                failure_records.append({
                    "kernel_id":           _fname,
                    "gpu":                 _gpu_name,
                    "compute_cap":         _compute_cap,
                    "phase_failed":        2,
                    "failure_type":        _ftype,
                    "is_hardware_failure": _ftype in HARDWARE_FAILURE_TYPES,
                    "stderr_excerpt":      _se[-2000:].replace("\n", " "),
                    "kernel_code":         _kernel_code,
                })
        _p2_records = failure_records[_p2_start:]
        _hw2 = sum(1 for r in _p2_records if r["is_hardware_failure"])
        print(
            f"phase2 failures recorded: {len(_p2_records)} total, {_hw2} hardware-attributable",
            flush=True,
        )

    # ---- Phase 3: efficiency ---------------------------------------------------
    print("\n" + "=" * 70 + "\n=== Phase 3: efficiency ===\n" + "=" * 70, flush=True)
    _t = time.perf_counter()
    eff_summary = "skipped (no surviving operators)"
    speedup = None
    per_kernel_speedups: list[float] = []
    if exec_survivors:
        perf_root = f"{REPO_DIR}/performance_metrics/perf_T"

        # 3a — generate per-op perf scripts.
        # capture_output so we can print what write_file.py actually did;
        # no check=True — a non-zero exit is logged but we carry on so the
        # scripts that *were* written still get benchmarked.
        write_proc = subprocess.run(
            [
                sys.executable,
                "run_bench/write_file.py",
                "--input_folder_path",
                str(call_acc_dir),
                "--results_path",
                str(perf_results_dir),
            ],
            cwd=perf_root,
            capture_output=True,
            text=True,
        )
        print(f"write_file.py exit={write_proc.returncode}", flush=True)
        if write_proc.stdout.strip():
            print(write_proc.stdout[:1000], flush=True)
        if write_proc.stderr.strip():
            print("[write_file stderr]", write_proc.stderr[:1000], flush=True)

        # Diagnostic: list what was actually written to perf_results_dir.
        perf_results_dir.mkdir(parents=True, exist_ok=True)
        all_written = list(perf_results_dir.iterdir())
        print(
            f"perf_results_dir contains {len(all_written)} items: "
            f"{[p.name for p in all_written[:10]]}",
            flush=True,
        )

        # write_file.py may write scripts into a subdirectory rather than
        # directly into perf_results_dir.  Walk the whole tree.
        perf_scripts_all = sorted(perf_results_dir.rglob("*.py"))
        print(f"found {len(perf_scripts_all)} .py scripts under perf_results_dir", flush=True)

        # If still zero, print the full directory tree for diagnosis.
        if not perf_scripts_all:
            print("[diag] full tree of perf_results_dir:", flush=True)
            for p in sorted(perf_results_dir.rglob("*")):
                print(f"  {p.relative_to(perf_results_dir)}", flush=True)
            # Also check whether write_file.py wrote into the CWD (perf_root)
            # instead of perf_results_dir — some versions do this.
            cwd_py = sorted(Path(perf_root).glob("tmp/*.py"))
            if cwd_py:
                print(
                    f"[diag] found {len(cwd_py)} .py files under {perf_root}/tmp — "
                    "using those instead",
                    flush=True,
                )
                perf_scripts_all = cwd_py

        # 3b — run each generated perf script in its own isolated subprocess.
        #
        # The upstream multiprocess_gpu_run.py pools all kernels together; if
        # one leaks GPU/CPU memory it OOM-kills the entire pool (exit 137) and
        # we lose every result.  Running one-at-a-time with a hard timeout and
        # a per-process memory ceiling lets bad kernels be skipped cleanly.
        # KERNEL_TIMEOUT, VIRT_MEM_BYTES, and _set_mem_limit are module-level.

        perf_scripts = perf_scripts_all
        print(
            f"\nrunning {len(perf_scripts)} perf scripts "
            f"(timeout={KERNEL_TIMEOUT}s each, mem<=12GiB each)",
            flush=True,
        )
        perf_skipped = 0
        for idx, script in enumerate(perf_scripts, 1):
            print(f"  [{idx:2d}/{len(perf_scripts)}] {script.name}", end=" ", flush=True)
            try:
                proc = subprocess.run(
                    [sys.executable, str(script)],
                    cwd=perf_root,
                    timeout=KERNEL_TIMEOUT,
                    preexec_fn=_set_mem_limit,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode == 0:
                    print("OK", flush=True)
                else:
                    snippet = (proc.stderr or proc.stdout or "")[:120].strip()
                    print(f"exit {proc.returncode} — {snippet}", flush=True)
                    perf_skipped += 1
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT (>{KERNEL_TIMEOUT}s) — skipped", flush=True)
                perf_skipped += 1
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR — {exc} — skipped", flush=True)
                perf_skipped += 1

        print(
            f"\nperf scripts: {len(perf_scripts) - perf_skipped} completed, "
            f"{perf_skipped} skipped",
            flush=True,
        )

        # 3c — compute speedup vs. the golden PyTorch numbers.
        # Only run 2_efficiency.py when at least one perf script finished;
        # an empty results dir causes a ZeroDivisionError inside that script.
        completed_count = len(perf_scripts) - perf_skipped
        if completed_count == 0:
            eff_summary = (
                "skipped: all perf scripts failed or timed out "
                f"({perf_skipped}/{len(perf_scripts)} skipped)"
            )
            print(eff_summary, flush=True)
        else:
            eff = subprocess.run(
                [
                    sys.executable,
                    "2_efficiency.py",
                    "--gen_folder",
                    str(perf_results_dir),
                ],
                cwd=f"{REPO_DIR}/EVAL/eval_T",
                capture_output=True,
                text=True,
            )
            eff_summary = eff.stdout
            if eff.stderr:
                # Filter out the ZeroDivisionError that fires when some ops
                # have no benchmark result — it's harmless if others do.
                filtered = [
                    l for l in eff.stderr.splitlines()
                    if "ZeroDivisionError" not in l and "avg" not in l
                ]
                if filtered:
                    eff_summary += "\n[stderr]\n" + "\n".join(filtered)
            for line in eff.stdout.splitlines():
                if line.startswith("speed up:"):
                    try:
                        speedup = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
            per_kernel_speedups = _parse_per_kernel_speedups(eff.stdout)

    timings["phase3_efficiency_s"] = round(time.perf_counter() - _t, 1)

    # ---- generation telemetry from the sidecar (absent for BYO uploads) ------
    gen_meta_file = _gen_meta_path(pred_full)
    generation = None
    if gen_meta_file.exists():
        try:
            generation = json.loads(gen_meta_file.read_text()).get("generation")
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not read {gen_meta_file}: {exc}", flush=True)

    speedup_stats = _speedup_stats(per_kernel_speedups)

    # ---- failure dataset -------------------------------------------------------
    by_type: dict[str, int] = {}
    for _r in failure_records:
        by_type[_r["failure_type"]] = by_type.get(_r["failure_type"], 0) + 1
    hw_failures   = sum(1 for _r in failure_records if _r["is_hardware_failure"])
    failure_analysis = {
        "total_failures":    len(failure_records),
        "hardware_failures": hw_failures,
        "code_failures":     len(failure_records) - hw_failures,
        "by_type":           by_type,
    }

    dataset_path = out_dir / "failure_dataset.jsonl"
    with dataset_path.open("w") as _f:
        for _r in failure_records:
            _f.write(json.dumps(_r, ensure_ascii=False) + "\n")
    print(f"wrote failure dataset -> {dataset_path} ({len(failure_records)} records)", flush=True)

    summary = {
        "model": model,
        "total_predictions": total,
        "generation_failures": gen_failures,
        "usable_predictions": total - gen_failures,
        "phase1_call_acc": {
            "passed": len(call_survivors),
            "rate": round(100 * len(call_survivors) / total, 2) if total else 0,
        },
        "phase2_exec_acc": {
            "passed": len(exec_survivors),
            "rate": round(100 * len(exec_survivors) / total, 2) if total else 0,
            "rate_among_phase1": (
                round(100 * len(exec_survivors) / len(call_survivors), 2)
                if call_survivors else None
            ),
        },
        "phase3_efficiency": {
            "mean_speedup": speedup,          # upstream arithmetic mean
            **speedup_stats,                  # geomean / median / %faster / N
            "raw_output_tail": eff_summary[-2000:],
        },
        "timing_s": timings,
        "generation": generation,
        "failure_analysis": failure_analysis,
        "artifacts_volume": VOLUME_NAME,
        "artifacts_subdir": output_subdir,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    data_volume.commit()
    print(f"\nwrote summary -> {summary_path}", flush=True)
    return summary

