"""Config, Modal app/image/volume — shared by all tritonbench modules."""
from __future__ import annotations

import os

import modal

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

APP_NAME = "tritonbench-t"
TRITONBENCH_REPO = "https://github.com/thunlp/TritonBench.git"

# Cheapest Modal GPU (compute capability 7.5 — Triton requires >= 7.0).
# Override at runtime via `--gpu A10` etc. on the local entrypoint.
DEFAULT_GPU = "T4"

VOLUME_NAME = "tritonbench-t-data"
DATA_DIR = "/data"           # mount point of the Modal Volume in the container
REPO_DIR = "/opt/TritonBench"

# Default model in OpenRouter format.  Any $0-cost model slug from
# https://openrouter.ai/models works here, e.g.:
#   "mistralai/mistral-7b-instruct"
#   "nvidia/llama-3.1-nemotron-70b-instruct:free"
#   "microsoft/phi-3-mini-128k-instruct:free"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"

# Name of the Modal Secret that holds OPENROUTER_API_KEY.
# Override with an env var if your existing secret is named differently:
#     export TRITONBENCH_LLM_SECRET=my-other-secret
LLM_SECRET_NAME = os.environ.get("TRITONBENCH_LLM_SECRET", "tritonbench-llm")

# Retry knobs for free-tier rate limits (429 / 503 / empty responses).
MAX_RETRIES = 2
RETRY_BASE_DELAY = 4.0   # seconds; doubles each attempt (exponential backoff)

# Per-kernel subprocess resource limits for isolated kernel runs.
KERNEL_TIMEOUT  = 60               # wall-clock seconds before killing a subprocess
VIRT_MEM_BYTES  = 12 * 1024 ** 3  # 12 GiB virtual-address ceiling

# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #

# 0_call_acc.py — wrong dataset filename (.json vs .jsonl), wrong test folder
# (G instead of T), and a hardcoded conda interpreter path.
PATCH_CALL_ACC = (
    f"""sed -i """
    f"""-e 's|^statis_path = .*|statis_path = "{REPO_DIR}/data/TritonBench_T_v1.jsonl"|' """
    f"""-e 's|^py_folder = .*|py_folder = "{REPO_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|py_interpreter = __import__("sys").executable|' """
    f"""{REPO_DIR}/EVAL/eval_T/0_call_acc.py"""
)

# 1_exe_acc.py — same hardcoded conda interpreter; gold_folder anchored to
# absolute path.
PATCH_EXE_ACC = (
    f"""sed -i """
    f"""-e 's|^gold_folder = .*|gold_folder = "{REPO_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|py_interpreter = __import__("sys").executable|' """
    f"""{REPO_DIR}/EVAL/eval_T/1_exe_acc.py"""
)

# multiprocess_gpu_run.py — assumes 8 GPUs; we have one.
PATCH_PERF = (
    f"""sed -i 's|^gpu_count = .*|gpu_count = 1|' """
    f"""{REPO_DIR}/performance_metrics/perf_T/run_bench/multiprocess_gpu_run.py"""
)


image = (
    modal.Image.from_registry(
        # Python 3.12: TritonBench's eval scripts use PEP-701 nested-quote
        # f-strings, which require >= 3.12 to parse.
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "triton==3.1.0",
        "tqdm==4.66.5",
        "numpy<2",
        "openai>=1.50",
        "psutil>=5.9",   # memory diagnostics
    )
    .run_commands(f"git clone --depth 1 {TRITONBENCH_REPO} {REPO_DIR}")
    .run_commands(PATCH_CALL_ACC, PATCH_EXE_ACC, PATCH_PERF)
    # ProcessPoolExecutor pickles workers by qualified module name, so the
    # eval scripts must be importable as plain `call_acc` / `exe_acc` from any
    # subprocess. Module names can't start with a digit, so symlink them.
    .run_commands(
        f"ln -s {REPO_DIR}/EVAL/eval_T/0_call_acc.py {REPO_DIR}/EVAL/eval_T/call_acc.py",
        f"ln -s {REPO_DIR}/EVAL/eval_T/1_exe_acc.py {REPO_DIR}/EVAL/eval_T/exe_acc.py",
    )
    # Ship the local package into the image (Modal does not auto-mount it).
    .add_local_python_source("tritonbench", "modal_app")
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# Target-hardware facts, used by both the generation prompt and the failure hints.
T4_HARDWARE = {
    "name": "NVIDIA T4",
    "compute_capability": "7.5",
    "arch": "Turing",
    "shared_mem_kb_per_block": 48,
    "supports_bf16": False,   # bf16 needs sm_80+ (Ampere)
    "supports_fp8": False,    # fp8 needs sm_89/sm_90 (Ada/Hopper)
    "supports_tf32": False,   # TF32 needs sm_80+
    "tensor_core_dtypes": ("fp16", "int8"),
    "constraints": (
        "No bf16 (requires sm_80+) — use fp32 or fp16 for storage and math.",
        "No fp8 (requires sm_89/sm_90).",
        "Shared memory must stay <= 48 KB per block.",
        "BLOCK_SIZE constants must be powers of 2.",
        "fp16 Tensor Cores need matmul dims that are multiples of 16.",
    ),
}

# Fix hint per failure category.
T4_FAILURE_HINTS = {
    "dtype_unsupported":   "T4 (sm_75) has no bf16/fp8 silicon; keep storage and math in fp32 (or fp16). bf16 needs sm_80+, fp8 needs sm_89+.",
    "shared_mem_overflow": "The tile exceeds T4's 48 KB shared memory; lower BLOCK_SIZE / tile dims so staged data fits.",
    "register_overflow":   "Kernel needs more registers than the SM provides; reduce per-thread work, unrolling, or block size.",
    "invalid_block_size":  "Triton requires power-of-2 block sizes; pick the nearest power of 2.",
    "numerical_mismatch":  "It compiles and runs but the output differs from the PyTorch reference; revisit the math/indexing/masking.",

    # Newly added categories
    "triton_api_misuse": "Triton ops live under triton.language (tl) and run only inside @triton.jit; removed APIs like tl.libdevice/tl.extra are gone in Triton 3.x. Import triton.language as tl, add @triton.jit to kernels, and replace missing attrs with tl.math or supported tl ops.",
    "triton_unsupported_construct": "Triton lowers a restricted subset of Python; dynamic lists/dicts, variable-length loops, and short-circuit booleans are not supported. Express control with masks/tl.where, use constexpr/static shapes, iterate over tl.arange ranges, and avoid Python-side flow in the kernel.",
    "triton_compilation_error": "Generic Triton compile failure on sm_75 usually stems from shapes/dtypes or pointer math Triton cannot lower. Minimize the kernel, make shapes constexpr, ensure pointer math uses int32/int64, add explicit dtypes, and compile a small tile first.",
    "model_abdication": "The code asserts the case is unsupported instead of implementing it. Remove the refusal and implement a masked, bounds-checked path that handles the test inputs.",
    "code_error": "Python failed before Triton could JIT (SyntaxError/NameError/ImportError). Fix the Python error: ensure symbols are imported/defined, syntax is valid, and only call tl.* inside @triton.jit kernels.",
    "other_runtime": "Runtime failed on T4 without a specific classifier; common causes are OOB or misaligned memory ops. Add bounds masks on every load/store, respect strides, and align pointers to element size.",

    # Additional hardware/runtime categories to fully cover classifier outputs
    "arch_unsupported": "The binary is not built for T4's sm_75. Avoid features that require newer SMs and ensure the kernel uses only sm_75-available ops and dtypes (fp32/fp16). Remove any hard-coded arch guards or sm_80+ intrinsics so Triton JIT targets sm_75.",
    "oom": "CUDA ran out of memory on T4. Reduce BLOCK_SIZE/tile sizes, stage fewer elements in shared memory, avoid large temporaries, and chunk the computation so working sets fit.",
    "illegal_memory_access": "Likely out-of-bounds or misaligned access. Add masks to every tl.load/tl.store, compute offsets with correct strides and element sizes, and pass mask/other=0 for out-of-bounds lanes.",
}


__all__ = [
    "APP_NAME", "TRITONBENCH_REPO", "DEFAULT_GPU", "VOLUME_NAME", "DATA_DIR",
    "REPO_DIR", "DEFAULT_MODEL", "LLM_SECRET_NAME", "MAX_RETRIES",
    "RETRY_BASE_DELAY", "KERNEL_TIMEOUT", "VIRT_MEM_BYTES",
    "image", "app", "data_volume", "T4_HARDWARE", "T4_FAILURE_HINTS",
]
