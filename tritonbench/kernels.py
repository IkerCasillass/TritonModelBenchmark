"""Isolated kernel execution + Triton/CUDA failure classification."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .core import *  # noqa: F401,F403

# Categories that indicate a hardware constraint rather than a code bug.
# Used to set `is_hardware_failure` in the failure dataset.
HARDWARE_FAILURE_TYPES: frozenset[str] = frozenset({
    "shared_mem_overflow",
    "register_overflow",
    "arch_unsupported",
    "dtype_unsupported",
    "oom",
    "illegal_memory_access",
})


def _classify_kernel_failure(stderr: str) -> str:
    """Classify Triton/CUDA compilation or execution stderr into a failure category.

    Distinct from _classify_failure, which handles LLM API errors.

    Ordering rule — most specific first, generic fallbacks last:
      1. Hardware-attributable (PTX / CUDA runtime limits)
      2. triton_api_misuse          — hallucinated or removed tl.* attributes
      3. triton_unsupported_construct — valid Python that Triton's compiler rejects
      4. triton_compilation_error   — catch-all for CompilationError (parent class;
                                      must come after the more specific subclasses)
      5. model_abdication           — wrapper asserts that explicitly refuse work
      6. invalid_block_size         — tight pattern; bare identifier is not enough
      7. numerical_mismatch
      8. code_error                 — Python-level: SyntaxError / NameError / ImportError
      9. other_runtime              — true fallback

    Patterns validated against Triton 3.1.0 / CUDA 12.4 error output.
    """
    s = stderr.lower()

    # --- 1. hardware-attributable ----------------------------------------
    if "out of resource" in s and "shared memory" in s:
        return "shared_mem_overflow"
    if "out of resource" in s and "regist" in s:
        # "regist" matches both "register" and "registers" in PTX compiler output.
        return "register_overflow"
    if "no kernel image is available" in s:
        # CUDA runtime: binary not compiled for this device's compute capability.
        return "arch_unsupported"
    if ("not supported" in s or "unsupported" in s) and any(
        t in s for t in ("bf16", "fp8", "float8")
    ):
        # Ambiguous: "not supported" appears in many error paths; requiring an
        # explicit dtype token (bf16/fp8/float8) narrows to GPU dtype limits.
        return "dtype_unsupported"
    if "cuda out of memory" in s or "out of memory" in s:
        return "oom"
    if "illegal memory access" in s:
        return "illegal_memory_access"

    # --- 2. triton_api_misuse --------------------------------------------
    # Triton 3.x removed tl.libdevice (→ tl.math) and tl.extra entirely.
    # Hallucinated attributes (tl.pow, tl.program, …) hit the same error path.
    # The exact string Triton emits when JIT-visiting an unknown attribute is:
    #   AttributeError: module 'triton.language' has no attribute '<name>'
    # tl.math sub-attributes produce the analogous message for that submodule.
    if (
        # Covers triton.language, triton.language.math, triton.language.extra, etc.
        # The closing quote is intentionally omitted so all tl.* submodules match.
        "attributeerror: module 'triton.language" in s
        # tl.* called outside @triton.jit (e.g. used as a type annotation).
        or "did you forget to add @triton.jit" in s
        # Same root cause — tl internals require _builder at compile time.
        or "_builder argument must be provided" in s
    ):
        return "triton_api_misuse"

    # --- 3. triton_unsupported_construct ---------------------------------
    # Triton's compiler rejects syntactically valid Python it cannot lower:
    # chained boolean operators, simultaneous comparisons, etc.
    if "triton.compiler.errors.unsupportedlanguageconstruct" in s:
        return "triton_unsupported_construct"

    # --- 4. triton_compilation_error -------------------------------------
    # Catch-all for CompilationError not covered above.  Checked AFTER
    # UnsupportedLanguageConstruct (a subclass) so the specific category wins
    # when both class names appear in the same traceback.
    if "triton.compiler.errors.compilationerror" in s:
        return "triton_compilation_error"

    # --- 5. model_abdication ---------------------------------------------
    # The LLM wrote a wrapper that explicitly asserts the test case isn't
    # supported.  A bare AssertionError with no message stays other_runtime
    # because there is no refusal text to match.
    if "assertionerror" in s and any(
        t in s for t in ("is supported", "not supported", "must be", "input must be")
    ):
        return "model_abdication"

    # --- 6. invalid_block_size -------------------------------------------
    # Require an explicit constraint-violation message alongside the identifier.
    # A bare "block_size" in stderr is just an identifier in a traceback frame
    # and fires on unrelated errors (false positive observed in production data).
    if "must be a power of 2" in s or (
        ("block_size" in s or "block size" in s)
        and any(t in s for t in ("too large", "exceeds", "expected a power"))
    ):
        return "invalid_block_size"

    # --- 7. numerical_mismatch -------------------------------------------
    if any(t in s for t in ("allclose", "mismatch", "incorrect", "not equal")):
        return "numerical_mismatch"

    # --- 8. code_error ---------------------------------------------------
    if any(t in s for t in ("syntaxerror", "nameerror", "importerror")):
        return "code_error"

    return "other_runtime"


def _set_mem_limit() -> None:
    """Pre-exec hook: cap virtual address space to VIRT_MEM_BYTES.

    Prevents a runaway kernel subprocess from OOM-killing the parent container.
    Best-effort — only effective on Linux; silently no-ops elsewhere.
    """
    import resource as _resource
    try:
        _resource.setrlimit(_resource.RLIMIT_AS, (VIRT_MEM_BYTES, VIRT_MEM_BYTES))
    except Exception:
        pass


def _probe_kernel_file(path: Path, timeout: int = KERNEL_TIMEOUT) -> tuple[int, str]:
    """Run *path* in an isolated subprocess; return (returncode, stderr).

    Mirrors the Phase 3 isolation pattern: a wall-clock limit (default
    KERNEL_TIMEOUT; callers may pass a shorter one) and _set_mem_limit
    virtual-memory ceiling.  Called before Phase 1 / Phase 2 upstream scripts run
    so stderr is captured before failing files are deleted.  stdout is discarded
    — only stderr carries failure information.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_set_mem_limit,
        )
        return proc.returncode, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, f"TimeoutExpired: kernel did not complete within {timeout}s"
    except Exception as exc:
        return -1, str(exc)

def _run_kernel_capture(path: Path) -> tuple[int, str, str]:
    """Like _probe_kernel_file but also returns stdout (needed for timing).

    Same KERNEL_TIMEOUT + _set_mem_limit isolation as Phase 1.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=KERNEL_TIMEOUT,
            preexec_fn=_set_mem_limit,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"TimeoutExpired: kernel did not complete within {KERNEL_TIMEOUT}s"
    except Exception as exc:
        return -1, "", str(exc)


# Appended to a gold module to time its wrapper: warm up 3×, then 10 timed
# runs, print the median wall-time in ms.  Runs under `python file.py`, after
