"""Pluggable generation backends (Strategy pattern).

The active backend is process-global, set once per run by ``generate_refine`` and
read by ``generate_kernel`` — this keeps ``generate_kernel``'s frozen signature
unchanged while letting a run choose where generation happens. Default is
``local`` so existing behaviour is untouched unless a run opts in.
"""
from __future__ import annotations

from typing import Protocol


class GenerationBackend(Protocol):
    def generate(self, messages: list[dict]) -> str: ...


class LocalConstrainedBackend:
    """In-process transformers + xgrammar on the eval GPU."""

    def generate(self, messages: list[dict]) -> str:
        from ..llm_local import _gen_constrained
        return _gen_constrained(messages)


class VLLMRemoteBackend:
    """Grammar-constrained vLLM on a dedicated GPU, called over RPC.

    Honours the module-level `_constrained` toggle; the local backend is always
    grammar-constrained, so the unconstrained ablation arm runs on this backend.
    """

    def generate(self, messages: list[dict]) -> str:
        from ..gen_service import ConstrainedGenerator
        return ConstrainedGenerator().generate.remote(messages, constrained=_constrained)


_BACKENDS: dict[str, type] = {
    "local": LocalConstrainedBackend,
    "vllm": VLLMRemoteBackend,
}

_active = "local"
_constrained = True


def set_active_backend(name: str) -> None:
    if name not in _BACKENDS:
        raise ValueError(f"unknown gen backend {name!r}; choices: {sorted(_BACKENDS)}")
    global _active
    _active = name


def set_constrained(flag: bool) -> None:
    global _constrained
    _constrained = bool(flag)


def get_backend(name: str | None = None) -> GenerationBackend:
    return _BACKENDS[name or _active]()
