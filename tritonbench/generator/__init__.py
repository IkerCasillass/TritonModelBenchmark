"""Kernel generation: prompt construction, model calls, and feedback-driven revision.

The integration point is ``generate_kernel(operator_id, history) -> code``.
"""
from .engine import generate_kernel  # noqa: F401
