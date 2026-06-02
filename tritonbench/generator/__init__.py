"""Phase C generator flow (owned by B/C/D).

The integration seam is a single function, `generate_kernel(operator_id, history)
-> code`, exported from `engine.py`. `prompting.py` (B) and `refinement.py` (C)
provide the pieces `engine.generate_kernel` (D) composes.
"""
from .engine import generate_kernel  # noqa: F401
