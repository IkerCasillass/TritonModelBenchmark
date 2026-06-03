"""Backward-compat shim. Real code lives in the tritonbench/ package.

Run e.g.  modal run modal_app.py::main  (or ::mutate_only, ::hardware_eval_only).
The import of tritonbench.entrypoints registers every @app.function.
"""
from tritonbench.core import app  # noqa: F401
from tritonbench.entrypoints import *  # noqa: F401,F403
