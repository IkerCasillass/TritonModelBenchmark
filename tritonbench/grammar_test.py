from pathlib import Path
import xgrammar as xgr
import os

print("CWD:", os.getcwd())

def load_ebnf(path: str) -> str:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"EBNF file not found: {path}")

    content = p.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"EBNF file is empty: {path}")

    return content

ebnf = load_ebnf("tritonbench/grammars/triton.ebnf")
compiled = xgr.Grammar.from_ebnf(ebnf)

print(compiled)