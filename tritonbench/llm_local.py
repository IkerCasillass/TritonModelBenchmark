"""Grammar-constrained local generation: Qwen + XGrammar.

Single public symbol: _gen_constrained(messages) -> str
Drop-in for _gen() in engine.py — returns raw text instead of GenResult.

Model and compiled grammar are initialised once on first call (lazy singleton).
On Modal this naturally aligns with @modal.enter() if you wrap the engine in
a @modal.cls — but that is the caller's concern, not this module's.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


# Model slug. Override at import time if needed:
#   import tritonbench.generator.llm_local as _loc; _loc._MODEL_ID = "..."
_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" # "Qwen/Qwen2.5-Coder-7B-Instruct"

_EBNF_PATH = Path(__file__).parent / "grammars" / "triton.ebnf"

@lru_cache(maxsize=None)
def _ebnf() -> str:
    return _EBNF_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Singleton: model + compiled grammar, loaded once per process
# ---------------------------------------------------------------------------

class _ConstrainedLLM:
    """Holds the loaded model and the grammar compiled against its tokenizer.

    _ensure_loaded() is called on first generate(); subsequent calls are free.
    Grammar compilation (~1-3 s) is the expensive step — it maps every token
    in the vocabulary to grammar-allowed continuations and must be bound to
    this tokenizer's vocab. Do it once.
    """

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._compiled_grammar = None  # sentinel: None means not yet loaded

    def _ensure_loaded(self) -> None:
        if self._compiled_grammar is not None:
            return

        import torch
        import xgrammar as xgr
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._model.eval()

        # Grammar compilation: parse EBNF, then bind to tokenizer vocab.
        # xgr.Grammar.from_ebnf() is fast (ms).
        # GrammarCompiler.compile_grammar() is slow (~1-3 s); result is reused.
        grammar = xgr.Grammar.from_ebnf(_ebnf())

        config = AutoConfig.from_pretrained(_MODEL_ID)

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self._tokenizer,
            vocab_size=config.vocab_size,
            )
        
        self._compiled_grammar = (
            xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
        )

    def generate(self, messages: list[dict], max_new_tokens: int = 2048) -> str:
        import torch
        from xgrammar.contrib.hf import LogitsProcessor

        self._ensure_loaded()

        # LogitsProcessor is stateful per output sequence — always create fresh.
        logits_processor = LogitsProcessor(self._compiled_grammar)

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                logits_processor=[logits_processor],
                max_new_tokens=max_new_tokens,
                do_sample=False,               # greedy — deterministic under constraints
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)


_llm = _ConstrainedLLM()


# ---------------------------------------------------------------------------
# Public symbol
# ---------------------------------------------------------------------------

def _gen_constrained(messages: list[dict], max_new_tokens: int = 2048) -> str:
    """Run Qwen with XGrammar-constrained decoding.

    Args:
        messages: OpenAI-style chat messages list (same format as _gen()).
        max_new_tokens: Hard cap on generated tokens.

    Returns:
        Raw decoded text — grammar guarantees it conforms to triton.ebnf.
        No markdown fences; _extract_code() in engine.py handles it safely.
    """
    return _llm.generate(messages, max_new_tokens)