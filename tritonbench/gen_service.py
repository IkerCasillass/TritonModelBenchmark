"""Grammar-constrained generation served by vLLM on a dedicated GPU.

Kept separate from the eval image: vLLM brings its own torch, and xgrammar ships
inside vLLM as a guided-decoding backend. The judge still runs on the eval GPU
(T4) — this container only produces text, so it carries no hardware-eval logic.
"""
from __future__ import annotations

from pathlib import Path

import modal

from .core import app

# Any HF slug; an ~9B fp16 (~18 GB) fits a 24 GB card (A10G / L4). If this slug
GEN_MODEL_ID = "Qwen/Qwen3.5-9B"
GEN_GPU = "A10G"

_GRAMMAR_PATH = Path(__file__).parent / "grammars" / "triton.ebnf"

# Persist the HF download across cold starts so the weights are fetched once.
_hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Latest vLLM: it bundles a self-consistent torch/transformers and supports recent
# model architectures. Pin to the resolved version once a build is confirmed green.
# VLLM_USE_FLASHINFER_SAMPLER=0: FlashInfer's sampler JIT-compiles a CUDA kernel at
# startup and needs nvcc, which this slim image lacks — use the native sampler.
gen_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)


@app.cls(
    gpu=GEN_GPU,
    image=gen_image,
    volumes={"/root/.cache/huggingface": _hf_cache},
    timeout=60 * 10,   # hard ceiling: no single generation should run longer
)
# One warm container batches the loop's concurrent generations.
@modal.concurrent(max_inputs=16)
class ConstrainedGenerator:
    @modal.enter()
    def _load(self) -> None:
        from vllm import LLM

        # max_model_len caps the context: the model defaults to 256K, whose KV
        # cache won't fit alongside the weights. Kernels + prompts need only a few K.
        self._llm = LLM(model=GEN_MODEL_ID, dtype="float16", gpu_memory_utilization=0.9,
                        enforce_eager=True, max_model_len=4096)
        self._grammar = _GRAMMAR_PATH.read_text(encoding="utf-8")
        self._struct_api = self._probe_structured_api(self._grammar)

    @staticmethod
    def _probe_structured_api(grammar: str) -> str | None:
        """Return the name of the grammar-constraint API this vLLM accepts.

        vLLM renamed `guided_decoding` -> `structured_outputs` across versions, so
        the right param can't be hard-coded. Probe once by constructing a throwaway
        SamplingParams (import success alone does not prove the kwarg is accepted).
        """
        from vllm import SamplingParams

        try:
            from vllm.sampling_params import StructuredOutputsParams
            SamplingParams(temperature=0.0, max_tokens=8,
                           structured_outputs=StructuredOutputsParams(grammar=grammar))
            print("[gen_service] structured-outputs API: structured_outputs", flush=True)
            return "structured_outputs"
        except Exception:
            pass
        try:
            from vllm.sampling_params import GuidedDecodingParams
            SamplingParams(temperature=0.0, max_tokens=8,
                           guided_decoding=GuidedDecodingParams(grammar=grammar, backend="xgrammar"))
            print("[gen_service] structured-outputs API: guided_decoding", flush=True)
            return "guided_decoding"
        except Exception:
            pass
        print("[gen_service] WARNING: no supported structured-outputs API found; "
              "constrained generation disabled (running unconstrained).", flush=True)
        return None

    def _structured_kwargs(self) -> dict:
        if self._struct_api == "structured_outputs":
            from vllm.sampling_params import StructuredOutputsParams
            return {"structured_outputs": StructuredOutputsParams(grammar=self._grammar)}
        if self._struct_api == "guided_decoding":
            from vllm.sampling_params import GuidedDecodingParams
            return {"guided_decoding": GuidedDecodingParams(grammar=self._grammar, backend="xgrammar")}
        return {}

    @modal.method()
    def generate(self, messages: list[dict], max_new_tokens: int = 2048,
                 constrained: bool = True) -> str:
        from vllm import SamplingParams

        tokenizer = self._llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # repetition_penalty breaks greedy degenerate loops (a CFG can't stop a
        # model repeating a valid statement).
        kwargs = {"temperature": 0.0, "max_tokens": max_new_tokens,
                  "repetition_penalty": 1.3}
        if constrained:
            kwargs.update(self._structured_kwargs())
        out = self._llm.generate([prompt], SamplingParams(**kwargs))
        return out[0].outputs[0].text
