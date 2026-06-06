"""Prompt construction for hardware-aware kernel generation.

These prompts drive the GRAMMAR-CONSTRAINED path (engine.generate_kernel +
grammars/triton.ebnf). They instruct the model to emit exactly what the grammar
permits — a @triton.jit kernel plus a plain host wrapper — and nothing the
grammar/pipeline supplies for free (imports are injected in engine._IMPORTS).
"""
from ..core import T4_HARDWARE

# Role/intro for the constrained path. Unlike llm.PROMPT_HEADER (the cloud path,
# which asks for a full self-contained, fenced module), this header asks only for
# kernel + wrapper logic; imports and the code-fence are handled elsewhere.
_CONSTRAINED_HEADER = (
    "You are an expert in Triton (OpenAI Triton) GPU programming. Given a "
    "functional description and a target signature, write a correct Triton kernel "
    "and a host wrapper function that launches it. The wrapper must fully match "
    "the described function signature."
)

# Hardware information used to build T4-specific prompts
_GPU_NAME = T4_HARDWARE["name"]
_CC = T4_HARDWARE["compute_capability"]
_ARCH = T4_HARDWARE["arch"]
_SMEM_KB = T4_HARDWARE["shared_mem_kb_per_block"]

# Build a list of unsupported datatypes based on hardware capabilities
_BANNED_DTYPES = [
    dtype
    for dtype, supported in [
        ("bf16", T4_HARDWARE["supports_bf16"]),
        ("fp8", T4_HARDWARE["supports_fp8"]),
        ("tf32", T4_HARDWARE["supports_tf32"]),
    ]
    if not supported
]

# Main hardware constraints that should guide kernel generation
_HARDWARE_RULES = f"""
Target GPU: {_GPU_NAME} (sm_{_CC.replace(".", "")}, {_ARCH})

Rules:
- Do not use {", ".join(_BANNED_DTYPES)}.
- Shared memory must stay below {_SMEM_KB} KB per block.
- BLOCK_SIZE values must be powers of 2.
- Use boundary masks on tl.load and tl.store.
- Prefer fp16 or fp32 when possible.
- Tensor Core dimensions should be multiples of 16.
"""

# Common mistakes observed in Triton kernels
_COMMON_MISTAKES = """
Write a complete kernel body: load the inputs (with masks), compute, and
tl.store the result. Each statement must make progress.

Avoid:
- Non power-of-two block sizes.
- Missing boundary masks.
- Out-of-bounds stores.
- Shared memory overflows.
- Generating explanations instead of code.
- Repeating the same statement or emitting no-op / placeholder lines.
- Leaving the kernel body empty.
"""

# Expected output format — must match exactly what grammars/triton.ebnf emits.
_OUTPUT_RULES = """
Output exactly two top-level definitions, in this order, separated by one blank line:
  1. The @triton.jit kernel function.
  2. A plain (undecorated) host wrapper function that allocates the output
     tensor(s) and launches the kernel with a grid, e.g. kernel_name[grid](...).

Compute the launch grid as a tuple and launch the kernel, e.g.
  grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
  kernel_name[grid](out, inp, n_elements, BLOCK_SIZE=BLOCK_SIZE)

Strict format rules:
- Do NOT write any import statements — torch, triton, and triton.language as tl
  are already imported.
- Do NOT use markdown fences, backticks, comments, or explanations.
- Do NOT include test code.
- The wrapper's name and parameters MUST match the signature in the description,
  and it must return the result the description specifies.
"""


def build_system_prompt() -> str:
    """Build the system prompt for the grammar-constrained path.

    Combines the constrained-path role header, the T4 hardware rules and
    common-mistake guidance (which steer the kernel body and stay valid under the
    grammar), and the kernel+wrapper output rules that match grammars/triton.ebnf.
    """
    return "\n\n".join([
        _CONSTRAINED_HEADER,
        _HARDWARE_RULES,
        _COMMON_MISTAKES,
        _OUTPUT_RULES,
    ])


def build_user_prompt(operator_id: str, instruction: str) -> str:
    """
    Builds the user prompt for a specific TritonBench operator.
    """
    return (
        f"Operator: {operator_id}\n\n"
        f"{instruction}\n\n"
        "Generate the Triton kernel and a host wrapper that matches the "
        "described function signature."
    )

