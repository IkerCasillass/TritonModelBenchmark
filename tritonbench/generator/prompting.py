"""Prompt construction for hardware-aware kernel generation."""
from ..core import T4_HARDWARE
from ..llm import PROMPT_HEADER

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
Avoid:
- Non power-of-two block sizes.
- Missing boundary masks.
- Out-of-bounds stores.
- Shared memory overflows.
- Generating explanations instead of code.
"""

# Expected output format for the generated response
_OUTPUT_RULES = """
Return a single Python code block only.
Do not include explanations.
Do not include test code.
"""


def build_system_prompt() -> str:
    """
    Builds the system prompt used during kernel generation.
    Combines the base Triton instructions with T4-specific constraints.
    """
    return "\n\n".join([
        PROMPT_HEADER,
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
        "Generate a valid Triton implementation."
    )

