from llm_local import _gen_constrained

messages = [
    {
        "role": "system",
        "content": """Generate exactly one Triton kernel. Output only raw Python code, no markdown fences.

@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tl.num_programs(0) * BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x, mask=mask)
"""
    }
]

# messages = [
#     {
#         "role": "system",
#         "content": "Generate exactly one empty Triton kernel."
#     }
# ]

result = _gen_constrained(messages)

print(result)