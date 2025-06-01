import triton
import triton.language as tl

@triton.jit
def maxpool_stride1_kernel(
    output_ptr, input_ptr, 
    N, C, H, W, 
    K,
    stride, 
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, 
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    n_off = tl.program_id(0) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_off = tl.program_id(1) * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    h_off = tl.program_id(2) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_off = tl.program_id(3) * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    n_mask = n_off < N
    c_mask = c_off < C
    h_mask = h_off < H
    w_mask = w_off < W

    mask = n_mask[:, None, None, None] & c_mask[None, :, None, None] & h_mask[None, None, :, None] & w_mask[None, None, None, :]

    n_off = n_off[:, None, None, None]
    c_off = c_off[None, :, None, None]
    h_off = h_off[None, None, :, None]
    w_off = w_off[None, None, None, :]

    max_vals = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), -float('inf'), dtype=tl.float32)

    for kh in range(K):
        for kw in range(K):
            h_idx = h_off + kh
            w_idx = w_off + kw
            h_valid = (h_idx >= 0) & (h_idx < H)
            w_valid = (w_idx >= 0) & (w_idx < W)
            valid = mask & h_valid & w_valid

            input_vals = tl.load(input_ptr + n_off*C*H*W + c_off*H*W + h_idx*W + w_idx, mask=valid, other=-float('inf'))
            max_vals = tl.maximum(max_vals, input_vals)

    tl.store(output_ptr + n_off*C*H*W + c_off*H*W + h_off*W + w_off, max_vals, mask=mask)

def maxpool_stride1_launch(x: torch.Tensor, kernel_size: int):
    N, C, H, W = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C = 4
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(C, BLOCK_SIZE_C), triton.cdiv(H, BLOCK_SIZE_H), triton.cdiv(W, BLOCK_SIZE_W))

    maxpool_stride1_kernel[grid](
        output, x, 
        N, C, H, W, 
        kernel_size,
        1, 
        BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W
    )

    return output