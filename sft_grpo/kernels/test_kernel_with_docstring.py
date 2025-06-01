import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Element-wise addition kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def launch_add(x: torch.Tensor, y: torch.Tensor):
    """
    Launch function for element-wise addition.
    
    Args:
        x: torch.Tensor of shape (N, C, H, W) - First input tensor
        y: torch.Tensor of shape (N, C, H, W) - Second input tensor
    
    Returns:
        torch.Tensor of shape (N, C, H, W) - Result of x + y
    """
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output 