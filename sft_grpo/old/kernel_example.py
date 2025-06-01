import torch
import triton
import triton.language as tl
print(torch.cuda.is_available())

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def launch(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def run():
    import inspect
    
    # Get the signature of the launch function
    sig = inspect.signature(launch)
    print("Function arguments for launch():")
    for param_name, param in sig.parameters.items():
        print(f"  {param_name}: {param.annotation}")
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Generate example arguments based on type annotations
    args = {}
    default_size = 1024  # Default tensor size
    
    for param_name, param in sig.parameters.items():
        annotation = param.annotation
        
        if annotation == torch.Tensor or str(annotation) == "<class 'torch.Tensor'>":
            # Generate a random tensor on the correct device
            tensor = torch.rand(default_size, device=DEVICE)
            args[param_name] = tensor
            print(f"Generated {param_name}: torch.Tensor of shape {tensor.shape} on {tensor.device}")
        
        elif hasattr(annotation, '__origin__') and annotation.__origin__ is torch.Tensor:
            # Handle more complex tensor type hints if any
            tensor = torch.rand(default_size, device=DEVICE)
            args[param_name] = tensor
            print(f"Generated {param_name}: torch.Tensor of shape {tensor.shape} on {tensor.device}")
        
        elif annotation == int:
            # Generate a reasonable integer value
            args[param_name] = default_size
            print(f"Generated {param_name}: {default_size}")
        
        elif annotation == float:
            # Generate a reasonable float value
            args[param_name] = 1.0
            print(f"Generated {param_name}: 1.0")
        
        else:
            # For unrecognized types, try to create a default tensor
            print(f"Warning: Unknown type annotation {annotation} for {param_name}, defaulting to tensor")
            tensor = torch.rand(default_size, device=DEVICE)
            args[param_name] = tensor
            print(f"Generated {param_name}: torch.Tensor of shape {tensor.shape} on {tensor.device}")
    
    print(f"\nCalling launch with generated arguments...")
    
    # Call the launch function with generated arguments
    output_triton = launch(**args)
    
    # Compute expected output using torch operations
    # For this specific example, we know it's addition, but this could be made more generic
    if len(args) >= 2:
        tensor_args = [v for v in args.values() if isinstance(v, torch.Tensor)]
        if len(tensor_args) >= 2:
            output_torch = tensor_args[0] + tensor_args[1]  # Assuming addition operation
            
            print(f"\nTorch result shape: {output_torch.shape}")
            print(f"Triton result shape: {output_triton.shape}")
            print(f'The maximum difference between torch and triton is '
                f'{torch.max(torch.abs(output_torch - output_triton))}')
        else:
            print(f"\nTriton result: {output_triton}")
    else:
        print(f"\nTriton result: {output_triton}")

if __name__ == "__main__":
    run()
