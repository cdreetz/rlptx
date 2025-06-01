#!/usr/bin/env python3
"""
Example usage of TritonSandbox for testing AI-generated Triton kernels.

This script demonstrates different ways to use the TritonSandbox:
1. Testing from a file
2. Testing from code string
3. Using as a context manager
4. Batch testing multiple kernels
"""

from triton_sandbox import TritonSandbox

def example_1_test_from_file():
    """Example 1: Test a kernel from a file"""
    print("=" * 60)
    print("Example 1: Testing kernel from file")
    print("=" * 60)
    
    with TritonSandbox() as sandbox:
        result = sandbox.test_kernel_from_file("kernel_example.py")
        print(f"Test result: {'PASSED' if result['success'] else 'FAILED'}")

def example_2_test_from_string():
    """Example 2: Test a kernel from a code string"""
    print("=" * 60)
    print("Example 2: Testing kernel from code string")
    print("=" * 60)
    
    # Example AI-generated kernel code
    kernel_code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Simple element-wise addition kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def launch_add(x: torch.Tensor, y: torch.Tensor):
    """Launch function for the add kernel"""
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

def run():
    """Test function"""
    torch.manual_seed(42)
    size = 1024
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    
    # Test our kernel
    result_triton = launch_add(x, y)
    
    # Compare with PyTorch
    result_torch = x + y
    
    print(f"Max difference: {torch.max(torch.abs(result_triton - result_torch))}")
    print("Test completed successfully!")
'''
    
    with TritonSandbox() as sandbox:
        result = sandbox.test_kernel(kernel_code)
        print(f"Test result: {'PASSED' if result['success'] else 'FAILED'}")

def example_3_manual_sandbox_management():
    """Example 3: Manual sandbox management"""
    print("=" * 60)
    print("Example 3: Manual sandbox management")
    print("=" * 60)
    
    sandbox = TritonSandbox(gpu_type="A100")
    try:
        sandbox.start()
        
        # Test multiple kernels with the same sandbox
        kernel_code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def multiply_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)

def launch_multiply(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    multiply_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=512)
    return output

def run():
    torch.manual_seed(123)
    x = torch.rand(2048, device="cuda")
    y = torch.rand(2048, device="cuda")
    result = launch_multiply(x, y)
    expected = x * y
    print(f"Multiplication test - Max diff: {torch.max(torch.abs(result - expected))}")
'''
        
        result = sandbox.test_kernel(kernel_code, verbose=False)
        print(f"Multiply kernel test: {'PASSED' if result['success'] else 'FAILED'}")
        
    finally:
        sandbox.stop()

def example_4_batch_testing():
    """Example 4: Batch testing multiple kernels"""
    print("=" * 60)
    print("Example 4: Batch testing multiple kernels")
    print("=" * 60)
    
    # Different kernel implementations to test
    kernels = {
        "vector_add": '''
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def launch(a, b):
    c = torch.empty_like(a)
    n_elements = c.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=1024)
    return c

def run():
    a = torch.rand(1000, device="cuda")
    b = torch.rand(1000, device="cuda")
    result = launch(a, b)
    expected = a + b
    print(f"Vector add test passed: {torch.allclose(result, expected)}")
''',
        
        "scalar_multiply": '''
import torch
import triton
import triton.language as tl

@triton.jit
def scalar_mul_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * scalar
    tl.store(output_ptr + offsets, output, mask=mask)

def launch(x, scalar):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    scalar_mul_kernel[grid](x, scalar, output, n_elements, BLOCK_SIZE=512)
    return output

def run():
    x = torch.rand(500, device="cuda")
    scalar = 3.14
    result = launch(x, scalar)
    expected = x * scalar
    print(f"Scalar multiply test passed: {torch.allclose(result, expected)}")
'''
    }
    
    results = {}
    with TritonSandbox() as sandbox:
        for kernel_name, kernel_code in kernels.items():
            print(f"\nTesting {kernel_name}...")
            try:
                result = sandbox.test_kernel(kernel_code, verbose=False)
                results[kernel_name] = result['success']
                print(f"{kernel_name}: {'PASSED' if result['success'] else 'FAILED'}")
            except Exception as e:
                print(f"{kernel_name}: ERROR - {e}")
                results[kernel_name] = False
    
    print(f"\nBatch test summary:")
    for kernel_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {kernel_name}")

def main():
    """Run all examples"""
    print("TritonSandbox Usage Examples")
    print("=" * 60)
    
    try:
        example_1_test_from_file()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_test_from_string()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_manual_sandbox_management()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_4_batch_testing()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")

if __name__ == "__main__":
    main() 