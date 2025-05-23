"""
Test script for the dataset loader component
"""

import json
import tempfile
from ..rldatasets import build_kernelbook_dataloaders, KernelBookLoader


def create_test_dataset():
    """Create a small test dataset with a few examples."""
    test_data = [
        {
            "query": "Write a Triton kernel that performs element-wise addition",
            "triton_kernel": """import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)
""",
            "pytorch_code": "torch.add(x, y)",
            "original_repo": "test/repo",
            "stars": 100,
            "license": ["MIT"]
        },
        {
            "query": "Write a Triton kernel that implements softmax",
            "triton_kernel": """import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    input_row = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute softmax
    row_max = tl.max(input_row, axis=0)
    numerator = tl.exp(input_row - row_max)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    
    tl.store(output_ptr + offsets, output, mask=mask)
""",
            "pytorch_code": "torch.nn.functional.softmax(x, dim=-1)",
            "original_repo": "test/repo2", 
            "stars": 250,
            "license": ["Apache-2.0"]
        },
        {
            "query": "Write a Triton kernel that performs matrix multiplication",
            "triton_kernel": """import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
""",
            "pytorch_code": "torch.matmul(a, b)",
            "original_repo": "test/repo3",
            "stars": 500,
            "license": ["MIT"]
        }
    ]
    return test_data


def test_kernelbook_loader():
    """Test the KernelBookLoader class directly."""
    print("Testing KernelBookLoader class...")
    
    # Create test data
    queries = [
        "Write a Triton kernel that adds two tensors",
        "Write a Triton kernel that computes softmax", 
        "Write a Triton kernel for matrix multiplication"
    ]
    
    kernels = [
        "import triton\n@triton.jit\ndef add_kernel(): pass",
        "import triton\n@triton.jit\ndef softmax_kernel(): pass",
        "import triton\n@triton.jit\ndef matmul_kernel(): pass"
    ]
    
    # Test sequential access
    print("\n1. Testing sequential access:")
    loader = KernelBookLoader(queries, kernels, random=False)
    print(f"Dataset size: {len(loader)}")
    print(f"System prompt length: {len(loader.system_prompt)} characters")
    
    for i, (query, kernel) in enumerate(loader):
        print(f"  Example {i+1}:")
        print(f"    Query: {query}")
        print(f"    Kernel: {kernel[:50]}...")
        if i >= 2:  # Only show first 3
            break
    
    # Test random access
    print("\n2. Testing random access:")
    loader_random = KernelBookLoader(queries, kernels, random=True)
    for i in range(3):
        query, kernel = next(loader_random)
        print(f"  Random example {i+1}: {query[:30]}...")
    
    # Test reset functionality
    print("\n3. Testing reset functionality:")
    loader.reset()
    query, kernel = next(loader)
    print(f"  After reset, first query: {query[:30]}...")


def test_build_dataloaders():
    """Test the build_kernelbook_dataloaders function."""
    print("\n" + "="*50)
    print("Testing build_kernelbook_dataloaders function...")
    
    # Create test dataset file
    test_data = create_test_dataset()
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, indent=2)
        temp_file = f.name
    
    try:
        # Test loading
        print(f"\n1. Loading dataset from {temp_file}")
        train_loader, test_loader = build_kernelbook_dataloaders(temp_file, test_split=0.3)
        
        print(f"   Train loader size: {len(train_loader)}")
        print(f"   Test loader size: {len(test_loader)}")
        print(f"   Total examples: {len(train_loader) + len(test_loader)}")
        
        # Test train loader
        print("\n2. Testing train loader (should be random):")
        train_loader.reset()
        for i in range(min(2, len(train_loader))):
            query, kernel = next(train_loader)
            print(f"   Train example {i+1}: {query}")
        
        # Test test loader  
        print("\n3. Testing test loader (should be sequential):")
        test_loader.reset()
        for i in range(min(2, len(test_loader))):
            query, kernel = next(test_loader)
            print(f"   Test example {i+1}: {query}")
            
        # Verify system prompt
        print(f"\n4. System prompt preview:")
        print(f"   {train_loader.system_prompt[:100]}...")
        
    finally:
        # Clean up
        import os
        os.unlink(temp_file)


def test_error_handling():
    """Test error handling in the loader."""
    print("\n" + "="*50)
    print("Testing error handling...")
    
    # Test with non-existent file
    try:
        build_kernelbook_dataloaders("non_existent_file.json")
        print("ERROR: Should have raised FileNotFoundError")
    except FileNotFoundError:
        print("✓ Correctly handled missing file")
    
    # Test with unsupported file format
    try:
        build_kernelbook_dataloaders("test.txt")
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly handled unsupported format: {e}")
    
    # Test empty loader
    empty_loader = KernelBookLoader([], [])
    print(f"✓ Empty loader size: {len(empty_loader)}")
    
    try:
        next(empty_loader)
        print("ERROR: Should have raised StopIteration")
    except StopIteration:
        print("✓ Correctly handled empty iteration")


if __name__ == "__main__":
    print("Testing Dataset Loader Components")
    print("=" * 50)
    
    # Run all tests
    test_kernelbook_loader()
    test_build_dataloaders()
    test_error_handling()
    
    print("\n" + "="*50)
    print("All dataset loader tests completed!")
