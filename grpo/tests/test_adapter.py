"""
Test script for the KernelBook dataset adapter
"""

import json
import tempfile
import os
from kernelbook_adapter import KernelBookAdapter


def create_mock_kernelbook_data():
    """Create mock KernelBook dataset entries for testing."""
    return [
        {
            "repo_name": "test/add_example",
            "stars": 123,
            "licenses": ["MIT"],
            "sha": "abc123",
            "pytorch_code": """
import torch
import torch.nn as nn

def element_wise_add(x, y):
    return torch.add(x, y)
""",
            "triton_code": """
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute and store
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
"""
        },
        {
            "repo_name": "test/matmul_example", 
            "stars": 456,
            "licenses": ["Apache-2.0"],
            "sha": "def456",
            "pytorch_implementation": """
import torch
import torch.nn.functional as F

class MatMul(nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)
""",
            "triton_kernel": """
import triton
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
    
    # Matrix multiplication logic
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    
    # Load and compute
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # ... matrix multiplication implementation
"""
        },
        {
            "repo_name": "test/relu_example",
            "stars": 89,
            "licenses": ["BSD-3-Clause"],
            "sha": "ghi789",
            "torch_code": """
import torch

def relu_activation(x):
    return torch.nn.functional.relu(x)
""",
            "optimized_kernel": """
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    output_vals = tl.maximum(input_vals, 0.0)
    tl.store(output_ptr + offsets, output_vals, mask=mask)
"""
        },
        {
            # This entry should be skipped - no triton code
            "repo_name": "test/no_triton",
            "stars": 10,
            "pytorch_only": """
import torch
def simple_add(x, y):
    return x + y
"""
        },
        {
            # This entry should be skipped - no pytorch code  
            "repo_name": "test/no_pytorch",
            "stars": 5,
            "triton_only": """
import triton
@triton.jit
def some_kernel(): pass
"""
        }
    ]


def test_dataset_loading():
    """Test loading different dataset formats."""
    print("Testing dataset loading...")
    
    # Create test data
    test_data = create_mock_kernelbook_data()
    
    # Test JSON loading
    print("\n1. Testing JSON loading:")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, indent=2)
        json_file = f.name
    
    try:
        adapter = KernelBookAdapter(json_file, "output.json")
        loaded_data = adapter.load_dataset()
        print(f"   ✓ Loaded {len(loaded_data)} entries from JSON")
        print(f"   ✓ First entry repo: {loaded_data[0]['repo_name']}")
    finally:
        os.unlink(json_file)
    
    # Test unsupported format
    print("\n2. Testing unsupported format:")
    try:
        adapter = KernelBookAdapter("test.txt", "output.json")
        adapter.load_dataset()
        print("   ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly handled unsupported format: {e}")


def test_query_generation():
    """Test natural language query generation."""
    print("\n" + "="*50)
    print("Testing query generation...")
    
    adapter = KernelBookAdapter("dummy.json", "output.json")
    
    # Test cases
    test_cases = [
        {
            "pytorch": "def simple_add(x, y): return x + y",
            "triton": """
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr):
    # kernel implementation
    pass
""",
            "expected_function": "add_kernel"
        },
        {
            "pytorch": "def matmul(a, b): return torch.matmul(a, b)", 
            "triton": """
@triton.jit
def matrix_multiply_kernel(a_ptr, b_ptr, c_ptr):
    # matrix multiplication
    pass
""",
            "expected_function": "matrix_multiply_kernel"
        },
        {
            "pytorch": "def relu(x): return F.relu(x)",
            "triton": """
@triton.jit
def relu_activation_kernel(input_ptr, output_ptr):
    pass
""",
            "expected_function": "relu_activation_kernel"
        },
        {
            "pytorch": "def softmax(x): return F.softmax(x)",
            "triton": """
# No function definition
pass
""",
            "expected_function": None
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing query generation:")
        query = adapter.generate_query(case["pytorch"], case["triton"])
        print(f"   PyTorch: {case['pytorch']}")
        print(f"   Generated query: {query}")
        
        if case["expected_function"]:
            expected_name = case["expected_function"].replace("_", " ")
            if expected_name in query:
                print(f"   ✓ Query contains expected function name")
            else:
                print(f"   ⚠ Query might not contain expected function name")
        else:
            print(f"   ✓ Handled case with no function definition")


def test_dataset_transformation():
    """Test full dataset transformation."""
    print("\n" + "="*50)
    print("Testing dataset transformation...")
    
    # Create test dataset
    test_data = create_mock_kernelbook_data()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, indent=2)
        input_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_file = f.name
    
    try:
        # Test transformation
        adapter = KernelBookAdapter(input_file, output_file)
        transformed_data = adapter.transform_dataset()
        
        print(f"\n1. Transformation results:")
        print(f"   Original entries: {len(test_data)}")
        print(f"   Transformed entries: {len(transformed_data)}")
        print(f"   Expected: 3 (entries with both PyTorch and Triton code)")
        
        # Verify structure
        print(f"\n2. Checking transformed entry structure:")
        if transformed_data:
            entry = transformed_data[0]
            required_fields = ["query", "pytorch_code", "triton_kernel", "original_repo", "stars", "license"]
            for field in required_fields:
                if field in entry:
                    print(f"   ✓ {field}: present")
                else:
                    print(f"   ✗ {field}: missing")
            
            print(f"\n3. Sample transformed entry:")
            print(f"   Repo: {entry['original_repo']}")
            print(f"   Query: {entry['query']}")
            print(f"   Triton kernel preview: {entry['triton_kernel'][:100]}...")
            print(f"   Stars: {entry['stars']}")
    
    finally:
        os.unlink(input_file)
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_full_processing():
    """Test the complete processing pipeline."""
    print("\n" + "="*50)
    print("Testing full processing pipeline...")
    
    # Create test dataset
    test_data = create_mock_kernelbook_data()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, indent=2)
        input_file = f.name
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_file = f.name
    
    try:
        # Test processing
        print(f"\n1. Processing dataset...")
        adapter = KernelBookAdapter(input_file, output_file)
        result = adapter.process()
        
        print(f"   ✓ Processing completed")
        print(f"   ✓ Returned {len(result)} entries")
        
        # Verify output file was created
        print(f"\n2. Checking output file:")
        if os.path.exists(output_file + ".json"):
            with open(output_file + ".json", 'r') as f:
                saved_data = json.load(f)
            print(f"   ✓ Output file created with {len(saved_data)} entries")
            
            # Verify data integrity
            if len(saved_data) == len(result):
                print(f"   ✓ File data matches returned data")
            else:
                print(f"   ✗ File data doesn't match returned data")
        else:
            print(f"   ✗ Output file not created")
    
    finally:
        os.unlink(input_file)
        for ext in [".json", ""]:
            out_path = output_file + ext
            if os.path.exists(out_path):
                os.unlink(out_path)


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n" + "="*50)
    print("Testing edge cases...")
    
    # Test empty dataset
    print("\n1. Testing empty dataset:")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([], f)
        empty_file = f.name
    
    try:
        adapter = KernelBookAdapter(empty_file, "output.json")
        result = adapter.process()
        print(f"   ✓ Empty dataset processed: {len(result)} entries")
    finally:
        os.unlink(empty_file)
    
    # Test malformed entries
    print("\n2. Testing malformed entries:")
    malformed_data = [
        {"repo_name": "test1"},  # Missing required fields
        {"random_field": "value"},  # No pytorch or triton code
        {"pytorch_code": "import torch", "no_triton": True},  # Only pytorch
        {"triton_kernel": "@triton.jit", "no_pytorch": True},  # Only triton
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(malformed_data, f)
        malformed_file = f.name
    
    try:
        adapter = KernelBookAdapter(malformed_file, "output.json") 
        result = adapter.process()
        print(f"   ✓ Malformed dataset processed: {len(result)} entries (should be 0)")
    finally:
        os.unlink(malformed_file)


if __name__ == "__main__":
    print("Testing KernelBook Dataset Adapter")
    print("=" * 50)
    
    # Run all tests
    test_dataset_loading()
    test_query_generation()
    test_dataset_transformation()
    test_full_processing()
    test_edge_cases()
    
    print("\n" + "="*50)
    print("All adapter tests completed!")
