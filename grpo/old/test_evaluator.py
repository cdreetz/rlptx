"""
Test script for the Triton evaluator component
"""

import torch
from evaluator import TritonEvaluator, get_evaluator


def create_test_completions():
    """Create test completions with various quality levels."""
    return {
        "perfect_kernel": [{
            "content": """
import triton
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
"""
        }],
        
        "syntax_error": [{
            "content": """
import triton
import triton.language as tl

@triton.jit
def broken_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE
    mask = offsets < n_elements  # Missing closing parenthesis above
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)
"""
        }],
        
        "missing_imports": [{
            "content": """
@triton.jit
def no_imports_kernel(
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
"""
        }],
        
        "wrong_functions": [{
            "content": """
import triton
import triton.language as tl

@triton.jit
def wrong_functions_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Using wrong function names
    x = tl.load_data(x_ptr + offsets, mask=mask)  # Should be tl.load
    y = tl.load_data(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.save_data(output_ptr + offsets, output, mask=mask)  # Should be tl.store
"""
        }],
        
        "no_decorator": [{
            "content": """
import triton
import triton.language as tl

def no_decorator_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: int,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)
"""
        }],
        
        "in_code_block": [{
            "content": """
Here's a Triton kernel for element-wise addition:

```python
import triton
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
```

This kernel efficiently performs element-wise addition.
"""
        }],
        
        "natural_language_only": [{
            "content": """
To implement element-wise addition in Triton, you would need to:
1. Use the @triton.jit decorator
2. Get the program ID
3. Load the input tensors
4. Perform the addition
5. Store the result

However, I cannot provide the actual implementation.
"""
        }]
    }


def create_reference_kernels():
    """Create reference kernels for comparison."""
    return [
        """
import triton
import triton.language as tl

@triton.jit
def reference_add_kernel(
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
"""
    ]


def test_evaluator_creation():
    """Test creating evaluators with different configurations."""
    print("Testing evaluator creation...")
    
    # Test without Triton
    print("\n1. Testing evaluator without Triton:")
    evaluator_no_triton = TritonEvaluator(can_run_triton=False)
    print(f"   ✓ Created evaluator (Triton available: {evaluator_no_triton.triton_available})")
    print(f"   ✓ Number of reward functions: {evaluator_no_triton.num_reward_functions}")
    
    # Test with Triton (if available)
    print("\n2. Testing evaluator with Triton:")
    try:
        import triton
        evaluator_with_triton = TritonEvaluator(can_run_triton=True)
        print(f"   ✓ Created evaluator (Triton available: {evaluator_with_triton.triton_available})")
    except ImportError:
        print("   ℹ Triton not installed, using fallback mode")
        evaluator_with_triton = TritonEvaluator(can_run_triton=True)
        print(f"   ✓ Created evaluator (Triton available: {evaluator_with_triton.triton_available})")
    
    # Test factory function
    print("\n3. Testing evaluator factory function:")
    evaluator_factory = get_evaluator("triton")
    print(f"   ✓ Created evaluator via factory")
    print(f"   ✓ Type: {type(evaluator_factory).__name__}")


def test_code_extraction():
    """Test code extraction from different response formats."""
    print("\n" + "="*50)
    print("Testing code extraction...")
    
    evaluator = TritonEvaluator(can_run_triton=False)
    completions = create_test_completions()
    
    test_cases = [
        ("perfect_kernel", "Should extract clean kernel code"),
        ("in_code_block", "Should extract code from markdown blocks"),
        ("natural_language_only", "Should handle non-code responses"),
    ]
    
    for completion_type, description in test_cases:
        print(f"\n{description}:")
        completion = completions[completion_type]
        extracted = evaluator._extract_code(completion[0]['content'])
        print(f"   Original length: {len(completion[0]['content'])} chars")
        print(f"   Extracted length: {len(extracted)} chars")
        print(f"   Starts with import: {'import triton' in extracted}")
        print(f"   Has @triton.jit: {'@triton.jit' in extracted}")
        print(f"   Preview: {extracted[:50]}...")


def test_compilation_rewards():
    """Test compilation reward computation."""
    print("\n" + "="*50)
    print("Testing compilation rewards...")
    
    evaluator = TritonEvaluator(can_run_triton=False)  # Use fallback mode
    completions = create_test_completions()
    
    test_cases = [
        ("perfect_kernel", "Perfect kernel", 0.8, 1.0),
        ("syntax_error", "Syntax error", 0.0, 0.3),
        ("missing_imports", "Missing imports", 0.4, 0.8),
        ("wrong_functions", "Wrong function names", 0.5, 0.9),
        ("no_decorator", "No @triton.jit decorator", 0.4, 0.8),
        ("natural_language_only", "Natural language only", 0.0, 0.2),
    ]
    
    for completion_type, description, min_expected, max_expected in test_cases:
        completion = completions[completion_type]
        rewards = evaluator._compilation_reward([completion])
        reward = rewards[0]
        
        print(f"\n{description}:")
        print(f"   Reward: {reward:.3f}")
        print(f"   Expected range: [{min_expected:.1f}, {max_expected:.1f}]")
        
        if min_expected <= reward <= max_expected:
            print(f"   ✓ Reward within expected range")
        else:
            print(f"   ⚠ Reward outside expected range")


def test_structural_similarity():
    """Test structural similarity computation."""
    print("\n" + "="*50)
    print("Testing structural similarity...")
    
    evaluator = TritonEvaluator(can_run_triton=False)
    completions = create_test_completions()
    reference_kernels = create_reference_kernels()
    
    test_cases = [
        ("perfect_kernel", "Perfect match", 0.8, 1.0),
        ("wrong_functions", "Wrong function names", 0.2, 0.6),
        ("no_decorator", "Missing decorator", 0.3, 0.7),
        ("natural_language_only", "No code", 0.0, 0.2),
    ]
    
    for completion_type, description, min_expected, max_expected in test_cases:
        completion = completions[completion_type]
        similarities = evaluator._structural_similarity([completion], reference_kernels)
        similarity = similarities[0]
        
        print(f"\n{description}:")
        print(f"   Similarity: {similarity:.3f}")
        print(f"   Expected range: [{min_expected:.1f}, {max_expected:.1f}]")
        
        if min_expected <= similarity <= max_expected:
            print(f"   ✓ Similarity within expected range")
        else:
            print(f"   ⚠ Similarity outside expected range")


def test_full_reward_computation():
    """Test complete reward computation."""
    print("\n" + "="*50)
    print("Testing full reward computation...")
    
    evaluator = TritonEvaluator(can_run_triton=False)
    completions = create_test_completions()
    reference_kernels = create_reference_kernels()
    
    # Test with multiple completions
    test_completions = [
        completions["perfect_kernel"][0],
        completions["syntax_error"][0],
        completions["missing_imports"][0],
    ]
    
    # Mock prompts (not used by this evaluator)
    mock_prompts = [[{"content": "Write a kernel"}]] * len(test_completions)
    mock_completions = [[comp] for comp in test_completions]
    answers = reference_kernels * len(test_completions)
    
    print(f"\n1. Computing rewards for {len(test_completions)} completions:")
    rewards_per_func, metrics = evaluator.compute_rewards(
        prompts=mock_prompts,
        completions=mock_completions,
        answer=answers,
        device="cpu"
    )
    
    print(f"   ✓ Rewards tensor shape: {rewards_per_func.shape}")
    print(f"   ✓ Expected shape: ({len(test_completions)}, {evaluator.num_reward_functions})")
    
    print(f"\n2. Metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    print(f"\n3. Individual completion rewards:")
    completion_names = ["Perfect kernel", "Syntax error", "Missing imports"]
    for i, name in enumerate(completion_names):
        total_reward = rewards_per_func[i].sum().item()
        breakdown = evaluator.get_reward_breakdown(rewards_per_func[i])
        print(f"   {name}:")
        print(f"     Total: {total_reward:.3f}")
        for reward_name, value in breakdown.items():
            print(f"     {reward_name}: {value:.3f}")


def test_reward_breakdown():
    """Test reward breakdown functionality."""
    print("\n" + "="*50)
    print("Testing reward breakdown...")
    
    evaluator = TritonEvaluator(can_run_triton=False)
    
    # Create test reward scores
    test_scores = torch.tensor([0.8, 0.6, 0.2])  # compilation, correctness, performance
    
    breakdown = evaluator.get_reward_breakdown(test_scores)
    
    print(f"\n1. Test scores: {test_scores.tolist()}")
    print(f"2. Breakdown:")
    expected_keys = ['compilation', 'correctness', 'performance']
    for key in expected_keys:
        if key in breakdown:
            print(f"   ✓ {key}: {breakdown[key]:.3f}")
        else:
            print(f"   ✗ Missing key: {key}")
    
    # Verify values match
    if (abs(breakdown['compilation'] - 0.8) < 1e-6 and
        abs(breakdown['correctness'] - 0.6) < 1e-6 and  
        abs(breakdown['performance'] - 0.2) < 1e-6):
        print(f"   ✓ All values match expected")
    else:
        print(f"   ✗ Values don't match expected")


def test_error_handling():
    """Test error handling in the evaluator."""
    print("\n" + "="*50)
    print("Testing error handling...")
    
    evaluator = TritonEvaluator(can_run_triton=False)
    
    # Test with empty completions
    print("\n1. Testing empty completions:")
    try:
        rewards = evaluator._compilation_reward([])
        print(f"   ✓ Handled empty completions: {len(rewards)} rewards")
    except Exception as e:
        print(f"   ✗ Error with empty completions: {e}")
    
    # Test with malformed completion
    print("\n2. Testing malformed completion:")
    malformed_completion = [{"wrong_key": "content"}]
    try:
        rewards = evaluator._compilation_reward([malformed_completion])
        print(f"   ? Handled malformed completion: {rewards}")
    except Exception as e:
        print(f"   ✓ Correctly caught error: {type(e).__name__}")
    
    # Test with unsupported evaluator name
    print("\n3. Testing unsupported evaluator:")
    try:
        bad_evaluator = get_evaluator("nonexistent")
        print(f"   ✗ Should have raised error")
    except NotImplementedError as e:
        print(f"   ✓ Correctly raised NotImplementedError: {e}")


def test_device_handling():
    """Test device handling for tensor operations."""
    print("\n" + "="*50)
    print("Testing device handling...")
    
    evaluator = TritonEvaluator(can_run_triton=False)
    completions = create_test_completions()
    reference_kernels = create_reference_kernels()
    
    # Test CPU device
    print("\n1. Testing CPU device:")
    mock_prompts = [[{"content": "test"}]]
    mock_completions = [completions["perfect_kernel"]]
    
    rewards_per_func, metrics = evaluator.compute_rewards(
        prompts=mock_prompts,
        completions=mock_completions,
        answer=reference_kernels,
        device="cpu"
    )
    
    print(f"   ✓ CPU computation successful")
    print(f"   ✓ Tensor device: {rewards_per_func.device}")
    print(f"   ✓ Tensor shape: {rewards_per_func.shape}")
    
    # Test CUDA device (if available)
    if torch.cuda.is_available():
        print("\n2. Testing CUDA device:")
        rewards_per_func_cuda, metrics_cuda = evaluator.compute_rewards(
            prompts=mock_prompts,
            completions=mock_completions,
            answer=reference_kernels,
            device="cuda"
        )
        
        print(f"   ✓ CUDA computation successful")
        print(f"   ✓ Tensor device: {rewards_per_func_cuda.device}")
        print(f"   ✓ Values match CPU: {torch.allclose(rewards_per_func, rewards_per_func_cuda.cpu())}")
    else:
        print("\n2. CUDA not available, skipping CUDA test")


if __name__ == "__main__":
    print("Testing Triton Evaluator Component")
    print("=" * 50)
    
    # Run all tests
    test_evaluator_creation()
    test_code_extraction() 
    test_compilation_rewards()
    test_structural_similarity()
    test_full_reward_computation()
    test_reward_breakdown()
    test_error_handling()
    test_device_handling()
    
    print("\n" + "="*50)
    print("All evaluator tests completed!")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("The TritonEvaluator has been tested with:")
    print("✓ Different code quality levels")
    print("✓ Various response formats (code blocks, plain text)")
    print("✓ Error handling and edge cases")
    print("✓ Device compatibility (CPU/CUDA)")
    print("✓ Reward computation and breakdown")
    print("\nThe evaluator focuses on practical metrics:")
    print("• Compilation success (most important)")
    print("• Structural similarity to reference")
    print("• Performance (placeholder for future implementation)")
