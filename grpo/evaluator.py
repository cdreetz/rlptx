"""
Triton Kernel Reward Evaluator for GRPO training.

Implements the 3-part reward system:
- 2.0 points: Kernel compiles successfully
- 1.0 points: Kernel produces correct output vs PyTorch baseline
- 0.5 points: Kernel includes requested optimizations
"""

import re
import torch
import triton
import triton.language as tl
import tempfile
import os
import sys
import importlib.util
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional


class RewardEvaluator(ABC):
    """Abstract base class for reward computation in RL training."""
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rewards for a batch of completions."""
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert raw reward scores tensor to a labeled dictionary."""
        pass


def get_evaluator(name: str) -> RewardEvaluator:
    """Get the appropriate reward evaluator for a given task."""
    if name.lower() == "triton_kernels":
        return TritonKernelEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")

class TritonKernelEvaluator(RewardEvaluator):
    """
    Reward evaluator for Triton kernel generation.
    
    Implements 3 reward functions:
    1. Compilation (2.0): Does the kernel compile and run?
    2. Correctness (1.0): Does output match PyTorch baseline?
    3. Optimizations (0.5): Does kernel include requested optimizations?
    """
    
    def __init__(self):
        self.num_reward_functions = 3

    def _extract_xml_answer(self, text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    
    def _extract_kernel_code(self, response: str) -> Optional[str]:
        """Extract kernel code from model response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # If no code blocks, return the whole response
        return response.strip()

    def _format_reward(self, completions: List[List[Dict[str, str]]]) -> List[float]:
        """Check if responses follow the required XML format."""
        pattern = r"^<thinking>\n.*?\n</thinking>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]['content'] for completion in completions]
        return [0.5 if bool(re.match(pattern, response, re.DOTALL)) else 0.0 for response in responses]
    
    def _compile_reward(self, completions: List[List[Dict[str, str]]], input_shapes: List[Tuple], output_shape: Tuple) -> List[float]:
        """Test if kernel codes compile and run using triton.do_bench."""
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(response) for response in responses]
        
        def _test_single_compilation(kernel_code: str) -> float:
            if not kernel_code:
                return 0.0
                
            try:
                # Quick check - must contain @triton.jit
                if '@triton.jit' not in kernel_code:
                    return 0.0
                
                # Write kernel code to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(kernel_code)
                    temp_file = f.name
                
                try:
                    # Load module from file
                    spec = importlib.util.spec_from_file_location("temp_kernel", temp_file)
                    module = importlib.util.module_from_spec(spec)
                    
                    # Add required imports to module namespace
                    import builtins
                    module.__builtins__ = builtins
                    module.triton = triton
                    module.tl = triton.language
                    module.torch = torch
                    sys.modules['temp_kernel'] = module
                    
                    # Execute the module
                    spec.loader.exec_module(module)
                    
                    # Find kernel function
                    kernel_func = None
                    for name in dir(module):
                        if not name.startswith('_'):
                            obj = getattr(module, name)
                            if callable(obj) and hasattr(obj, '__name__'):
                                if hasattr(obj, '__module__') and obj.__module__ == 'temp_kernel':
                                    kernel_func = obj
                                    break
                    
                    if kernel_func is None:
                        return 0.0
                    
                    # Create dummy tensors based on input/output shapes
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    inputs = [torch.randn(shape, device=device, dtype=torch.float32) for shape in input_shapes]
                    output = torch.empty(output_shape, device=device, dtype=torch.float32)
                    
                    # Determine grid size and block size based on shapes
                    if len(output_shape) == 1:  # 1D output (elementwise or reduction)
                        N = output_shape[0]
                        BLOCK_SIZE = 256
                        grid = (triton.cdiv(N, BLOCK_SIZE),)
                        args = (*inputs, output, N, BLOCK_SIZE)
                    elif len(output_shape) == 2:  # 2D output (matmul)
                        M, N = output_shape
                        if len(input_shapes) == 2 and len(input_shapes[0]) == 2:  # Likely matmul
                            K = input_shapes[0][1]
                            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
                            grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
                            # Matmul args: A, B, C, M, N, K, strides...
                            args = (*inputs, output, M, N, K, 
                                   inputs[0].stride(0), inputs[0].stride(1),
                                   inputs[1].stride(0), inputs[1].stride(1), 
                                   output.stride(0), output.stride(1),
                                   BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
                        else:  # 2D reduction
                            M, N = input_shapes[0]
                            BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 32
                            grid = (triton.cdiv(M, BLOCK_SIZE_M),)
                            args = (*inputs, output, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
                    else:
                        return 0.0
                    
                    # Use triton.do_bench to test compilation and execution
                    def benchmark_fn():
                        kernel_func[grid](*args)
                    
                    # Run benchmark - if this succeeds, kernel compiles and runs
                    _ = triton.do_bench(benchmark_fn, warmup=1, rep=1)
                    
                    return 2.0  # Compilation success worth 2.0 points
                    
                except Exception as e:
                    return 0.0
                    
                finally:
                    # Clean up
                    if 'temp_kernel' in sys.modules:
                        del sys.modules['temp_kernel']
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        
            except Exception as e:
                return 0.0
        
        return [_test_single_compilation(kernel_code) for kernel_code in kernel_codes]
    
    def _correctness_reward(self, completions: List[List[Dict[str, str]]], input_shapes: List[Tuple], output_shape: Tuple, operation: str) -> List[float]:
        """Test if kernels produce correct output compared to PyTorch baseline."""
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(response) for response in responses]
        
        # First get compilation results to extract kernel functions
        compilation_results = self._test_kernel_compilation_batch(kernel_codes, input_shapes, output_shape)
        kernel_funcs = [result[2] if result[0] else None for result in compilation_results]
        
        def _test_single_correctness(kernel_func: Optional[callable]) -> float:
            if kernel_func is None:
                return 0.0
                
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Generate test inputs
                inputs = [torch.randn(shape, device=device, dtype=torch.float32) for shape in input_shapes]
                kernel_output = torch.empty(output_shape, device=device, dtype=torch.float32)
                
                # Run kernel (reuse benchmark setup from compilation test)
                if len(output_shape) == 1:  # 1D output
                    N = output_shape[0]
                    BLOCK_SIZE = 256
                    grid = (triton.cdiv(N, BLOCK_SIZE),)
                    kernel_func[grid](*inputs, kernel_output, N, BLOCK_SIZE)
                elif len(output_shape) == 2:  # 2D output
                    M, N = output_shape
                    if len(input_shapes) == 2 and len(input_shapes[0]) == 2:  # Matmul
                        K = input_shapes[0][1]
                        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
                        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
                        kernel_func[grid](*inputs, kernel_output, M, N, K,
                                        inputs[0].stride(0), inputs[0].stride(1),
                                        inputs[1].stride(0), inputs[1].stride(1),
                                        kernel_output.stride(0), kernel_output.stride(1),
                                        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
                    else:  # 2D reduction
                        M, N = input_shapes[0]
                        BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 32
                        grid = (triton.cdiv(M, BLOCK_SIZE_M),)
                        kernel_func[grid](*inputs, kernel_output, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
                
                # Compute PyTorch reference
                if operation in ['add', 'subtract', 'multiply', 'divide']:
                    if operation == 'add':
                        reference = inputs[0] + inputs[1]
                    elif operation == 'subtract':
                        reference = inputs[0] - inputs[1]
                    elif operation == 'multiply':
                        reference = inputs[0] * inputs[1]
                    elif operation == 'divide':
                        reference = inputs[0] / inputs[1]
                elif operation in ['relu', 'gelu', 'sigmoid']:
                    if operation == 'relu':
                        reference = torch.relu(inputs[0])
                    elif operation == 'gelu':
                        reference = torch.nn.functional.gelu(inputs[0])
                    elif operation == 'sigmoid':
                        reference = torch.sigmoid(inputs[0])
                elif operation.endswith('_reduction'):
                    base_op = operation.replace('_reduction', '')
                    if base_op == 'sum':
                        reference = torch.sum(inputs[0], dim=-1)
                    elif base_op == 'max':
                        reference = torch.max(inputs[0], dim=-1)[0]
                    elif base_op == 'min':
                        reference = torch.min(inputs[0], dim=-1)[0]
                    elif base_op == 'mean':
                        reference = torch.mean(inputs[0], dim=-1)
                elif operation == 'matmul':
                    reference = torch.matmul(inputs[0], inputs[1])
                else:
                    return 0.0
                
                # Compare outputs with tolerance
                if torch.allclose(kernel_output, reference, rtol=1e-4, atol=1e-4):
                    return 1.0  # Correctness worth 1.0 points
                else:
                    return 0.0
                    
            except Exception as e:
                return 0.0
        
        return [_test_single_correctness(kernel_func) for kernel_func in kernel_funcs]
    
    def _optimization_reward(self, completions: List[List[Dict[str, str]]], optimization_level: str) -> List[float]:
        """Test if kernels include requested optimizations."""
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(response) for response in responses]
        
        def _test_single_optimization(kernel_code: str) -> float:
            if not kernel_code:
                return 0.0
                
            if optimization_level == 'none':
                return 0.5  # No optimizations requested, give full points
            
            optimization_keywords = {
                'basic': ['BLOCK_SIZE', 'mask', 'tl.load', 'tl.store'],
                'advanced': ['shared_memory', 'tile', 'vectorization', 'coalesced']
            }
            
            keywords = optimization_keywords.get(optimization_level, [])
            found_keywords = [kw for kw in keywords if kw in kernel_code]
            
            if optimization_level == 'basic':
                # For basic optimizations, require at least 3/4 keywords
                if len(found_keywords) >= 3:
                    return 0.5
                else:
                    return 0.0
            elif optimization_level == 'advanced':
                # For advanced optimizations, require at least 2/4 keywords
                if len(found_keywords) >= 2:
                    return 0.5
                else:
                    return 0.0
            else:
                return 0.0
        
        return [_test_single_optimization(kernel_code) for kernel_code in kernel_codes]
    
    def _test_kernel_compilation_batch(self, kernel_codes: List[str], input_shapes: List[Tuple], output_shape: Tuple) -> List[Tuple[bool, str, Optional[callable]]]:
        """Helper method for correctness testing - returns compilation results with kernel functions."""
        def _test_single_compilation(kernel_code: str) -> Tuple[bool, str, Optional[callable]]:
            if not kernel_code:
                return (False, "No kernel code provided", None)
                
            try:
                # Quick check - must contain @triton.jit
                if '@triton.jit' not in kernel_code:
                    return (False, "No @triton.jit decorator found in code", None)
                
                # Write kernel code to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(kernel_code)
                    temp_file = f.name
                
                try:
                    # Load module from file
                    spec = importlib.util.spec_from_file_location("temp_kernel", temp_file)
                    module = importlib.util.module_from_spec(spec)
                    
                    # Add required imports to module namespace
                    import builtins
                    module.__builtins__ = builtins
                    module.triton = triton
                    module.tl = triton.language
                    module.torch = torch
                    sys.modules['temp_kernel'] = module
                    
                    # Execute the module
                    spec.loader.exec_module(module)
                    
                    # Find kernel function
                    kernel_func = None
                    for name in dir(module):
                        if not name.startswith('_'):
                            obj = getattr(module, name)
                            if callable(obj) and hasattr(obj, '__name__'):
                                if hasattr(obj, '__module__') and obj.__module__ == 'temp_kernel':
                                    kernel_func = obj
                                    break
                    
                    if kernel_func is None:
                        return (False, "No function found in module", None)
                    
                    # Create dummy tensors based on input/output shapes
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    inputs = [torch.randn(shape, device=device, dtype=torch.float32) for shape in input_shapes]
                    output = torch.empty(output_shape, device=device, dtype=torch.float32)
                    
                    # Determine grid size and block size based on shapes
                    if len(output_shape) == 1:  # 1D output (elementwise or reduction)
                        N = output_shape[0]
                        BLOCK_SIZE = 256
                        grid = (triton.cdiv(N, BLOCK_SIZE),)
                        args = (*inputs, output, N, BLOCK_SIZE)
                    elif len(output_shape) == 2:  # 2D output (matmul)
                        M, N = output_shape
                        if len(input_shapes) == 2 and len(input_shapes[0]) == 2:  # Likely matmul
                            K = input_shapes[0][1]
                            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
                            grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
                            # Matmul args: A, B, C, M, N, K, strides...
                            args = (*inputs, output, M, N, K, 
                                   inputs[0].stride(0), inputs[0].stride(1),
                                   inputs[1].stride(0), inputs[1].stride(1), 
                                   output.stride(0), output.stride(1),
                                   BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
                        else:  # 2D reduction
                            M, N = input_shapes[0]
                            BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 32
                            grid = (triton.cdiv(M, BLOCK_SIZE_M),)
                            args = (*inputs, output, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
                    else:
                        return (False, "Unsupported tensor dimensionality", None)
                    
                    # Use triton.do_bench to test compilation and execution
                    def benchmark_fn():
                        kernel_func[grid](*args)
                    
                    # Run benchmark - if this succeeds, kernel compiles and runs
                    _ = triton.do_bench(benchmark_fn, warmup=1, rep=1)
                    
                    return (True, f"Kernel '{kernel_func.__name__}' successfully benchmarked", kernel_func)
                    
                except Exception as e:
                    return (False, f"Benchmark failed: {str(e)}", None)
                    
                finally:
                    # Clean up
                    if 'temp_kernel' in sys.modules:
                        del sys.modules['temp_kernel']
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        
            except Exception as e:
                return (False, f"Setup error: {str(e)}", None)
        
        return [_test_single_compilation(kernel_code) for kernel_code in kernel_codes]
    
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,  # answer should be dict with 'input_shapes', 'output_shape', 'operation', 'optimization_level'
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the given completions."""
        
        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Extract kernel specs from answer
        input_shapes = answer['input_shapes']
        output_shape = answer['output_shape']
        operation = answer['operation']
        optimization_level = answer['optimization_level']
        
        all_scores = [
            self._compile_reward(completions, input_shapes, output_shape),
            self._correctness_reward(completions, input_shapes, output_shape, operation),
            self._optimization_reward(completions, optimization_level),
        ]
        
        # Populate rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        
        # Calculate success rates
        compilation_scores = rewards_per_func[:, 0]
        correctness_scores = rewards_per_func[:, 1]
        optimization_scores = rewards_per_func[:, 2]
        
        num_compiled = (compilation_scores == 2.0).sum().item()
        num_correct = (correctness_scores == 1.0).sum().item()
        num_optimized = (optimization_scores == 0.5).sum().item()
        
        compilation_rate = num_compiled / num_completions
        correctness_rate = num_correct / num_completions
        optimization_rate = num_optimized / num_completions
        
        metrics = {
            "rewards/compilation_reward_func": reward_per_func[0].item(),
            "rewards/correctness_reward_func": reward_per_func[1].item(),
            "rewards/optimization_reward_func": reward_per_func[2].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "compilation_rate": compilation_rate,
            "correctness_rate": correctness_rate,
            "optimization_rate": optimization_rate
        }
        
        return rewards_per_func, metrics
    
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'compilation': reward_scores[0].item(),
            'correctness': reward_scores[1].item(),
            'optimization': reward_scores[2].item()
        }


def get_evaluator(name: str) -> RewardEvaluator:
    """Get the appropriate reward evaluator for a given task."""
    if name.lower() == "triton_kernels":
        return TritonKernelEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")
