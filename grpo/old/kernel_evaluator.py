"""
Reward evaluator for CUDA/Triton kernel optimization (simplified v1).

This evaluator implements reward functions that assess the quality, performance,
and correctness of generated kernel code without enforcing specific reasoning patterns.
Focused on pass@1 functionality and performance metrics.
"""

import re
import torch
import subprocess
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

class RewardEvaluator(ABC):
    """
    Abstract base class for reward computation in RL training.
    
    This class defines the interface for reward evaluators that can be used
    to score model completions during RL training. Implement this class to
    create custom reward functions for different tasks.
    """
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rewards for a batch of completions.
        
        Args:
            prompts: List of prompt messages in chat format
                    [{"role": "user", "content": "..."}, ...]
            completions: List of completion messages in chat format
                        [{"role": "assistant", "content": "..."}, ...]
            answer: Ground truth answer(s) for the prompts (could be reference kernel)
            device: Device to place tensors on ("cpu" or "cuda")
            
        Returns:
            rewards_per_func: Tensor of shape (num_completions, num_reward_functions)
                            containing individual reward function scores
            metrics: Dictionary of aggregated metrics including mean rewards
                    per function and total reward
        """
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """
        Convert raw reward scores tensor to a labeled dictionary.
        
        Args:
            reward_scores: Tensor of raw scores from compute_rewards
            
        Returns:
            Dictionary mapping reward function names to their scores
        """
        pass


def get_evaluator(name: str) -> RewardEvaluator:
    """
    Get the appropriate reward evaluator for a given task.
    
    Args:
        name: Name of the task/dataset to get evaluator for
        
    Returns:
        RewardEvaluator instance for the specified task
        
    Raises:
        NotImplementedError: If evaluator for given task is not implemented
    """
    if name.lower() == "cuda_matmul":
        return CUDAMatmulEvaluator()
    elif name.lower() == "triton_matmul":
        return TritonMatmulEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")


class KernelEvaluator(RewardEvaluator):
    """
    Base class for kernel evaluators with common functionality.
    """
    
    def __init__(self):
        self.num_reward_functions = 5
        
    def _extract_kernel_code(self, text: str) -> str:
        """Extract kernel code from the completion."""
        # Extract code between ```cuda or ```python tags
        pattern = r'```(?:cuda|python|triton)(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
            
        # If no matches with language specifier, try without
        pattern = r'```(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
            
        return text  # Return the whole text if no code blocks found
    
    def _compilation_reward(self, completions: List[Dict[str, str]]) -> List[float]:
        """Reward for successful compilation."""
        raise NotImplementedError("Must be implemented by subclass")

    def _correctness_reward(self, completions: List[Dict[str, str]], reference_outputs: Any) -> List[float]:
        """Reward for producing correct outputs compared to reference."""
        raise NotImplementedError("Must be implemented by subclass")

    def _performance_reward(self, completions: List[Dict[str, str]]) -> List[float]:
        """Reward for performance (execution time)."""
        raise NotImplementedError("Must be implemented by subclass")
        
    def _memory_efficiency_reward(self, completions: List[Dict[str, str]]) -> List[float]:
        """Reward for memory usage efficiency."""
        raise NotImplementedError("Must be implemented by subclass")
        
    def _code_quality_reward(self, completions: List[Dict[str, str]]) -> List[float]:
        """Reward for code quality metrics."""
        raise NotImplementedError("Must be implemented by subclass")


class CUDAMatmulEvaluator(KernelEvaluator):
    """
    Reward evaluator for CUDA matrix multiplication kernels (simplified v1).
    
    Implements reward functions specifically for CUDA matmul kernels:
    - Compilation success (2.0 points)
    - Correctness (3.0 points for matching reference output)
    - Performance (4.0 points for execution time, scaled)
    - Memory efficiency (1.0 point)
    - Code quality metrics (1.0 point)
    """
    
    def __init__(self):
        super().__init__()
        self.benchmark_sizes = [(128, 128), (1024, 1024), (4096, 4096)]
        
    def _compilation_reward(self, completions: List[List[Dict[str, str]]]) -> List[float]:
        """Check if CUDA kernel compiles successfully (2.0 points)."""
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(r) for r in responses]
        
        results = []
        for code in kernel_codes:
            # Write code to temporary file
            with open("temp_kernel.cu", "w") as f:
                f.write(code)
                
            # Try to compile with nvcc
            try:
                result = subprocess.run(
                    ["nvcc", "-c", "temp_kernel.cu"], 
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                # Give 2.0 reward if compilation succeeds
                results.append(2.0 if result.returncode == 0 else 0.0)
            except Exception:
                results.append(0.0)
                
        return results
    
    def _correctness_reward(self, completions: List[List[Dict[str, str]]], reference_kernel: str) -> List[float]:
        """
        Check if kernel produces correct matrix multiplication results (3.0 points).
        Compares against torch.matmul or a reference kernel.
        """
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(r) for r in responses]
        
        results = []
        for code in kernel_codes:
            # Skip if compilation failed
            if not self._does_compile(code):
                results.append(0.0)
                continue
                
            # Test with small matrices first for correctness
            try:
                # Create test harness that calls the kernel with test data
                # This would compile and run the kernel and compare results
                # with a reference implementation (e.g., torch.matmul)
                
                # For v1, simplify by checking if output matches for test cases
                correct, accuracy = self._test_kernel_correctness(code)
                
                # Scale reward based on accuracy (0.0-3.0)
                if correct:
                    results.append(3.0)  # Perfectly correct
                else:
                    # Scale based on accuracy (e.g., numerical precision differences)
                    # accuracy is between 0 and 1
                    results.append(accuracy * 3.0)
            except Exception:
                results.append(0.0)
                
        return results
    
    def _performance_reward(self, completions: List[List[Dict[str, str]]]) -> List[float]:
        """
        Measure execution time performance compared to baseline (4.0 points max).
        Returns scaled score where:
        - 1.0 = same as baseline
        - 2.0 = 1.5x faster than baseline
        - 3.0 = 2x faster than baseline
        - 4.0 = 3x or more faster than baseline
        """
        # Implementation would benchmark the kernel across different sizes
        # and compare to a baseline implementation (cuBLAS or reference kernel)
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(r) for r in responses]
        
        results = []
        baseline_times = self._get_baseline_times()
        
        for code in kernel_codes:
            # Skip if compilation or correctness failed
            if not self._does_compile(code):
                results.append(0.0)
                continue
                
            # Run benchmarks
            try:
                times = self._benchmark_kernel(code)
                # Calculate speedup relative to baseline
                speedups = [baseline / kernel for baseline, kernel in zip(baseline_times, times)]
                relative_perf = np.mean(speedups)
                
                # Scale reward based on performance
                if relative_perf <= 1.0:
                    # Linear scaling from 0.0 to 1.0 for worse than baseline
                    reward = max(0.0, relative_perf)
                elif relative_perf <= 1.5:
                    # 1.0-2.0 points for up to 1.5x baseline
                    reward = 1.0 + 2.0 * (relative_perf - 1.0) / 0.5
                elif relative_perf <= 2.0:
                    # 2.0-3.0 points for up to 2x baseline
                    reward = 2.0 + (relative_perf - 1.5) / 0.5
                else:
                    # 3.0-4.0 points for 2x+ baseline, max at 3x
                    reward = 3.0 + min(1.0, (relative_perf - 2.0))
                
                results.append(reward)
            except Exception:
                results.append(0.0)
                
        return results
                
    def _memory_efficiency_reward(self, completions: List[List[Dict[str, str]]]) -> List[float]:
        """Reward for memory usage efficiency (1.0 point max)."""
        # This would measure memory bandwidth utilization using nvprof or similar
        # Higher utilization = better score
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(r) for r in responses]
        
        # Placeholder implementation
        results = []
        for code in kernel_codes:
            # Skip if compilation failed
            if not self._does_compile(code):
                results.append(0.0)
                continue
                
            try:
                memory_utilization = self._measure_memory_utilization(code)
                # Scale to 0.0-1.0 range
                results.append(min(1.0, memory_utilization))
            except Exception:
                results.append(0.0)
                
        return results
        
    def _code_quality_reward(self, completions: List[List[Dict[str, str]]]) -> List[float]:
        """Reward based on code quality metrics (1.0 point max)."""
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(r) for r in responses]
        
        results = []
        for code in kernel_codes:
            # Analyze code for quality metrics - each worth 0.2 points:
            # - Use of shared memory
            # - Thread coarsening
            # - Loop unrolling
            # - Bank conflict avoidance
            # - Proper synchronization
            
            score = 0.0
            if "shared" in code or "__shared__" in code:
                score += 0.2  # Use of shared memory
            if "pragma unroll" in code or "#pragma unroll" in code:
                score += 0.2  # Loop unrolling
            if "syncthreads" in code or "__syncthreads" in code:
                score += 0.2  # Proper synchronization
            if "coalesced" in code or any(r in code for r in ["threadIdx.x + blockDim.x * i", "threadIdx.x + i * blockDim.x"]):
                score += 0.2  # Coalesced memory access patterns
            if "blockDim.x * blockDim.y" in code or "gridDim" in code:
                score += 0.2  # Proper use of thread/block dimensions
                
            results.append(score)
                
        return results
    
    def _does_compile(self, code: str) -> bool:
        """Helper to check if code compiles."""
        with open("temp_kernel.cu", "w") as f:
            f.write(code)
            
        try:
            result = subprocess.run(
                ["nvcc", "-c", "temp_kernel.cu"], 
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _test_kernel_correctness(self, code: str) -> tuple[bool, float]:
        """
        Test if kernel produces correct results for test cases.
        
        Returns:
            tuple: (is_perfectly_correct, accuracy_score)
                - is_perfectly_correct: True if output exactly matches reference
                - accuracy_score: Float between 0.0-1.0 indicating partial correctness
        """
        # Placeholder implementation
        # Would compile kernel with a test harness and compare against reference
        
        # For v1 simplified evaluator:
        # 1. Hard-code a few test cases with known results
        # 2. Run kernel on these inputs
        # 3. Compare against expected outputs
        # 4. Return (perfect_match, similarity_score)
        
        return True, 1.0  # Placeholder
        
    def _get_baseline_times(self) -> List[float]:
        """Get baseline execution times for benchmark sizes."""
        # Placeholder - would use cuBLAS or other reference implementation
        return [0.1, 1.0, 10.0]  # Fictional baseline times
        
    def _benchmark_kernel(self, code: str) -> List[float]:
        """Run benchmark on the kernel for different matrix sizes."""
        # Placeholder implementation
        return [0.2, 2.0, 20.0]  # Fictional benchmark times
        
    def _measure_memory_utilization(self, code: str) -> float:
        """Measure memory bandwidth utilization."""
        # Placeholder implementation
        return 0.5  # 50% utilization
    
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,  # This would be reference kernel or specs
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the given completions."""

        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

        # Compute all reward functions
        all_scores = [
            self._compilation_reward(completions),
            self._correctness_reward(completions, answer),
            self._performance_reward(completions),
            self._memory_efficiency_reward(completions),
            self._code_quality_reward(completions)
        ]
        
        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        
        # Calculate success rate (compilation + correctness)
        compilation_scores = rewards_per_func[:, 0]  # First reward function is compilation
        correctness_scores = rewards_per_func[:, 1]  # Second is correctness
        
        num_compiled = (compilation_scores == 1.0).sum().item()
        compilation_rate = num_compiled / num_completions
        
        num_correct = (correctness_scores == 2.0).sum().item()
        correctness_rate = num_correct / num_completions
        
        metrics = {
            "rewards/compilation_reward": reward_per_func[0].item(),
            "rewards/correctness_reward": reward_per_func[1].item(), 
            "rewards/performance_reward": reward_per_func[2].item(),
            "rewards/memory_efficiency_reward": reward_per_func[3].item(),
            "rewards/code_quality_reward": reward_per_func[4].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "compilation_rate": compilation_rate,
            "correctness_rate": correctness_rate
        }
        
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'compilation': reward_scores[0].item(),
            'correctness': reward_scores[1].item(),
            'performance': reward_scores[2].item(),
            'memory_efficiency': reward_scores[3].item(),
            'code_quality': reward_scores[4].item()
        }


class TritonMatmulEvaluator(KernelEvaluator):
    """
    Reward evaluator for Triton matrix multiplication kernels.
    
    Similar to the CUDA evaluator but adapted for Triton's specific patterns
    and optimization techniques.
    """
    
    def __init__(self):
        super().__init__()
        # Implementation would be similar to CUDAMatmulEvaluator but
        # adapted for Triton-specific patterns and optimizations
        pass
    
    # Similar methods as CUDAMatmulEvaluator but customized for Triton
    # ...
