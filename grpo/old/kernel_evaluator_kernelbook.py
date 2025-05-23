"""
Reward evaluator for Triton kernel optimization.

This evaluator implements reward functions to assess the quality, performance,
and correctness of generated Triton kernels based on PyTorch code.
"""

import re
import sys
import torch
import subprocess
import tempfile
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional


class RewardEvaluator(ABC):
    """
    Abstract base class for reward computation in RL training.
    
    This class defines the interface for reward evaluators used to score
    model completions during RL training with GRPO.
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
            answer: Ground truth answer(s) for the prompts (reference Triton code)
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
    if name.lower() == "triton_kernels" or name.lower() == "kernelbook":
        return TritonKernelEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")


class TritonKernelEvaluator(RewardEvaluator):
    """
    Reward evaluator for Triton kernels converted from PyTorch code.
    
    Implements reward functions to assess Triton kernel quality:
    - Compilation success (2.0 points)
    - Functional correctness (3.0 points)
    - Performance (4.0 points)
    - Memory efficiency (1.0 point)
    - Code quality (1.0 point)
    """
    
    def __init__(self):
        self.num_reward_functions = 5
        
    def _extract_kernel_code(self, text: str) -> str:
        """Extract Triton kernel code from the completion."""
        # Extract code between triple backticks with triton or python
        pattern = r'```(?:triton|python)(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
            
        # Try without language specified
        pattern = r'```(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
            
        return text  # Return the whole text if no code blocks found
    
    def _compilation_reward(self, completions: List[List[Dict[str, str]]], 
                            pytorch_code: Optional[str] = None) -> List[float]:
        """
        Test if the Triton kernel compiles successfully (2.0 points).
        
        Args:
            completions: List of model completions
            pytorch_code: Original PyTorch code for context (optional)
            
        Returns:
            List of reward scores (0.0 or 2.0)
        """
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(r) for r in responses]
        
        results = []
        for code in kernel_codes:
            try:
                # Create temporary file with the kernel code
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
                    # Add necessary imports
                    full_code = "import triton\nimport triton.language as tl\nimport torch\n\n" + code
                    f.write(full_code.encode('utf-8'))
                    temp_path = f.name
                
                # Try to compile with Python
                result = subprocess.run(
                    [sys.executable, "-c", f"import triton; import torch; exec(open('{temp_path}').read())"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                # Clean up temp file
                os.unlink(temp_path)
                
                # Give 2.0 reward if compilation succeeds
                if result.returncode == 0:
                    results.append(2.0)
                else:
                    # Check if it's a minor import error but structure looks right
                    if "triton.autotune" in code and "Error" in result.stderr:
                        # Partial credit for minor issues
                        results.append(1.0)
                    else:
                        results.append(0.0)
                        
            except Exception as e:
                results.append(0.0)
                
        return results
    
    def _correctness_reward(self, completions: List[List[Dict[str, str]]], 
                           reference_solution: Dict) -> List[float]:
        """
        Check if kernel produces functionally correct results (3.0 points).
        
        For v1, we primarily check for structural similarity between the 
        generated Triton code and the reference solution, as full execution
        verification would require more complex infrastructure.
        
        Args:
            completions: List of model completions
            reference_solution: Dictionary containing reference Triton code
            
        Returns:
            List of reward scores (0.0 to 3.0)
        """
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(r) for r in responses]
        reference_code = reference_solution["triton_code"]
        
        results = []
        for code in kernel_codes:
            # Skip if compilation completely failed
            if not self._does_compile(code):
                results.append(0.0)
                continue
                
            # For v1, use similarity-based assessment
            similarity_score = self._assess_structural_similarity(code, reference_code)
            
            # Scale similarity to reward (0.0-3.0)
            reward = similarity_score * 3.0
            results.append(reward)
                
        return results
    
    def _performance_reward(self, completions: List[List[Dict[str, str]]], 
                           reference_solution: Dict) -> List[float]:
        """
        Assess potential performance characteristics (4.0 points max).
        
        Since actual execution benchmarking is complex for v1, we analyze
        the code for performance-enhancing patterns and compare against the
        reference solution.
        
        Args:
            completions: List of model completions
            reference_solution: Dictionary with reference solution
            
        Returns:
            List of reward scores (0.0 to 4.0)
        """
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(r) for r in responses]
        reference_code = reference_solution["triton_code"]
        
        results = []
        for code in kernel_codes:
            # Skip if compilation failed
            if not self._does_compile(code):
                results.append(0.0)
                continue
                
            # Check for performance optimization patterns
            autotune_score = self._check_autotune_configs(code)
            tiling_score = self._check_tiling_patterns(code)
            memory_score = self._check_memory_patterns(code)
            
            # Compare against reference solution's patterns
            ref_similarity = self._assess_performance_similarity(code, reference_code)
            
            # Compute weighted performance score (0.0-4.0)
            # 40% for autotuning, 30% for tiling, 20% for memory patterns, 10% for similarity
            performance_score = (0.4 * autotune_score + 
                                0.3 * tiling_score + 
                                0.2 * memory_score +
                                0.1 * ref_similarity) * 4.0
            
            results.append(performance_score)
                
        return results
                
    def _memory_efficiency_reward(self, completions: List[List[Dict[str, str]]], 
                                 reference_solution: Dict) -> List[float]:
        """
        Assess memory optimization patterns (1.0 point max).
        
        Args:
            completions: List of model completions
            reference_solution: Dictionary with reference solution
            
        Returns:
            List of reward scores (0.0 to 1.0)
        """
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(r) for r in responses]
        reference_code = reference_solution["triton_code"]
        
        results = []
        for code in kernel_codes:
            # Skip if compilation failed
            if not self._does_compile(code):
                results.append(0.0)
                continue
                
            # Check memory access patterns
            score = 0.0
            
            # Check for block access patterns
            if "tl.load" in code and ("[:, None]" in code or "[:, :]" in code):
                score += 0.5
                
            # Check for proper masking in loads/stores
            if "mask=" in code:
                score += 0.3
                
            # Check for shared memory or scratch usage
            if "tl.zeros" in code:
                score += 0.2
                
            results.append(min(1.0, score))
                
        return results
        
    def _code_quality_reward(self, completions: List[List[Dict[str, str]]]) -> List[float]:
        """
        Assess code quality and best practices (1.0 point max).
        
        Args:
            completions: List of model completions
            
        Returns:
            List of reward scores (0.0 to 1.0)
        """
        responses = [completion[0]['content'] for completion in completions]
        kernel_codes = [self._extract_kernel_code(r) for r in responses]
        
        results = []
        for code in kernel_codes:
            score = 0.0
            
            # Check for proper docstring
            if '"""' in code or "'''" in code:
                score += 0.2
                
            # Check for descriptive variable names
            non_descriptive = ['a', 'b', 'c', 'x', 'y', 'z']
            descriptive_vars = [v for v in re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code) 
                              if len(v) > 1 and v not in non_descriptive]
            if len(descriptive_vars) > 5:  # At least 5 descriptive variable names
                score += 0.2
                
            # Check for consistent formatting
            if code.count('    ') > code.count('\t'):  # Consistent space indentation
                score += 0.2
                
            # Check for comments
            if code.count('#') >= 3:  # At least 3 comments
                score += 0.2
                
            # Check for type hints
            if ': tl.' in code:
                score += 0.2
                
            results.append(min(1.0, score))
                
        return results
    
    def _does_compile(self, code: str) -> bool:
        """
        Helper to check if Triton code compiles.
        
        Args:
            code: Triton kernel code to check
            
        Returns:
            Boolean indicating whether code compiles
        """
        try:
            # Create temporary file with the kernel code
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
                # Add necessary imports
                full_code = "import triton\nimport triton.language as tl\nimport torch\n\n" + code
                f.write(full_code.encode('utf-8'))
                temp_path = f.name
            
            # Try to compile with Python
            result = subprocess.run(
                [sys.executable, "-c", f"import triton; import torch; exec(open('{temp_path}').read())"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return result.returncode == 0
            
        except Exception:
            return False

    def _assess_structural_similarity(self, generated_code: str, reference_code: str) -> float:
        """
        Assess structural similarity between generated code and reference solution.
        
        Args:
            generated_code: Generated Triton kernel code
            reference_code: Reference Triton kernel code
            
        Returns:
            Float between 0.0 and 1.0 indicating similarity
        """
        similarity = 0.0
        
        # Check for key function signatures
        ref_functions = re.findall(r'@triton\.jit\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)', reference_code)
        gen_functions = re.findall(r'@triton\.jit\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)', generated_code)
        
        if ref_functions and any(rf in generated_code for rf in ref_functions):
            similarity += 0.3
            
        # Check for similar function parameters
        ref_params = re.findall(r'def\s+[a-zA-Z0-9_]+\((.*?)\)', reference_code, re.DOTALL)
        gen_params = re.findall(r'def\s+[a-zA-Z0-9_]+\((.*?)\)', generated_code, re.DOTALL)
        
        if ref_params and gen_params:
            ref_param_count = len(ref_params[0].split(','))
            gen_param_count = len(gen_params[0].split(','))
            
            # Calculate parameter count similarity
            param_similarity = 1.0 - min(abs(ref_param_count - gen_param_count) / max(ref_param_count, 1), 1.0)
            similarity += 0.2 * param_similarity
            
        # Check for similar tl operations
        ref_ops = re.findall(r'tl\.([a-zA-Z_][a-zA-Z0-9_]*)', reference_code)
        gen_ops = re.findall(r'tl\.([a-zA-Z_][a-zA-Z0-9_]*)', generated_code)
        
        if ref_ops and gen_ops:
            common_ops = set(ref_ops).intersection(set(gen_ops))
            op_similarity = len(common_ops) / len(set(ref_ops))
            similarity += 0.3 * op_similarity
            
        # Check for similar code length (not too short or too long)
        ref_lines = len(reference_code.split('\n'))
        gen_lines = len(generated_code.split('\n'))
        
        length_ratio = min(gen_lines / max(ref_lines, 1), max(ref_lines, 1) / max(gen_lines, 1))
        similarity += 0.2 * length_ratio
        
        return min(1.0, similarity)

    def _check_autotune_configs(self, code: str) -> float:
        """
        Check for proper autotuning configurations in Triton code.
        
        Args:
            code: Triton kernel code
            
        Returns:
            Float between 0.0 and 1.0 indicating quality of autotuning
        """
        score = 0.0
        
        # Check for autotune decorator
        if "@triton.autotune" in code:
            score += 0.5
            
            # Check for multiple configs
            configs = code.count("triton.Config")
            if configs >= 3:
                score += 0.3  # Good number of configs
            elif configs >= 1:
                score += 0.1  # At least one config
                
            # Check for proper num_stages
            if "num_stages=" in code:
                score += 0.1
                
            # Check for proper key parameter
            if "key=[" in code:
                score += 0.1
        
        return min(1.0, score)

    def _check_tiling_patterns(self, code: str) -> float:
        """
        Check for effective tiling patterns in Triton code.
        
        Args:
            code: Triton kernel code
            
        Returns:
            Float between 0.0 and 1.0 indicating quality of tiling
        """
        score = 0.0
        
        # Check for BLOCK_SIZE parameters
        block_sizes = re.findall(r'BLOCK_SIZE_([A-Z])', code)
        if block_sizes:
            score += 0.4
            
            # Check for multiple dimensions
            if len(set(block_sizes)) >= 2:
                score += 0.3
                
        # Check for proper tile parameter usage  
        if "tl.constexpr" in code:
            score += 0.3
            
        return min(1.0, score)

    def _check_memory_patterns(self, code: str) -> float:
        """
        Check for efficient memory access patterns in Triton code.
        
        Args:
            code: Triton kernel code
            
        Returns:
            Float between 0.0 and 1.0 indicating quality of memory patterns
        """
        score = 0.0
        
        # Check for efficient load patterns
        if "tl.load" in code:
            score += 0.3
            
            # Check for proper masking
            if "mask=" in code:
                score += 0.2
                
        # Check for efficient store patterns
        if "tl.store" in code:
            score += 0.3
            
            # Check for proper masking
            if "mask=" in code:
                score += 0.2
                
        return min(1.0, score)

    def _assess_performance_similarity(self, generated_code: str, reference_code: str) -> float:
        """
        Compare performance characteristics between generated and reference code.
        
        Args:
            generated_code: Generated Triton kernel code
            reference_code: Reference Triton kernel code
            
        Returns:
            Float between 0.0 and 1.0 indicating performance similarity
        """
        gen_autotune = self._check_autotune_configs(generated_code)
        ref_autotune = self._check_autotune_configs(reference_code)
        
        gen_tiling = self._check_tiling_patterns(generated_code)
        ref_tiling = self._check_tiling_patterns(reference_code)
        
        gen_memory = self._check_memory_patterns(generated_code)
        ref_memory = self._check_memory_patterns(reference_code)
        
        # Calculate weighted similarity
        similarity = 0.4 * min(gen_autotune / max(ref_autotune, 0.1), 1.0)
        similarity += 0.4 * min(gen_tiling / max(ref_tiling, 0.1), 1.0)
        similarity += 0.2 * min(gen_memory / max(ref_memory, 0.1), 1.0)
        
        return min(1.0, similarity)
    
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Dict,  # Reference solution with Triton code
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the given completions
