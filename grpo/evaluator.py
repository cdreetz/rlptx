"""
Abstract base class and implementations for reward computation in RL training.
"""

import re
import os
import torch
import tempfile
import subprocess
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

class RewardEvaluator(ABC):
    """
    Abstract base class for reward computation in RL training.
    
    This class defines the interface for reward evaluators that can be used
    to score model completions during RL training. Implement this class to
    create custom reward functions for different tasks.
    
    The main methods that need to be implemented are:
    - compute_rewards: Computes rewards for a batch of completions
    - get_reward_breakdown: Converts raw reward scores to a labeled dictionary
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
            answer: Ground truth answer(s) for the prompts
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


class TritonEvaluator(RewardEvaluator):
    """
    Reward evaluator for Triton kernel generation.
    
    Implements reward functions for:
    - Code compilation
    - Functional correctness
    - Performance (if available)
    """
    
    def __init__(self, can_run_triton: bool = False):
        """
        Initialize the TritonEvaluator.
        
        Args:
            can_run_triton: Whether Triton is available for actual compilation
        """
        self.num_reward_functions = 3
        self.can_run_triton = can_run_triton
        
        # Try to import triton if available
        self.triton_available = False
        if self.can_run_triton:
            try:
                import triton
                self.triton = triton
                self.triton_available = True
            except ImportError:
                print("Warning: Triton not available. Using fallback evaluation.")
    
    def _extract_code(self, text: str) -> str:
        """
        Extract code from the model's response, ignoring explanations.
        
        Args:
            text: Model response text
            
        Returns:
            Extracted code
        """
        # code block
        code_block_match = re.search(r'```(?:python|triton)?\s*(.*?)```', text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # If no code block, try to extract just the code part
        # Look for import statements as starting points
        import_match = re.search(r'(import\s+triton.*?)$', text, re.DOTALL)
        if import_match:
            return import_match.group(1).strip()
        
        # If all else fails, return the entire text (might be just code)
        return text.strip()
    
    def _compilation_reward(self, completions: List[List[Dict[str, str]]]) -> List[float]:
        """
        Reward for successful kernel compilation.
        
        Args:
            completions: Model completions
            
        Returns:
            List of reward values
        """
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_code(r) for r in responses]
        
        rewards = []
        
        for code in extracted:
            if self.triton_available:
                # Attempt actual compilation
                reward = self._try_compile_kernel(code)
            else:
                # Fallback: Basic syntax check and pattern recognition
                reward = self._syntax_check(code)
                
            rewards.append(reward)
            
        return rewards
    
    def _try_compile_kernel(self, code: str) -> float:
        """
        Try to actually compile the kernel using Triton.
        
        Args:
            code: The kernel code to compile
            
        Returns:
            Reward value between 0 and 1
        """
        # Write code to a temporary file
        temp_filename = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
                temp_filename = temp_file.name
                # Add necessary imports if not present
                if "import triton" not in code:
                    temp_file.write("import triton\n")
                if "import triton.language" not in code and "from triton.language" not in code:
                    temp_file.write("import triton.language as tl\n")
                temp_file.write(code)
            
            # Run the file to check for compilation
            try:
                result = subprocess.run(
                    ['python', temp_filename],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
                
                # Check if there were errors
                if result.returncode != 0:
                    # Analyze error type
                    stderr = result.stderr.decode('utf-8')
                    
                    # Give partial credit for different error types
                    if "SyntaxError" in stderr:
                        return 0.0  # Syntax error
                    elif "AttributeError" in stderr:
                        return 0.1  # Attribute error (e.g., wrong function names)
                    elif "TypeError" in stderr:
                        return 0.3  # Type error (wrong argument types)
                    else:
                        return 0.1  # Other errors
                else:
                    return 1.0  # Successful compilation
                    
            except subprocess.TimeoutExpired:
                return 0.4  # Took too long but might be partially correct
                
        except Exception as e:
            print(f"Error testing compilation: {e}")
            return 0.0
        finally:
            # Clean up
            if temp_filename and os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def _syntax_check(self, code: str) -> float:
        """
        Fallback method to check syntax when Triton is not available.
        
        Args:
            code: The kernel code to check
            
        Returns:
            Reward value between 0 and 1
        """
        # Check for basic Python syntax
        try:
            compile(code, '<string>', 'exec')
            syntax_correct = True
        except SyntaxError:
            syntax_correct = False
            
        # Check for necessary Triton components
        has_triton_import = "import triton" in code
        has_language_import = "triton.language" in code or "import triton.language" in code
        has_decorator = "@triton.jit" in code
        has_kernel_function = bool(re.search(r'def\s+\w+\s*\(', code))
        has_memory_ops = "tl.load" in code and "tl.store" in code
        
        # Calculate score based on presence of key components
        score = 0.0
        if syntax_correct:
            score += 0.5
        if has_triton_import and has_language_import:
            score += 0.1
        if has_decorator:
            score += 0.1
        if has_kernel_function:
            score += 0.1
        if has_memory_ops:
            score += 0.2
            
        return min(score, 1.0)
    
    def _functional_correctness(self, completions: List[List[Dict[str, str]]], reference_kernels: List[str]) -> List[float]:
        """
        Reward for functional correctness.
        
        Args:
            completions: Model completions
            reference_kernels: Reference kernel implementations
            
        Returns:
            List of reward values
        """
        if self.triton_available:
            # This would execute kernels and compare outputs
            # For now, return a simplified check
            return self._structural_similarity(completions, reference_kernels)
        else:
            # Fallback to structural similarity
            return self._structural_similarity(completions, reference_kernels)
    
    def _structural_similarity(self, completions: List[List[Dict[str, str]]], reference_kernels: List[str]) -> List[float]:
        """
        Measure structural similarity between generated and reference kernels.
        
        Args:
            completions: Model completions
            reference_kernels: Reference implementations
            
        Returns:
            Similarity scores
        """
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_code(r) for r in responses]
        
        rewards = []
        
        for code, ref_kernel in zip(extracted, reference_kernels):
            # Clean whitespace and comments
            code_clean = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            code_clean = re.sub(r'\s+', ' ', code_clean).strip()
            
            ref_clean = re.sub(r'#.*$', '', ref_kernel, flags=re.MULTILINE)
            ref_clean = re.sub(r'\s+', ' ', ref_clean).strip()
            
            # Extract function signature pattern
            code_sig = re.search(r'def\s+\w+\s*\([^)]*\)', code_clean)
            ref_sig = re.search(r'def\s+\w+\s*\([^)]*\)', ref_clean)
            
            # Extract triton-specific functions used
            code_funcs = set(re.findall(r'tl\.\w+', code_clean))
            ref_funcs = set(re.findall(r'tl\.\w+', ref_clean))
            
            # Calculate scores
            signature_score = 0.0
            if code_sig and ref_sig:
                # Count matching parameter names
                code_params = re.findall(r'[\w_]+(?=\s*[,:])', code_sig.group(0))
                ref_params = re.findall(r'[\w_]+(?=\s*[,:])', ref_sig.group(0))
                
                if ref_params:
                    matches = sum(1 for p in code_params if p in ref_params)
                    signature_score = min(matches / len(ref_params), 1.0) * 0.5
            
            # Function usage score
            func_score = 0.0
            if ref_funcs:
                matches = len(code_funcs.intersection(ref_funcs))
                func_score = min(matches / len(ref_funcs), 1.0) * 0.5
            
            # Combined score
            similarity = signature_score + func_score
            rewards.append(min(similarity, 1.0))
            
        return rewards
    
    def _performance_reward(self, completions: List[List[Dict[str, str]]], reference_kernels: List[str]) -> List[float]:
        """
        Reward for kernel performance (placeholder).
        
        Args:
            completions: Model completions
            reference_kernels: Reference kernel implementations
            
        Returns:
            List of performance reward values
        """
        # Real implementation would benchmark kernels
        # For now, return neutral scores
        return [0.0] * len(completions)

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,  # Reference kernel implementations
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rewards for generated Triton kernels.
        
        Args:
            prompts: List of prompt messages in chat format
            completions: List of completion messages in chat format
            answer: Reference Triton kernel implementations
            device: Device to place tensors on
            
        Returns:
            rewards_per_func: Tensor of shape (num_completions, num_reward_functions)
            metrics: Dictionary of aggregated metrics
        """
        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

        # Compute all reward functions
        all_scores = [
            self._compilation_reward(completions),
            self._functional_correctness(completions, answer),
            self._performance_reward(completions, answer)
        ]
        
        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        
        # Calculate compile_success_rate (main success metric)
        compile_scores = rewards_per_func[:, 0]  # First reward function is compilation
        num_compiled = (compile_scores > 0.9).sum().item()  # Count successful compilations
        compile_rate = num_compiled / num_completions
        
        metrics = {
            "rewards/compilation": reward_per_func[0].item(),
            "rewards/correctness": reward_per_func[1].item(),
            "rewards/performance": reward_per_func[2].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "compile_success_rate": compile_rate,
        }
        
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'compilation': reward_scores[0].item(),
            'correctness': reward_scores[1].item(),
            'performance': reward_scores[2].item()
        }


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
    if name.lower() in ["triton", "kernelbook"]:
        # Check if Triton is available
        try:
            import triton
            print("Triton is available - will use for compilation testing")
            return TritonEvaluator(can_run_triton=True)
        except ImportError:
            print("Triton not available - using fallback evaluation")
            return TritonEvaluator(can_run_triton=False)
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")
