"""
Abstract base class and implementations for reward computation in RL training.

"""

import re
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from evaluate_template import evaluate

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
    if name.lower() == "kernelbench":
        return KernelBenchEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")

def extract_kernel_methods(text: str) -> tuple[str, str]:
    if "triton_kernel" in text and "tritton_wrapper" in text:
        kernel_start = text.find("triton_kernel") 
        wrapper_start = text.find("tritton_wrapper")
        
        wrapper_lines = text[wrapper_start:].split('\n')
        wrapper_end = wrapper_start
        for i, line in enumerate(wrapper_lines):
            if 'return' in line:
                wrapper_end += sum(len(l) + 1 for l in wrapper_lines[:i+1])
                break
        if wrapper_end == wrapper_start:  # No return found
            wrapper_end = len(text)
            
        kernel_code = text[kernel_start:wrapper_start].strip()
        wrapper_code = text[wrapper_start:wrapper_end].strip()
        return kernel_code, wrapper_code
    else:
        return text, ""

class KernelBenchEvaluator(RewardEvaluator):
    """
    Reward evaluator for the KernelBench dataset.
    
    Implements reward functions for:
    - Compiles
    - Correctness
    - Performance
    """
    
    def __init__(self):
        self.num_reward_functions = 3
    
    def _compiles_reward(self, prompts, completions, answer) -> List[float]:
        """Reward for compiling."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [extract_kernel_methods(r) for r in responses]
        results = [evaluate(*ext, answer) if ext[0] and ext[1] else {'compiles': False} for ext in extracted]
        return [2.0 if result['compiles'] else 0.0 for result in results]

    def _correctness_reward(self, prompts, completions, answer) -> List[float]:
        """Reward for correctness."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [extract_kernel_methods(r) for r in responses]
        results = [evaluate(*ext, answer) if ext[0] and ext[1] else {'correct': False} for ext in extracted]
        return [1.0 if result['correct'] else 0.0 for result in results]

    def _perf_reward(self, prompts, completions, answer) -> List[float]:
        """Reward for performance compared to torch."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [extract_kernel_methods(r) for r in responses]
        results = [evaluate(*ext, answer) if ext[0] and ext[1] else {'correct': False, 'speedup': None} for ext in extracted]
        return [min(1.0, max(0.0, result['speedup'])) if result['correct'] and result['speedup'] is not None else 0.0 for result in results]

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the given completions."""

        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

        # Compute all reward functions
        all_scores = [
            self._compiles_reward(prompts, completions, answer),
            self._correctness_reward(prompts, completions, answer),
            self._perf_reward(prompts, completions, answer),
        ]
        
        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        
        # Calculate accuracy (perfect correctness score)
        compiles_scores = rewards_per_func[:, 0]
        correctness_scores = rewards_per_func[:, 1]
        perf_scores = rewards_per_func[:, 2]
        num_compiles = (compiles_scores == 2.0).sum().item()
        accuracy = num_compiles / num_completions
        
        metrics = {
            "rewards/compiles_reward_func": reward_per_func[0].item(),
            "rewards/correctness_reward_func": reward_per_func[1].item(),
            "rewards/perf_reward_func": reward_per_func[2].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "accuracy": accuracy
        }
        
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'compiles': reward_scores[0].item(),
            'correctness': reward_scores[1].item(),
            'performance': reward_scores[2].item(),
        }