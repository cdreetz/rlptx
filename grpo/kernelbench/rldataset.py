"""
Dataset loader for Triton kernel generation training.

Loads kernel generation prompts and specifications for GRPO training.
"""

import json
import random
import numpy as np
from typing import Tuple, Dict, Any, List
from tqdm import tqdm
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset

class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    """

    def __init__(self, random: bool = False) -> None:
        self.random = random
        self.current_index = 0

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        pass

    @abstractmethod
    def __iter__(self) -> 'DataLoader':
        """Return self as iterator."""
        return self

    @abstractmethod
    def __next__(self) -> Any:
        """Return the next item(s) in the dataset."""
        pass

class KernelBenchLoader(DataLoader):
    """
    A loader class that provides iteration over Triton kernel generation prompts.

    This class loads kernel generation tasks from a JSON dataset and provides
    both sequential and random access to prompts and their specifications.

    Attributes:
        prompts (List[str]): List of kernel generation prompts
        specs (List[Dict]): List of corresponding kernel specifications
        random (bool): If True, returns items randomly; if False, returns sequentially
        current_index (int): Current position for sequential access
    """

    def __init__(self, prompts: List[str], answers: List[str] = None, random: bool = False) -> None:
        super().__init__(random)
        self.prompts = prompts
        self.answers = answers
        self.pre_prompt = """You are a helpful assistant tasked with writing PyTorch code as Triton kernels.

            Name the kernel method as "triton_kernel" and the wrapper as "tritton_wrapper".

            Example:
            ```python
            import triton
            import triton.language as tl

            @triton.jit
            def triton_kernel(
                a_ptr, b_ptr, c_ptr, 
                stride_am, stride_ak, stride_bn, 
                stride_bk, stride_cm, stride_cn, 
                BLOCK_SIZE_M: tl.constexpr,  BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
                ...

            def triton_wrapper(A, B):
                ...
                return C
            ```

            Your output should include a 'triton_kernel' and a 'triton_wrapper' method.
            You don't need to explain anything just write the kernel and wrapper.

            Torch Code: """

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self) -> 'KernelBenchLoader':
        return self

    def __next__(self) -> Tuple[str, Dict[str, Any]]:
        if self.current_index >= len(self.prompts):
            raise StopIteration

        if self.random:
            idx = random.randint(0, len(self.prompts) - 1)
        else:
            idx = self.current_index
            self.current_index += 1

        # Format the question with the pre-prompt and the actual torch code
        formatted_question = self.pre_prompt + self.prompts[idx]
        
        return formatted_question, self.answers[idx]

    def reset(self):
        """Reset iterator to beginning."""
        self.current_index = 0

def build_kernelbench_dataloaders(test_split: float = 0.1, level_filter: int = None) -> Tuple[KernelBenchLoader, KernelBenchLoader]:
    data = load_dataset('ScalingIntelligence/KernelBench', 'default')["level_1"]

    questions = []
    answers = []
    for i in tqdm(range(len(data)), desc="Processing"):
        # Filter by level if specified
        if level_filter is not None and data[i]['level'] != level_filter:
            continue
            
        questions.append(data[i]['code'])
        answers.append(data[i]['code'])  # torch baseline code is the same as the question

    print(f"Loaded {len(questions)} examples" + (f" (level {level_filter} only)" if level_filter is not None else ""))

    total_samples = len(questions)
    test_size = int(total_samples * test_split)

    # Generate random indices for test set
    test_indices = random.sample(range(total_samples), test_size)
    test_indices_set = set(test_indices)

    # Convert to numpy arrays for easier indexing
    questions = np.array(questions)
    answers = np.array(answers)

    # Create boolean mask for test indices
    test_mask = np.zeros(total_samples, dtype=bool)
    test_mask[list(test_indices_set)] = True

    # Split using boolean indexing
    test_questions = questions[test_mask]
    test_answers = answers[test_mask]
    train_questions = questions[~test_mask]
    train_answers = answers[~test_mask]

    # Setup data loaders
    # Training loader uses random=True for variety during training
    train_loader = KernelBenchLoader(train_questions.tolist(), train_answers.tolist(), random=True)
    test_loader = KernelBenchLoader(test_questions.tolist(), test_answers.tolist(), random=False)

    return train_loader, test_loader

def get_dataloaders(dataset_name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.

    Args:
        dataset_name: Name/path of the dataset to load
        **kwargs: Additional arguments (e.g., test_split)

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders

    Raises:
        ValueError: If dataset file not found or invalid
    """
    if dataset_name.lower() == 'kernelbench':
        return build_kernelbench_dataloaders(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Provide 'triton_kernels' or path to JSON file.")

if __name__ == "__main__":
    # Test the loader
    train_loader, test_loader = get_dataloaders('kernelbench')

