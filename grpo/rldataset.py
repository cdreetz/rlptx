"""
Dataset loader for Triton kernel generation training.

Loads kernel generation prompts and specifications for GRPO training.
"""

import json
import random
import numpy as np
from typing import Tuple, Dict, Any, List
from abc import ABC, abstractmethod


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


SYSTEM_PROMPT = """You are an expert at writing high-performance Triton kernels.

Write clean, efficient Triton code that:
- Includes proper imports (triton, triton.language as tl)
- Uses appropriate BLOCK_SIZE constants
- Handles edge cases with proper masking
- Includes the @triton.jit decorator

Provide only the kernel code without explanation."""


class TritonKernelLoader(DataLoader):
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

    def __init__(self, prompts: List[str], specs: List[Dict[str, Any]], random: bool = False) -> None:
        super().__init__(random)
        self.prompts = prompts
        self.specs = specs
        self.system_prompt = SYSTEM_PROMPT

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self) -> 'TritonKernelLoader':
        return self

    def __next__(self) -> Tuple[str, Dict[str, Any]]:
        if self.current_index >= len(self.prompts):
            raise StopIteration

        if self.random:
            idx = random.randint(0, len(self.prompts) - 1)
        else:
            idx = self.current_index
            self.current_index += 1

        return self.prompts[idx], self.specs[idx]

    def reset(self):
        """Reset iterator to beginning."""
        self.current_index = 0


def load_kernel_dataset(dataset_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Load kernel dataset from JSON file.

    Args:
        dataset_path: Path to JSON file containing kernel examples

    Returns:
        prompts: List of kernel generation prompts
        specs: List of kernel specifications with input_shapes, output_shape, etc.
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    prompts = []
    specs = []

    for item in data:
        # Extract just the task description from the full prompt
        # The full prompt contains system instructions, we just want the task
        full_prompt = item['prompt']

        # Find the task description (starts after "Task: ")
        if "Task: " in full_prompt:
            task_start = full_prompt.find("Task: ") + len("Task: ")
            task_end = full_prompt.find("\n\nRequirements:")
            if task_end == -1:
                task_end = full_prompt.find("\nRequirements:")

            if task_end != -1:
                task_prompt = full_prompt[task_start:task_end].strip()
            else:
                # Fallback: use everything after "Task: "
                task_prompt = full_prompt[task_start:].strip()
        else:
            # Fallback: use the full prompt
            task_prompt = full_prompt

        prompts.append(task_prompt)

        # Create spec dictionary with all needed info for evaluation
        spec = {
            'input_shapes': item['input_shapes'],
            'output_shape': item['output_shape'],
            'operation': item['operation'],
            'optimization_level': item['optimization_level']
        }
        specs.append(spec)

    return prompts, specs


def build_triton_dataloaders(dataset_path: str, test_split: float = 0.1) -> Tuple[TritonKernelLoader, TritonKernelLoader]:
    """
    Build train and test data loaders from kernel dataset.

    Args:
        dataset_path: Path to JSON dataset file
        test_split: Fraction of data to use for testing

    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
    """
    prompts, specs = load_kernel_dataset(dataset_path)

    # Filter to only include examples that compiled successfully (optional)
    # You might want to remove this filter if you want the model to learn from failures too
    # compiled_indices = [i for i, spec in enumerate(specs) if spec.get('compiles', False)]
    # prompts = [prompts[i] for i in compiled_indices]
    # specs = [specs[i] for i in compiled_indices]

    # Randomly split into train/test sets
    total_samples = len(prompts)
    test_size = int(total_samples * test_split)

    # Generate random indices for test set
    test_indices = random.sample(range(total_samples), test_size)
    test_indices_set = set(test_indices)

    # Convert to numpy arrays for easier indexing
    prompts = np.array(prompts)
    specs = np.array(specs)

    # Create boolean mask for test indices
    test_mask = np.zeros(total_samples, dtype=bool)
    test_mask[list(test_indices_set)] = True

    # Split using boolean indexing
    test_prompts = prompts[test_mask]
    test_specs = specs[test_mask]
    train_prompts = prompts[~test_mask]
    train_specs = specs[~test_mask]

    # Setup data loaders
    # Training loader uses random=True for variety during training
    train_loader = TritonKernelLoader(train_prompts.tolist(), test_specs.tolist(), random=True)
    test_loader = TritonKernelLoader(test_prompts.tolist(), test_specs.tolist(), random=False)

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
    if dataset_name.lower() == 'triton_kernels':
        # Default to looking for dataset in current directory
        dataset_path = kwargs.get('dataset_path', 'triton_kernels_dataset_v5.json')
        test_split = kwargs.get('test_split', 0.1)
        return build_triton_dataloaders(dataset_path, test_split)
    elif dataset_name.endswith('.json'):
        # Direct path to JSON file
        test_split = kwargs.get('test_split', 0.1)
        return build_triton_dataloaders(dataset_name, test_split)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Provide 'triton_kernels' or path to JSON file.")


if __name__ == "__main__":
    # Test the loader
    train_loader, test_loader = get_dataloaders('triton_kernels_dataset_v5.json')

    print(f"Train set size: {len(train_loader)}")
    print(f"Test set size: {len(test_loader)}")

    # Test iteration
    prompt, spec = next(train_loader)
    print(f"\nExample prompt: {prompt}")
    print(f"Example spec: {spec}")
