"""
Dataset loaders for GRPO training
"""

import json
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    This class defines the interface that all dataset loaders should implement.
    Specific dataset loaders should inherit from this class and implement the
    required methods.
    
    Attributes:
        random (bool): If True, returns items randomly; if False, returns sequentially
        current_index (int): Current position for sequential access
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
        
    def reset(self) -> None:
        """Reset iteration to the beginning of the dataset."""
        self.current_index = 0


class KernelBookLoader(DataLoader):
    """
    A loader class for the KernelBook dataset with natural language queries.
    
    Provides iteration over natural language query -> Triton kernel pairs,
    following the same interface as other dataset loaders.
    
    Attributes:
        queries: List of natural language query strings
        kernels: List of corresponding Triton kernel implementations
        random: If True, returns pairs randomly; if False, returns sequentially
        current_index: Current position for sequential access
    """
    
    def __init__(self, queries: List[str], kernels: List[str], random: bool = False) -> None:
        """
        Initialize the KernelBook loader.
        
        Args:
            queries: List of natural language queries
            kernels: List of corresponding Triton kernel implementations
            random: Whether to return examples randomly or sequentially
        """
        super().__init__(random)
        self.queries = queries
        self.kernels = kernels
        self.system_prompt = """
        You are a CUDA kernel expert specializing in Triton, a Python DSL for writing high-performance GPU kernels. 
        When given a request, write a correct and optimized Triton kernel implementation.

        Focus on:
        1. Correct use of @triton.jit decorator
        2. Proper memory access patterns with tl.load and tl.store
        3. Efficient kernel organization with program_id

        Write only the code implementation without additional explanations.
        """
        
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.queries)
        
    def __iter__(self) -> 'KernelBookLoader':
        """Return self as iterator."""
        return self
        
    def __next__(self) -> Tuple[str, str]:
        """
        Return the next query and kernel pair.
        
        Returns:
            Tuple containing (query, kernel)
        
        Raises:
            StopIteration: When all examples have been iterated through
        """
        if self.current_index >= len(self.queries):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.queries) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        return self.queries[idx], self.kernels[idx]


def build_kernelbook_dataloaders(dataset_path: str, test_split: float = 0.1) -> Tuple[KernelBookLoader, KernelBookLoader]:
    """
    Load and split the KernelBook dataset into train and test loaders.
    
    Args:
        dataset_path: Path to the transformed dataset (JSON or Parquet)
        test_split: Fraction of data to use for testing
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Load dataset
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    elif dataset_path.endswith('.parquet'):
        import pandas as pd
        data = pd.read_parquet(dataset_path).to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")
    
    # Extract queries and kernels
    queries = []
    kernels = []
    
    for item in data:
        queries.append(item['query'])
        kernels.append(item['triton_kernel'])
    
    # Split into train and test sets
    total_samples = len(queries)
    test_size = int(total_samples * test_split)
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_samples), test_size)
    test_indices_set = set(test_indices)
    
    # Convert to numpy arrays for easier indexing
    queries_array = np.array(queries)
    kernels_array = np.array(kernels)
    
    # Create boolean mask for test indices
    test_mask = np.zeros(total_samples, dtype=bool)
    test_mask[list(test_indices_set)] = True
    
    # Split using boolean indexing
    test_queries = queries_array[test_mask]
    test_kernels = kernels_array[test_mask]
    
    train_queries = queries_array[~test_mask]
    train_kernels = kernels_array[~test_mask]
    
    # Create and return data loaders
    train_loader = KernelBookLoader(
        train_queries.tolist(),
        train_kernels.tolist(),
        random=True  # Randomize training examples
    )
    
    test_loader = KernelBookLoader(
        test_queries.tolist(),
        test_kernels.tolist(),
        random=False  # Sequential for testing
    )
    
    print(f"Loaded KernelBook dataset: {len(train_loader)} training examples, {len(test_loader)} testing examples")
    return train_loader, test_loader


def get_dataloaders(dataset_name: str, dataset_path: str = "kernelbook_nl_queries.json") -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name: Name of the dataset to load
        dataset_path: Path to the dataset file (for KernelBook)
        
    Returns:
        Tuple of (train_loader, test_loader)
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name.lower() in ["kernelbook", "triton"]:
        return build_kernelbook_dataloaders(dataset_path)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Try 'kernelbook'.")
