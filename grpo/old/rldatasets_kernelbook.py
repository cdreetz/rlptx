"""
Data loaders for the Kernel Book dataset (PyTorch to Triton conversion).
"""

import os
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple, Any, List, Dict, Optional
from abc import ABC, abstractmethod


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


SYSTEM_PROMPT = """
You are an expert in writing high-performance Triton kernels. Your task is to convert PyTorch code to optimized Triton implementations.

Focus on:
- Effective use of Triton's programming model
- Proper tiling strategies
- Memory hierarchy optimization
- Auto-tuning annotations
- Maximizing performance

Write clear, efficient Triton code that correctly implements the functionality of the provided PyTorch code.
"""

class KernelBookLoader(DataLoader):
    """
    A loader class that provides iteration over Kernel Book dataset examples.
    
    This class implements both sequential and random access to PyTorch-to-Triton
    translation tasks through standard Python iterator protocols.
    
    Attributes:
        data (List[Dict]): List of dataset entries with PyTorch and Triton code
        random (bool): If True, returns problems randomly; if False, returns sequentially
        current_index (int): Current position in the lists for sequential access
    """
    
    def __init__(self, data: List[Dict], random: bool = False) -> None:
        super().__init__(random)
        self.data = data
        self.system_prompt = SYSTEM_PROMPT
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __iter__(self) -> 'KernelBookLoader':
        return self
        
    def __next__(self) -> tuple[Dict, Dict]:
        """
        Returns the next (pytorch_code, triton_code) pair.
        
        Returns:
            tuple: (input_data, reference_solution)
                - input_data: Dict containing PyTorch code and metadata
                - reference_solution: Dict containing Triton code and metadata
        """
        if self.current_index >= len(self.data):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.data) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        entry = self.data[idx]
        
        # Format input data (PyTorch code and relevant context)
        input_data = {
            "pytorch_code": entry["pytorch_code"],
            "repo_name": entry.get("repo_name", ""),
            "description": self._generate_task_description(entry),
            "repo_link": entry.get("repo_link", ""),
            "id": entry.get("id", str(idx))
        }
        
        # Format reference solution (Triton code)
        reference_solution = {
            "triton_code": entry["triton_code"],
            "id": entry.get("id", str(idx)),
            "performance_metrics": entry.get("performance_metrics", {})
        }
        
        return input_data, reference_solution

    def _generate_task_description(self, entry: Dict) -> str:
        """
        Generate a task description based on the entry.
        
        For Kernel Book dataset, we create a task description that
        explains what the PyTorch code does and what needs to be translated.
        
        Args:
            entry: Dictionary containing dataset entry
            
        Returns:
            String containing task description
        """
        # Extract module name and functionality from code if possible
        module_name = "module"
        if "class " in entry["pytorch_code"]:
            module_parts = entry["pytorch_code"].split("class ")[1].split("(")[0].strip()
            if module_parts:
                module_name = module_parts
        
        description = f"""
Convert the following PyTorch code to an optimized Triton kernel:

```python
{entry["pytorch_code"]}
```

Create a high-performance Triton implementation that maintains the same functionality as the PyTorch code.
Your solution should follow Triton best practices with appropriate tiling, memory access patterns, 
and auto-tuning configurations.
"""
        return description

    def reset(self):
        self.current_index = 0


def load_kernelbook_dataset(dataset_path: str) -> List[Dict]:
    """
    Load the Kernel Book dataset from the given path.
    
    Args:
        dataset_path: Path to the Kernel Book dataset file (JSON or Parquet)
        
    Returns:
        List of dictionaries containing dataset entries
    """
    # Determine file type and load accordingly
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    elif dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")
    
    # Process and clean the data
    processed_data = []
    for entry in data:
        # Ensure required fields exist
        if "pytorch_code" not in entry or "triton_code" not in entry:
            continue
            
        # Clean and process the code as needed
        processed_entry = {
            "pytorch_code": entry["pytorch_code"].strip(),
            "triton_code": entry["triton_code"].strip(),
            "id": entry.get("id", ""),
            "repo_name": entry.get("repo_name", ""),
            "repo_link": entry.get("repo_link", ""),
            "licenses": entry.get("licenses", []),
            "stars": entry.get("stars", 0)
        }
        
        # Add performance metrics if available
        if "performance_metrics" in entry:
            processed_entry["performance_metrics"] = entry["performance_metrics"]
        
        processed_data.append(processed_entry)
    
    return processed_data


def build_kernelbook_dataloaders(dataset_path: str, test_size: float = 0.2, 
                                random_seed: int = 42) -> Tuple[KernelBookLoader, KernelBookLoader]:
    """
    Build train and test data loaders for the Kernel Book dataset.
    
    Args:
        dataset_path: Path to the Kernel Book dataset
        test_size: Proportion of data to use for testing
        random_seed: Random seed for reproducible splitting
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load the dataset
    print(f"Loading Kernel Book dataset from {dataset_path}...")
    data = load_kernelbook_dataset(dataset_path)
    print(f"Loaded {len(data)} entries")
    
    # Randomly split into train/test sets
    total_samples = len(data)
    test_count = int(total_samples * test_size)
    
    # Generate random indices for test set
    indices = list(range(total_samples))
    random.shuffle(indices)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    # Split data into train and test sets
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    
    print(f"Split into {len(train_data)} training and {len(test_data)} testing examples")

    # Setup data loaders 
    trainloader = KernelBookLoader(train_data, random=True)
    testloader = KernelBookLoader(test_data, random=False)
    
    return trainloader, testloader


def get_dataloaders(dataset_name: str) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load or a path to the dataset file
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported or file not found
    """
    if dataset_name.lower() == 'kernelbook':
        # Default path for the Kernel Book dataset
        dataset_path = os.environ.get('KERNELBOOK_PATH', 'dataset_permissive.json')
        return build_kernelbook_dataloaders(dataset_path)
    
    elif os.path.exists(dataset_name):
        # Assume dataset_name is a direct path to the dataset file
        return build_kernelbook_dataloaders(dataset_name)
    
    else:
        raise ValueError(f"Dataset {dataset_name} not found or not supported.")


if __name__ == "__main__":
    # Example usage
    trainloader, testloader = get_dataloaders('kernelbook')
    
    # Get one example to verify loading works
    print("Fetching one example from training data...")
    input_data, reference_solution = next(trainloader)
    
    print(f"PyTorch code snippet: {input_data['pytorch_code'][:200]}...")
    print(f"Triton code snippet: {reference_solution['triton_code'][:200]}...")
