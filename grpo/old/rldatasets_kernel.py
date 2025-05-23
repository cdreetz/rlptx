"""
Data loaders for kernel programming tasks using the KernelBench dataset.
"""

import os
import random
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod
from typing import Tuple, Any, List, Dict



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
You are an expert CUDA/Triton kernel programmer. Your task is to write high-performance, optimized kernels based on the given specifications. Focus on writing efficient, correct code that will compile and run with optimal performance.

For CUDA kernels, prioritize:
- Proper use of shared memory
- Coalesced memory access
- Thread and block level parallelism
- Memory bank conflict avoidance
- Loop unrolling

For Triton kernels, prioritize:
- Effective use of Triton's programming model
- Proper tiling strategies
- Memory hierarchy optimization
- Auto-tuning annotations
"""


class KernelBenchLoader(DataLoader):
    """
    A loader class that provides iteration over kernel programming problems.
    
    This class implements both sequential and random access to kernel programming
    tasks through standard Python iterator protocols.
    
    Attributes:
        tasks (List[Dict]): List of kernel programming tasks
        solutions (List[Dict]): List of corresponding reference solutions
        random (bool): If True, returns problems randomly; if False, returns sequentially
        current_index (int): Current position in the lists for sequential access
    """
    
    def __init__(self, tasks: List[Dict], solutions: List[Dict], random: bool = False) -> None:
        super().__init__(random)
        self.tasks = tasks
        self.solutions = solutions
        self.system_prompt = SYSTEM_PROMPT
        
    def __len__(self) -> int:
        return len(self.tasks)
        
    def __iter__(self) -> 'KernelBenchLoader':
        return self
        
    def __next__(self) -> tuple[str, Dict]:
        if self.current_index >= len(self.tasks):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.tasks) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        return self.tasks[idx]['description'], self.solutions[idx]

    def reset(self):
        self.current_index = 0 


def build_cuda_matmul_dataloaders(dataset_path: str = "kernelbench_data") -> Tuple[KernelBenchLoader, KernelBenchLoader]:
    """
    Build train and test data loaders for CUDA matrix multiplication kernels.
    
    Args:
        dataset_path: Path to the KernelBench dataset
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Load the dataset from local path
    # In a real implementation, this would load from the KernelBench dataset
    # For now, we'll simulate with placeholder data
    
    tasks = []
    solutions = []
    
    # Example task structure
    task_template = {
        "id": "",
        "type": "matmul",
        "description": "",
        "input_shapes": [],
        "input_dtypes": [],
        "output_shape": [],
        "output_dtype": ""
    }
    
    solution_template = {
        "id": "",
        "code": "",
        "performance_metrics": {
            "execution_time": 0.0,
            "memory_usage": 0.0,
            "throughput": 0.0
        }
    }
    
    # Create sample tasks
    for i in range(100):
        # Vary matrix sizes
        M, N, K = random.choice([(128, 128, 128), (256, 256, 256), 
                                  (512, 512, 512), (1024, 1024, 1024)])
        
        task = task_template.copy()
        task["id"] = f"matmul_{i}"
        task["description"] = f"""
Write a CUDA kernel for efficient matrix multiplication C = A * B where:
- A is a matrix of shape ({M}, {K})
- B is a matrix of shape ({K}, {N})
- C is the output matrix of shape ({M}, {N})
- All matrices are in row-major layout
- Data type is float

Optimize for:
1. Memory coalescing
2. Shared memory usage
3. Thread block optimization
4. Handling the matrix boundaries correctly

Include a kernel launch configuration that maximizes performance.
"""
        task["input_shapes"] = [[M, K], [K, N]]
        task["input_dtypes"] = ["float32", "float32"]
        task["output_shape"] = [M, N]
        task["output_dtype"] = "float32"
        
        solution = solution_template.copy()
        solution["id"] = f"matmul_{i}"
        solution["code"] = """
__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // Example CUDA kernel (simplified)
    const int TILE_SIZE = 32;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"""
        solution["performance_metrics"] = {
            "execution_time": random.uniform(0.5, 10.0),
            "memory_usage": random.uniform(100, 1000),
            "throughput": random.uniform(1, 100)
        }
        
        tasks.append(task)
        solutions.append(solution)
    
    # Randomly split into train/test sets
    total_samples = len(tasks)
    test_size = int(total_samples * 0.2)  # 20% for test set
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_samples), test_size)
    test_indices_set = set(test_indices)
    
    # Split using indices
    test_tasks = [tasks[i] for i in test_indices]
    test_solutions = [solutions[i] for i in test_indices]
    train_tasks = [tasks[i] for i in range(total_samples) if i not in test_indices_set]
    train_solutions = [solutions[i] for i in range(total_samples) if i not in test_indices_set]

    # Setup data loaders 
    trainloader = KernelBenchLoader(train_tasks, train_solutions, random=True)
    testloader = KernelBenchLoader(test_tasks, test_solutions, random=False)
    
    return trainloader, testloader


def build_triton_matmul_dataloaders(dataset_path: str = "kernelbench_data") -> Tuple[KernelBenchLoader, KernelBenchLoader]:
    """
    Build train and test data loaders for Triton matrix multiplication kernels.
    
    Args:
        dataset_path: Path to the KernelBench dataset
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Similar to the CUDA loader but with Triton-specific tasks
    tasks = []
    solutions = []
    
    # Example task structure
    task_template = {
        "id": "",
        "type": "matmul",
        "description": "",
        "input_shapes": [],
        "input_dtypes": [],
        "output_shape": [],
        "output_dtype": ""
    }
    
    solution_template = {
        "id": "",
        "code": "",
        "performance_metrics": {
            "execution_time": 0.0,
            "memory_usage": 0.0,
            "throughput": 0.0
        }
    }
    
    # Create sample tasks
    for i in range(100):
        # Vary matrix sizes
        M, N, K = random.choice([(128, 128, 128), (256, 256, 256), 
                                  (512, 512, 512), (1024, 1024, 1024)])
        
        task = task_template.copy()
        task["id"] = f"triton_matmul_{i}"
        task["description"] = f"""
Write a Triton kernel for efficient matrix multiplication C = A * B where:
- A is a matrix of shape ({M}, {K})
- B is a matrix of shape ({K}, {N})
- C is the output matrix of shape ({M}, {N})
- All matrices are in row-major layout
- Data type is float32

Optimize for:
1. Effective tiling strategy
2. Memory hierarchy optimization
3. Auto-tuning parameters
4. Handling the matrix boundaries correctly

Use Triton's block semantics and include proper auto-tuning configurations.
"""
        task["input_shapes"] = [[M, K], [K, N]]
        task["input_dtypes"] = ["float32", "float32"]
        task["output_shape"] = [M, N]
        task["output_dtype"] = "float32"
        
        solution = solution_template.copy()
        solution["id"] = f"triton_matmul_{i}"
        solution["code"] = """
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The strides of the matrices
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block start indices
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate to compute a block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load matrices from device memory
        a = tl.load(a_ptr + (offs_am[:, None] * stride_am + (k * BLOCK_SIZE_K + offs_k[None, :]) * stride_ak), 
                   mask=(offs_am[:, None] < M) & ((k * BLOCK_SIZE_K + offs_k[None, :]) < K))
        b = tl.load(b_ptr + ((k * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn),
                   mask=((k * BLOCK_SIZE_K + offs_k[:, None]) < K) & (offs_bn[None, :] < N))
        
        # Matrix multiplication
        acc += tl.dot(a, b)
    
    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c = acc.to(tl.float32)
    
    # Write result
    tl.store(c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn),
             c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
"""
