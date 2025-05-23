"""
Adapter for KernelBook dataset to create natural language query -> Triton kernel pairs
"""

import json
import pandas as pd
import os
from typing import List, Dict, Any


class KernelBookAdapter:
    """
    Adapter for converting KernelBook dataset to natural language queries paired with Triton kernels.
    """
    
    def __init__(self, dataset_path: str, output_path: str):
        """
        Initialize with dataset paths.
        
        Args:
            dataset_path: Path to original KernelBook dataset (JSON or Parquet)
            output_path: Path to save the processed dataset
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load the dataset based on file extension.
        
        Returns:
            List of dataset entries
        """
        if self.dataset_path.endswith('.json'):
            with open(self.dataset_path, 'r') as f:
                return json.load(f)
        elif self.dataset_path.endswith('.parquet'):
            return pd.read_parquet(self.dataset_path).to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {self.dataset_path}")
    
    def generate_query(self, pytorch_code: str, triton_kernel: str) -> str:
        """
        Generate a natural language query for a kernel.
        Uses function name extraction as a simple approach.
        
        Args:
            pytorch_code: PyTorch implementation (for context)
            triton_kernel: Triton kernel implementation
            
        Returns:
            A natural language query
        """
        # Extract function name as fallback
        function_name = "implements this functionality"
        
        # Look for function name in triton kernel
        for line in triton_kernel.split("\n"):
            if "def " in line and "(" in line:
                func_name = line.split("def ")[1].split("(")[0].strip()
                if func_name:
                    function_name = f"implements {func_name.replace('_', ' ')}"
                break
        
        # Simple template
        return f"Write a Triton kernel that {function_name}"
    
    def transform_dataset(self) -> List[Dict[str, Any]]:
        """
        Transform the dataset to include natural language queries.
        
        Returns:
            Transformed dataset with NL queries and Triton kernels
        """
        original_data = self.load_dataset()
        transformed_data = []
        
        for entry in original_data:
            # Extract Triton kernels
            triton_kernels = []
            for key, value in entry.items():
                # Direct string fields
                if isinstance(value, str) and "triton.language" in value:
                    triton_kernels.append(value)
                # Nested dictionaries
                elif isinstance(value, dict):
                    for subvalue in value.values():
                        if isinstance(subvalue, str) and "triton.language" in subvalue:
                            triton_kernels.append(subvalue)
                # Lists of dictionaries
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            for subvalue in item.values():
                                if isinstance(subvalue, str) and "triton.language" in subvalue:
                                    triton_kernels.append(subvalue)
            
            # Find PyTorch code
            pytorch_code = None
            for key, value in entry.items():
                if isinstance(value, str) and ("import torch" in value or "torch.nn" in value):
                    pytorch_code = value
                    break
            
            if not pytorch_code or not triton_kernels:
                continue
            
            # Generate query for each Triton kernel
            for triton_kernel in triton_kernels:
                query = self.generate_query(pytorch_code, triton_kernel)
                
                transformed_entry = {
                    "query": query,
                    "pytorch_code": pytorch_code,
                    "triton_kernel": triton_kernel,
                    "original_repo": entry.get("repo_name", ""),
                    "stars": entry.get("stars", 0),
                    "license": entry.get("licenses", []),
                }
                transformed_data.append(transformed_entry)
        
        return transformed_data
    
    def save_transformed_dataset(self, transformed_data: List[Dict[str, Any]], format: str = "json") -> None:
        """
        Save the transformed dataset.
        
        Args:
            transformed_data: Transformed dataset
            format: Output format ("json" or "parquet")
        """
        if format == "json" or format == "both":
            json_path = self.output_path.replace(".parquet", "") + ".json"
            with open(json_path, 'w') as f:
                json.dump(transformed_data, f, indent=2)
                
        if format == "parquet" or format == "both":
            parquet_path = self.output_path.replace(".json", "") + ".parquet"
            df = pd.DataFrame(transformed_data)
            df.to_parquet(parquet_path, index=False)
    
    def process(self, format: str = "json") -> List[Dict[str, Any]]:
        """
        Process the dataset: transform and save.
        
        Args:
            format: Output format ("json", "parquet", or "both")
            
        Returns:
            Transformed dataset
        """
        transformed_data = self.transform_dataset()
        self.save_transformed_dataset(transformed_data, format)
        print(f"Transformed {len(transformed_data)} entries from KernelBook dataset")
        return transformed_data


# Example usage
if __name__ == "__main__":
    adapter = KernelBookAdapter(
        dataset_path="dataset_permissive.json",
        output_path="kernelbook_nl_queries.json"
    )
    transformed_data = adapter.process()
    
    # Display sample
    if transformed_data:
        print("\nSample entry:")
        sample = transformed_data[0]
        print(f"Query: {sample['query']}")
        print(f"Triton kernel (first 200 chars): {sample['triton_kernel'][:200]}...")
