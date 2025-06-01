import re
import json
import ast
from typing import List, Dict, Tuple
from triton_sandbox import TritonSandbox


class DataQualityFilter:
    def __init__(self):
        self.required_patterns = [
            r'@triton\.jit',
            r'<kernel>',
            r'</kernel>',
            r'<launch_fn>',
            r'</launch_fn>'
        ]
        
    def extract_code_blocks(self, text: str) -> Tuple[str, str]:
        """Extract kernel and launch function from XML tags"""
        kernel_match = re.search(r'<kernel>(.*?)</kernel>', text, re.DOTALL)
        launch_match = re.search(r'<launch_fn>(.*?)</launch_fn>', text, re.DOTALL)
        
        kernel_code = kernel_match.group(1).strip() if kernel_match else ""
        launch_code = launch_match.group(1).strip() if launch_match else ""
        
        return kernel_code, launch_code
    
    def has_required_structure(self, text: str) -> bool:
        """Check if response has required XML structure"""
        return all(re.search(pattern, text) for pattern in self.required_patterns)
    
    def is_valid_python(self, code: str) -> bool:
        """Check if code is syntactically valid Python"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def has_triton_imports(self, code: str) -> bool:
        """Check if code has necessary triton imports"""
        triton_patterns = [
            r'import triton',
            r'import triton\.language',
            r'from triton'
        ]
        return any(re.search(pattern, code) for pattern in triton_patterns)
    
    def validate_example(self, example: Dict) -> Dict:
        """Validate a single training example"""
        response = example.get("response", "")
        
        # Basic structure check
        has_structure = self.has_required_structure(response)
        
        # Extract code blocks
        kernel_code, launch_code = self.extract_code_blocks(response)
        
        # Syntax validation
        kernel_valid = self.is_valid_python(kernel_code) if kernel_code else False
        launch_valid = self.is_valid_python(launch_code) if launch_code else False
        
        # Import check
        has_imports = self.has_triton_imports(launch_code) if launch_code else False
        
        validation_result = {
            "has_structure": has_structure,
            "kernel_valid": kernel_valid,
            "launch_valid": launch_valid,
            "has_imports": has_imports,
            "kernel_code": kernel_code,
            "launch_code": launch_code,
            "overall_valid": has_structure and kernel_valid and launch_valid
        }
        
        return {**example, "validation": validation_result}
    
    def filter_dataset(self, dataset_path: str, output_path: str, 
                      min_quality_score: float = 0.7) -> List[Dict]:
        """Filter dataset by quality metrics"""
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = [json.loads(line.strip()) for line in f]
        
        print(f"Loaded {len(dataset)} examples")
        
        # Validate each example
        validated_dataset = []
        for i, example in enumerate(dataset):
            print(f"Validating {i+1}/{len(dataset)}")
            validated_example = self.validate_example(example)
            validated_dataset.append(validated_example)
        
        # Filter by quality
        filtered_dataset = []
        for example in validated_dataset:
            validation = example["validation"]
            
            # Calculate quality score
            quality_score = sum([
                validation["has_structure"],
                validation["kernel_valid"], 
                validation["launch_valid"],
                validation["has_imports"]
            ]) / 4.0
            
            if quality_score >= min_quality_score:
                example["quality_score"] = quality_score
                filtered_dataset.append(example)
        
        print(f"Filtered dataset: {len(filtered_dataset)}/{len(dataset)} examples passed")
        
        # Save filtered dataset
        with open(output_path, 'w') as f:
            for example in filtered_dataset:
                f.write(json.dumps(example) + '\n')
        
        return filtered_dataset
    
    def test_executable_examples(self, dataset_path: str, output_path: str,
                                max_test_count: int = 50) -> List[Dict]:
        """Test examples for executability using sandbox"""
        
        # Load filtered dataset
        with open(dataset_path, 'r') as f:
            dataset = [json.loads(line.strip()) for line in f]
        
        print(f"Testing executability for up to {max_test_count} examples")
        
        executable_dataset = []
        test_count = 0
        
        with TritonSandbox() as sandbox:
            for i, example in enumerate(dataset):
                if test_count >= max_test_count:
                    break
                    
                validation = example["validation"]
                if not validation["overall_valid"]:
                    continue
                
                print(f"Testing example {test_count+1}/{max_test_count}: {example['id']}")
                
                # Combine kernel and launch code
                full_code = validation["launch_code"]
                if validation["kernel_code"]:
                    full_code = validation["kernel_code"] + "\n\n" + validation["launch_code"]
                
                try:
                    result = sandbox.test_kernel(full_code, verbose=False)
                    example["executable"] = result["success"]
                    example["execution_result"] = {
                        "stdout": result["stdout"][:500],  # Truncate for storage
                        "stderr": result["stderr"][:500]
                    }
                    
                    if result["success"]:
                        executable_dataset.append(example)
                        print(f"  ✓ Executable")
                    else:
                        print(f"  ✗ Not executable")
                        
                except Exception as e:
                    print(f"  ✗ Error testing: {e}")
                    example["executable"] = False
                    example["execution_result"] = {"error": str(e)}
                
                test_count += 1
        
        print(f"Executable examples: {len(executable_dataset)}/{test_count} tested")
        
        # Save executable dataset
        with open(output_path, 'w') as f:
            for example in executable_dataset:
                f.write(json.dumps(example) + '\n')
        
        return executable_dataset


if __name__ == "__main__":
    filter_tool = DataQualityFilter()
    
    # Step 1: Basic quality filtering
    print("Step 1: Basic quality filtering...")
    filtered_data = filter_tool.filter_dataset(
        "sft_training_data.jsonl", 
        "sft_filtered_data.jsonl"
    )
    
    # Step 2: Test executability (sample)
    #print("\nStep 2: Testing executability...")
    #executable_data = filter_tool.test_executable_examples(
    #    "sft_filtered_data.jsonl",
    #    "sft_executable_data.jsonl",
    #    max_test_count=50
    #)
    
    print(f"\nFinal dataset stats:")
    #print(f"- Executable examples: {len(executable_data)}")
    print(f"- Ready for SFT training!")