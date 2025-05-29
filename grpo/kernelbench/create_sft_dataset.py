#!/usr/bin/env python3
"""
Generate SFT dataset by having API model write Triton kernels and verify they work
"""

import os
import json
import time
import random
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
from evaluate_template import evaluate
from evaluator import extract_kernel_methods

class APIKernelGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("LLAMA_API_KEY"),
            base_url=os.getenv("LLAMA_BASE_URL"),
        )
    
    def generate_kernel(self, torch_code):
        """Generate Triton kernel from torch code using API"""
        
        prompt = f"""Convert this PyTorch model to a Triton kernel implementation.

Your response must contain EXACTLY two functions:
1. A function named `triton_kernel` decorated with @triton.jit
2. A function named `triton_wrapper` that calls the kernel

Here's the exact pattern to follow:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def triton_kernel(
    input_ptr,
    output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    output = x  # MODIFY THIS LINE FOR YOUR OPERATION
    
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_wrapper(input_tensor):
    output = torch.empty_like(input_tensor)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    triton_kernel[grid](input_tensor, output, n_elements, BLOCK_SIZE)
    return output
```

Now convert this PyTorch code:

{torch_code}

Only respond with the Python code. No explanations."""

        try:
            response = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
                
        except Exception as e:
            print(f"API call failed: {e}")
            return None

def load_kernelbench_examples(num_examples=200, level_filter=1):
    """Load examples from KernelBench dataset"""
    
    print(f"Loading KernelBench examples (level {level_filter})...")
    data = load_dataset('ScalingIntelligence/KernelBench', 'default')["level_1"]
    
    examples = []
    for i in range(min(num_examples, len(data))):
        if data[i]['level'] == level_filter:
            examples.append({
                'torch_code': data[i]['code'],
                'level': data[i]['level']
            })
    
    print(f"Loaded {len(examples)} examples")
    return examples

def verify_kernel(torch_code, triton_response):
    """Verify that the generated Triton kernel works correctly"""
    
    try:
        # Extract kernel and wrapper methods
        kernel_code, wrapper_code = extract_kernel_methods(triton_response)
        
        if not kernel_code or not wrapper_code:
            return False, "Could not extract triton_kernel and triton_wrapper methods"
        
        # Evaluate the kernel
        result = evaluate(kernel_code, wrapper_code, torch_code)
        
        # Check if it compiles and is correct
        if result['compiles'] and result['correct']:
            return True, f"Success! Speedup: {result.get('speedup', 'N/A')}"
        elif result['compiles']:
            return False, f"Compiles but incorrect: {result.get('error', 'Unknown error')}"
        else:
            return False, f"Compilation failed: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return False, f"Verification error: {str(e)}"

def create_sft_dataset(
    num_examples=100,
    level_filter=1,
    output_file="sft_dataset.json",
    attempts_per_example=3
):
    """Create SFT dataset by generating and verifying Triton kernels"""
    
    # Load KernelBench examples
    torch_examples = load_kernelbench_examples(num_examples * 2, level_filter)  # Load extra in case some fail
    random.shuffle(torch_examples)
    
    # Initialize API generator
    generator = APIKernelGenerator()
    
    sft_dataset = []
    successful = 0
    
    for i, example in enumerate(tqdm(torch_examples)):
        if successful >= num_examples:
            break
            
        torch_code = example['torch_code']
        
        # Try multiple times per example
        for attempt in range(attempts_per_example):
            print(f"\nExample {i+1}, Attempt {attempt+1}")
            print(f"Torch code preview: {torch_code[:100]}...")
            
            # Generate Triton kernel
            triton_response = generator.generate_kernel(torch_code)
            
            if not triton_response:
                print("API call failed")
                continue
            
            # Verify it works
            works, message = verify_kernel(torch_code, triton_response)
            print(f"Verification: {message}")
            
            if works:
                # Add to dataset
                sft_dataset.append({
                    'torch_code': torch_code,
                    'triton_code': triton_response,
                    'level': example['level'],
                    'verification_message': message
                })
                
                successful += 1
                print(f"✓ Success! Dataset now has {successful}/{num_examples} examples")
                break  # Move to next example
            else:
                print(f"✗ Failed: {message}")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save dataset
    print(f"\nSaving {len(sft_dataset)} examples to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(sft_dataset, f, indent=2)
    
    print(f"SFT dataset creation complete!")
    print(f"Successfully generated {len(sft_dataset)} working examples")
    
    return sft_dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create SFT dataset")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--level_filter", type=int, default=1, help="KernelBench level to use")
    parser.add_argument("--output_file", type=str, default="sft_dataset.json", help="Output file")
    parser.add_argument("--attempts_per_example", type=int, default=3, help="Max attempts per example")
    
    args = parser.parse_args()
    
    create_sft_dataset(
        num_examples=args.num_examples,
        level_filter=args.level_filter,
        output_file=args.output_file,
        attempts_per_example=args.attempts_per_example
    )