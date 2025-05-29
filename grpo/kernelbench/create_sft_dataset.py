#!/usr/bin/env python3
"""
Generate SFT dataset by having API model write Triton kernels - NO VERIFICATION
"""

import os
import json
import time
import random
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class APIKernelGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("LLAMA_API_KEY"),
            base_url=os.getenv("LLAMA_BASE_URL"),
        )
    
    def generate_kernel(self, torch_code):
        """Generate Triton kernel from torch code using API"""
        
        prompt = f"""Convert this PyTorch model to a Triton kernel implementation.

CRITICAL: Respond with ONLY Python code. No explanations, no markdown, no text.

Your response must be valid Python code that contains EXACTLY these two functions:
1. A function named `triton_kernel` decorated with @triton.jit
2. A function named `triton_wrapper` that calls the kernel

Required pattern:

import torch
import triton
import triton.language as tl

@triton.jit
def triton_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    # MODIFY THE COMPUTATION HERE
    output = x
    
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_wrapper(input_tensor):
    output = torch.empty_like(input_tensor)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    triton_kernel[grid](input_tensor, output, n_elements, BLOCK_SIZE)
    return output

PyTorch code to convert:
{torch_code}

Respond with only the Python code."""

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

def create_sft_dataset(
    num_examples=100,
    level_filter=1,
    output_file="sft_dataset.json"
):
    """Create SFT dataset by generating Triton kernels WITHOUT VERIFICATION"""
    
    # Load KernelBench examples
    torch_examples = load_kernelbench_examples(num_examples, level_filter)
    random.shuffle(torch_examples)
    
    # Initialize API generator
    generator = APIKernelGenerator()
    
    sft_dataset = []
    
    print(f"Generating SFT dataset with Llama API (NO VERIFICATION)...")
    
    for i, example in enumerate(tqdm(torch_examples[:num_examples])):
        torch_code = example['torch_code']
        
        print(f"\nExample {i+1}")
        print(f"Torch code preview: {torch_code[:100]}...")
        
        # Generate Triton kernel
        triton_response = generator.generate_kernel(torch_code)
        
        if not triton_response:
            print("API call failed, skipping...")
            continue
        
        # Just save it, no verification
        sft_dataset.append({
            'torch_code': torch_code,
            'triton_code': triton_response,
            'level': example['level'],
            'api_provider': 'llama'
        })
        
        print(f"✓ Generated! Dataset now has {len(sft_dataset)} examples")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save dataset
    print(f"\nSaving {len(sft_dataset)} examples to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(sft_dataset, f, indent=2)
    
    print(f"SFT dataset creation complete!")
    print(f"Generated {len(sft_dataset)} examples (unverified)")
    
    return sft_dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create SFT dataset")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--level_filter", type=int, default=1, help="KernelBench level to use")
    parser.add_argument("--output_file", type=str, default="sft_dataset.json", help="Output file")
    
    args = parser.parse_args()
    
    create_sft_dataset(
        num_examples=args.num_examples,
        level_filter=args.level_filter,
        output_file=args.output_file
    )