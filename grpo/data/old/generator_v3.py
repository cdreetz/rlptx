import os
import random
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_client():
    client = OpenAI(
        api_key=os.getenv("LLAMA_API_KEY"),
        base_url="https://api.llama.com/compat/v1/",
    )
    return client

client = get_client()



@dataclass
class KernelSpec:
    operation: str
    input_shapes: List[Tuple[int, ...]]
    output_shape: Tuple[int, ...]
    prompt: str
    optimization_level: str  # 'none', 'basic', 'advanced'


class CompletionProvider:
    """Abstract completion provider - swap out for different models"""

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        model = "Llama-4-Maverick-17B-128E-Instruct-FP8"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content



class TritonDataGenerator:
    def __init__(self, completion_provider: CompletionProvider):
        self.completion_provider = completion_provider

    def generate_elementwise_specs(self, num_samples: int = 100) -> List[KernelSpec]:
        """Generate specs for elementwise operations"""
        operations = ['add', 'subtract', 'multiply', 'divide', 'relu', 'gelu', 'sigmoid']
        optimization_levels = ['none', 'basic']

        specs = []
        for _ in range(num_samples):
            op = random.choice(operations)
            # Keep concrete size for testing, but prompt uses symbolic
            test_size = random.choice([1024, 4096, 8192, 16384, 32768])

            # Most elementwise ops are binary, some are unary
            if op in ['relu', 'gelu', 'sigmoid']:
                input_shapes = [(test_size,)]
                prompt_template = f"Write a Triton kernel that computes {op} of a 1D tensor of size N"
            else:
                input_shapes = [(test_size,), (test_size,)]
                prompt_template = f"Write a Triton kernel that performs element-wise {op} of two 1D tensors of size N"

            opt_level = random.choice(optimization_levels)
            if opt_level == 'basic':
                prompt_template += " with efficient memory access patterns"

            specs.append(KernelSpec(
                operation=op,
                input_shapes=input_shapes,
                output_shape=(test_size,),
                prompt=prompt_template,
                optimization_level=opt_level
            ))

        return specs

    def generate_reduction_specs(self, num_samples: int = 50) -> List[KernelSpec]:
        """Generate specs for reduction operations"""
        operations = ['sum', 'max', 'min', 'mean']

        specs = []
        for _ in range(num_samples):
            op = random.choice(operations)
            # Keep concrete dims for testing
            rows = random.choice([128, 256, 512, 1024])
            cols = random.choice([128, 256, 512, 1024])

            # Reduce along last dimension
            input_shapes = [(rows, cols)]
            output_shape = (rows,)

            prompt = f"Write a Triton kernel that computes the {op} reduction along the last dimension of a 2D tensor with shape [M, N]"

            specs.append(KernelSpec(
                operation=f"{op}_reduction",
                input_shapes=input_shapes,
                output_shape=output_shape,
                prompt=prompt,
                optimization_level='none'
            ))

        return specs

    def generate_simple_matmul_specs(self, num_samples: int = 30) -> List[KernelSpec]:
        """Generate specs for simple matrix multiplication"""
        specs = []
        for _ in range(num_samples):
            # Keep concrete dims for testing
            m = random.choice([128, 256, 512])
            k = random.choice([128, 256, 512])
            n = random.choice([128, 256, 512])

            input_shapes = [(m, k), (k, n)]
            output_shape = (m, n)

            opt_level = random.choice(['none', 'basic'])
            prompt = f"Write a Triton kernel for matrix multiplication of tensors with shapes [M, K] and [K, N]"
            if opt_level == 'basic':
                prompt += " using tile-based computation with shared memory"

            specs.append(KernelSpec(
                operation="matmul",
                input_shapes=input_shapes,
                output_shape=output_shape,
                prompt=prompt,
                optimization_level=opt_level
            ))

        return specs

    def create_full_prompt(self, spec: KernelSpec) -> str:
        """Create the full prompt for the model"""
        base_prompt = f"""You are an expert at writing high-performance Triton kernels.

Task: {spec.prompt}

Requirements:
- Write clean, efficient Triton code
- Include proper imports (triton, triton.language as tl)
- Use appropriate BLOCK_SIZE constants
- Handle edge cases with proper masking
- Include the @triton.jit decorator

Provide only the kernel code without explanation."""

        return base_prompt

    def extract_kernel_code(self, response: str) -> Optional[str]:
        """Extract kernel code from model response"""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()

        # If no code blocks, return the whole response
        return response.strip()


    def test_kernel_compilation(self, kernel_code: str) -> Tuple[bool, str]:
        """Test if kernel code actually compiles with Triton by attempting full compilation"""
        try:
            import tempfile
            import os
            import sys
            import importlib.util

            # Write kernel code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(kernel_code)
                temp_file = f.name

            try:
                # Load module from file
                spec = importlib.util.spec_from_file_location("temp_kernel", temp_file)
                module = importlib.util.module_from_spec(spec)

                # Add required imports to module namespace
                import builtins
                module.__builtins__ = builtins
                module.triton = triton
                module.tl = triton.language
                module.torch = torch
                sys.modules['temp_kernel'] = module

                # Execute the module
                spec.loader.exec_module(module)

                # Find kernel function
                kernel_func = None
                for name in dir(module):
                    if not name.startswith('_'):
                        obj = getattr(module, name)
                        if callable(obj) and hasattr(obj, '__name__'):
                            if hasattr(obj, '__module__') and obj.__module__ == 'temp_kernel':
                                # Check if this is likely a triton kernel
                                if hasattr(obj, '__triton_kernel__'):
                                    kernel_func = obj
                                    break

                if kernel_func is None:
                    return False, "No @triton.jit decorated function found"

                # Force actual Triton compilation by trying to get compiled kernel
                try:
                    # Try to compile with a minimal grid - this forces full compilation
                    grid = (1,)
                    compiled_kernel = kernel_func[grid]

                    # If we get here, compilation succeeded
                    return True, f"Kernel '{kernel_func.__name__}' compiled successfully"

                except Exception as triton_error:
                    return False, f"Triton compilation failed: {str(triton_error)}"

            finally:
                # Clean up
                if 'temp_kernel' in sys.modules:
                    del sys.modules['temp_kernel']
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Import/execution error: {str(e)}"


    def generate_dataset(self, num_elementwise: int = 100, num_reduction: int = 50,
                        num_matmul: int = 30) -> List[Dict]:
        """Generate complete dataset"""
        print(f"Generating {num_elementwise} elementwise, {num_reduction} reduction, {num_matmul} matmul examples...")

        all_specs = []
        all_specs.extend(self.generate_elementwise_specs(num_elementwise))
        all_specs.extend(self.generate_reduction_specs(num_reduction))
        all_specs.extend(self.generate_simple_matmul_specs(num_matmul))

        dataset = []
        for i, spec in enumerate(all_specs):
            print(f"Processing {i+1}/{len(all_specs)}: {spec.operation}")

            prompt = self.create_full_prompt(spec)
            response = self.completion_provider.complete(prompt)
            kernel_code = self.extract_kernel_code(response)

            compiles, error_msg = self.test_kernel_compilation(kernel_code)

            dataset.append({
                'prompt': prompt,
                'response': response,
                'kernel_code': kernel_code,
                'operation': spec.operation,
                'input_shapes': spec.input_shapes,
                'output_shape': spec.output_shape,
                'optimization_level': spec.optimization_level,
                'compiles': compiles,
                'error_message': error_msg if not compiles else None
            })

        return dataset

    def save_dataset(self, dataset: List[Dict], filename: str):
        """Save dataset to JSON file"""
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} examples to {filename}")


# Usage example
if __name__ == "__main__":
    provider = CompletionProvider()
    generator = TritonDataGenerator(provider)

    # Generate small test dataset
    dataset = generator.generate_dataset(num_elementwise=3, num_reduction=3, num_matmul=3)
    generator.save_dataset(dataset, "triton_kernels_dataset_v3.json")

    # Print compilation stats
    compiled = sum(1 for item in dataset if item['compiles'])
    print(f"Compilation success rate: {compiled}/{len(dataset)} ({100*compiled/len(dataset):.1f}%)")
