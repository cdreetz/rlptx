#!/usr/bin/env python3
import os
import subprocess
import time
import json
import re
import argparse
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
DEFAULT_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_OUTPUT_DIR = "generated_ptx"

class PtxBenchmarker:
    def __init__(self, api_key=None, model=DEFAULT_MODEL, output_dir=DEFAULT_OUTPUT_DIR):
        """Initialize the PTX benchmarker with OpenAI API key and configuration."""
        if api_key is None:
            api_key = os.environ.get("LLAMA_API_KEY")
            if api_key is None:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=api_key,base_url="https://api.llama.com/compat/v1/",
)
        self.model = model
        self.output_dir = output_dir
        self.best_performance = float('inf')  # Lower is better for time
        self.best_ptx_file = None
        self.results_history = []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate_ptx(self, prompt, temperature=DEFAULT_TEMPERATURE):
        """Generate PTX code using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in CUDA PTX assembly programming. Generate efficient, correct PTX code for matrix operations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating PTX: {e}")
            return None
    
    def extract_ptx_code(self, response_text):
        """Extract PTX code from the AI response."""
        # First try to get code from markdown code blocks
        code_block_pattern = r"```(?:ptx)?\s*([\s\S]+?)```"
        matches = re.findall(code_block_pattern, response_text)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks found, look for sections that look like PTX
        ptx_indicators = [".version", ".target", ".visible .entry", ".reg", "ld.param"]
        lines = response_text.split('\n')
        start_idx = None
        
        for i, line in enumerate(lines):
            if any(indicator in line for indicator in ptx_indicators) and start_idx is None:
                start_idx = i
                break
        
        if start_idx is not None:
            # Extract from the first indicator to the end
            return '\n'.join(lines[start_idx:]).strip()
        
        # If we can't identify PTX code, return the whole response
        return response_text
    
    def save_ptx_to_file(self, ptx_code, filename):
        """Save PTX code to a file."""
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w') as f:
            f.write(ptx_code)
        return file_path
    
    def benchmark_ptx(self, ptx_file_path):
        """Run the benchmark on a PTX file and return the results."""
        try:
            result = subprocess.run(
                ["./benchmark_gemm", ptx_file_path],
                capture_output=True,
                text=True,
                check=False
            )
            
            # Parse the output to extract performance metrics
            output = result.stdout
            
            # Extract average time
            time_match = re.search(r"Average time per run = ([0-9.]+) ms", output)
            avg_time = float(time_match.group(1)) if time_match else float('inf')
            
            # Check verification
            verification_passed = "Verification: PASSED" in output
            
            benchmark_result = {
                "ptx_file": ptx_file_path,
                "avg_time_ms": avg_time,
                "verification_passed": verification_passed,
                "output": output,
                "error": result.stderr if result.stderr else None
            }
            
            self.results_history.append(benchmark_result)
            
            # Update best performance if this is better
            if verification_passed and avg_time < self.best_performance:
                self.best_performance = avg_time
                self.best_ptx_file = ptx_file_path
            
            return benchmark_result
        except Exception as e:
            print(f"Error benchmarking PTX file {ptx_file_path}: {e}")
            return {
                "ptx_file": ptx_file_path,
                "avg_time_ms": float('inf'),
                "verification_passed": False,
                "error": str(e)
            }
    
    def generate_feedback_prompt(self, ptx_code, benchmark_result, original_prompt):
        """Generate a prompt to improve the PTX code based on benchmark results."""
        feedback = f"""
I generated the following PTX code for a 4x4 GEMM operation:

```ptx
{ptx_code}
```

When benchmarked, it produced the following results:
- Average execution time: {benchmark_result['avg_time_ms']:.6f} ms
- Verification: {"PASSED" if benchmark_result['verification_passed'] else "FAILED"}

{"Here's the complete benchmark output:" if benchmark_result.get('output') else ""}
{benchmark_result.get('output', '')}

{"Here's the error output:" if benchmark_result.get('error') else ""}
{benchmark_result.get('error', '')}

Please analyze this PTX code and suggest improvements to make it more efficient.
If verification failed, please fix the correctness issues.

Original task: {original_prompt}

Your response should include:
1. Analysis of the current code's performance or issues
2. Complete improved PTX code (not just changes)
3. Explanation of why your changes should improve performance

Important notes:
- The matrices are 4x4 and stored in row-major format
- The entry point must be named "gemm4x4"
- The PTX must take three parameters: A, B, and C (input matrices A, B, and output matrix C)
- The PTX should compute C = A * B
- Focus on efficient register usage, instruction scheduling, and memory access patterns
"""
        return feedback
    
    def run_optimization_loop(self, initial_prompt, max_iterations=DEFAULT_MAX_ITERATIONS, temperature=DEFAULT_TEMPERATURE):
        """Run an iterative optimization loop to improve PTX performance."""
        print(f"Starting optimization loop with max {max_iterations} iterations")
        
        # Generate initial PTX code
        current_prompt = initial_prompt
        timestamp = int(time.time())
        iteration = 0
        
        while iteration < max_iterations:
            print(f"\nIteration {iteration + 1}/{max_iterations}")
            
            # Generate PTX code
            response = self.generate_ptx(current_prompt, temperature)
            if not response:
                print("Failed to generate PTX code. Trying again with different temperature.")
                temperature = random.uniform(0.5, 0.9)  # Try with different temperature
                continue
            
            # Extract and save PTX code
            ptx_code = self.extract_ptx_code(response)
            file_name = f"gemm4x4_gen_{timestamp}_iter{iteration}.ptx"
            ptx_file_path = self.save_ptx_to_file(ptx_code, file_name)
            print(f"Generated PTX saved to {ptx_file_path}")
            
            # Benchmark the generated PTX
            benchmark_result = self.benchmark_ptx(ptx_file_path)
            print(f"Benchmark results: time={benchmark_result['avg_time_ms']:.6f}ms, verification={'PASSED' if benchmark_result['verification_passed'] else 'FAILED'}")
            
            # If verification failed or the performance is worse than the best, generate feedback
            if not benchmark_result['verification_passed'] or benchmark_result['avg_time_ms'] > self.best_performance:
                current_prompt = self.generate_feedback_prompt(ptx_code, benchmark_result, initial_prompt)
            else:
                print(f"New best performance: {benchmark_result['avg_time_ms']:.6f} ms")
            
            iteration += 1
        
        # Save a summary of results
        self.save_results_summary()
        
        if self.best_ptx_file:
            print(f"\nOptimization completed. Best PTX file: {self.best_ptx_file}")
            print(f"Best performance: {self.best_performance:.6f} ms")
            return self.best_ptx_file, self.best_performance
        else:
            print("\nOptimization completed, but no valid PTX was generated.")
            return None, float('inf')
    
    def save_results_summary(self):
        """Save a summary of all benchmark results."""
        summary_file = os.path.join(self.output_dir, "benchmark_summary.json")
        
        # Sort results by performance (best first)
        sorted_results = sorted(
            [r for r in self.results_history if r["verification_passed"]],
            key=lambda x: x["avg_time_ms"]
        )
        
        with open(summary_file, 'w') as f:
            json.dump({
                "best_performance_ms": self.best_performance if self.best_performance != float('inf') else None,
                "best_ptx_file": self.best_ptx_file,
                "all_results": self.results_history,
                "sorted_valid_results": sorted_results
            }, f, indent=2)
        
        print(f"Results summary saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='AI-driven PTX Generator and Benchmarker')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY environment variable)')
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'OpenAI model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS, 
                        help=f'Maximum number of optimization iterations (default: {DEFAULT_MAX_ITERATIONS})')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help=f'Temperature for generation (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save generated PTX files (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--prompt', default=None, help='Custom prompt for PTX generation')
    
    args = parser.parse_args()
    
    # Initialize benchmarker
    benchmarker = PtxBenchmarker(
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir
    )
    
    # Default prompt if none provided
    default_prompt = """
Generate efficient PTX code for a 4x4 matrix multiplication (GEMM) operation.

The function should:
1. Take three parameters: pointers to matrices A, B, and C
2. Compute C = A * B where A, B, and C are 4x4 matrices of single-precision floats
3. Matrices are stored in row-major format
4. The PTX should be optimized for performance on modern NVIDIA GPUs

The entry point should be named "gemm4x4" and the PTX version should be 7.5 targeting sm_86.

For reference, here's a simplified example of what the C equivalent would be:

```c
void gemm4x4(float* A, float* B, float* C) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += A[i*4+k] * B[k*4+j];
            }
            C[i*4+j] = sum;
        }
    }
}
```

Consider optimizations like:
- Efficient register usage
- Instruction scheduling to hide latencies
- Tiling/blocking if applicable
- Minimizing memory access patterns
- Using fused multiply-add (fma) instructions

Provide only the complete, executable PTX code.
"""
    
    initial_prompt = args.prompt if args.prompt else default_prompt
    
    # Run the optimization loop
    best_ptx, best_performance = benchmarker.run_optimization_loop(
        initial_prompt=initial_prompt,
        max_iterations=args.max_iterations,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main()
