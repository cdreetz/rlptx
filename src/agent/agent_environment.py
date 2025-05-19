# agent_environment.py
import os
import subprocess
import numpy as np
import json
import time
import tempfile
import random
from anthropic import Anthropic

# Configuration
class Config:
    num_iterations = 1000
    matrix_sizes = [4, 8, 16, 32]  # Different sizes to test
    num_runs_per_benchmark = 100
    gpu_arch = "sm_70"  # Adjust to your GPU
    log_dir = "ptx_evolution_logs"
    best_implementations_dir = "best_implementations"
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # Curriculum learning params
    difficulty_levels = {
        0: {"matrix_size": 4, "optimizations": ["naive"]},
        100: {"matrix_size": 8, "optimizations": ["naive", "1d_tiling"]},
        250: {"matrix_size": 16, "optimizations": ["naive", "1d_tiling", "2d_tiling"]},
        500: {"matrix_size": 32, "optimizations": ["naive", "1d_tiling", "2d_tiling", "shared_memory"]},
        750: {"matrix_size": 32, "optimizations": ["all"]}
    }

# Initialize configuration
config = Config()

# Create directories
os.makedirs(config.log_dir, exist_ok=True)
os.makedirs(config.best_implementations_dir, exist_ok=True)

# Initialize Anthropic client
client = Anthropic(api_key=config.anthropic_api_key)

# Class to manage PTX generation and testing
class PTXEvolutionEnvironment:
    def __init__(self, config):
        self.config = config
        self.current_iteration = 0
        self.best_performances = {}  # Track best performance for each matrix size
        self.implementation_history = []
        self.performance_history = []
        self.current_difficulty = 0
        
        # Load best implementations if they exist
        self.load_best_implementations()
    
    def load_best_implementations(self):
        for size in self.config.matrix_sizes:
            best_file = os.path.join(self.config.best_implementations_dir, f"best_gemm_{size}x{size}.json")
            if os.path.exists(best_file):
                with open(best_file, 'r') as f:
                    data = json.load(f)
                    self.best_performances[size] = data
    
    def save_best_implementation(self, matrix_size, ptx_code, benchmark_results):
        data = {
            "ptx_code": ptx_code,
            "benchmark_results": benchmark_results,
            "iteration": self.current_iteration,
            "timestamp": time.time()
        }
        
        with open(os.path.join(self.config.best_implementations_dir, f"best_gemm_{matrix_size}x{matrix_size}.json"), 'w') as f:
            json.dump(data, f, indent=2)
        
        # Also save as .ptx file for easy access
        with open(os.path.join(self.config.best_implementations_dir, f"best_gemm_{matrix_size}x{matrix_size}.ptx"), 'w') as f:
            f.write(ptx_code)
    
    def get_current_requirements(self):
        # Find the appropriate difficulty level
        current_level = 0
        for level in sorted(self.config.difficulty_levels.keys()):
            if self.current_iteration >= level:
                current_level = level
        
        return self.config.difficulty_levels[current_level]
    
    def generate_prompt(self, matrix_size, previous_code=None, previous_performance=None):
        requirements = self.get_current_requirements()
        optimizations = requirements["optimizations"]
        
        # Build a context with information about past performances if available
        context = ""
        if previous_code and previous_performance:
            context = f"""
            Your previous implementation for {matrix_size}x{matrix_size} matrix multiplication achieved:
            - Execution time: {previous_performance['execution_time']:.6f} ms
            - GFLOPS: {previous_performance['gflops']:.2f}

            Here's your previous code:

            {previous_code}


            Please improve upon this implementation.
            """
                    
                    # Create different prompts based on difficulty level
                    optimization_prompt = ""
                    if "all" in optimizations:
                        optimization_prompt = "Use any optimization techniques you think would be effective."
                    else:
                        optimization_prompt = f"Focus on these optimization techniques: {', '.join(optimizations)}."
                    
                    prompt = f"""
            Write an optimized PTX implementation for a {matrix_size}x{matrix_size} GEMM (General Matrix Multiplication).

            {context}

            {optimization_prompt}

            Consider:
            1. Register usage and reuse
            2. Memory access patterns
            3. Instruction-level parallelism
            4. Avoiding bank conflicts
            5. Thread coarsening if appropriate

            The goal is to maximize performance (GFLOPS) while maintaining numerical correctness.
            Return only the PTX code without explanations.
            """
        
        return prompt
    
    def generate_ptx_code(self, matrix_size, previous_code=None, previous_performance=None):
        prompt = self.generate_prompt(matrix_size, previous_code, previous_performance)
        
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                temperature=0.7,
                system="You are an expert in GPU programming and PTX code generation. Your task is to generate high-performance PTX code for matrix multiplication.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract PTX code from the response
            response_text = response.content[0].text
            
            # Simple extraction - assuming the model returns just the PTX code
            # In practice, you might need more robust extraction
            ptx_code = response_text.strip()
            
            # Clean up any markdown code blocks if present
            if ptx_code.startswith("```") and ptx_code.endswith("```"):
                ptx_code = "\n".join(ptx_code.split("\n")[1:-1])
            
            return ptx_code
        
        except Exception as e:
            print(f"Error generating PTX code: {e}")
            # Return a simple fallback implementation if generation fails
            return self.get_fallback_ptx(matrix_size)
    
    def get_fallback_ptx(self, matrix_size):
        # Simple naive implementation as fallback
        return f"""
.version 7.0
.target {self.config.gpu_arch}
.address_size 64

.visible .entry gemm{matrix_size}x{matrix_size}(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C
) {{
    // Naive implementation
    // (Basic implementation would go here)
}}
"""
    
    def compile_and_benchmark(self, ptx_code, matrix_size):
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.ptx', delete=False) as ptx_file:
            ptx_file.write(ptx_code.encode())
            ptx_filename = ptx_file.name
        
        # Create benchmark code
        benchmark_code = self.generate_benchmark_code(matrix_size)
        with tempfile.NamedTemporaryFile(suffix='.cu', delete=False) as cu_file:
            cu_file.write(benchmark_code.encode())
            cu_filename = cu_file.name
        
        try:
            # Compile PTX to cubin
            cubin_filename = ptx_filename.replace('.ptx', '.cubin')
            compile_cmd = f"nvcc -arch={self.config.gpu_arch} -cubin {ptx_filename} -o {cubin_filename}"
            subprocess.run(compile_cmd, shell=True, check=True)
            
            # Compile benchmark
            benchmark_binary = cu_filename.replace('.cu', '')
            benchmark_cmd = f"nvcc -arch={self.config.gpu_arch} {cu_filename} -o {benchmark_binary} --fatbin-options \"-link\" --device-link {cubin_filename}"
            subprocess.run(benchmark_cmd, shell=True, check=True)
            
            # Run benchmark
            result = subprocess.run(f"{benchmark_binary}", shell=True, capture_output=True, text=True, check=True)
            
            # Parse benchmark results
            return self.parse_benchmark_results(result.stdout)
            
        except subprocess.CalledProcessError as e:
            print(f"Error in compilation or benchmarking: {e}")
            print(f"STDERR: {e.stderr}")
            return {
                "status": "failed",
                "error": str(e),
                "execution_time": float('inf'),
                "gflops": 0.0
            }
        finally:
            # Clean up temporary files
            for filename in [ptx_filename, cu_filename, cubin_filename, benchmark_binary]:
                if os.path.exists(filename):
                    os.unlink(filename)
    
    def generate_benchmark_code(self, matrix_size):
        return f"""
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// External declaration for the PTX kernel
extern "C" __global__ void gemm{matrix_size}x{matrix_size}(float* A, float* B, float* C);

// CPU reference implementation for verification
void gemm{matrix_size}x{matrix_size}_cpu(float* A, float* B, float* C) {{
    for (int i = 0; i < {matrix_size}; i++) {{
        for (int j = 0; j < {matrix_size}; j++) {{
            float sum = 0.0f;
            for (int k = 0; k < {matrix_size}; k++) {{
                sum += A[i*{matrix_size}+k] * B[k*{matrix_size}+j];
            }}
            C[i*{matrix_size}+j] = sum;
        }}
    }}
}}

// Utility function to check if results match
bool verify_result(float* expected, float* actual, int size) {{
    const float epsilon = 1e-5;
    for (int i = 0; i < size; i++) {{
        if (fabs(expected[i] - actual[i]) > epsilon) {{
            printf("VERIFICATION FAILED: Mismatch at index %d: expected %f, got %f\\n", 
                   i, expected[i], actual[i]);
            return false;
        }}
    }}
    printf("VERIFICATION PASSED\\n");
    return true;
}}

int main() {{
    // Matrix size
    const int N = {matrix_size};
    const int size = N * N;
    
    // Allocate host memory
    float *h_A, *h_B, *h_C, *h_C_ref;
    h_A = (float*)malloc(size * sizeof(float));
    h_B = (float*)malloc(size * sizeof(float));
    h_C = (float*)malloc(size * sizeof(float));
    h_C_ref = (float*)malloc(size * sizeof(float));
    
    // Initialize matrices with random values
    srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < size; i++) {{
        h_A[i] = (float)(rand() % 10);
        h_B[i] = (float)(rand() % 10);
    }}
    
    // Compute reference result on CPU
    gemm{matrix_size}x{matrix_size}_cpu(h_A, h_B, h_C_ref);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));
    
    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Determine grid and block dimensions (simple case for now)
    dim3 grid(1, 1);
    dim3 block(32, 1);  // Adjust based on matrix size
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int NUM_RUNS = {self.config.num_runs_per_benchmark};
    
    // Warmup
    gemm{matrix_size}x{matrix_size}<<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; i++) {{
        gemm{matrix_size}x{matrix_size}<<<grid, block>>>(d_A, d_B, d_C);
    }}
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate performance metrics
    float avg_time_ms = milliseconds / NUM_RUNS;
    float gflops = (2.0f * N * N * N) / (avg_time_ms * 1e6);  // 2*N^3 FLOPs for GEMM
    
    printf("BENCHMARK_RESULT\\n");
    printf("matrix_size: %d\\n", N);
    printf("execution_time_ms: %f\\n", avg_time_ms);
    printf("gflops: %f\\n", gflops);
    
    // Verify result
    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
    bool passed = verify_result(h_C_ref, h_C, size);
    printf("verification: %s\\n", passed ? "passed" : "failed");
    
    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return passed ? 0 : 1;
}}
"""
    
    def parse_benchmark_results(self, output):
        results = {
            "status": "failed",
            "execution_time": float('inf'),
            "gflops": 0.0,
            "verification": "failed"
        }
        
        try:
            if "BENCHMARK_RESULT" in output:
                for line in output.split('\n'):
                    if "execution_time_ms:" in line:
                        results["execution_time"] = float(line.split(': ')[1])
                    elif "gflops:" in line:
                        results["gflops"] = float(line.split(': ')[1])
                    elif "verification:" in line:
                        results["verification"] = line.split(': ')[1]
                
                results["status"] = "success" if results["verification"] == "passed" else "verification_failed"
                
            return results
        except Exception as e:
            print(f"Error parsing benchmark results: {e}")
            return results
    
    def log_iteration(self, iteration, matrix_size, ptx_code, benchmark_results):
        log_file = os.path.join(self.config.log_dir, f"iteration_{iteration:04d}.json")
        
        data = {
            "iteration": iteration,
            "matrix_size": matrix_size,
            "ptx_code": ptx_code,
            "benchmark_results": benchmark_results,
            "timestamp": time.time()
        }
        
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def run_iteration(self):
        # Select matrix size based on curriculum
        requirements = self.get_current_requirements()
        matrix_size = requirements["matrix_size"]
        
        print(f"\n=== Iteration {self.current_iteration} (Matrix Size: {matrix_size}x{matrix_size}) ===")
        
        # Get previous best implementation for this size if it exists
        previous_code = None
        previous_performance = None
        if matrix_size in self.best_performances:
            previous_code = self.best_performances[matrix_size]["ptx_code"]
            previous_performance = self.best_performances[matrix_size]["benchmark_results"]
            print(f"Previous best for {matrix_size}x{matrix_size}: {previous_performance['gflops']:.2f} GFLOPS")
        
        # Generate new PTX code
        ptx_code = self.generate_ptx_code(matrix_size, previous_code, previous_performance)
        
        # Compile and benchmark
        benchmark_results = self.compile_and_benchmark(ptx_code, matrix_size)
        
        # Log results
        self.log_iteration(self.current_iteration, matrix_size, ptx_code, benchmark_results)
        
        print(f"Status: {benchmark_results['status']}")
        print(f"Execution time: {benchmark_results['execution_time']:.6f} ms")
        print(f"GFLOPS: {benchmark_results['gflops']:.2f}")
        
        # Update best implementation if this one is better
        if (benchmark_results["status"] == "success" and 
            (matrix_size not in self.best_performances or 
             benchmark_results["gflops"] > self.best_performances[matrix_size]["benchmark_results"]["gflops"])):
            
            print(f"New best implementation for {matrix_size}x{matrix_size}!")
            self.best_performances[matrix_size] = {
                "ptx_code": ptx_code,
                "benchmark_results": benchmark_results
            }
            self.save_best_implementation(matrix_size, ptx_code, benchmark_results)
        
        # Record history for RL training
        self.implementation_history.append({
            "iteration": self.current_iteration,
            "matrix_size": matrix_size,
            "ptx_code": ptx_code,
            "benchmark_results": benchmark_results
        })
        
        # Increment iteration counter
        self.current_iteration += 1
    
    def run(self):
        while self.current_iteration < self.config.num_iterations:
            self.run_iteration()
            # Optional: small delay to avoid API rate limits
            time.sleep(1)
        
        # Print summary at the end
        print("\n=== Final Results ===")
        for size in sorted(self.best_performances.keys()):
            perf = self.best_performances[size]["benchmark_results"]["gflops"]
            print(f"Matrix size {size}x{size}: {perf:.2f} GFLOPS")

# Main function
def main():
    env = PTXEvolutionEnvironment(config)
    env.run()

if __name__ == "__main__":
    main()
