// Save this as benchmark_gemm.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// External declarations for the PTX kernels
extern "C" __global__ void gemm4x4(float* A, float* B, float* C);
extern "C" __global__ void gemm4x4_1d_tiled(float* A, float* B, float* C);

// CPU reference implementation for verification
void gemm4x4_cpu(float* A, float* B, float* C) {
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

// Utility function to check if results match
bool verify_result(float* expected, float* actual, int size) {
    const float epsilon = 1e-5;
    for (int i = 0; i < size; i++) {
        if (fabs(expected[i] - actual[i]) > epsilon) {
            printf("Mismatch at index %d: expected %f, got %f\n", i, expected[i], actual[i]);
            return false;
        }
    }
    return true;
}

// Benchmark function
void benchmark_kernel(void (*kernel)(float*, float*, float*), float* d_A, float* d_B, float* d_C, 
                      const char* kernel_name, int num_runs) {
    // Warmup
    kernel<<<1, 1>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = a 0; i < num_runs; i++) {
        kernel<<<1, 1>>>(d_A, d_B, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("%s: average time per run = %f ms (over %d runs)\n", 
           kernel_name, milliseconds / num_runs, num_runs);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Allocate host memory
    float *h_A, *h_B, *h_C, *h_C_ref;
    h_A = (float*)malloc(16 * sizeof(float));
    h_B = (float*)malloc(16 * sizeof(float));
    h_C = (float*)malloc(16 * sizeof(float));
    h_C_ref = (float*)malloc(16 * sizeof(float));
    
    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < 16; i++) {
        h_A[i] = (float)(rand() % 10);
        h_B[i] = (float)(rand() % 10);
    }
    
    // Compute reference result on CPU
    gemm4x4_cpu(h_A, h_B, h_C_ref);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 16 * sizeof(float));
    cudaMalloc(&d_B, 16 * sizeof(float));
    cudaMalloc(&d_C, 16 * sizeof(float));
    
    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 16 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Benchmark and verify naive implementation
    benchmark_kernel(gemm4x4, d_A, d_B, d_C, "Naive GEMM", 1000);
    cudaMemcpy(h_C, d_C, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Naive GEMM: %s\n", verify_result(h_C_ref, h_C, 16) ? "PASSED" : "FAILED");
    
    // Benchmark and verify tiled implementation
    benchmark_kernel(gemm4x4_1d_tiled, d_A, d_B, d_C, "1D Tiled GEMM", 1000);
    cudaMemcpy(h_C, d_C, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("1D Tiled GEMM: %s\n", verify_result(h_C_ref, h_C, 16) ? "PASSED" : "FAILED");
    
    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
