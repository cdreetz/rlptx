#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

// Function prototype for the PTX kernel
extern "C" __global__ void _Z16gemm4x4_1d_tiledPfS_S_(float* A, float* B, float* C);

// Helper function to initialize matrix
void init_matrix(float* mat, int size, float val) {
    for (int i = 0; i < size; i++) {
        mat[i] = val;
    }
}

// Helper function to verify result
bool verify_result(float* C, int size, float expected) {
    for (int i = 0; i < size; i++) {
        if (fabs(C[i] - expected) > 1e-5) {
            printf("Verification failed at index %d: got %f, expected %f\n", 
                   i, C[i], expected);
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 4; // Matrix dimension
    const int SIZE = N * N;
    const int BYTES = SIZE * sizeof(float);

    // Host matrices
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(BYTES);
    h_B = (float*)malloc(BYTES);
    h_C = (float*)malloc(BYTES);

    // Initialize input matrices
    init_matrix(h_A, SIZE, 1.0f);
    init_matrix(h_B, SIZE, 2.0f);
    init_matrix(h_C, SIZE, 0.0f);

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, BYTES);
    cudaMalloc(&d_B, BYTES);
    cudaMalloc(&d_C, BYTES);

    // Copy data to device
    cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice);

    // Launch kernel and measure time
    const int NUM_RUNS = 1000;
    float total_time = 0;

    for (int i = 0; i < NUM_RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        _Z16gemm4x4_1d_tiledPfS_S_<<<1, 1>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        total_time += duration.count();
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, BYTES, cudaMemcpyDeviceToHost);

    // Verify result (expected value is 8.0 for 4x4 matrix with A=1.0 and B=2.0)
    bool passed = verify_result(h_C, SIZE, 8.0f);

    // Print timing results
    float avg_time = total_time / NUM_RUNS;
    printf("Average kernel execution time: %f ms\n", avg_time);
    printf("Verification %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
