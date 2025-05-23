#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void gemm1024x1024_1d_tiled(float* A, float* B, float* C, int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes one element of C
    if (row < N && col < N) {
        float sum = 0.0f;
        
        // 1D tiling loop over K dimension
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

// Host function to launch the kernel
void launch_gemm1024x1024(float* d_A, float* d_B, float* d_C) {
    const int N = 1024;
    
    // Define grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                        (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    gemm1024x1024_1d_tiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
}
