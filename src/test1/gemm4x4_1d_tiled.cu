#include <cuda_runtime.h>

__global__ void gemm4x4_1d_tiled(float* A, float* B, float* C) {
    // Initialize result matrix to zeros
    float c[16] = {0.0f};
    
    // 1D tiling loop over K dimension
    for (int k = 0; k < 4; k++) {
        // Load one column from each row of A
        float a_tile[4];
        a_tile[0] = A[0*4 + k];    // A[0][k]
        a_tile[1] = A[1*4 + k];    // A[1][k]
        a_tile[2] = A[2*4 + k];    // A[2][k]
        a_tile[3] = A[3*4 + k];    // A[3][k]
        
        // Load one full row from B
        float b_tile[4];
        b_tile[0] = B[k*4 + 0];    // B[k][0]
        b_tile[1] = B[k*4 + 1];    // B[k][1]
        b_tile[2] = B[k*4 + 2];    // B[k][2]
        b_tile[3] = B[k*4 + 3];    // B[k][3]
        
        // Compute outer product of current tiles
        // and accumulate into result
        
        // Row 0 of C
        c[0*4 + 0] += a_tile[0] * b_tile[0];  // C[0][0]
        c[0*4 + 1] += a_tile[0] * b_tile[1];  // C[0][1]
        c[0*4 + 2] += a_tile[0] * b_tile[2];  // C[0][2]
        c[0*4 + 3] += a_tile[0] * b_tile[3];  // C[0][3]
        
        // Row 1 of C
        c[1*4 + 0] += a_tile[1] * b_tile[0];  // C[1][0]
        c[1*4 + 1] += a_tile[1] * b_tile[1];  // C[1][1]
        c[1*4 + 2] += a_tile[1] * b_tile[2];  // C[1][2]
        c[1*4 + 3] += a_tile[1] * b_tile[3];  // C[1][3]
        
        // Row 2 of C
        c[2*4 + 0] += a_tile[2] * b_tile[0];  // C[2][0]
        c[2*4 + 1] += a_tile[2] * b_tile[1];  // C[2][1]
        c[2*4 + 2] += a_tile[2] * b_tile[2];  // C[2][2]
        c[2*4 + 3] += a_tile[2] * b_tile[3];  // C[2][3]
        
        // Row 3 of C
        c[3*4 + 0] += a_tile[3] * b_tile[0];  // C[3][0]
        c[3*4 + 1] += a_tile[3] * b_tile[1];  // C[3][1]
        c[3*4 + 2] += a_tile[3] * b_tile[2];  // C[3][2]
        c[3*4 + 3] += a_tile[3] * b_tile[3];  // C[3][3]
    }
    
    // Store results back to global memory
    for (int i = 0; i < 16; i++) {
        C[i] = c[i];
    }
}
