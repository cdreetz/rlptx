#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to compile and load PTX at runtime
CUmodule loadPTX(const char* ptx_filename) {
    FILE* ptx_file = fopen(ptx_filename, "rb");
    if (!ptx_file) {
        fprintf(stderr, "Error opening PTX file: %s\n", ptx_filename);
        exit(1);
    }
    
    // Get file size
    fseek(ptx_file, 0, SEEK_END);
    size_t size = ftell(ptx_file);
    rewind(ptx_file);
    
    // Read file content
    char* ptx = (char*)malloc(size + 1);
    fread(ptx, 1, size, ptx_file);
    ptx[size] = '\0';
    fclose(ptx_file);
    
    printf("PTX file size: %zu bytes\n", size);
    
    // Initialize CUDA driver API
    cuInit(0);
    
    // Create module from PTX
    CUmodule module;
    CUresult result = cuModuleLoadData(&module, ptx);
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        fprintf(stderr, "Error loading PTX module: %s\n", error_str);
        free(ptx);
        exit(1);
    }
    
    free(ptx);
    return module;
}

// CPU reference implementation for verification (small subset only for testing)
void gemm_cpu_subset(float* A, float* B, float* C, int N, int subset_size) {
    for (int i = 0; i < subset_size; i++) {
        for (int j = 0; j < subset_size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
}

// Utility function to check if results match (subset only)
bool verify_result_subset(float* expected, float* actual, int N, int subset_size) {
    const float epsilon = 1e-5;
    bool match = true;
    for (int i = 0; i < subset_size; i++) {
        for (int j = 0; j < subset_size; j++) {
            int idx = i*N + j;
            if (fabs(expected[idx] - actual[idx]) > epsilon) {
                printf("Mismatch at [%d,%d]: expected %f, got %f\n", 
                       i, j, expected[idx], actual[idx]);
                match = false;
                // Only report up to 5 mismatches to avoid flooding output
                if (--subset_size <= 0) return match;
            }
        }
    }
    return match;
}

int main() {
    const char* ptx_file = "gemm1024x1024_1d_tiled_generated.ptx";
    const int N = 1024;
    const size_t matrix_size = N * N * sizeof(float);
    const int verify_subset = 10; // Only verify a small subset to save time
    
    // Initialize CUDA Driver API
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    
    // Get device properties
    int maxThreadsPerBlock;
    cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
    printf("Max threads per block: %d\n", maxThreadsPerBlock);
    
    CUcontext context;
    cuCtxCreate(&context, 0, device);
    
    // Load PTX module
    CUmodule module = loadPTX(ptx_file);
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "_Z22gemm1024x1024_1d_tiledPfS_S_i");
    
    // Get kernel launch configuration
    int minGridSize, blockSize;
    cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0, 0);
    printf("Optimal block size: %d\n", blockSize);
    
    // Allocate host memory
    float *h_A, *h_B, *h_C, *h_C_ref;
    cudaMallocHost(&h_A, matrix_size);
    cudaMallocHost(&h_B, matrix_size);
    cudaMallocHost(&h_C, matrix_size);
    cudaMallocHost(&h_C_ref, matrix_size);
    
    // Initialize matrices with random values
    printf("Initializing matrices...\n");
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i*N+j] = (float)(rand() % 10) / 10.0f;
            h_B[i*N+j] = (float)(rand() % 10) / 10.0f;
            h_C[i*N+j] = 0.0f;
            h_C_ref[i*N+j] = 0.0f;
        }
    }
    
    // Compute reference result for subset on CPU
    printf("Computing reference result for subset...\n");
    gemm_cpu_subset(h_A, h_B, h_C_ref, N, verify_subset);
    
    // Allocate device memory
    CUdeviceptr d_A, d_B, d_C;
    cudaError_t cuda_status;
    
    cuda_status = cudaMalloc((void**)&d_A, matrix_size);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_A: %s\n", cudaGetErrorString(cuda_status));
        exit(1);
    }
    
    cuda_status = cudaMalloc((void**)&d_B, matrix_size);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_B: %s\n", cudaGetErrorString(cuda_status));
        exit(1);
    }
    
    cuda_status = cudaMalloc((void**)&d_C, matrix_size);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_C: %s\n", cudaGetErrorString(cuda_status));
        exit(1);
    }
    
    // Copy input matrices from host to device
    printf("Copying data to GPU...\n");
    cudaMemcpy((void*)d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
    
    // Set up kernel parameters
    int n_value = N;  // Use a non-const local variable
    void* args[] = { &d_A, &d_B, &d_C, &n_value };
    
    // Define grid and block dimensions
    int BLOCK_SIZE = 32;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                        (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
                        
    printf("Launching kernel with grid: (%d,%d), block: (%d,%d)\n", 
            blocksPerGrid.x, blocksPerGrid.y, 
            threadsPerBlock.x, threadsPerBlock.y);
    
    // Warmup
    printf("Running warmup...\n");
    cuLaunchKernel(kernel, 
                  blocksPerGrid.x, blocksPerGrid.y, 1,
                  threadsPerBlock.x, threadsPerBlock.y, 1,
                  0, 0, args, 0);
    cudaDeviceSynchronize();
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int num_runs = 10; // Fewer runs due to larger problem size
    printf("Running %d iterations...\n", num_runs);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        cuLaunchKernel(kernel, 
                      blocksPerGrid.x, blocksPerGrid.y, 1,
                      threadsPerBlock.x, threadsPerBlock.y, 1,
                      0, 0, args, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Total time for %d runs = %f ms\n", num_runs, milliseconds);
    printf("Average time per run = %f ms\n", milliseconds / num_runs);
    
    // Calculate performance metrics
    float avg_time_sec = (milliseconds / 1000.0f) / num_runs;
    float flops = 2.0f * N * N * N; // 2 operations per multiply-add
    float gflops = (flops / avg_time_sec) / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // Copy result from device to host
    printf("Copying results back to host...\n");
    cudaMemcpy(h_C, (void*)d_C, matrix_size, cudaMemcpyDeviceToHost);
    
    // Verify subset of the result
    printf("Verifying subset of results...\n");
    bool passed = verify_result_subset(h_C_ref, h_C, N, verify_subset);
    printf("Verification: %s\n", passed ? "PASSED" : "FAILED");
    
    // Free memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_C_ref);
    cudaFree((void*)d_A);
    cudaFree((void*)d_B);
    cudaFree((void*)d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    
    return 0;
} 