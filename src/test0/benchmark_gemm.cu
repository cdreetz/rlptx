/ benchmark_gemm.cu
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Function to compile and load PTX at runtime
// Update to loadPTX function in benchmark_gemm.cu for better error reporting
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
    
    // Print first few characters for debugging
    printf("First 100 chars of PTX file:\n%.*s...\n\n", 100, ptx);
    
    // Initialize CUDA driver API
    cuInit(0);
    
    // Create module from PTX
    CUmodule module;
    CUresult result = cuModuleLoadData(&module, ptx);
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        fprintf(stderr, "Error loading PTX module: %s\n", error_str);
        
        // Print more info about the PTX for debugging
        fprintf(stderr, "PTX file size: %zu bytes\n", size);
        fprintf(stderr, "Please check your PTX syntax\n");
        
        free(ptx);
        exit(1);
    }
    
    free(ptx);
    return module;
}

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

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <ptx_file>\n", argv[0]);
        return 1;
    }
    
    const char* ptx_file = argv[1];
    
    // Initialize CUDA Driver API
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext context;
    cuCtxCreate(&context, 0, device);
    
    // Load PTX module
    CUmodule module = loadPTX(ptx_file);
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "gemm4x4");
    
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
    CUdeviceptr d_A, d_B, d_C;
    cuMemAlloc(&d_A, 16 * sizeof(float));
    cuMemAlloc(&d_B, 16 * sizeof(float));
    cuMemAlloc(&d_C, 16 * sizeof(float));
    
    // Copy input matrices from host to device
    cuMemcpyHtoD(d_A, h_A, 16 * sizeof(float));
    cuMemcpyHtoD(d_B, h_B, 16 * sizeof(float));
    
    // Set up kernel parameters
    void* args[] = { &d_A, &d_B, &d_C };
    
    // Warmup
    cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
    cuCtxSynchronize();
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int num_runs = 1000;
    
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Average time per run = %f ms (over %d runs)\n", milliseconds / num_runs, num_runs);
    
    // Copy result from device to host
    cuMemcpyDtoH(h_C, d_C, 16 * sizeof(float));
    
    // Verify result
    bool passed = verify_result(h_C_ref, h_C, 16);
    printf("Verification: %s\n", passed ? "PASSED" : "FAILED");
    
    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    
    return 0;
}
