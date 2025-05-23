#!/bin/bash

# Step 1: Compile the CUDA file to generate PTX
echo "Compiling CUDA kernel to PTX..."
nvcc -ptx src/test2/gemm4x4_1d_tiled.cu -o src/test2/gemm1024x1024.ptx

# Step 2: Compile the benchmark program
echo "Compiling benchmark program..."
nvcc -o benchmark_gemm1024x1024 src/test2/benchmark_gemm1024x1024.cu -lcuda

# Step 3: Run the benchmark
echo "Running benchmark..."
./benchmark_gemm1024x1024 