`nvcc -ptx gemm1024x1024_1d_tiled.cu -o gemm1024x1024_1d_tiled_generated.ptx`

`nvcc -o benchmark_gemm1024x1024 src/test2/benchmark_gemm1024x1024.cu -lcuda` 
`./benchmark_gemm1024x1024`