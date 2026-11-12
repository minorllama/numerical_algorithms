# CUDA C++ Monte Carlo

This is the `c++` version of the Oxford Monte Carlo example. 

## Build and run

Build with a CUDA toolkit that includes cuRAND. Run the GPU examples on an NVIDIA GPU with a compatible driver.

```bash
nvcc -O3 -std=c++17 -lineinfo -gencode arch=compute_89,code=sm_89 monte_carlo.cu -lcurand -o monte-carlo
./monte-carlo --help
./monte-carlo buffered 4097 100 verify
./monte-carlo device 131072 100
./monte-carlo polynomial 131072
```

The `nvcc` should be `13.0` and above as there are known issues with `C++` headers, the `-gencode` will need to be adjusted for the GPU.  

Arguments: mode, path count, and steps per path with defaults `buffered 131072 100`. The block size is 128 and the seed is 1234. Polynomial mode uses 200 samples per path regardless of the steps argument. If the toolkit libraries are outside the linker’s search path, add `-L $cudapath/lib64` or the corresponding toolkit library directory to the build command.

Defining `CPP_CPU_TEST` will turn this into a cpu only binary *to do!*. 

