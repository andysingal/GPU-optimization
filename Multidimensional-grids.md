A grid consists of one or more blocks, and each block consists of one or more threads. All threads in a block share the same block index, which can be accessed via the blockIdx (built-in) variable. Each thread also has a thread index, which can be accessed via the threadIdx (built-in) variable. The execution configuration parameters in a kernel call statement specify the dimensions of the grid and the dimensions of each block. These dimensions are available via the gridDim and blockDim (built-in) variables.

When calling a kernel, the program needs to specify the size of the grid and the blocks in each dimension. These are specified by using the execution configuration parameters (within <<<…>>>) of the kernel call statement. The first execution configuration parameter specifies the dimensions of the grid in number of blocks. The second specifies the dimensions of each block in number of threads.

For example, the following host code can be used to call the vecAddkernel() kernel function and generate a 1D grid that consists of 32 blocks, each of which consists of 128 threads. The total number of threads in the grid is 128*32=4096

```
dim3 dimGrid(32, 1, 1)
dim3 dimBlock(128, 1, 1)

vecAddKernel<<<dimGrid, dimBlock>>>(...);
```

```
dim3 dimGrid(ceil(n/256.0), 1, 1);
dim3 dimBlock(256.0,1,1)

vecAddKernel<<<dimGrid, dimBlock>>>(...);
```

In this example the programmer chose to fix the block size at 256. The value of variable n at kernel call time will determine dimension of the grid. If n is equal to 1000, the grid will consist of four blocks. If n is equal to 4000, the grid will have 16 blocks. In each case, there will be enough threads to cover all the vector elements. Once the grid has been launched, the grid and block dimensions will remain the same until the entire grid has finished execution.
