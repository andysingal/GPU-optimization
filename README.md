# GPU-optimization

[SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips](https://pytorch.org/blog/superoffload-unleashing-the-power-of-large-scale-llm-training-on-superchips/)

### Definitions
- Thread: A thread is a simplified view of how a processor executes a sequential program in modern computers. A thread consists of the code of the program, the point in the code that is being executed, and the values of its variables and data structures. The execution of a thread is sequential as far as a user is concerned. One can use a source-level debugger to monitor the progress of a thread by executing one statement at a time, looking at the statement that will be executed next and checking the values of the variables and data structures as the execution progresses.
- Loop Parallelism: The answer is that the loop is now replaced with the grid of threads. The entire grid forms the equivalent of the loop. Each thread in the grid corresponds to one iteration of the original loop. This is sometimes referred to as loop parallelism, in which iterations of the original sequential code are executed by threads in parallel.

When a program’s host code calls a kernel, the CUDA runtime system launches a grid of threads that are organized into a two-level hierarchy. Each grid is organized as an array of thread blocks, which we will refer to as blocks for brevity. All blocks of a grid are of the same size; each block can contain up to 1024 threads on current systems. 



One can think of each phone line as a CUDA thread, with the area code as the value of blockIdx and the seven-digital local number as the value of threadIdx. This hierarchical organization allows the system to have a very large number of phone lines while preserving “locality” for calling the same area. That is, when dialing a phone line in the same area, a caller only needs to dial the local number. As long as we make most of our calls within the local area, we seldom need to dial the area code. If we occasionally need to call a phone line in another area, we dial 1 and the area code, followed by the local number. (This is the reason why no local number in any area should start with a 1.) The hierarchical organization of CUDA threads also offers a form of locality. 

```py
// Compute vector sum C = A + B
// Each thread performs one pair-wise addition
__global__
void vecAddKernel(float* A, Float* B, float* C, int n) {
    int i = threadIdx + blockDim.x * blockIdx.x;
    if (i < n) {
           c[i] = A[i] + B[i];
         }
}
```
Qualified Keyword    Callable From        Executed On   Executed By
__host__(default)    Host                 Host          Caller host thread 
__global__           Host(or Device)      Device        New grid of device threads
__device__           Device               Device        Caller device thread 

The “__device__” keyword indicates that the function being declared is a CUDA device function. A device function executes on a CUDA device and can be called only from a kernel function or another device function. The device function is executed by the device thread that calls it and does not result in any new device threads being launched.

The “__host__” keyword indicates that the function being declared is a CUDA host function. A host function is simply a traditional C function that executes on the host and can be called only from another host function.



## Courses 

[cuda-course](https://github.com/Infatoshi/cuda-course/tree/master)
