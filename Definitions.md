1. Block Scheduling - When a kernel is called, the CUDA runtime system launches a grid of threads that execute the kernel code.These threads are assigned to SMs on a block-by-block basis. That is, all threads in a block are simultaneously assigned to the same SM.


Multiple blocks are likely to be simultaneously assigned to the same SM. However, blocks need to reserve hardware resources to excute, so only a limited number of blocks can be simultaneously assigned to give SM. With a limited number of SMs and a limited number of blocks that can be simultaneously assigned to each SM, there is a limit on total number of blocks that can be simultaneously executed in a CUDA device. To ensure that all blocks in a grid get executed, the runtime system maintains a list of blocks that need to execute and assigns new blocks to SMs when previously assigned blocks complete execution.

**** The assignment of threads to SMs on a block-by-block basis guarantees that threads in the same block are scheduled simultaneously on the same SM. This guarantee makes it possible for threads in the same block to interact with each other in ways that threads across different blocks cannot


2. Barrier Synchronization - It  is a simple and popular method for coordinating parallel activities.

There are N threads in the block. Time goes from left to right. Some of the threads reach the barrier synchronization statement early, and some reach it much later. The ones that reach the barrier early will wait for those that arrive late. When the latest one arrives at the barrier, all threads can continue their execution. With barrier synchronization, “no one is left behind.”

**** In CUDA, if a __syncthreads() statement is present, it must be executed by all threads in a block.

Caution:  In general, incorrect usage of barrier synchronization can result in incorrect result, or in threads waiting for each other forever, which is referred to as a deadlock. It is the responsibility of the programmer to avoid such inappropriate use of barrier synchronization.

This leads us to an important tradeoff in the design of CUDA barrier synchronization. By not allowing threads in different blocks to perform barrier synchronization with each other, the CUDA runtime system can execute blocks in any order relative to each other, since none of them need to wait for each other. This flexibility enables scalable implementations

3. Wraps: Thread scheduling in CUDA GPUs is a hardware implementation concept and therefore must be discussed in the context of specific hardware implementations. In most implementations to date, once a block has been assigned to an SM, it is further divided into 32-thread units called warps. The size of warps is implementation specific and can vary in future generations of GPUs. Knowledge of warps can be helpful in understanding and optimizing the performance of CUDA applications on particular generations of CUDA devices.

<strong>For a block whose size is not a multiple of 32, the last warp will be padded with inactive threads to fill up the 32 thread positions. For example, if a block has 48 threads, it will be partitioned into two warps, and the second warp will be padded with 16 inactive threads.</strong>

