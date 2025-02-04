#### accelerated systems

accelerated systems are *heterogeneous* systems. they are composed of both CPUs and GPUs. 
as normal systems, these systems also run CPU programs, but the launched CPU programs utilize the GPU for parallelism. 
as everyone knows GPUs are extremely powerful when it comes to batch-processing.

initialization, as usual is performed on the CPU. but, the work will be done on the GPU. CUDA allows the CPU to interact and use the GPU.

CPU and GPU also perform work parallely. while GPU is working on some large batch processing, CPU can also perform its own work synchronously with it or asynchronous/independent of it. 
***
CUDA provides extensions (like writing the CUDA based code - kind of like compiled assembly code in other high level languages) in many languages.

we will use C/C++ here.

**.cu** is the extension for CUDA based accelerated programs.
***
essentially, there are two functions that we have to define (don't know whether it is mandatory to define both of them), CPU_Function and GPU_Function.

things to remember:

1. \_\_global\_\_ : this keyword is used to tell that the function gets executed via a gpu. it also means that it can be called by any means, cpu or gpu.
2. generally, we call the cpu code as *host* code and gpu code as *device* code.
3. \_\_global\_\_ functions should have the return type as *void*.
4. a gpu function is generally called as *kernel*. a *kernel* expects a *execution configuration*. *execution config* is something which tells the gpu about the **thread hierarchy**. thread hierarchy is a simple way of saying how many thread grouping we should have and the number of threads in each thread group.


to make the cpu synchronize with the gpu while the gpu kernel is being processed, we can use the function *cudaDeviceSynchronize()* to make the cpu wait for the gpu kernel to complete execution instead of asynchronously processing its own tasks.
***
- a global function call (device code) should be configured.
- a non global function call (host code) cannot be configured. (you can't pass exec configs to a cpu)
- cpu functions always run first then, gpu takes over. (maybe not, maybe parallel, i don't know. only for the time being, i am assuming this.)
***
#### compilation
- cuda ships with the cuda compiler :) which is the nvcc. nvidia cuda compiler. don't know why it is not ncc.
***
#### parallel kernels
- exec config syntax is <<<number_of_blocks, number_of_threads_per_block>>>
- a kernel is executed in every thread of each and every block.
- threads are grouped into a block. blocks are grouped into a grid.

things to alter the functioning within the grid:
- *girdDim.x* : number of blocks in the grid.
- *blockId.x* : index of the block.
- *blockDim.x* : number of threads in the block.
- all blocks in the grid contains the same number of threads.
- *threadIdx.x* : index of the thread.
- there is a limit to the number of threads in a thread block. 1024.
***
#### memory allocation
- cudaMallocManaged - malloc
- cudaFree - free
- we can use the data that is allocated by the host in both the host and the device.
- cuda based memory alloc handles many intricacies that are involved in this accelerated memory allocation.



