Getting started
===============

This section will discuss about loading specific modules to run CUDA jobs. 
The first one you need is 

``module load cuda-<version_number>``

Curerntly ADA has CUDA/12.2.2. However other versions could be requested if need be.
One would also need GCC flags, this can be done with 

``module load gcc-uoneasy/<version_number>``

There can be two ways of using a GPU node, depending on the user's requriement. One is the batch mode, and one is the interactive session. 
The first one will require SLURM based commands as discussed below to be able to submit a job. In either case, the command 

``nvidia-smi`` can be used to see if the hardware requested has NVIDIA-cards or not. The output will look something as follows, 

.. code-block:: bash

    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  NVIDIA A100 80GB PCIe          On  | 00000000:01:00.0 Off |                   On |
    | N/A   23C    P0              40W / 300W |     87MiB / 81920MiB |     N/A      Default |
    |                                         |                      |              Enabled |
    +-----------------------------------------+----------------------+----------------------+
    |   1  NVIDIA A100 80GB PCIe          On  | 00000000:02:00.0 Off |                   On |
    | N/A   23C    P0              40W / 300W |     87MiB / 81920MiB |     N/A      Default |
    |                                         |                      |              Enabled |
    +-----------------------------------------+----------------------+----------------------+
    |   2  NVIDIA A100 80GB PCIe          On  | 00000000:61:00.0 Off |                   On |
    | N/A   21C    P0              38W / 300W |     87MiB / 81920MiB |     N/A      Default |
    |                                         |                      |              Enabled |
    +-----------------------------------------+----------------------+----------------------+
    |   3  NVIDIA A100 80GB PCIe          On  | 00000000:62:00.0 Off |                   On |
    | N/A   22C    P0              39W / 300W |     87MiB / 81920MiB |     N/A      Default |
    |                                         |                      |              Enabled |
    +-----------------------------------------+----------------------+----------------------+
    |   4  NVIDIA A100 80GB PCIe          On  | 00000000:81:00.0 Off |                   On |
    | N/A   23C    P0              41W / 300W |     87MiB / 81920MiB |     N/A      Default |
    |                                         |                      |              Enabled |
    +-----------------------------------------+----------------------+----------------------+
    |   5  NVIDIA A100 80GB PCIe          On  | 00000000:82:00.0 Off |                   On |
    | N/A   22C    P0              39W / 300W |     87MiB / 81920MiB |     N/A      Default |
    |                                         |                      |              Enabled |
    +-----------------------------------------+----------------------+----------------------+
    |   6  NVIDIA A100 80GB PCIe          On  | 00000000:E1:00.0 Off |                   On |
    | N/A   23C    P0              39W / 300W |     87MiB / 81920MiB |     N/A      Default |
    |                                         |                      |              Enabled |
    +-----------------------------------------+----------------------+----------------------+
    |   7  NVIDIA A100 80GB PCIe          On  | 00000000:E2:00.0 Off |                   On |
    | N/A   22C    P0              39W / 300W |     87MiB / 81920MiB |     N/A      Default |
    |                                         |                      |              Enabled |
    +-----------------------------------------+----------------------+----------------------+

    +---------------------------------------------------------------------------------------+
    | MIG devices:                                                                          |
    +------------------+--------------------------------+-----------+-----------------------+
    | GPU  GI  CI  MIG |                   Memory-Usage |        Vol|      Shared           |
    |      ID  ID  Dev |                     BAR1-Usage | SM     Unc| CE ENC DEC OFA JPG    |
    |                  |                                |        ECC|                       |
    |==================+================================+===========+=======================|
    |  0    7   0   0  |              12MiB /  9728MiB  | 14      0 |  1   0    0    0    0 |
    |                  |               0MiB / 16383MiB  |           |                       |
    +------------------+--------------------------------+-----------+-----------------------+

    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |  No running processes found                                                           |
    +---------------------------------------------------------------------------------------+


This is an output from the Ampere-q partition, indicating there are 8 GPU cards, indexed from 0 to 7. Similarly the compiler used to compile CUDA programs, known as nvcc or (Nvidia Cuda Compiler) can also be checked with the command, 

``nvcc --help``

.. code-block:: bash

    Usage  : nvcc [options] <inputfile>

    Options for specifying the compilation phase
    ============================================
    More exactly, this option specifies up to which stage the input files must be compiled,
    according to the following compilation trajectories for different input file types:
            .c/.cc/.cpp/.cxx : preprocess, compile, link
            .o               : link
            .i/.ii           : compile, link
            .cu              : preprocess, cuda frontend, PTX assemble,
                            merge with host C code, compile, link
            .gpu             : cicc compile into cubin
            .ptx             : PTX assemble into cubin.
    .
    .
    .

Submit a basic CUDA program in batch. 
=====================================

Copy the following folder to your directory. This will have a file ``01-hello-gpu.cu`` and a SLURM file ``submit.slurm`` which will submit the job on the Ampere-q partition. If not, save the following code, as ``01-hello-gpu.cu`` in youur directory. 
The files should also contain the solutions by the filename ``01-hello-gpu-solution.cu``

.. code-block:: CUDA
    :caption: Very first CUDA exercise

    #include <stdio.h>

    void helloCPU()
    {
    printf("Hello from the CPU.\n");
    }

    /*
    * Refactor the `helloGPU` definition to be a kernel
    * that can be launched on the GPU. Update its message
    * to read "Hello from the GPU!"
    */

    void helloGPU()
    {
    printf("Hello also from the GPU.\n");
    }

    int main()
    {

    /*
    * Refactor this call to `helloGPU` so that it launches
    * as a kernel on the GPU.
    */

    /*
    * Add code below to synchronize on the completion of the
    * `helloGPU` kernel completion before continuing the CPU
    * thread.
    */
    }


In this progarm two functions are provided, one is to be run from CPU and the other from GPU. Take your time to see if you can write a main function to solve the problem above. 

Once, you have a result, the code is compiled as follows, if you are on an interactive session. 

.. code-block:: bash

    nvcc 01-hello-gpu.cu -o 01hello.out -gencode arch=compute_80,code=sm_80

This should create an executable file by the name of 01hello.out (you can name this anything else, as the output flag is ``-o``)

In an interactive session, this is run by, 

.. code-block:: bash

    ./01hello.out

And while submitting via a batch submission, the SLURM script can look like the following, 

.. code-block:: bash

    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --job-name=hellogpu
    #SBATCH --time=00:05:00
    #SBATCH --partition=ampereq
    #SBATCH --gres=gpu:8          ### you can change this number accordingly but cannot exceed 8 (on ampereq)

    module load cuda-12.2.2
    module load gcc-uoneasy/8.3.0

    nvcc 01-hello-gpu-solution.cu -o 01hello.out -gencode arch=compute_80,code=sm_80

    ./01hello.out


The solution is provided below and in ``01-hello-gpu-solution.cu``, as follows:

.. code-block:: cpp

    #include <stdio.h>
    void helloCPU()
    {
        printf("Hello from the CPU.\n");
    }
    /*
    * The addition of `__global__` signifies that this function
    * should be launced on the GPU.
    */
    __global__ void helloGPU()
    {
        printf("Hello from the GPU.\n");
    }

    int main()
    {
        helloCPU();
    /*
    * Add an execution configuration with the <<<...>>> syntax
    * will launch this function as a kernel on the GPU.
    * The numbers inside the <<<gridDim, blockDim>>> will be discussed further.
    */

        helloGPU<<<1, 32>>>(); //change these numbers and have fun.

    /*
    * `cudaDeviceSynchronize` will block the CPU stream until
    * all GPU kernels have completed.
    */

        cudaDeviceSynchronize();
    }



The output should then look something like the following, depending on the number provided in the kernels. or ``<<<1,16>>>`` in the case below. 

.. code-block:: cpp

    Hello from the CPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.
    Hello also from the GPU.



Congratulations on having run the first CUDA code. 


