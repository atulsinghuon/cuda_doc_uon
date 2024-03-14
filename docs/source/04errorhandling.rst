Error handling in CUDA
======================

CUDA provides inbuilt functions with its API that are helpful for error handling. (`Explore more here <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html>`_)

The following are a few methods that are discussed with an example. This example simply doubles an arrays and checks with the help of functions, 
``cudaGetLastError()``, to assess if the required pre-defined condition has been met. 

.. cpp:function:: cudaError_t cudaGetLastError(void)

    Returns the last error that has been produced by any of the runtime calls in the same instance of the CUDA Runtime library in the host thread and resets it to cudaSuccess.
    Note: Multiple instances of the CUDA Runtime library can be present in an application when using a library that statically links the CUDA Runtime. 

   :return: Return cudaSuccess

.. cpp:function:: cudaError_t cudaGetErrorString(void)

    Returns the description string for an error code. If the error code is not recognized, "unrecognized error code" is returned.

    :param: Error code to convert to string. 
    :return: pointer to null terminated strings. 


See the following example. 

.. tabs::

   .. code-tab:: cuda

             #include <stdio.h>

            void init(int *a, int N)
            {
                int i;
                for (i = 0; i < N; ++i)
                {
                    a[i] = i;
                }
            }

            __global__ void doubleElements(int *a, int N){

                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int stride = gridDim.x * blockDim.x;

                /*
                * The previous code (now commented out) attempted
                * to access an element outside the range of `a`.
                */

                // for (int i = idx; i < N + stride; i += stride)
                for (int i = idx; i < N; i += stride)
                {
                    a[i] *= 2;
                }
            }

            bool checkElementsAreDoubled(int *a, int N)
            {
                int i;
                for (i = 0; i < N; ++i)
                {
                    if (a[i] != i*2) return false;
                }
                return true;
            }

            int main()
            {
                int N = 1000000;
                int *a;

                size_t size = N * sizeof(int);
                cudaMallocManaged(&a, size);

                init(a, N);

                /*
                * The previous code (now commented out) attempted to launch
                * the kernel with more than the maximum number of threads per
                * block, which is 1024.
                */

                size_t threads_per_block = 1024;
                /* size_t threads_per_block = 1024; */
                size_t number_of_blocks = 32;

                cudaError_t syncErr, asyncErr;

                doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);

                /*
                * Catch errors for both the kernel launch above and any
                * errors that occur during the asynchronous `doubleElements`
                * kernel execution.
                */

                syncErr = cudaGetLastError();
                asyncErr = cudaDeviceSynchronize();

                /*
                * Print errors should they exist.   
                */

                if (syncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(syncErr));
                if (asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

                bool areDoubled = checkElementsAreDoubled(a, N);
                printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

                cudaFree(a);
            }

   .. code-tab:: make

         # Compiler options
         NVCC = nvcc
         GENCODE = -gencode arch=compute_80,code=sm_80

         # Target executable
         TARGET = err.out

         # Source files #This is the name of the saved code. Change this if you change the file name.
         SRCS = error_handling.cpp

         # Rule to build the executable
         $(TARGET): $(SRCS)
                 $(NVCC) $(SRCS) -o $(TARGET) $(GENCODE)

         # Clean rule
         clean:
                 rm -f $(TARGET)

   .. code-tab:: slurm

         #!/bin/bash
         #SBATCH --nodes=1 
         #SBATCH --job-name=errorhandle
         #SBATCH --time=00:10:00
         #SBATCH --partition=ampere-mq     
         #SBATCH --gres=gpu:1

         module load cuda-12.2.2
         module load gcc-uoneasy/8.3.0

         make
         
         #The executable will be named after the "-o" flag in the #TARGET variable inside makefile. 
         ./err.out 

   .. code-tab:: bash Solution

         All elements were doubled? True