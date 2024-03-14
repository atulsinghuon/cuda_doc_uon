Memory Management
==================


The two commands that are important to understand memory management between host (CPU) and device (GPU) are ``cudaMalloc()`` and ``cudaMallocManaged()``

Lets take a look at them. 

.. cpp:function:: cudaError_t cudaMallocManaged(void** ptr, size_t size)

    Allocates memory that is accessible to both the host and the device. 
   
   :param ptr: Pointer to allocated memory.
   :param size: Requested allocation size in bytes.
   :return: Returns `cudaSuccess` if successful, otherwise returns an error code.


.. code-block:: cpp

    //Example of using the above function is as follows, 
    float *a;
    int N = 2048;
    cudaMallocManaged(&a, N*sizeof(float));
    // do device computations
    cudaFree(a);

While the other function ``cudaMalloc()`` requires a manual ``cudaMemcpy()`` function , as this is a manual data transfer function. 


.. cpp:function:: cudaError_t cudaMalloc(void** ptr, size_t size) 

    Allocates pointer T of size nBytes manually to be used by the GPU. The allocated memory is suitably aligned for any type of variable. 
   
   :param ptr: Pointer to allocated memory.
   :param size: Requested allocation size in bytes.
   :return: Returns `cudaSuccess` if successful, otherwise returns an error code.


.. cpp:function:: cudaError_t cudaMemcpy(void* destination, void* source, size_t nBytes, enum cudaMemcpyKind dir)

    Transfers data manually from destination to source, or vice-versa. 

   :param destination: The destination of the copied data.
   :param source: The source of the data.
   :param nBytes: Size of the data transferred. 
   :param cudaMemcpyKind: {cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost}
   :return: Returns `cudaSuccess` if successful, otherwise returns an error code.


See an example below, that copies the data to device, launches a kernel and copies the data back to the host. 

.. code-block:: c

    float *a_host, *a_device;
    int N = 2048;
    //fill a
    // allocate memory
    cudaMalloc(&a_device, N*sizeof(float));

    // copy memory to device
    cudaMemcpy(a_device, a_host, N*sizeof(float), cudaMemcpyHostToDevice);

    // perform gpu computation
    my_kernel_function<<<8, 16>>>(a_d, N);

    // copy back the return back to host
    cudaMemcpy(a_host, a_device, N*sizeof(float), cudaMemcpyDeviceToHost);


Vector addition example
-----------------------
Now lets perform a simple vector addition in CUDA using the threads and kernel function knowledge from above. 

This code of vector addition, first defines the ``vec_add`` kernel function that will perform the operation of 

``z[i] = x[i] + y[i]``

with i being the thread index for a kernel function. Then we define host and device pointers, ``h_x``, ``h_y``, ``h_z`` and ``d_x``, ``d_y``, ``d_z`` respectively. After populating the arrays with significantly large numbers, we user ``cudaMalloc`` and ``cudaMemcpy`` to transfer the the allocated data to device. 

We then launch the kernel function, once the memory is present in the device. Then, copy the results back to the host using ``cudaMemcpy`` again. 
And once the operation has been viewed, we free the memory. 

.. tabs::


  .. code-tab:: cuda

            #include <stdio.h>
            #include <unistd.h>
            #include <stdlib.h>
            #include <math.h>

            __global__ void Vec_add(float x[], float y[], float z[], int n) {
            int thread_id = threadIdx.x;
            if (thread_id < n){
                z[thread_id] = x[thread_id] + y[thread_id];
            }
            }

            int main(int argc, char* argv[]) {

            int n, m;
            float *h_x, *h_y, *h_z;
            float *d_x, *d_y, *d_z;
            size_t size;

            /* Define vector length */
            n = 10000000;
            m = 200;
            size = n*sizeof(float);

            // Allocate memory for the vectors on host memory.
            h_x = (float*) malloc(size);
            h_y = (float*) malloc(size);
            h_z = (float*) malloc(size);

            for (int i = 0; i < n; i++) {
                h_x[i] = i+1;
                h_y[i] = n-i;
            }

            // Print original vectors.
            printf("h_x = ");
            for (int i = 0; i < m; i++){
                printf("%.1f ", h_x[i]);
            }
            printf("\n\n");
            printf("h_y = ");
            for (int i = 0; i < m; i++){
                printf("%.1f ", h_y[i]);
            }
            printf("\n\n");

                /* Allocate vectors in device memory */
            cudaMalloc(&d_x, size);
            cudaMalloc(&d_y, size);
            cudaMalloc(&d_z, size);

            /* Copy vectors from host memory to device memory */
            cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

            /* Kernel Call */
            Vec_add<<<1,1024>>>(d_x, d_y, d_z, n);

            cudaDeviceSynchronize();
            cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);
            printf("The sum is: \n");
            for (int i = 0; i < m; i++){
                printf("%.1f ", h_z[i]);
            }
            printf("\n");


            /* Free device memory */
            cudaFree(d_x);
            cudaFree(d_y);
            cudaFree(d_z);
            /* Free host memory */
            free(h_x);
            free(h_y);
            free(h_z);

            return 0;
            } 

  .. code-tab:: make

            # Compiler options
            NVCC = nvcc
            GENCODE = -gencode arch=compute_80,code=sm_80

            # Target executable
            TARGET = vecadd.out

            # Source files #This is the name of the saved code. Change this if you change the file name.
            SRCS = vec_add.cu

            # Rule to build the executable
            $(TARGET): $(SRCS)
                    $(NVCC) $(SRCS) -o $(TARGET) $(GENCODE)

            # Clean rule
            clean:
                    rm -f $(TARGET)

  .. code-tab:: slurm

            #!/bin/bash
            #SBATCH --nodes=1
            #SBATCH --job-name=vecadd
            #SBATCH --time=00:05:00
            #SBATCH --partition=ampere-mq
            #SBATCH --gres=gpu:1          ### you can change this number accordingly but cannot exceed 8 (on ampereq)

            module load cuda-12.2.2
            module load gcc-uoneasy/8.3.0

            ##nvcc vec_add.cu -o vecadd.out -gencode arch=compute_80,code=sm_80

            make

            ./vecadd.out

  .. code-tab:: bash Solution

            h_x = 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0 96.0 97.0 98.0 99.0 100.0 101.0 102.0 103.0 104.0 105.0 106.0 107.0 108.0 109.0 110.0 111.0 112.0 113.0 114.0 115.0 116.0 117.0 118.0 119.0 120.0 121.0 122.0 123.0 124.0 125.0 126.0 127.0 128.0 129.0 130.0 131.0 132.0 133.0 134.0 135.0 136.0 137.0 138.0 139.0 140.0 141.0 142.0 143.0 144.0 145.0 146.0 147.0 148.0 149.0 150.0 151.0 152.0 153.0 154.0 155.0 156.0 157.0 158.0 159.0 160.0 161.0 162.0 163.0 164.0 165.0 166.0 167.0 168.0 169.0 170.0 171.0 172.0 173.0 174.0 175.0 176.0 177.0 178.0 179.0 180.0 181.0 182.0 183.0 184.0 185.0 186.0 187.0 188.0 189.0 190.0 191.0 192.0 193.0 194.0 195.0 196.0 197.0 198.0 199.0 200.0

            h_y = 10000000.0 9999999.0 9999998.0 9999997.0 9999996.0 9999995.0 9999994.0 9999993.0 9999992.0 9999991.0 9999990.0 9999989.0 9999988.0 9999987.0 9999986.0 9999985.0 9999984.0 9999983.0 9999982.0 9999981.0 9999980.0 9999979.0 9999978.0 9999977.0 9999976.0 9999975.0 9999974.0 9999973.0 9999972.0 9999971.0 9999970.0 9999969.0 9999968.0 9999967.0 9999966.0 9999965.0 9999964.0 9999963.0 9999962.0 9999961.0 9999960.0 9999959.0 9999958.0 9999957.0 9999956.0 9999955.0 9999954.0 9999953.0 9999952.0 9999951.0 9999950.0 9999949.0 9999948.0 9999947.0 9999946.0 9999945.0 9999944.0 9999943.0 9999942.0 9999941.0 9999940.0 9999939.0 9999938.0 9999937.0 9999936.0 9999935.0 9999934.0 9999933.0 9999932.0 9999931.0 9999930.0 9999929.0 9999928.0 9999927.0 9999926.0 9999925.0 9999924.0 9999923.0 9999922.0 9999921.0 9999920.0 9999919.0 9999918.0 9999917.0 9999916.0 9999915.0 9999914.0 9999913.0 9999912.0 9999911.0 9999910.0 9999909.0 9999908.0 9999907.0 9999906.0 9999905.0 9999904.0 9999903.0 9999902.0 9999901.0 9999900.0 9999899.0 9999898.0 9999897.0 9999896.0 9999895.0 9999894.0 9999893.0 9999892.0 9999891.0 9999890.0 9999889.0 9999888.0 9999887.0 9999886.0 9999885.0 9999884.0 9999883.0 9999882.0 9999881.0 9999880.0 9999879.0 9999878.0 9999877.0 9999876.0 9999875.0 9999874.0 9999873.0 9999872.0 9999871.0 9999870.0 9999869.0 9999868.0 9999867.0 9999866.0 9999865.0 9999864.0 9999863.0 9999862.0 9999861.0 9999860.0 9999859.0 9999858.0 9999857.0 9999856.0 9999855.0 9999854.0 9999853.0 9999852.0 9999851.0 9999850.0 9999849.0 9999848.0 9999847.0 9999846.0 9999845.0 9999844.0 9999843.0 9999842.0 9999841.0 9999840.0 9999839.0 9999838.0 9999837.0 9999836.0 9999835.0 9999834.0 9999833.0 9999832.0 9999831.0 9999830.0 9999829.0 9999828.0 9999827.0 9999826.0 9999825.0 9999824.0 9999823.0 9999822.0 9999821.0 9999820.0 9999819.0 9999818.0 9999817.0 9999816.0 9999815.0 9999814.0 9999813.0 9999812.0 9999811.0 9999810.0 9999809.0 9999808.0 9999807.0 9999806.0 9999805.0 9999804.0 9999803.0 9999802.0 9999801.0

            The sum is:
            10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0 10000001.0





 
