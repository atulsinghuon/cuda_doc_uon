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


Scale vector example
--------------------
Now lets perform a simple vector addition in CUDA using the threads and kernel function knowledge from above. 

This code of vector addition, first defines the ``vec_add`` kernel function that will perform the operation of 

``z[i] = x[i] + y[i]``

with i being the thread index for a kernel function. Then we define host and device pointers, ``h_x``, ``h_y``, ``h_z`` and ``d_x``, ``d_y``, ``d_z`` respectively. After populating the arrays with significantly large numbers, we user ``cudaMalloc`` and ``cudaMemcpy`` to transfer the the allocated data to device. 

We then launch the kernel function, once the memory is present in the device. Then, copy the results back to the host using ``cudaMemcpy`` again. 
And once the operation has been viewed, we free the memory. 

.. code-block:: cpp

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

 
