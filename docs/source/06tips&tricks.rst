Improving GPU utilization
-------------------------

The general workflow of GPU program as mentioned in :ref:`Overview of using GPU programs` is that

1. Copy data from CPU memory to GPU memory.
2. Transfer program. (The code that tells the processors of what to do with the device memory.)
3. Load the GPU program , execute on streaming processors (SMs), get cached data from device (GPU) memory; write back the results.
4. Copy the results back to the host memory.

This is even more effective if the GPU transfers of memory can be minimized, while keeping the thread utilization, so that none of them sit idle due to a lack of memory. 
The code/algorithm itself should be able to run on GPU. 

This is usually simply done by print statements and sometimes even through profiling. See their help guidres here `Nvidia Nsight systems <https://docs.nvidia.com/nsight-systems/>`_.
This feature is available via the `cuda-uoneasy/12.1.1` module

.. code-block:: bash

    $ nsys --version
    NVIDIA Nsight Systems version 2023.1.2.43-32377213v0

    $ ncu --version
    NVIDIA (R) Nsight Compute Command Line Profiler
    Copyright (c) 2018-2023 NVIDIA Corporation
    Version 2023.1.1.0 (build 32678585) (public-release)

The gui based profiling can be done through the visualization nodes and loading the `cuda-uoneasy/12.1.1` module. 


Resources
---------

Some other additional resources are mentioned below, 

1. `Programming guide <https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf>`_
2. `Best practices guide <https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf>`_.
3. `CUDA training series <https://www.youtube.com/playlist?list=PL6RdenZrxrw-zNX7uuGppWETdxt_JxdMj>`_
4.  Oak Ridge National Laboratory `CUDA training <https://www.olcf.ornl.gov/cuda-training-series/>`_.







