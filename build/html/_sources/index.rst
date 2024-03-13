.. CUDA on ADA documentation master file, created by
   sphinx-quickstart on Tue Mar  5 13:12:55 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CUDA on ADA's documentation!
=======================================

This documentaion will cover the basics of GPU and CUDA (C/c++) for users to efficiently use the GPU hardware currently available on ADA. 

What is a GPU?
==============

Graphics Processing unit better known as GPU were originally designed for image manipulation on a computer screen. GPUs, or graphics processing units, were originally used to process data for computer displays. As time evolved, GPUs became powerful enough to accelarate scientific computing. 

GPUs are almost always used along with CPUs where the main function of a program is being run by the CPU while specific computation intensive functions assigned to the GPU.

Hardware wise, a CPU as in a laptop or desktop system may have 8-12 to 24 CPU cores, however, a GPU can go to thousands of processors, all of which can be utilized in parallel with help of some programming. 
The domains of machine learning, neural networks and solving differential equations etc, specifically benefit from a GPU hardware resource as this can significantly outperform a traditional CPU. 

.. figure:: images/GPUchip.png

.. figure:: images/cpuvsgpu.png

The above image also gives an intutive comparison when a computation (passenger) needs to be performed (commuted).

Lastly, like a CPU, a GPU has separate structures for execution units and memory. This simply means that the process flows for using GPUs is defined differently as follows. 

Overview of using GPU programs
==============================

The following steps give a brief of how GPU programs work (or should be written).

1. Copy data from CPU memory to GPU memory.
2. Transfer program. (The code that tells the processors of what to do with the device memory.)
3. Load the GPU program , execute on streaming processors (SMs), get cached data from device (GPU) memory; write back the results.
4. Copy the results back to the host memory.

.. figure:: images/step1step2.png
.. figure:: images/step3.png
.. figure:: images/step4.png


GPU Resources at University of Nottingham
=========================================


See the `Partitions link <https://uniofnottm.sharepoint.com/sites/DigitalResearch/SitePages/Ada-Commands-Partitions-and-Resources.aspx#partitions>`_ to look at the various hardware resources available for HPC jobs at University of Nottingham.

There are 3 GPU partitions on this list. Namely, 

+----------------------+-----------------------------------------------------+
| GPU partitions       | Properties                                          |
+======================+=====================================================+
| ampereq              | no permissions                                      |
+----------------------+-----------------------------------------------------+
| ampere-devq          | execute                                             |
+----------------------+-----------------------------------------------------+
| ampere-mq            | write                                               |
+----------------------+-----------------------------------------------------+


As all of these partitions contain the current state-of-the-art NVIDIA A100 cards, they all have the following properties. 

.. code-block:: bash

    CUDADevice with properties:

                      Name: 'NVIDIA A100 80GB PCIe'
                     Index: 1
         ComputeCapability: '8.0'
            SupportsDouble: 1
             DriverVersion: 12.2000
            ToolkitVersion: 11.2000
        MaxThreadsPerBlock: 1024
          MaxShmemPerBlock: 49152
        MaxThreadBlockSize: [1024 1024 64]
               MaxGridSize: [2.1475e+09 65535 65535]
                 SIMDWidth: 32
               TotalMemory: 8.5175e+10
           AvailableMemory: 8.4519e+10
       MultiprocessorCount: 108
              ClockRateKHz: 1410000
               ComputeMode: 'Default'
      GPUOverlapsTransfers: 1
    KernelExecutionTimeout: 0
          CanMapHostMemory: 1
           DeviceSupported: 1
           DeviceAvailable: 1
            DeviceSelected: 1

While the CPU properties on these nodes is as follows, 

.. code-block:: bash

   Architecture:        x86_64
   CPU op-mode(s):      32-bit, 64-bit
   Byte Order:          Little Endian
   CPU(s):              96
   On-line CPU(s) list: 0-95
   Thread(s) per core:  1
   Core(s) per socket:  48
   Socket(s):           2
   NUMA node(s):        16
   Vendor ID:           AuthenticAMD
   CPU family:          25
   Model:               17
   Model name:          AMD EPYC 9454 48-Core Processor
   Stepping:            1
   CPU MHz:             2750.000
   CPU max MHz:         3810.7910
   CPU min MHz:         1500.0000
   BogoMIPS:            5492.04
   Virtualization:      AMD-V
   L1d cache:           32K
   L1i cache:           32K
   L2 cache:            1024K
   L3 cache:            32768K
   NUMA node0 CPU(s):   0-5
   NUMA node1 CPU(s):   6-11
   NUMA node2 CPU(s):   12-17
   NUMA node3 CPU(s):   18-23
   NUMA node4 CPU(s):   24-29
   NUMA node5 CPU(s):   30-35
   NUMA node6 CPU(s):   36-41
   NUMA node7 CPU(s):   42-47
   NUMA node8 CPU(s):   48-53
   NUMA node9 CPU(s):   54-59
   NUMA node10 CPU(s):  60-65
   NUMA node11 CPU(s):  66-71
   NUMA node12 CPU(s):  72-77
   NUMA node13 CPU(s):  78-83
   NUMA node14 CPU(s):  84-89
   NUMA node15 CPU(s):  90-95


Feel free to inquire about other GPU related properties through the command,

``nvidia-smi -q | less``

Finally, a streaming multiprocessor or SM's configuration for the A100 card is shown below. 

.. figure:: images/A100arch.png
   :align: center
   :alt: Your image alt text

   Diagram of streaming multiprocessor for `NVIDIA A100 GPU <https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/>`_ card.


Another important feature that is useful while writing/compiling CUDA programs is the compute_capability of the hardware, which in case of A100s, is 8.0. This can be obtained from the ``nvcc --help`` command after loading the cuda module with ``module load cuda/12.2.2``. This will also become more clear with examples. 






Chapters
========

The documentation covers the following aspects of CUDA and GPU programming.

.. toctree::

    01Helloworld


More information can be obtained from folllowing Nvidia's pages [1]_ and [2]_.

[1] `Programming guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>`_
[2] `Runtime API <https://docs.nvidia.com/cuda/cuda-runtime-api/index.html>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


