# GPUMatrixInversion
My final project in the technion class 236509 - Advanced topics in computer architecture

# Running instructions
In order to run the CUDA code, a GPU is required.
To compile it, the command is:
"""
nvcc matrixInversion_gpu.cu -o gpu_mi
"""
And to run:
"""
./gpu_mi
"""
An L.txt file will be created with a random matrix, size depending on the inner code, and the inverse of that matrix in inv.txt.

Similarly, in order to use the classical CPU impalemtion, use:
"""
nvcc matrixInversion_cpu.cpp -o cpu_mi3
./cpu_mi
"""
It will read L.txt and create inverse.txt file with the inverted matrix.

In order to run the pytorch comparison, pytorch is required. It can be run by:
"""
python torch_compare.py > time_results.txt
"""
time_results.txt will hold the runtime results in 2 columns, one for CPU and one for GPU.

