#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int N = 5;
    size_t size = N * sizeof(int); // this will calculate the total bytes that will be needed : 5 * 4 = 20 bytes needed here to store
    int h_array[5] = {1,2,3,4,5};// Host CPU Memory
    int *d_array; //Device GPU Pointer

    //Allocate GPU memory
    cudaMalloc(&d_array, size);
    //copy data from CPU to GPU
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
    //copy back from gpu to cpu
    int h_result[5];
    cudaMemcpy(h_result, d_array, size, cudaMemcpyDeviceToHost);
    //Print Results
    for (int i = 0 ; i<N; i++)
        std::cout << h_result[i] << " ";
    std::cout << std::endl;

    return 0;
}