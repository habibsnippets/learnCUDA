#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

//define the kernel which will run the part of the code on your gpu
__global__ void scaleKernel(const float* d_A, float* d_C, float alpha, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        d_C[idx] = alpha * d_A[idx]; // main function that we want to execute
    }
}


//now allocate and inititalize the inputs
int main()
{
    // what we want to do ?
    // we want to write a program that does the following:
    // C[i] = alpha * A[i] for i = 0 to N-1

    int N = 1000;
    float alpha = 3.14f;
    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    //allocate to host CPU
    float *h_A = new float[N];
    float *h_C = new float[N]; //ouptut

    //init the input
    for (int i =0; i< N ; ++i){
        h_A[i] = static_cast<float>(i) * 0.001f;
    }
    //device pointers
    float *d_A = nullptr;
    float *d_C = nullptr;

    //allocate device memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);



    //copy from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    //device kernel


    //launch and run kernel
    int threadsPerBlock = 256;
    int blocks = (N+ threadsPerBlock - 1) / threadsPerBlock;

    //call the scaleKernel 
    scaleKernel<<<blocks, threadsPerBlock>>>(d_A, d_C, alpha, N);

    //wait for the kernel to finish before you copy the results from gpu to the cpu again
    cudaDeviceSynchronize();

    //after computing the results get them back to the cpu
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);





    //print for few examples
    for (int i =0; i< 10; ++i)
        std::cout<< i << h_A[i] << " * " << alpha << " = " << h_C[i] << std::endl;

    //now free the spaces - Cleanup
    cudaFree(d_A);
    cudaFree(d_C);
    
    delete[] h_A;
    delete[] h_C;

    return 0;



}