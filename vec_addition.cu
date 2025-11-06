//Vector Addition in CUDA - First Kernel
#include <iostream>
#include <cuda_runtime.h>

//GPU Kernel
__global__ void vectorAdd(const float *A,const float *B, float *C, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<n)
        C[idx] = A[idx] + B[idx];
}

int main()
{
    int N = 1000;
    size_t size = N * sizeof(float);
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];


    //Initialize
    for (int i = 0; i <N; ++i)
    {
        h_A[i]=i;
        h_B[i]= 2 * i;
    }

    //Allocate GPU memory
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    //Copy data from CPU to the GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //Launch Kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock -1 )/ threadsPerBlock;
    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    //Copy the rese 2: Image Processisults back after getting from the above vectorAdd function
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    //print few results to check them
    for (int i =0; i< 10; ++i)
        std::cout << h_A[i] << "+" << h_B[i] << " = " << h_C[i] << std::endl;

    //Free the GPU Memory - cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}

