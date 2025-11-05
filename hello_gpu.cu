#include <iostream>

// we use global - it will tell cuda that this function is to be run on the GPU
__global__ void helloFromGPU(){
    printf("Hello from GPU!", threadIdx.x, blockIdx.x);
}

int main(){
    //now we will launch 2 block with 4 threads
    helloFromGPU<<<2,4>>>();
    //wait for threads to finish
    cudaDeviceSynchronize();

    std::cout<<"Hello from CPU!"<<std::endl;
    return 0;
}