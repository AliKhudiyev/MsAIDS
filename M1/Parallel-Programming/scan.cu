#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string.h>

using namespace std;

#define N 100

// 3 2 4 5
// 3 5 6 9
// 

__global__ void scan(float* A, float* B){
    if(!threadIdx.x)
        B[0] = 0;
    B[threadIdx.x+1] = A[threadIdx.x];
    __syncthreads();

    for(int i=1; i<blockDim.x; ++i){
        if(threadIdx.x >= i)
            B[threadIdx.x+1] += A[threadIdx.x-i];
        else break;
        __syncthreads();
    }
}

int main(){
    srand(time(0));

    float A[N], B[N+1], B_[N+1] = { 0 };
    constexpr size_t n_byte = N*sizeof(float);

    for(size_t i=0; i<N; ++i){
        A[i] = rand() % 10;
        B_[i+1] = B_[i] + A[i];
    }

    float* d_A, * d_B;
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_A, n_byte);
    cudaMalloc(&d_B, n_byte+sizeof(float));

    cudaMemcpy(d_A, A, n_byte, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    scan<<<1, N>>>(d_A, d_B);
    cudaEventRecord(stop);
    cudaMemcpy(B, d_B, n_byte+sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("--- Took %f ms to execute ---\n", milliseconds);
    printf("--- Effective Bandwidth (GB/s): %f ---\n", N*4*3/milliseconds/1e6);

    cudaFree(d_A);
    cudaFree(d_B);
    
    printf("%d\n", memcmp(B, B_, n_byte+sizeof(float)));
    return 0;
}