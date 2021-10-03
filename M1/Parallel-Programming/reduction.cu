#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define N (1024*10)

// 
__global__ void reduction(float* A){
    for(size_t i=blockDim.x/2, j=blockDim.x%2, tmp=0; i; tmp=i+j, i=tmp/2, j=tmp%2){
        if(threadIdx.x < i)
            A[1024*blockIdx.x+threadIdx.x] += A[1024*blockIdx.x+threadIdx.x+i+j];
        __syncthreads();
    }
    
    for(size_t i=gridDim.x/2, j=gridDim.x%2, tmp=0; i; tmp=i+j, i=tmp/2, j=tmp%2){
        if(blockIdx.x == 0 && threadIdx.x < i)
            A[1024*threadIdx.x] += A[1024*(threadIdx.x+i+j)];
        __syncthreads();
    }
}

// 
__global__ void reduction2(float* A){
    for(size_t i=blockDim.x/2, j=blockDim.x%2, tmp=0; i; tmp=i+j, i=tmp/2, j=tmp%2){
        if(threadIdx.x < i)
            A[1024*blockIdx.x+threadIdx.x] += A[1024*blockIdx.x+threadIdx.x+i+j];
        __syncthreads();
    }
    
    for(size_t i=gridDim.x/2, j=gridDim.x%2, tmp=0; i; tmp=i+j, i=tmp/2, j=tmp%2){
        if(blockIdx.x == 0 && threadIdx.x < i)
            A[1024*threadIdx.x] += A[1024*(threadIdx.x+i+j)];
        __syncthreads();
    }
}

// 
__global__ void reduction3(float* A){
    for(size_t i=blockDim.x/2, j=blockDim.x%2, tmp=0; i; tmp=i+j, i=tmp/2, j=tmp%2){
        if(threadIdx.x < i)
            A[1024*blockIdx.x+threadIdx.x] += A[1024*blockIdx.x+threadIdx.x+i+j];
        __syncthreads();
    }
    
    for(size_t i=gridDim.x/2, j=gridDim.x%2, tmp=0; i; tmp=i+j, i=tmp/2, j=tmp%2){
        if(blockIdx.x == 0 && threadIdx.x < i)
            A[1024*threadIdx.x] += A[1024*(threadIdx.x+i+j)];
        __syncthreads();
    }
}

// 
__global__ void reduction4(float* A){
    for(size_t i=blockDim.x/2, j=blockDim.x%2, tmp=0; i; tmp=i+j, i=tmp/2, j=tmp%2){
        if(threadIdx.x < i)
            A[1024*blockIdx.x+threadIdx.x] += A[1024*blockIdx.x+threadIdx.x+i+j];
        __syncthreads();
    }
    
    for(size_t i=gridDim.x/2, j=gridDim.x%2, tmp=0; i; tmp=i+j, i=tmp/2, j=tmp%2){
        if(blockIdx.x == 0 && threadIdx.x < i)
            A[1024*threadIdx.x] += A[1024*(threadIdx.x+i+j)];
        __syncthreads();
    }
}

int main(){
    srand(time(0));
    float A[N], kernel_sum, sum = 0.f;
    constexpr size_t n_byte = N*sizeof(float);

    for(size_t i=0; i<N; ++i){
        A[i] = rand() % 100;
        // printf("%.1f\t", A[i]);
        sum += A[i];
    }   // printf("\n");

    float* d_A;
    cudaMalloc((void**)&d_A, n_byte);

    cudaMemcpy(d_A, A, n_byte, cudaMemcpyHostToDevice);
    reduction<<<10, 1024>>>(d_A);
    cudaMemcpy(&kernel_sum, d_A, sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f - %f\n", kernel_sum, sum);
    printf("%f\n", kernel_sum - sum);
    return 0;
}