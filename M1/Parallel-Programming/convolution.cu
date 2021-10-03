#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define N (10*10)
#define M (3*3)

__global__ void convolve(float* A, float* B, float* mask, int len){
    float tmp = 0, elem = 0;
    for(size_t j=0; j<len; ++j, elem=0){
        if(threadIdx.x+j>=len/2 && threadIdx.x+j-len/2<blockDim.x)
            elem = A[threadIdx.x+j-len/2];
        tmp += elem * mask[j];
    }
    B[threadIdx.x] = tmp;
}

__global__ void convolve2D(float* A, float* B, float* mask, int width, int height){
    float tmp = 0, elem = 0;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(int i=0; i<height; ++i){
        for(int j=0; j<width; ++j, elem=0){
            if(blockIdx.x+i>=height/2 && blockIdx.x+i-height/2<gridDim.x &&
               threadIdx.x+j>=width/2 && threadIdx.x+j-width/2<blockDim.x)
                elem = A[(blockIdx.x+i-height/2)*blockDim.x+(threadIdx.x+j-width/2)];
            tmp += elem * mask[i*width+j];
        }
    }
    B[index] = tmp;
}

int main(){
    srand(time(0));

    float A[N], B[N], /* B_[N] = { 0 }, */ mask[M];
    constexpr size_t n_byte = N*sizeof(float);

    for(size_t i=0; i<M; mask[i++]=rand()%5/*, printf("%.1f\t", mask[i-1])*/);
    // printf("\n");
    for(size_t i=0; i<N; ++i){
        A[i] = rand() % 10;
        // printf("%.1f\t", A[i]);
    }   // printf("\n");

    float* d_A, * d_B, * d_mask;
    
    cudaMalloc(&d_A, n_byte);
    cudaMalloc(&d_B, n_byte);
    cudaMalloc(&d_mask, M*sizeof(float));

    cudaMemcpy(d_A, A, n_byte, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, M*sizeof(float), cudaMemcpyHostToDevice);

    // convolve<<<1, N>>>(d_A, d_B, d_mask, M);
    convolve2D<<<10, 10>>>(d_A, d_B, d_mask, 3, 3);
    cudaMemcpy(B, d_B, n_byte, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_mask);

    // for(size_t i=0; i<N; ++i){
    //     float tmp = 0, elem = 0;
    //     for(size_t j=0; j<M; ++j, elem=0){
    //         if(i+j>=M/2 && i+j-M/2<N) elem = A[i+j-M/2];
    //         tmp += elem * mask[j];
    //     }
    //     B_[i] = tmp;
    // }

    // for(size_t i=0; i<N; printf("%f\t", B_[i++]));
    // printf("\n");
    // for(size_t i=0; i<N; printf("%f\t", B[i++]));
    // printf("\n");

    for(size_t i=0; i<3; ++i){
        for(size_t j=0; j<3; ++j){
            printf("%.1f ", mask[i*3+j]);
        }   printf("\n");
    }   printf("\n");
    for(size_t i=0; i<10; ++i){
        for(size_t j=0; j<10; ++j){
            printf("%.1f ", A[i*10+j]);
        }   printf("\n");
    }   printf("\n");
    for(size_t i=0; i<10; ++i){
        for(size_t j=0; j<10; ++j){
            printf("%.1f ", B[i*10+j]);
        }   printf("\n");
    }

    // printf("%d\n", memcmp(B, B_, n_byte));

    return 0;
}