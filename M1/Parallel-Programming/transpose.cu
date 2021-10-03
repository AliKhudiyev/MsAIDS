#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 5
#define M 5

__global__ void transpose(float* A, float* B){
    B[blockIdx.x*M + threadIdx.x] = A[threadIdx.x*N + blockIdx.x];
}

int main(){
    float A[N][M], B[M][N];
    constexpr size_t n_byte = N*M*sizeof(float);

    for(size_t i=0; i<N*M; ++i){
        A[i/N][i%M] = rand() % 100;
    }

    float* d_A, * d_B;

    cudaMalloc((void**)&d_A, n_byte);
    cudaMalloc((void**)&d_B, n_byte);

    cudaMemcpy(d_A, A, n_byte, cudaMemcpyHostToDevice);
    transpose<<<N, M>>>(d_A, d_B);
    cudaMemcpy(B, d_B, n_byte, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    for(size_t i=0; i<N*M; ++i){
        if(i%M == 0) printf("\n");
        printf("%f\t", A[i/M][i%M]);
    }
    printf("\n");
    for(size_t i=0; i<N*M; ++i){
        if(i%M == 0) printf("\n");
        printf("%f\t", B[i/M][i%M]);
    }
    printf("\n\n");

    return 0;
}