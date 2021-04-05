#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define N 4
#define M 4
#define K 4

#define Width 2

__global__ void matmul(float* A, float* B, float* C){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp = 0;
    for(size_t i=0; i<Width; ++i){
        tmp += A[row * Width + i] * B[i * Width + col];
    }
    C[row * Width + col] = tmp;
}

int main(){
    srand(time(0));
    float A[N][M], B[M][K], C[N][K];

    for(size_t i=0; i<N*M; ++i){
        A[i/M][i%M] = rand() % N;
    }
    for(size_t i=0; i<M*K; ++i){
        B[i/K][i%K] = rand() % M;
    }

    float* d_A, * d_B, * d_C;

    cudaMalloc((void**)&d_A, N*M*sizeof(float));
    cudaMalloc((void**)&d_B, M*K*sizeof(float));
    cudaMalloc((void**)&d_C, N*K*sizeof(float));

    cudaMemcpy(d_A, A, N*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M*K*sizeof(float), cudaMemcpyHostToDevice);

    dim3 dim_grid(2, 2, 1);
    dim3 dim_block(2, 2, 1);

    matmul<<<dim_grid, dim_block>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, N*K*sizeof(float), cudaMemcpyDeviceToHost);

    float C_[N][K];
    for(size_t i=0; i<N; ++i){
        for(size_t j=0; j<K; ++j){
            C_[i][j] = 0;
            for(size_t t=0; t<M; ++t){
                C_[i][j] += A[i][t] * B[t][j];
            }
        }
    }

    printf("Ground Truth:\n");
    for(int i=0; i<N; ++i){
        for(int j=0;; i<K++i){
            printf("%f\t", C[i][j]);
        }   printf("\n");
    }   printf("\n\n");
    printf("Result:\n");
    for(int i=0; i<N; ++i){
        for(int j=0;; i<K++i){
            printf("%f\t", C[i][j]);
        }   printf("\n");
    }   printf("\n\n");

    printf("%d\n", memcmp(C, C_, N*K*sizeof(float)));
    return 0;
}