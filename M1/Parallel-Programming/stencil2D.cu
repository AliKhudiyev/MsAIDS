#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

__global__ void stencil2D(float* in, float* out, size_t R, size_t N){
    size_t i = blockIdx.x, j = threadIdx.x;

    out[(R+i)*N+(R+j)] = in[(R+i)*N+(R+j)];
    for(size_t r=0; r<R; ++r){
        out[(R+i)*N+(R+j)] += in[(R+i-(r+1))*N + (R+j)];
        out[(R+i)*N+(R+j)] += in[(R+i+(r+1))*N + (R+j)];
        out[(R+i)*N+(R+j)] += in[(R+i)*N + ((R+j)-(r+1))];
        out[(R+i)*N+(R+j)] += in[(R+i)*N + ((R+j)+(r+1))];
    }
}

int main(){
    const size_t N = 10, R = 1;
    constexpr size_t n_byte = N*N*sizeof(float);
    float in[N][N], out[N][N];

    for(size_t i=0; i<N*N; ++i){
        in[i/N][i%N] = rand() % N;
    }
    memset(out, 0, n_byte);

    float* d_in, * d_out;

    cudaMalloc((void**)&d_in, n_byte);
    cudaMalloc((void**)&d_out, n_byte);

    cudaMemcpy(d_in, in, n_byte, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, n_byte, cudaMemcpyHostToDevice);

    stencil2D<<<N-2*R, N-2*R>>>(d_in, d_out, R, N);

    cudaMemcpy(in, d_in, n_byte, cudaMemcpyDeviceToHost);
    cudaMemcpy(out, d_out, n_byte, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    float out_[N][N];
    memset(out_, 0, n_byte);
    for(size_t i=0; i<N-2*R; ++i){
        for(size_t j=0; j<N-2*R; ++j){
            out_[R+i][R+j] = in[R+i][R+j];
            for(size_t r=0; r<R; ++r){
                out_[R+i][R+j] += in[R+i - (r+1)][R+j];
                out_[R+i][R+j] += in[R+i + (r+1)][R+j];
                out_[R+i][R+j] += in[R+i][R+j - (r+1)];
                out_[R+i][R+j] += in[R+i][R+j + (r+1)];
            }
        }
    }
    
    printf("%d\n", memcmp(out, out_, n_byte));
    return 0;
}