#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

using namespace std;

__global__ void add_vector(float* in, float* out, int R, int N)
{
    if(threadIdx.x+2*R >= N) return ;

    out[R+threadIdx.x] = 0;
    for(int i=0; i<2*R+1; ++i){
        out[R+threadIdx.x] += in[threadIdx.x+i];
    }
}

int main(int argc, char** argv){
    srand(time(0));

    const size_t N = 10;
    const size_t R = 3;
    constexpr size_t n_byte = N * sizeof(float);

    float in[N], out[N];

    for(size_t i=0; i<N; ++i){
        in[i] = rand() % N;
    }
    memset(out, 0, N*sizeof(float));

    float* d_in, * d_out;

    cudaMalloc((void**)&d_in, n_byte);
    cudaMalloc((void**)&d_out, n_byte);

    cudaMemcpy(d_in, in, n_byte, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, n_byte, cudaMemcpyHostToDevice);

    add_vector<<<1, N>>>(d_in, d_out, R, N);

    cudaMemcpy(in, d_in, n_byte, cudaMemcpyDeviceToHost);
    cudaMemcpy(out, d_out, n_byte, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    float out_[N];
    memset(out_, 0, N*sizeof(float));
    for(size_t i=R; i<N-R; ++i){
        out_[i] = 0.f;
        for(size_t j=0; j<2*R+1; ++j){
            out_[i] += in[j+i-R];
        }
    }

    printf("%d\n", memcmp(out, out_, n_byte));
    return 0;
}