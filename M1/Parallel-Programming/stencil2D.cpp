#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

#define N 10
#define R 1

int main(){
    float buf_in[N][N], buf_out[N][N];

    for(size_t i=0; i<N*N; ++i){
        buf_in[i/N][i%N] = rand() % N;
    }
    memset(buf_out, 0, N*N*sizeof(float));

    for(size_t i=0; i<N-2*R; ++i){
        for(size_t j=0; j<N-2*R; ++j){
            buf_out[R+i][R+j] = buf_in[R+i][R+j];
            for(size_t r=0; r<R; ++r){
                buf_out[R+i][R+j] += buf_in[R+i - (r+1)][R+j];
                buf_out[R+i][R+j] += buf_in[R+i + (r+1)][R+j];
                buf_out[R+i][R+j] += buf_in[R+i][R+j - (r+1)];
                buf_out[R+i][R+j] += buf_in[R+i][R+j + (r+1)];
            }
        }
    }

    for(size_t i=0; i<N; ++i){
        for(size_t j=0; j<N; ++j){
            printf("%f\t", buf_in[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    for(size_t i=0; i<N; ++i){
        for(size_t j=0; j<N; ++j){
            printf("%f\t", buf_out[i][j]);
        }
        printf("\n");
    }
    return 0;
}