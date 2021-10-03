#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define N 10
#define M 10
#define K 10

int main(){
    srand(time(0));
    float A[N][M], B[M][K], C[N][K];

    for(size_t i=0; i<N*M; ++i){
        A[i/M][i%M] = rand() % N;
    }
    for(size_t i=0; i<M*K; ++i){
        B[i/K][i%K] = rand() % M;
    }

    for(size_t i=0; i<N; ++i){
        for(size_t j=0; j<K; ++j){
            C[i][j] = 0;
            for(size_t t=0; t<M; ++t){
                C[i][j] += A[i][t] * B[t][j];
            }
        }
    }

    for(size_t i=0; i<N; ++i){
        for(size_t j=0; j<M; ++j){
            printf("%f\t", A[i][j]);
        }   printf("\n");
    }   printf("\n");
    for(size_t i=0; i<M; ++i){
        for(size_t j=0; j<K; ++j){
            printf("%f\t", B[i][j]);
        }   printf("\n");
    }   printf("\n");
    for(size_t i=0; i<N; ++i){
        for(size_t j=0; j<K; ++j){
            printf("%f\t", C[i][j]);
        }   printf("\n");
    }
    return 0;
}