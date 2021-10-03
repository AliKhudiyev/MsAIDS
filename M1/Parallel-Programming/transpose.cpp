#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 5
#define M 5

int main(){
    float A[N][M], B[M][N];
    
    for(size_t i=0; i<N*M; ++i){
        A[i/M][i%M] = rand() % 100;
    }

    for(size_t i=0; i<N; ++i){
        for(size_t j=0; j<M; ++j){
            B[j][i] = A[i][j];
        }
    }

    float C[N][M];
    for(size_t i=0; i<N; ++i){
        for(size_t j=0; j<M; ++j){
            C[j][i] = B[i][j];
        }
    }

    /*
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
    */
    printf("%d\n", memcmp(A, C, N*M*sizeof(float)));

    return 0;
}