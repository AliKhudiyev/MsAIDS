#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define N (10*10)
#define M (3*3)

void convolve2D(float* A, float* B, float* mask, int w, int h, int width, int height){
    for(int r=0; r<h; ++r){
        for(int c=0; c<w; ++c){
            float tmp = 0, elem = 0;
            int index = r*w+c;
            for(int i=0; i<height; ++i){
                for(int j=0; j<width; ++j, elem=0){
                    if(r+i>=height/2 && r+i-height/2<h &&
                        c+j>=width/2 && c+j-width/2<w)
                        elem = A[(r+i-height/2)*w+(c+j-width/2)];
                    tmp += elem * mask[i*width+j];
                }
            }
            B[index] = tmp;
        }
    }
}

int main(){
    srand(time(0));

    float A[N], B[N] = { 0 }, mask[M];
    constexpr size_t n_byte = N*sizeof(float);

    for(size_t i=0; i<M; mask[i++]=rand()%5/*, printf("%.1f\t", mask[i-1])*/);
    // printf("\n");
    for(size_t i=0; i<N; ++i){
        A[i] = rand() % 5;
        // printf("%.1f\t", A[i]);
    }   // printf("\n");

    for(size_t i=0; i<N; ++i){
        float tmp = 0, elem = 0;
        for(size_t j=0; j<M; ++j, elem=0){
            if(i+j>=M/2 && i+j-M/2<N) elem = A[i+j-M/2];
            tmp += elem * mask[j];
        }
        B[i] = tmp;
    }

    convolve2D(A, B, mask, 10, 10, 3, 3);

    for(int i=0; i<3; ++i){
        for(int j=0; j<3; ++j){
            printf("%.1f ", mask[i*3+j]);
        }   printf("\n");
    }   printf("\n");

    for(int i=0; i<10; ++i){
        for(int j=0; j<10; ++j){
            printf("%.1f ", A[i*10+j]);
        }   printf("\n");
    }   printf("\n");

    for(int i=0; i<10; ++i){
        for(int j=0; j<10; ++j){
            printf("%.1f ", B[i*10+j]);
        }   printf("\n");
    }   printf("\n");
    // for(size_t i=0; i<N; printf("%f\t", B[i++]));
    // printf("\n");

    return 0;
}