#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define N 5

int main(){
    srand(time(0));

    float A[N], B[N+1] = { 0 };
    constexpr size_t n_byte = N*sizeof(float);

    for(size_t i=0; i<N; ++i){
        A[i] = rand() % 100;
        B[i+1] = B[i] + A[i];

        printf("%f\t", A[i]);
    }   printf("\n");
    for(size_t i=0; i<N+1; printf("%f\t", B[i++]));
    printf("\n");

    return 0;
}