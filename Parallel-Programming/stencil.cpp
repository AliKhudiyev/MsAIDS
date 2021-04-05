#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

#define N 100
#define R 3

int main(){
    float buf_in[N], buf_out[N];

    for(size_t i=0; i<N; ++i){
        buf_in[i] = rand() % 100;
    }
    memset(buf_out, 0, N*sizeof(float));

    for(size_t i=R; i<N-R; ++i){
        buf_out[i] = 0.f;
        for(size_t j=0; j<2*R+1; ++j){
            buf_out[i] += buf_in[j+i-R];
        }
    }
    return 0;
}