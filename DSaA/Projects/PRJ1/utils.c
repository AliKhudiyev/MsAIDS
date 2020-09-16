#include "utils.h"

int tell_index(double* arr, double val, unsigned size){
    int index = -1;

    for(unsigned i=0; i<size; ++i){
        if(fabs(arr[i]-val) < epsilon){
            index = i;
        }
    }

    return index;
}

void copy_arr(double* dst, const double* src, unsigned size){
    for(unsigned i=0; i<size; ++i){
        dst[i] = src[i];
    }
}

void print_arr(double* arr, unsigned size){
    for(unsigned i=0; i<size; ++i){
        printf("%lf ", arr[i]);
    }
}

double gamma(unsigned val){
    double ans = 1;
    for(unsigned i=2; i<val; ++i) ans *= i;
    return ans;
}

double gamma_div_2(unsigned val){
    return sqrt(M_PI) / pow(2.0, ((double)val-1)/2.) * gamma(gamma(val-2));
}