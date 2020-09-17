#include "utils.h"
#include <memory.h>

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

void clear_str(char* str, unsigned size){
    unsigned i = size, pass = 1;
    for(; i; --i){
        if(str[i-1] == '\n') str[i-1] = '\0';
        else if(str[i-1] == ' ' && pass) str[i-1] = '\0';
        else if(str[i-1] != ' ') pass = 0;
    }
}

double gammad(int val){
    double ans = 1;
    for(int i=2; i<val; ++i) ans *= i;
    return ans;
}

double gamma_div_2d(int val){
    return sqrt(M_PI) / pow(2.0, ((double)val-1.)/2.) * gammad(gammad(val-2));
}

double calc_integral_of(func_t f, double beg, double end, unsigned n_iter){
    double area = 0.0;
    double dx = (end-beg)/(double)n_iter;

    for(unsigned i=0; i<n_iter; ++i){
        area += f(beg+i*dx) * dx;
    }

    return area;
}

double find_integral_boundary_of(func_t f, double area, double boundary, double dx, boundary_t type){
    double boundary2 = boundary, curr_area = 0.0;

    while(curr_area < area){
        curr_area += f(boundary2) * dx;
        boundary2 += type * dx;
    }

    return boundary2;
}

double cio(dens_t f, Input input_lwr, Input input_ppr, unsigned n_iter, unsigned arg_index){
    double area = 0.0;
    double dx = (input_ppr.args[arg_index] - input_lwr.args[arg_index]) / (double)n_iter;

    printf("dx: %.10lf\n", dx);

    for(unsigned i=0; i<n_iter; ++i){
        area += f(&input_lwr) * dx;
        input_lwr.args[arg_index] += dx;
    }

    return area;
}

double fibo(dens_t f, double area, Input input, unsigned arg_index, double dx, boundary_t type){
    double curr_area = 0.0;

    int i=0;

    while(curr_area < area){
        curr_area += f(&input) * dx;
        input.args[arg_index] += type * dx;
    }

    return input.args[arg_index];
}