#include "utils.h"
#include <memory.h>

//

double _beta(const Input* input){
    double x = input->args[0];
    double y = input->args[1];
    double t = input->args[2];

    return pow(t, x-1) * pow(1-t, y-1);
}

// ===============================

int is_equal(double a, double b){
    return (fabs(a-b) < epsilon)? 0 : (a>b)? 1 : -1;
}

int tell_index(double* arr, double val, unsigned size){
    for(unsigned i=0; i<size; ++i){
        if(fabs(arr[i]-val) < epsilon){
            return i;
        }
    }
    return -1;
}

int tell_index2(const double* arr1, const double* arr2, double val1, double val2, unsigned size){
    for(unsigned i=0; i<size; ++i){
        if(!is_equal(arr1[i], val1) && !is_equal(arr2[i], val2)){
            return i;
        }
    }
    return -1;
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

double beta(double x, double y, double dt){
    double area = 0.0, t = 0.0;
    Input input = {t, x, y};

    for(; t<1.0; t+=dt){
        input.args[0] = dt;
        area += _beta(&input) * dt;
    }

    return area;
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