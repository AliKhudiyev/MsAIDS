
#ifndef _UTILITY_
#define _UTILITY_

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define epsilon                     10e-6
#define INTEGRAL_LOWER_BOUNDARY     -1
#define INTEGRAL_UPPER_BOUNDARY     1

typedef struct{
    double args[3];
    // args can be: arg, row, col (to calculate probability, i.e. calc_prob([args], distribution)
    // args can also be parameters for probability density functions
}Input;

typedef double(*func_t)(double x);
typedef double(*dens_t)(const Input* input);
typedef int boundary_t;

/* Returns:
 * a == b   =>  0
 * a > b    =>  1
 * a < b    => -1
*/
int is_equal(double a, double b);
int tell_index(double* arr, double val, unsigned size);
int tell_index2(const double* arr1, const double* arr2, double val1, double val2, unsigned size);
void copy_arr(double* dst, const double* src, unsigned size);
void print_arr(double* arr, unsigned size);
void clear_str(char* str, unsigned size);
double gammad(int val);
double gamma_div_2d(int val);
double beta(double x, double y, double dt);
double calc_integral_of(func_t f, double beg, double end, unsigned n_iter);
double find_integral_boundary_of(func_t f, double area, double boundary, double dx, boundary_t type);
double cio(dens_t f, Input input_lwr, Input input_ppr, unsigned n_iter, unsigned arg_index);
double fibo(dens_t f, double area, Input input, unsigned arg_index, double dx, boundary_t type);

#endif
