
#ifndef _UTILITY_
#define _UTILITY_

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define epsilon 10e-6

int tell_index(double* arr, double val, unsigned size);
void copy_arr(double* dst, const double* src, unsigned size);
void print_arr(double* arr, unsigned size);
double gamma(unsigned val);
double gamma_div_2(unsigned val);

#endif
