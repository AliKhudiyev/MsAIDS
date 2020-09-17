
#ifndef _DISTRIBUTION_
#define _DISTRIBUTION_

#include "utils.h"

#define MAX_DIST_TYPES 5

extern const char _files[5][100];

typedef enum{
    Normal_d = 0,
    Student_T_d,
    Chi_Square_d,
    Binomial_d,
    F_d
}Dist_T;

typedef struct{
    // meta-data
    unsigned n_col, n_row;

    // data
    double *col_vals, *row_vals;
    double** arr;
}Dist_Table;

typedef struct{
    // meta-data
    Dist_T type;
    unsigned n_arg;
    // n_arg - number of arg_vals

    // data
    double* arg_vals;
    // arg_vals - third arguments of the probability function,
    // or the indices for each probability table
    Dist_Table* table;
}Prob_Dist;

typedef struct{
    double min_col_val, col_val_dif;
    unsigned n_col_val;

    double min_row_val, row_val_dif;
    unsigned n_row_val;

    double min_arg_val, arg_val_dif;
    unsigned n_arg_val;
}Dist_Param;

Prob_Dist* init_dist_table(const char* filepath, Dist_T type);
double calc_prob(const Input* input, const Prob_Dist* distribution);
void print_columns(const Prob_Dist* distribution);
void print_rows(const Prob_Dist* distribution);
void print_distribution(const Prob_Dist* distribution);
void free_dist_table(Dist_Table* table);
void free_distribution(Prob_Dist* distribution);
void save_probability_table(const char* filepath, Dist_T type, Dist_Param param);

// To Be Removed
double calc_chi_square(const Input* input);
double calc_student_t(const Input* input);
double calc_f(const Input* input);

#endif
