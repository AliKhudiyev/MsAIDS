
#ifndef _DISTRIBUTION_
#define _DISTRIBUTION_

#define MAX_DIST_TYPES 2

#define INIT()                                              \
    ;\
    const char* filepaths[] = {                             \
        "Loi\ Normal.csv",                                  \
        "Binomial_Distribution.csv"                         \
    };                                                      \
    Prob_Dist distributions[MAX_DIST_TYPES];                \
    for(int i=0; i< MAX_DIST_TYPES; ++i){                   \
        distributions[i].type = (Dist_T)i;                  \
        init_dist_table(distributions+i, filepaths[i]);     \
    }

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
double calc_prob(double col_arg, double row_arg, const Prob_Dist* distribution);
void print_columns(const Prob_Dist* distribution);
void print_rows(const Prob_Dist* distribution);
void print_distribution(const Prob_Dist* distribution);
void free_dist_table(Dist_Table* table);
void free_distribution(Prob_Dist* distribution);
void save_probability_table(const char* filepath, Dist_T type, Dist_Param param);

#endif
