
#ifndef _DISTRIBUTION_
#define _DISTRIBUTION_

#define MAX_DIST_TYPES 2

#define INIT()                                              \
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
    Normal = 0,
    Binomial,
    Student_t,
    Chi_square,
    F
}Dist_T;

typedef struct{
    // meta-data
    unsigned n_col, n_row;

    // data
    double *col_vals, *row_vals;
    double** arr;
}Dist_Table;

typedef struct{
    Dist_T type;
    Dist_Table table;
}Prob_Dist;

void init_dist_table(Prob_Dist* distribution, const char* filepath);
double calc_prob(double col_arg, double row_arg, const Prob_Dist* distribution);
void print_columns(const Prob_Dist* distribution);
void print_rows(const Prob_Dist* distribution);
void print_distribution(const Prob_Dist* distribution);

#endif
