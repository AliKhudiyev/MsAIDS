#include "distribution.h"
#include "csv.h"

#include <omp.h>
#include <time.h>

// Distribution Utilities

const char _files[5][100] = {
    "../distributions/Loi Normal.csv",
    "../distributions/Student_T.csv",
    "../distributions/Chi_Square.csv",
    "../distributions/Binomial_Distribution.csv",
    "../distributions/F.csv"
};

// k - degree of freedom
double calc_chi_square(const Input* input){
    double k = input->args[1];
    double x = input->args[2];

    if(x > 0.0){
        return pow(x, k/2.0-1) * pow(M_E, -x/2.0) / (pow(2.0, k/2.0) * tgamma(k/2.0));
    }
    return 0.0;
}

// n - degree of freedom
double calc_student_t(const Input* input){
    double n = input->args[1];
    double t = input->args[2];

    return pow(1 + pow(t, 2.0)/n, -(1+n)/2.0) * (tgamma((1.0+n)/2.0) / (sqrt(M_PI*n) * tgamma(n/2.0)));
}

// d1, d2 - degrees of freedom
double calc_f(const Input* input){
    double x = input->args[0];
    double d1 = input->args[1], d2 = input->args[2];
    
    return sqrt(pow(d1*x, d1) * pow(d2, d2) / pow(d1*x+d2, d1+d2)) / (x * beta(d1/2.0, d2/2.0, epsilon, 1.0));
}

// =========================

Prob_Dist* init_dist_table(const char* filepath, Dist_T type){
    set_delim(';');

    Prob_Dist* distribution = malloc(sizeof(Prob_Dist));
    CSV* csv = read_csv(filepath);
    unsigned gap = 1;

    // Pre-processing
    distribution->n_arg = 0;
    if(type > 2){
        gap = 2;
        distribution->n_arg = csv->nrow-gap;
        distribution->arg_vals = malloc((csv->nrow-gap) * sizeof(double));
        for(unsigned i=gap; i<csv->nrow; ++i){
            if(strncmp(csv->context[i][0], NULL_CELL, 1)){
                distribution->arg_vals[i-gap] = atof(csv->context[i][0]);
            } else{
                distribution->arg_vals[i-gap] = distribution->arg_vals[i-gap-1];
            }
        }
    }

    // Allocting memory for Prob_Dist
    distribution->table = malloc(sizeof(Dist_Table));
        // Initializing meta-data for Dist_Table
        distribution->table->n_col = csv->ncol - gap;
        distribution->table->n_row = csv->nrow - gap;
        // -------------------------------------
    distribution->table->col_vals = malloc(distribution->table->n_col * sizeof(double));
    distribution->table->row_vals = malloc(distribution->table->n_row * sizeof(double));
    distribution->table->arr = malloc(distribution->table->n_row * sizeof(double));
    for(unsigned i=0; i<distribution->table->n_row; ++i){
        distribution->table->arr[i] = malloc(distribution->table->n_col * sizeof(double));
    }

    // Initializing column and row values
    for(unsigned c=gap; c<csv->ncol; ++c){
        distribution->table->col_vals[c-gap] = atof(csv->context[0][c]);
    }
    for(unsigned r=gap; r<csv->nrow; ++r){
        distribution->table->row_vals[r-gap] = atof(csv->context[r][gap-1]);
    }
    
    // Initializing rest of the data
    for(unsigned r=gap; r<csv->nrow; ++r){
        for(unsigned c=gap; c<csv->ncol; ++c){
            distribution->table->arr[r-gap][c-gap] = atof(csv->context[r][c]);
        }
    }

    free_csv(csv);

    return distribution;
}

double calc_prob(const Input* input, const Prob_Dist* distribution){
    int row = tell_index(distribution->table->row_vals, input->args[1], distribution->table->n_row);
    int col = tell_index(distribution->table->col_vals, input->args[2], distribution->table->n_col);

    if(distribution->n_arg){
        row = tell_index2(distribution->arg_vals, distribution->table->row_vals, input->args[0], input->args[1], distribution->table->n_row);
    }

    if(col < 0 || row < 0){
        return -1.;
    }

    return distribution->table->arr[row][col];
}

void print_columns(const Prob_Dist* distribution){
    for(unsigned i=0; i<distribution->table->n_col; ++i){
        printf("%lf ", distribution->table->col_vals[i]);
    }
}

void print_rows(const Prob_Dist* distribution){
    for(unsigned i=0; i<distribution->table->n_row; ++i){
        printf("%lf ", distribution->table->row_vals[i]);
    }
}

void print_distribution(const Prob_Dist* distribution){
    for(unsigned r=0; r<distribution->table->n_row; ++r){
        for(unsigned c=0; c<distribution->table->n_col; ++c){
            printf("%lf ", distribution->table->arr[r][c]);
        }   printf("\n");
    }
}

void free_dist_table(Dist_Table* table){
    free((void*)table->col_vals);
    free((void*)table->row_vals);
    for(unsigned i=0; i<table->n_row; ++i) free((void*)table->arr[i]);
    free((void*)table->arr);
    free((void*)table);
}

void free_distribution(Prob_Dist* distribution){
    if(distribution->n_arg) free((void*)distribution->arg_vals);
    free_dist_table(distribution->table);
    free((void*)distribution);
}

double func(const Input* input){
    double x = input->args[0] * epsilon;
    double d1 = input->args[1], d2 = input->args[2];
    
    return sqrt(pow(d1*x, d1) * pow(d2, d2) / pow(d1*x+d2, d1+d2)) / (input->args[0] * beta(d1/2.0, d2/2.0, epsilon, 1.0));
}

void save_probability_table(const char* filepath, Dist_T type, Dist_Param param){
    set_delim(';');
    
    double** table, area = 0.0;
    unsigned gap = 1, arg_index = (type == F_d)? 0 : 2;
    unsigned n_arg_val = param.n_arg_val? param.n_arg_val : 1;
    Input input;
    dens_t funcs[] = {NULL, calc_student_t, calc_chi_square, NULL, calc_f};

    if(type > 2) gap = 2;

    param.n_col_val += gap;
    unsigned n_row_val = n_arg_val * param.n_row_val + gap;
    double coef = 1.0;

    // Allocating and initializing table
    table = (double**)malloc((n_row_val-gap) * sizeof(double*));

    #pragma omp parallel for
    for(unsigned i=0; i<n_arg_val; ++i){
        input.args[0] = param.min_arg_val + i * param.arg_val_dif;
        
        for(unsigned j=0; j<param.n_row_val; ++j){
            input.args[1] = param.min_row_val + j * param.row_val_dif;
            table[param.n_row_val * i + j] = (double*)malloc((param.n_col_val-gap) * sizeof(double));
            
            for(unsigned t=0; t<param.n_col_val-gap; ++t){
                input.args[2] = param.min_col_val + t * param.col_val_dif;
                if(type == Student_T_d){
                    area = (1.0 - (param.min_col_val + t * param.col_val_dif)) / 2.0;
                } else if(type == Chi_Square_d){
                    area = 1.0 - (param.min_col_val + t * param.col_val_dif);
                } else{
                    area = param.min_arg_val + i * param.arg_val_dif;
                }
                input.args[arg_index] = (type != F_d? 0.0 : epsilon*20);
                if(!arg_index) coef = 10 * (input.args[1]+input.args[2]);
                
                table[param.n_row_val * i + j][t] = fibo(funcs[type], area, input, arg_index, coef*epsilon, INTEGRAL_UPPER_BOUNDARY);
            }
        }
    }

    CSV* csv = malloc(sizeof(CSV));
    
    // Initialization
    csv->nrow = n_row_val;
    csv->ncol = param.n_col_val;
    csv->context = (char***)malloc(csv->nrow * sizeof(char**));
    for(unsigned r=0; r<csv->nrow; ++r){
        csv->context[r] = (char**)malloc(csv->ncol * sizeof(char*));
        for(unsigned c=0; c<csv->ncol; ++c){
            csv->context[r][c] = (char*)malloc(MAX_CHARS/10);
        }
    }

    // Initializing columns
    for(unsigned c=0; c<gap; ++c){
        strcpy(csv->context[0][c], NULL_CELL);
    }
    for(unsigned i=gap; i<csv->ncol; ++i){
        sprintf(csv->context[0][i], "%.6lf", param.min_col_val + (i-gap) * param.col_val_dif);
    }

    // Initializing rows
    for(unsigned r=0; r<gap; ++r){
        strcpy(csv->context[r][0], NULL_CELL);
    }
    for(unsigned i=gap; i<param.n_arg_val+gap; ++i){
        sprintf(csv->context[gap + param.n_row_val * (i-gap)][0], "%.6lf", param.min_arg_val + (i-gap) * param.arg_val_dif);
        for(unsigned j=0; j<param.n_row_val; ++j){
            sprintf(csv->context[gap + param.n_row_val * (i-gap) + j][1], "%.6lf", param.min_row_val + j * param.row_val_dif);
        }
    }
    for(unsigned i=gap; !param.n_arg_val && i<csv->nrow; ++i){
        sprintf(csv->context[i][0], "%.6lf", param.min_row_val + (i-gap) * param.row_val_dif);
    }

    // Initializing rest of the data (or table)
    for(unsigned r=gap; r<csv->nrow; ++r){
        for(unsigned c=gap; c<csv->ncol; ++c){
            sprintf(csv->context[r][c], "%.6lf", table[r-gap][c-gap]);
        }
    }

    write_csv(filepath, csv);

    free_csv(csv);
}