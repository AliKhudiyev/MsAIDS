#include "distribution.h"
#include "csv.h"

#include <math.h>

// Non-user friendly functions or Utilities

#define epsilon 10e-6

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

// ========================

void init_dist_table(Prob_Dist* distribution, const char* filepath){
    set_delim(';');
    printf("Delim: %s\n", _csv_delim);

    CSV csv;
    read_csv(filepath, &csv);

    // Initializing meta-data for Dist_Table
    distribution->table.n_col = csv.ncol - 1;
    distribution->table.n_row = csv.nrow - 1;

    // Allocting memory for Prob_Dist
    distribution->table.col_vals = malloc(distribution->table.n_col * sizeof(double));
    distribution->table.row_vals = malloc(distribution->table.n_row * sizeof(double));
    distribution->table.arr = malloc(distribution->table.n_row * sizeof(double));
    for(unsigned i=0; i<distribution->table.n_row; ++i){
        distribution->table.arr[i] = malloc(distribution->table.n_col * sizeof(double));
    }

    // Initializing column and row values
    for(unsigned c=1; c<csv.ncol; ++c){
        distribution->table.col_vals[c-1] = csv.context[0][c];
    }
    for(unsigned r=1; r<csv.nrow; ++r){
        distribution->table.row_vals[r-1] = csv.context[r][0];
    }

    // Initializing rest of the data
    for(unsigned r=1; r<csv.nrow; ++r){
        for(unsigned c=1; c<csv.ncol; ++c){
            distribution->table.arr[r-1][c-1] = csv.context[r][c];
        }
    }
}

double calc_prob(double col_arg, double row_arg, const Prob_Dist* distribution){
    int col = tell_index(distribution->table.col_vals, col_arg, distribution->table.n_col);
    int row = tell_index(distribution->table.row_vals, row_arg, distribution->table.n_row);

    if(col < 0 || row < 0){
        return -1.;
    }

    return distribution->table.arr[row][col];
}

void print_columns(const Prob_Dist* distribution){
    for(unsigned i=0; i<distribution->table.n_col; ++i){
        printf("%lf ", distribution->table.col_vals[i]);
    }
}

void print_rows(const Prob_Dist* distribution){
    for(unsigned i=0; i<distribution->table.n_row; ++i){
        printf("%lf ", distribution->table.row_vals[i]);
    }
}

void print_distribution(const Prob_Dist* distribution){
    for(unsigned r=0; r<distribution->table.n_row; ++r){
        for(unsigned c=0; c<distribution->table.n_col; ++c){
            printf("%lf ", distribution->table.arr[r][c]);
        }   printf("\n");
    }
}