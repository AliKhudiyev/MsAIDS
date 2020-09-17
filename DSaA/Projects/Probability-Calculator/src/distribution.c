#include "distribution.h"
#include "csv.h"

// Distribution Utilities

const char _files[5][30] = {
    "Loi Normal.csv",
    "Student_T.csv",
    "Chi_Square.csv",
    "Binomial.csv",
    "F.csv"
};

// k - degree of freedom
double calc_chi_square(const Input* input){
    double x = input->args[0];
    double k = input->args[1];

    if(x > 0.0){
        return pow(x, k/2.0-1) * pow(M_E, -x/2.0) / (pow(2.0, k/2.0) * tgamma(k/2.0));
    }
    return 0.0;
}

// n - degree of freedom
double calc_student_t(const Input* input){
    double t = input->args[0];
    double n = input->args[1];

    return pow(1 + pow(t, 2.0)/n, -(1+n)/2.0) * (tgamma((1.0+n)/2.0) / (sqrt(M_PI*n) * tgamma(n/2.0)));
}

double calc_f(const Input* input){
    return 0.0;
}

// =========================

Prob_Dist* init_dist_table(const char* filepath, Dist_T type){
    set_delim(';');

    Prob_Dist* distribution = malloc(sizeof(Prob_Dist));
    CSV* csv = read_csv(filepath);
    print_csv(csv);

    // Allocting memory for Prob_Dist
    distribution->table = malloc(sizeof(Dist_Table));
        // Initializing meta-data for Dist_Table
        distribution->table->n_col = csv->ncol - 1;
        distribution->table->n_row = csv->nrow - 1;
        // -------------------------------------
    distribution->table->col_vals = malloc(distribution->table->n_col * sizeof(double));
    distribution->table->row_vals = malloc(distribution->table->n_row * sizeof(double));
    distribution->table->arr = malloc(distribution->table->n_row * sizeof(double));
    for(unsigned i=0; i<distribution->table->n_row; ++i){
        distribution->table->arr[i] = malloc(distribution->table->n_col * sizeof(double));
    }

    // Initializing column and row values
    for(unsigned c=1; c<csv->ncol; ++c){
        distribution->table->col_vals[c-1] = atof(csv->context[0][c]);
    }
    for(unsigned r=1; r<csv->nrow; ++r){
        distribution->table->row_vals[r-1] = atof(csv->context[r][0]);
    }
    
    // Initializing rest of the data
    for(unsigned r=1; r<csv->nrow; ++r){
        for(unsigned c=1; c<csv->ncol; ++c){
            distribution->table->arr[r-1][c-1] = atof(csv->context[r][c]);
        }
    }

    return distribution;
}

double calc_prob(const Input* input, const Prob_Dist* distribution){
    int index = tell_index(distribution->arg_vals, input->args[0], distribution->n_arg);
    int row = tell_index(distribution->table->row_vals, input->args[1], distribution->table->n_row);
    int col = tell_index(distribution->table->col_vals, input->args[2], distribution->table->n_col);

    if(!distribution->n_arg) index = 0;

    printf("%d: %d, %d\n", index, row, col);

    if(col < 0 || row < 0){
        return -1.;
    }

    return distribution->table[index].arr[row][col];
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
    free((void*)distribution->arg_vals);
    for(unsigned i=0; i<distribution->n_arg; ++i){
        free_dist_table(distribution->table);
    }
    free((void*)distribution);
}

void save_probability_table(const char* filepath, Dist_T type, Dist_Param param){
    ;
}