#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "csv.h"
#include "distribution.h"

// To check inputs and make sure everything is correct
// In case of ill-formed inputs, the usage is printed
void check_inputs(int argc, const char** argv);

double f(double x){
    return x;
}

double f2(const Input* input){
    return input->args[0];
}

int main(int argc, const char** argv){

    // check_inputs(argc, argv);

    // Prob_Dist* distribution = init_dist_table("Binomial_Distribution.csv", Normal_d);

    CSV* csv = read_csv("../distributions/100_Sales_Records.csv");
    print_csv(csv);

    Prob_Dist* distributions[MAX_DIST_TYPES];

    {
        Dist_Param params[3];

        for(int i=0, j=0; i<MAX_DIST_TYPES; ++i){
            if(i && i!=Binomial_d) save_probability_table(_files[i], (Dist_T)i, params[j++]);
            // distributions[i] = init_dist_table(_files[0], (Dist_T)i);
            distributions[i] = malloc(sizeof(Prob_Dist));
        }
    }

    Input input;

    // if(atoi(argv[1]) > 2){
    //     printf("Give [arg]: ");
    //     scanf("%lf", &input.args[0]);
    // }
    // printf("Give [column argument] and [row argument]: ");
    // scanf("%lf %lf", &input.args[1], &inputs.args[2]);

    // double prob = calc_prob(&input, distributions[atoi(argv[1])]);

    // if(prob >= 0.f){
    //     printf("Probabily: %lf\n", prob);
    // } else{
    //     printf("Wrong arguments, there is no such entry!\n");
    // }

    for(int i=0; i<MAX_DIST_TYPES; ++i) free_distribution((void*)distributions[i]);

    return 0;
}

void check_inputs(int argc, const char** argv){
    if(argc != 2 || atoi(argv[1]) > 4){
        fprintf(stderr, "%s", 
        "Usage: ./main [distribution type]\n    \
        Distribution types:\n                   \
        \t[0] Normal\n                          \
        \t[1] Student's T\n                     \
        \t[2] Chi Square\n                      \
        \t[3] Binomial\n                        \
        \t[4] F\n                               \
        ");
        exit(1);
    }
}