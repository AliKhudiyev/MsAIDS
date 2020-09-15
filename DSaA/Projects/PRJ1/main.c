#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "csv.h"
#include "distribution.h"

// To check inputs and make sure everything is correct
// In case of ill-formed inputs, the usage is printed
void check_inputs(int argc, const char** argv);

int main(int argc, const char** argv){

    check_inputs(argc, argv);

    // INIT();

    double colarg, rowarg;

    printf("Give [column argument] and [row argument]: ");
    scanf("%lf %lf", &colarg, &rowarg);

    // double prob = calc_prob(colarg, rowarg, distributions[atoi(argv[1])]);

    // if(prob >= 0.f){
    //     printf("Probabily: %lf\n", prob);
    // } else{
    //     printf("Wrong arguments!\n");
    // }

    // CSV csv;

    // read_csv("hey.csv", &csv);
    // print_csv(&csv);

    Prob_Dist distribution;

    distribution.type = Normal;

    init_dist_table(&distribution, "Loi\ Normal.csv");

    // print_columns(&distribution);
    // printf("\n");
    // print_rows(&distribution);
    // printf("\n");
    // print_distribution(&distribution);
    // printf("\n");

    printf("Prob: %lf\n", calc_prob(colarg, rowarg, &distribution));

    return 0;
}

void check_inputs(int argc, const char** argv){
    if(argc != 2 || atoi(argv[1]) > 4){
        fprintf(stderr, "%s", 
        "Usage: ./main [distribution type]\n    \
        Distribution types:\n                   \
        \t[0] Normal\n                          \
        \t[1] Binomial\n                        \
        \t[2] Student's T\n                     \
        \t[3] Chi Square\n                      \
        \t[4] F\n                               \
        ");
        exit(1);
    }
}