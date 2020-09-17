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

    // CSV* csv = read_csv("../distributions/100_Sales_Records.csv");
    // print_csv(csv);
    // write_csv("new.csv", csv);

    Prob_Dist* distributions[MAX_DIST_TYPES];

    {
        Dist_Param params[3] = {
            {
                0.0, 1.0, 3,
                0.0, 2.0, 5,
                0.0, 0.0, 0,
            },
            {
                0.0, 0.0, 1,
                0.0, 0.0, 1,
                0.0, 0.0, 0,
            },
            {
                0.0, 1.0, 4,
                0.0, 2.0, 5,
                5.0, 5.0, 3,
            }
        };

        for(int i=0, j=0; i<MAX_DIST_TYPES; ++i){
            if(i && i!=Binomial_d){
                // printf("saving dist(%d) in %s\n", i, _files[i]);
                save_probability_table(_files[i], (Dist_T)i, params[j++]);
            }
            distributions[i] = init_dist_table(_files[i], (Dist_T)i);
        }
    }

    Input input;
    int opt = atoi(argv[1]);

    if(opt == Normal_d){
        printf("Give z: ");
        scanf("%lf", &input.args[0]);
        // args[0] -> args[1] + args[2]
        input.args[1] = 0.1 * floor(input.args[0]/0.1);
        input.args[2] = input.args[0]-input.args[1];
    }
    else if(opt > 2 && opt < 5){
        printf("Give [arg]: ");
        scanf("%lf", &input.args[0]);
    }

    if(opt){
        printf("Give [column argument] and [row argument]: ");
        scanf("%lf %lf", &input.args[1], &input.args[2]);
    }

    double prob = calc_prob(&input, distributions[opt]);

    if(prob >= 0.f){
        printf("Probabily: %lf\n", prob);
    } else{
        printf("Wrong arguments, there is no such entry!\n");
    }

    // for(int i=0; i<MAX_DIST_TYPES; ++i) free_distribution((void*)distributions[i]);

    return 0;
}

void check_inputs(int argc, const char** argv){
    if(argc != 2 || argv[1][0] < '0' || argv[1][0] > '4'){
        fprintf(stderr, "%s", 
        "Usage: ./main [distribution type]\n\
Distribution types:\n\
\t[0] Normal\n\
\t[1] Student's T\n\
\t[2] Chi Square\n\
\t[3] Binomial\n\
\t[4] F\n\
");
        exit(1);
    }
}