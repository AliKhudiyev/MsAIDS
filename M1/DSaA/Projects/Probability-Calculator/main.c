#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

#include "csv.h"
#include "distribution.h"

// To check inputs and make sure everything is correct
// In case of ill-formed inputs, the usage is printed
void check_inputs(int argc, const char** argv);

int main(int argc, const char** argv){

    check_inputs(argc, argv);

    Prob_Dist* distributions[MAX_DIST_TYPES];

    {
        Dist_Param params[3] = {
            {
                0.1, 0.1, 9,
                1.0, 1.0, 30,
                0.0, 0.0, 0,
            },
            {
                0.1, 0.1, 9,
                1.0, 1.0, 30,
                0.0, 0.0, 0,
            },
            {
                2.0, 1.0, 9,
                1.0, 1.0, 10,
                0.95, 0.04, 1,
            }
        };
        
        #pragma omp parallel for
        for(int i=0; i<MAX_DIST_TYPES; ++i){
            if(!strcmp(argv[1], "-g") && i && i!=Binomial_d){
                save_probability_table(_files[i], (Dist_T)i, params[(int)floor((double)i/2.0)]);
            } else if(strcmp(argv[1], "-g")){
                distributions[i] = init_dist_table(_files[i], (Dist_T)i);
            }
        }
    }

    if(strcmp(argv[1], "-g")){
        Input input;
        int opt = atoi(argv[1]);

        if(opt == Normal_d){
            printf("Give z: ");
            scanf("%lf", &input.args[0]);
            // args[0] -> args[1] + args[2]
            input.args[1] = 0.1 * floor(fabs(input.args[0])/0.1);
            input.args[2] = fabs(input.args[0])-input.args[1];
        }
        else if(opt > 2 && opt < 5){
            printf("Give [arg]: ");
            scanf("%lf", &input.args[0]);
        }

        if(opt){
            printf("Give [row argument] and [column argument]: ");
            scanf("%lf %lf", &input.args[1], &input.args[2]);
        }

        double prob = calc_prob(&input, distributions[opt]);

        if(prob >= 0.f){
            printf("Probabily: %lf\n", input.args[0] >= 0.0 ? prob : 1.0 - prob);
        } else{
            printf("Wrong arguments, there is no such entry!\n");
        }

        for(int i=0; i<MAX_DIST_TYPES; ++i) free_distribution((void*)distributions[i]);
    }
    
    return 0;
}

void check_inputs(int argc, const char** argv){
    if(argc != 2 || 
        (strcmp(argv[1], "-g") && 
        (argv[1][0] < '0' || argv[1][0] > '4'))){
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