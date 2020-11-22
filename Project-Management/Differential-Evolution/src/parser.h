
#pragma once

#include <stdio.h>
// #include <iostream>
#include <string>
#include <getopt.h>

#define NO_GOAL             0
#define GOAL_GLOBAL_MIN     -1
#define GOAL_GLOBAL_MAX     1

#define NO_VISUAL 0

#define NO_OPTIMIZATION     0   // ST_NA (Single Thread + Non-Adaptive)
#define OPTIMIZATION_ST_SA  1   // Single Thread + Self-Adaptive
#define OPTIMIZATION_MT_NA  2   // Multi Thread + Non-Adaptive
#define OPTIMIZATION_MT_MA  3   // Multi Thread + Self-Adaptive

#define CUSTOM_FUNCTION     0
#define ACKLEY_FUNCTION     1
#define RASTRIGIN_FUNCTION  2
#define ROSENBROCK_FUNCTION 3
#define SCHWEFEL_FUNCTION   4

struct ArgumentList{
    static int help_flag;
    static int verbose_flag;
    static int visual_flag;

    static int population_size;
    static int n_generation;
    static int n_benchmark_run;
    static int goal;
    static double threshold, expected_y;
    static double f, p;
    static int optimization;
    static int function;

    static void parse(int argc, char* const* argv){
        int c;

        while (1)
        {
        static struct option long_options[] =
            {
            /* These options set a flag. */
            {"verbose",         no_argument,            &ArgumentList::verbose_flag, 1},
            {"help",            no_argument,            &ArgumentList::help_flag, 1},
            {"visual",          no_argument,            &ArgumentList::visual_flag, 1},
            {"global-min",      no_argument,            &ArgumentList::goal, GOAL_GLOBAL_MIN},
            {"global-max",      no_argument,            &ArgumentList::goal, GOAL_GLOBAL_MAX},
            /* These options don’t set a flag.
                We distinguish them by their indices. */
            {"population",      optional_argument,      0, 's'},
            {"generation",      optional_argument,      0, 'g'},
            {"threshold",       optional_argument,      0, 't'},
            {"benchmark-run",   optional_argument,      0, 'r'},
            {"optimization",    optional_argument,      0, 'o'},
            {"function",        optional_argument,      0, 'b'},
            {0, 0, 0, 0}
            };
        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long (argc, argv, "f:p:t:o:r:g:s:b:",
                        long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;

        switch (c){
            case 0:
            /* If this option set a flag, do nothing else now. */
            if (long_options[option_index].flag != 0)
                break;
            printf ("option %s", long_options[option_index].name);
            if (optarg)
                printf (" with arg %s", optarg);
            printf ("\n");
            break;

            case 'f':
            // printf ("option -f with value [%s]\n", optarg);
            ArgumentList::f = std::atof(optarg);
            break;

            case 'p':
            // printf ("option -p with value [%s]\n", optarg);
            ArgumentList::p = std::atof(optarg);
            break;
            
            case 't':
            // printf ("option -t with value [%s]\n", optarg);
            ArgumentList::threshold = std::atof(optarg);
            break;

            case 'o':
            // printf ("option -o with value [%s]\n", optarg);
            ArgumentList::optimization = std::atoi(optarg);
            break;
            
            case 'r':
            // printf ("option -r with value [%s]\n", optarg);
            ArgumentList::n_benchmark_run = std::atoi(optarg);
            break;
            
            case 'g':
            // printf ("option -g with value [%s]\n", optarg);
            ArgumentList::n_generation = std::atoi(optarg);
            break;

            case 's':
            // printf ("option -s with value [%s]\n", optarg);
            ArgumentList::population_size = std::atoi(optarg);
            break;

            case 'b':
            // printf ("option -b with value [%s]\n", optarg);
            ArgumentList::function = std::atoi(optarg);
            break;

            case '?':
            /* getopt_long already printed an error message. */
            break;

            default:
            abort();
            }
        }

    /* Instead of reporting ‘--verbose’
        and ‘--brief’ as they are encountered,
        we report the final status resulting from them. */

        if(ArgumentList::help_flag){
            printf("Help!\n");
            exit(0);
        }
        
        if (ArgumentList::verbose_flag)
            puts ("verbose flag is set");

        if(ArgumentList::n_generation < 1 && ArgumentList::threshold < 0){
            ArgumentList::n_generation = 1000;
        } else if(ArgumentList::threshold >= 0 && ArgumentList::goal == NO_GOAL){
            ArgumentList::n_generation = 0;
        }
        // printf("GOAL: %d\n", ArgumentList::goal);
        // printf("Population size: %d\n", ArgumentList::population_size);
        // if(ArgumentList::n_generation > 0)
        //     printf("Number of generations: %d\n", ArgumentList::n_generation);
        // else if(ArgumentList::threshold >= 0)
        //     printf("Threshold: %lf\n", ArgumentList::threshold);
        // printf("Bencmark runs: %d\n", ArgumentList::n_benchmark_run);
        // printf("Visualization: %d\n", ArgumentList::visual_flag);
        // printf("Optimization: %d\n", ArgumentList::optimization);

        if(ArgumentList::goal == NO_GOAL){
            printf("f(X) = ");
            scanf("%lf", &ArgumentList::expected_y);
        }

    /* Print any remaining command line arguments (not options). */
        if (optind < argc)
        {
            printf ("non-option ARGV-elements: ");
            while (optind < argc)
                printf ("%s ", argv[optind++]);
            putchar ('\n');
        }
    }

    static bool is_generation_driven(){
        if(ArgumentList::goal == NO_GOAL && ArgumentList::threshold >= 0) return false;
        return true;
    }
};

int ArgumentList::help_flag = 0;
int ArgumentList::verbose_flag = 0;
int ArgumentList::visual_flag = NO_VISUAL;

int ArgumentList::population_size = 10;
int ArgumentList::n_generation = 100;
int ArgumentList::n_benchmark_run = 1;
int ArgumentList::goal = NO_GOAL;
double ArgumentList::threshold = -1.0;
double ArgumentList::expected_y = 0.0;
double ArgumentList::f = 0.8;
double ArgumentList::p = 0.9;
int ArgumentList::optimization = NO_OPTIMIZATION;
int ArgumentList::function = CUSTOM_FUNCTION;
