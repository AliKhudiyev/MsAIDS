#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>

#include "parser.h"
#include "dea.h"

using namespace std;

ostream& operator<<(ostream& out, const vector<double>& vec){
    out<<"(";
    for(size_t i=0; i<vec.size(); ++i){
        cout<<vec[i];
        if(i < vec.size() - 1) cout<<", ";
    }
    out<<")";
    return out;
}

double mean_(const vector<double>& vals){
    double result = 0;
    
    for(const auto& val: vals){
        // printf("%.10lf\n", val);
        result += val;
    }

    return result / static_cast<double>(vals.size());
}

double std_(const vector<double>& vals){
    double result = 0;
    double mean = mean_(vals);
    
    for(const auto& val: vals){
        result += pow(val - mean, 2);
    }

    return sqrt(result / static_cast<double>(vals.size()));
}

int main(int argc, char* const* argv){
    // Parsing arguments from the command line and initializing the settings
    ArgumentList::parse(argc, argv);
    srand(time(nullptr));

    /* = = = = = = = Initializing settings for benchmark functions = = = = = = = */
    size_t n_dimension = 30;
    double interval[2] = { -32.768, 32.768 };
    vector<double> global_mins;

    Function::function.set(ArgumentList::variables, ArgumentList::function);
    switch (ArgumentList::benchmark)
    {
    case CUSTOM_FUNCTION:
        Function::calculate = Function::_calculate_;
        n_dimension = ArgumentList::variables.size();
        break;
    
    case ACKLEY_FUNCTION:
        Function::calculate = Function::Benchmark::ackley;
        break;
    
    case RASTRIGIN_FUNCTION:
        Function::calculate = Function::Benchmark::rastrigin;
        interval[0] = -5.12;
        interval[1] = 5.12;
        break;
    
    case ROSENBROCK_FUNCTION:
        Function::calculate = Function::Benchmark::rosenbrock;
        interval[0] = -5;
        interval[1] = 10;
        break;
    
    case SCHWEFEL_FUNCTION:
        Function::calculate = Function::Benchmark::schwefel;
        interval[0] = -500;
        interval[1] = 500;
        break;
    
    default:
        break;
    }
    /* = = = = = = = = = = = = = = */

    /* = = = = = = = Initializing number of threads = = = = = = = */
    size_t n_thread = 1;
    if(ArgumentList::optimization >= 2){
        n_thread = omp_get_max_threads();
        // cout<<n_thread<<'\n';
    }
    /* = = = = = = = = = = = = = = */
    
    /* = = = = = = = Running the DEA = = = = = = = */
    #pragma omp parallel for num_threads(n_thread)
    for(size_t n_run=0; n_run<ArgumentList::n_benchmark_run; ++n_run){
        space_t input_space;
        ofstream log;
        if(ArgumentList::visual_flag && !n_run){ log.open("evolution.log"); }

        unsigned int n_generation = 0;
        double multiplication_factor = ArgumentList::f;
        double probability_crossover = ArgumentList::p;
        double best_y = 0;

        set<unsigned int> indices;
        input_t target, inputs[2];
        input_t mutant, trial;

        initialize_input_space(input_space, n_dimension, ArgumentList::population_size, interval);
        if(ArgumentList::visual_flag) log<<ArgumentList::population_size<<endl;

        while(++n_generation){
            best_y = Function::calculate(input_space[best_match(input_space, ArgumentList::goal, ArgumentList::expected_y)]);

            if(ArgumentList::visual_flag && !n_run){
                for(const auto& input: input_space){
                    for(const auto& coord: input){
                        log<<coord<<", ";
                    }   log<<Function::calculate(input)<<endl;
                }
            }

            if(ArgumentList::verbose_flag){
                cout<<"\n[ Generation "<<n_generation<<" ]"<<endl;
                for(const auto& input: input_space){
                    for(const auto& coord: input){
                        cout<<coord<<", ";
                    }   cout<<endl;
                }

                input_t input = input_space[best_match(input_space, ArgumentList::goal, ArgumentList::expected_y)];
                cout<<"Best fit: "<<input<<endl;
            }

            for(size_t i=0; i<input_space.size(); ++i){
                input_t& parent = input_space[i];
                get_random_inputs(input_space, i, target, inputs[0], inputs[1]);
                mutant = mutate(target, inputs[0], inputs[1], multiplication_factor, interval);
                trial = crossover(mutant, parent, probability_crossover);
                select(trial, parent, ArgumentList::goal, ArgumentList::expected_y);
            }
        
            if(!ArgumentList::is_generation_driven()){
                double error = fabs(best_y-ArgumentList::expected_y);
                if(error <= ArgumentList::threshold) break;
            } else if(n_generation == ArgumentList::n_generation) break;
        }

        if(ArgumentList::visual_flag && !n_run){ log.close(); }
        best_y = Function::calculate(input_space[best_match(input_space, ArgumentList::goal, ArgumentList::expected_y)]);
        global_mins.push_back(best_y);
        cout<<"[ Generation "<<n_generation<<" ]: f"<<input_space[best_match(input_space, ArgumentList::goal, ArgumentList::expected_y)]<<" = "<<best_y<<endl;
    }

    cout<<"Mean: "<<mean_(global_mins)<<"\tStd: "<<std_(global_mins)<<endl;
    if(ArgumentList::visual_flag && n_dimension <= 2){
        vector<double> input(2);
        double step = (interval[1] - interval[0]) / 128;
        ofstream out("function.out");
        for(double i=interval[0]; i<interval[1]; i+=step){
            for(double j=interval[0]; j<interval[1]; j+=step){
                input[0] = i;
                input[1] = j;
                out<<i<<", ";
                if(n_dimension == 2){ out<<j<<", "; }
                out<<Function::calculate(input)<<endl;
                if(n_dimension < 2){ j = interval[1]; }
            }
        }
        out.close();
    }
    /* = = = = = = = = = = = = = = */

    return 0;
}
