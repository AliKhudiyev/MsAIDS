#include <iostream>
#include <set>
#include <random>
#include <vector>

#include "parser.h"
#include "dea.h"

using namespace std;

#define N_DIMENSION 2

ostream& operator<<(ostream& out, const vector<double>& vec){
    out<<"(";
    for(size_t i=0; i<vec.size(); ++i){
        cout<<vec[i];
        if(i < vec.size() - 1) cout<<", ";
    }
    out<<")";
    return out;
}

int main(int argc, char* const* argv){
    ArgumentList::parse(argc, argv);

    for(size_t n_run=0; n_run<ArgumentList::n_benchmark_run; ++n_run){
        space_t input_space;

        unsigned int n_generation = 0;
        double multiplication_factor = ArgumentList::f;
        double probability_crossover = ArgumentList::p;
        double best_y = 0;

        set<unsigned int> indices;
        input_t target, inputs[2];
        input_t mutant, trial;

        initialize_input_space(input_space, N_DIMENSION, ArgumentList::population_size, -32, 32);

        while(++n_generation){
            best_y = Function::calculate(input_space[best_match(input_space)]);

            if(ArgumentList::verbose_flag){
                cout<<"\n[ Generation "<<n_generation<<" ]"<<endl;
                for(const auto& input: input_space){
                    for(const auto& coord: input){
                        cout<<coord<<", ";
                    }   cout<<endl;
                }

                input_t input = input_space[best_match(input_space)];
                cout<<"Best fit: "<<input<<endl;
            }

            for(size_t i=0; i<input_space.size(); ++i){
                input_t& parent = input_space[i];
                get_random_inputs(input_space, i, target, inputs[0], inputs[1]);
                mutant = mutate(target, inputs[0], inputs[1], multiplication_factor);
                trial = crossover(mutant, parent, probability_crossover);
                select(trial, parent, ArgumentList::goal, ArgumentList::expected_y);
            }
        
            if(!ArgumentList::is_generation_driven()){
                double error = fabs(best_y-ArgumentList::expected_y);
                if(error <= ArgumentList::threshold) break;
            } else if(n_generation == ArgumentList::n_generation) break;
        }

        cout<<"[ Generation "<<n_generation<<" ]: f"<<input_space[best_match(input_space)]<<" = "<<best_y<<endl;
    }

    return 0;
}
