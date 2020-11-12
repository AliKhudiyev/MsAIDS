#include <iostream>

using namespace std;

#include <set>
#include <random>
#include <vector>
#include "function.h"

#define N_DIMENSION 2

using input_t = vector<double>;
using space_t = vector<input_t>;

void initialize_input_space(space_t& input_space, size_t dimension, size_t population_size, double lower_limit, double upper_limit){
    srand(time(nullptr));
    
    default_random_engine generator;
    uniform_real_distribution<double> distribution(lower_limit, upper_limit);

    for(size_t i=0; i<population_size; ++i){
        input_t input;
        for(size_t j=0; j<dimension; ++j){
            input.push_back(distribution(generator));
        }
        input_space.push_back(input);
    }
}

void get_random_inputs( const space_t& input_space, 
                        size_t excluded_index, 
                        input_t& target, 
                        input_t& input1, 
                        input_t& input2 ){
    size_t size = input_space.size();
    input_t* inputs[3] = {&target, &input1, &input2};
    set<size_t> unique;
    unique.insert(excluded_index);
    for(size_t i=0; i<3; ){
        size_t index = rand()%size;
        if(!unique.count(index)){
            *(inputs[i++]) = input_space[index];
            unique.insert(index);
        }
    }
}

input_t mutate(const input_t& target, const input_t input1, const input_t input2, double f){
    input_t mutant;

    for(size_t i=0; i<target.size(); ++i){
        double coord = target[i] + f * (input1[i] - input2[i]);
        if(coord < -32) coord = -32;
        else if(coord > 32) coord = 32;
        mutant.push_back(coord);
    }

    return mutant;
}

input_t crossover(const input_t& parent, const input_t& mutant, double probability_crossover){
    input_t trial = parent;

    // default_random_engine generator;
    // uniform_real_distribution<double> distribution(0, 1);
    rand();

    for(size_t i=0; i<trial.size(); ++i){
        if(((double)rand())/((double)RAND_MAX) <= probability_crossover){ // TO DO || rand()%(trial.size()) == i
            trial[i] = mutant[i];
        }
    }

    return trial;
}

void select(const input_t& trial, input_t& parent){
    if(Function::calculate(trial) < Function::calculate(parent)){
        parent = trial;
    }
}

size_t best_match(const space_t& input_space){
    size_t index = 1;
    double ans = Function::calculate(input_space[0]);

    for(size_t i=1; i<input_space.size()-1; ++i){
        const input_t& input = input_space[i];
        double tmp = Function::calculate(input);
        if(tmp < ans){
            ans = tmp;
            index = i;
        }
    }

    return index;
}

int main(int argc, const char** argv){
    space_t input_space;

    bool silent_run = false;
    unsigned int n_generation = 300;
    unsigned int n_population = 5;
    double multiplication_factor = 0.8;
    double probability_crossover = 0.9;

    set<unsigned int> indices;
    input_t target, inputs[2];

    input_t mutant, trial;

    initialize_input_space(input_space, N_DIMENSION, n_population, -32, 32);
    // cout<<input_space.size()<<endl;
    
    for(unsigned int gen_index=0; gen_index<n_generation; ++gen_index){
        if(!silent_run){
            cout<<"\n[ Generation "<<gen_index+1<<"]"<<endl;
            for(const auto& input: input_space){
                for(const auto& coord: input){
                    cout<<coord<<", ";
                }   cout<<endl;
            }

            cout<<"Best fit: (";
            input_t input = input_space[best_match(input_space)];
            for(size_t j=0; j<input.size(); ++j){
                cout<<input[j];
                if(j < input.size() - 1) cout<<", ";
            }
            cout<<")\n";
        }

        for(size_t i=0; i<input_space.size(); ++i){
            input_t& parent = input_space[i];
            // cout<<"=========>>>\nParent agent: "<<parent[0]<<", "<<parent[1]<<endl;
            get_random_inputs(input_space, i, target, inputs[0], inputs[1]);
            // cout<<"> Target agent: "<<target[0]<<", "<<target[1]<<endl;
            // cout<<">> Agent 1: "<<inputs[0][0]<<", "<<inputs[0][1]<<endl;
            // cout<<">> Agent 2: "<<inputs[1][0]<<", "<<inputs[1][1]<<endl;
            mutant = mutate(target, inputs[0], inputs[1], multiplication_factor);
            // cout<<"Mutant: "<<mutant[0]<<", "<<mutant[1]<<endl;
            trial = crossover(mutant, parent, probability_crossover);
            // cout<<"Trial: "<<trial[0]<<", "<<trial[1]<<endl;
            select(trial, parent);
            // cout<<"Selected: "<<parent[0]<<", "<<parent[1]<<"\n<<<========\n";
        }
    }

    return 0;
}
