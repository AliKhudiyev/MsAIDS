
#pragma once

#include <vector>
#include <set>
#include <random>

#include "function.h"

using input_t = std::vector<double>;
using space_t = std::vector<input_t>;

void initialize_input_space(space_t& input_space, size_t dimension, size_t population_size, double lower_limit, double upper_limit){
    srand(time(nullptr));
    
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(lower_limit, upper_limit);

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
    std::set<size_t> unique;
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

    for(size_t i=0; i<trial.size(); ++i){
        if(((double)rand())/((double)RAND_MAX) <= probability_crossover){ // TO DO || rand()%(trial.size()) == i
            trial[i] = mutant[i];
        }
    }

    return trial;
}

void select(const input_t& trial, input_t& parent, int goal=-1, double desired_y=0){
    if(goal == -1){
        if(Function::calculate(trial) < Function::calculate(parent)){
            parent = trial;
        }
    } else if(goal == 1){
        if(Function::calculate(trial) > Function::calculate(parent)){
            parent = trial;
        }
    } else{
        if(fabs(Function::calculate(trial)-desired_y) < fabs(Function::calculate(parent)-desired_y)){
            parent = trial;
        }
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