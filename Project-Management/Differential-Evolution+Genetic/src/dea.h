
#pragma once

#include <vector>
#include <set>
#include <random>

#include "function.h"

using input_t = std::vector<double>;
using space_t = std::vector<input_t>;

/*
 * This function initializes the input space uniform randomly.
*/
void initialize_input_space(space_t& input_space, size_t dimension, size_t population_size, const double* interval){
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(interval[0], interval[1]);

    for(size_t i=0; i<population_size; ++i){
        input_t input;
        double tmp = interval[0] + (double)rand() / (double)RAND_MAX * (interval[1] - interval[0]);
        for(size_t j=0; j<dimension; ++j){
            input.push_back(distribution(generator)); // distribution(generator)
        }
        input_space.push_back(input);
    }
}

/*
 * target, input1 and input2 are selected randomly from the input_space;
 * all of them are different from one another;
 * none of them can be chosen as input_space[excluded_index].
*/
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

/*
 * Returns a mutant derived from the 3 points(inputs) obtained from the function get_random_inputs(...);
 * f - scaling factor for the mutation;
 * interval - an array of lower and upper boundaries(i.e. interval[0] = -1, interval[1] = 1).
*/
input_t mutate(const input_t& target, const input_t input1, const input_t input2, double f, const double* interval){
    input_t mutant;

    for(size_t i=0; i<target.size(); ++i){
        double coord = target[i] + f * (input1[i] - input2[i]);
        if(coord < interval[0]) coord = interval[0];
        else if(coord > interval[1]) coord = interval[1];
        mutant.push_back(coord);
    }

    return mutant;
}

/*
 * Returns a trial derived from the parent and the mutant obtained from the function mutate(...);
 * probabilitiy_crossover - crossover probability for uniform random recombination process.
*/
input_t crossover(const input_t& parent, const input_t& mutant, double probability_crossover){
    input_t trial = parent;

    // std::default_random_engine generator;
    // std::uniform_real_distribution<double> distribution(0, 1);

    for(size_t i=0; i<trial.size(); ++i){
        if(((double)rand())/((double)RAND_MAX) <= probability_crossover){ // || rand()%(trial.size()) == i
            trial[i] = mutant[i];
        }
    }

    return trial;
}

/*
 * The trial obtained from the function crossover(...) is replaced by its parent if necessary;
 * goal - global minimum(-1) / global maximum(+1) / specific point(0)
 * desired_y - if goal is 0 then this is the point we want to approximate
*/
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

/*
 * Returns the index of the best candidate solution;
 * goal - global minimum(-1) / global maximum(+1) / specific point(0)
 * desired_y - if goal is 0 then this is the point we want to approximate
*/
size_t best_match(const space_t& input_space, int goal=-1, double desired_y=0){
    size_t index = 0;
    double ans = Function::calculate(input_space[0]);

    for(size_t i=1; i<input_space.size()-1; ++i){
        const input_t& input = input_space[i];
        double tmp = Function::calculate(input);
        
        if(goal == -1 && tmp < ans){
            ans = tmp;
            index = i;
        }
        else if(goal == 1 && tmp > ans){
            ans = tmp;
            index = i;
        }
        else if(!goal && abs(tmp-desired_y) < abs(ans-desired_y)){
            ans = tmp;
            index = i;
        }
    }

    return index;
}
