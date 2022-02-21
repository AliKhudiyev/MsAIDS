
#pragma once

#include <vector>
#include <random>
#include <map>

#define f1 mutation_rates[0]
#define f2 mutation_rates[1]
#define f3 mutation_rates[2]
#define f4 mutation_rates[3]

using float_t = double;
using agent_t = std::vector<float_t>;
using popul_t = std::vector<agent_t>;
using func_t  = float_t (*)(const agent_t&);
using mem_t   = std::vector<float_t>;

/* = = = = = Hyperparameters = = = = = */
size_t population_size = 10;
float mutation_rates[4] = { 0.5, 0.5, 0.5, 0.5 };
float crossover_rate = 0.5;
size_t dimension = 10;
unsigned char benchmark_function_index = 0;
size_t best_index = 0;

constexpr size_t max_objective_evaluation(){ return 1000 * dimension; }

/* = = = = = DEA = = = = = */
void initialize(popul_t& popul);
void evaluate(const popul_t& popul, mem_t& mem);
void mutate(agent_t& mutant, const size_t& parent_index, const size_t* indices, const popul_t& popul);
void crossover(agent_t& trial, const agent_t& parent);
void select(agent_t& trial, agent_t& parent, const size_t parent_index, float* const mem);

/* = = = = = Benchmark functions = = = = = */
float_t rastrigin(const agent_t& agent, const agent_t& optimum);
float_t rosenbrock(const agent_t& agent, const agent_t& optimum);
float_t griewank(const agent_t& agent, const agent_t& optimum);
float_t sphere(const agent_t& agent, const agent_t& optimum);

func_t bfuncs[4] = { rastrigin, rosenbrock, griewank, sphere };
const char* bfunc_names[4] = { "shifted rastrigin", "shifted rosenbrock", 
							   "shifted griewank", "shifted sphere" };
std::pair<float_t, float_t> domains[4] {
	std::make_pair(-5., 5.),
	std::make_pair(-100., 100.),
	std::make_pair(-600., 600.),
	std::make_pair(-100., 100.),
};

