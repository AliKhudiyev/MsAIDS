#include <cstdio>
#include <chrono>
#include "dea.h"

using namespace std::chrono::high_resolution_clock;
using namespace std::chrono::duration;

// usage: ./main [dimension] [population_size] (benchmark_function_index)
// if `benchmark_function_index` is not given, then all functions are targeted
int main(int argc, char** argv){
	// handling the command line arguments
	int bfend = 4;
	if(argc < 3)
		exit(1);
	else{
		dimension = atol(argv[1]);
		population_size = atol(argv[2]);
		if(argc == 4){
			benchmark_function_index = atoi(argv[3]);
			bfend = benchmark_function_index + 1;
		}
	}

	// defining the number of benchmark runs
	const unsigned n_run = 25;
	// defining the global minimum of the selected benchmark function
	const agent_t optimum(population_size, 0);

	// allocating population
	popul_t population(population_size);
	agent_t agent(dimension);
	// allocating volatile memory for dp
	mem_t memory(population_size);
	float_t best_results[4][n_run];
	size_t indices[5];

	for(; benchmark_function_index < bfend; ++benchmark_function_index){
		float_t mean = 0, var = 0;
		auto time = high_resolution_clock::now();

		for(size_t run=0, n_eval=0; run<n_run; ++run, n_eval=0){
			// initializing the population randomly
			initialize(population);
			// evaluating the fitness of population
			evaluate(population, memory);
			
			while((n_eval += population_size) <= max_objective_evaluation()){
				for(size_t i=0; i<population.size(); ++i){
					const agentt& parent = population[i];

					// turning agent to mutant
					mutate(agent, i, indices, population);
					// turning mutant to trial
					crossover(agent, parent);
					// replacing parent with trial if necessary
					select(agent, parent, i, memory);
				}
			}

			best_results[benchmark_function_index][run] = memory[best_index];
			mean += memory[best_index] / n_run;
		}

		duration<float_t> dt = high_resolution_clock::now() - time;
		for(size_t i=0; i<n_run; ++i)
			var += pow(best_results[benchmark_function_index][i] - mean, 2) / population_size;
		printf("Benchmark function: %s\nMean: %.2e\nStd: %.2e\n Time(sec): %.3lf\n", 
				bfunc_names[benchmark_function_index-1], mean, sqrt(var), dt.count());
	}
	return 0;
}
