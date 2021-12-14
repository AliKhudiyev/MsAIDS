#include <cmath>
#include "dea.h"

void initialize(popul_t& popul){
	for(size_t i=0; i<popul.size(); ++i){
		popul[i].resize(dimension);
		for(size_t j=0; j<dimension; ++j)
			// TODO
			popul[i][j] = 0.f;
	}
}

void evaluate(const popul_t& popul, mem_t& mem){
	best_index = 0;
	for(size_t i=0; i<popul.size(); ++i){
		mem[i] = bfuncs[benchmark_function_index](popul[i]);
		if(mem[best_index] > mem[i])
			best_index = i;
	}
}

void mutate(agent_t& mutant, const size_t& parent_index, const size_t* indices, const popul_t& popul){
	const agent_t& parent = popul[parent_index];
	const agent_t& best = popul[best_index];
	const agent_t& r1 = popul[indices[0]];
	const agent_t& r2 = popul[indices[1]];
	const agent_t& r3 = popul[indices[2]];
	const agent_t& r4 = popul[indices[3]];
	const agent_t& r5 = popul[indices[4]];

	for(size_t i=0; i<mutant.size(); ++i)
		mutant[i] = parent[i] + f1 * (best[i] - parent[i]) + 
			f2 * (r1 - parent) + f3 * (r2 - r3) + f4 * (r4 - r5);
}

void crossover(agent_t& trial, const agent_t& parent){
	for(size_t i=0; i<trial.size(); ++i){
		if(0)
			trial[i] = parent[i];
	}
}

void select(agent_t& trial, agentt& parent, const size_t& parent_index, mem_t& mem){
	float_t tmp = bfuncs[benchmark_function_index](trial);
	if(tmp < mem[parent_index]){
		parent = trial;
		mem[parent_index] = tmp;
		if(tmp < mem[best_index])
			best_index = parent_index;
	}
}

float_t rastrigin(const agent_t& agent, const agent_t& optimum){
	float_t res = 0;
	float_t coord;
	for(size_t i=0; i<agent.size(); ++i){
		coord = agent[i] - optimum[i];
		res += coord * coord - 10 * cos(2 * M_PI * coord);
	}
	return res + 10 * agent.size();
}

float_t rosenbrock(const agent_t& agent, const agent_t& optimum){
	float_t res = 0;
	float_t coords[2] = { agent[0] - optimum[0] };
	for(size_t i=1; i<agent.size(); ++i, coords[0]=coords[1]){
		coords[1] = agent[i] - optimum[i];
		res += 100 * (coords[0] * coords[0] - coords[1]) * (coords[0] * coords[0] - coords[1]) + 
			(coords[0] - 1) * (coords[0] - 1);
	}
	return res;
}

float_t griewank(const agent_t& agent, const agent_t& optimum){
	float_t ress[2] = { 0, 1 };
	float_t coord;
	for(size_t i=0; i<agent.size(); ++i){
		coord = agent[i] - optimum[i];
		ress[0] += coord * coord / 4000.f;
		ress[1] *= cos(coord / sqrt(i + 1));
	}
	return ress[0] + ress[1] + 1;
}

float_t sphere(const agent_t& agent, const agent_t& optimum){
	float_t res = 0;
	float_t coord;
	for(size_t i=0; i<agent.size(); ++i){
		coord = agent[i] - optimum[i];
		res += coord * coord;
	}
	return res;
}
