/* Copyright 2017 Ian Rankin
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
 * to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//
//  testMain.cpp
//
// This is a test code to show an example usage of Differential Evolution

#include <stdio.h>

#include "DifferentialEvolution.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <float.h>

#define BENCH_RUNS 25


int main(int argc, char** argv)
{
	if(argc < 3){
		fprintf(stderr, "./programDE [popSize] [dim]\n");
		return 1;
	}

	int popSize = atoi(argv[1]);
	int dim = atoi(argv[2]);
	int numGen = 10000/popSize*dim;
	float CR = 0.8;
	float F[] = { 0.25, 0.25, 0.2, 0.2 };

	printf("popSize: %d, dim: %d, numGen: %d\n", popSize, dim, numGen);

	// 0 - rastrigin
	// 1 - rosenbrock
	// 2 - griewank
	// 3 - sphere

    // create the min and max bounds for the search space.
	float *minBounds = (float*)malloc(dim * sizeof(float));
	float *maxBounds = (float*)malloc(dim * sizeof(float));

	for(int i=0; i<dim; ++i){
		if(COST_SELECTOR == COST_RASTRIGIN){
			minBounds[i] = -5;
			maxBounds[i] = 5;
		}
		else if(COST_SELECTOR == COST_ROSENBROCK){
			minBounds[i] = -100;
			maxBounds[i] = 100;
		}
		else if(COST_SELECTOR == COST_GRIEWANK){
			minBounds[i] = -600;
			maxBounds[i] = 600;
		}
		else{ // sphere
			minBounds[i] = -100;
			maxBounds[i] = 100;
		}
	}

	printf("min, max : %f, %f\n", minBounds[0], maxBounds[0]);
    
    // a random array or data that gets passed to the cost function.
    float *arr = (float*)malloc((dim+1)*sizeof(float)); // [3] = {2.5, 2.6, 2.7};
	for(int i=0; i<=dim; ++i)
		arr[0] = 0;
    
    // data that is created in host, then copied to a device version for use with the cost function.
    struct data x;
    struct data *d_x;
    gpuErrorCheck(cudaMalloc(&x.arr, sizeof(float) * (dim+1)));
    unsigned long size = sizeof(struct data);
    gpuErrorCheck(cudaMalloc((void **)&d_x, size));
    x.v = 3;
    x.dim = dim + 1;
    gpuErrorCheck(cudaMemcpy(x.arr, (const void *)arr, sizeof(float) * (dim+1), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_x, (void *)&x, sizeof(struct data), cudaMemcpyHostToDevice));

	std::vector<float> best_individual(dim);
	std::vector<float> costs(BENCH_RUNS);
	float mean = 0, std = 0, best_cost = FLT_MAX;
	float time = 0;
    
    // Create the minimizer with a popsize of 192, 50 generations, Dimensions = 2, CR = 0.9, F = 2
	for(int i=0; i<BENCH_RUNS; ++i){
		DifferentialEvolution minimizer(popSize, numGen, dim, CR, F, minBounds, maxBounds);
		
		auto start = std::chrono::high_resolution_clock::now();
		// get the result from the minimizer
		std::vector<float> result = minimizer.fmin(d_x);
		auto duration = std::chrono::high_resolution_clock::now() - start;
		time += std::chrono::duration<float, std::milli>(duration).count() / 
			(1000 * BENCH_RUNS);

		costs[i] = minimizer.getBestCost();
		if(costs[i] < best_cost){
			best_individual = result;
			best_cost = costs[i];
		}

		printf("%d: Result = (", i);
		for(const auto& r: result)
			printf("%.3f,", r);
		printf(")\nFitness: %f\n", costs[i]);
		mean += costs[i] / BENCH_RUNS;
	}

	float tmp = 0;
	for(int i=0; i<BENCH_RUNS; ++i)
		tmp += powf(costs[i] - mean, 2);
	std = sqrtf(tmp / BENCH_RUNS);

	printf("= = = = = = =\nMean: %f\nStd: %f\nBest: %f\nTime(s): %f\n",
			mean, std, best_cost, time);
    return 0;
}
