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
//  DifferentialEvolution.hpp
//
// This class is a wrapper to make calls to the cuda differential evolution code easier to work with.
// It handles all of the internal memory allocation for the differential evolution and holds them
// as device memory for the GPU
//
// Example wrapper usage:
//
// float mins[3] = {0,-1,-3.14};
// float maxs[3] = {10,1,3.13};
//
// DifferentialEvolution minimizer(64,100, 3, 0.9, 0.5, mins, maxs);
//
// minimizer.fmin(NULL);
//
//////////////////////////////////////////////////////////////////////////////////////////////
// However if needed to pass arguements then an example usage is:
//
// // create the min and max bounds for the search space.
// float minBounds[2] = {-50, -50};
// float maxBounds[2] = {100, 200};
//
// // a random array or data that gets passed to the cost function.
// float arr[3] = {2.5, 2.6, 2.7};
//
// // data that is created in host, then copied to a device version for use with the cost function.
// struct data x;
// struct data *d_x;
// gpuErrorCheck(cudaMalloc(&x.arr, sizeof(float) * 3));
// unsigned long size = sizeof(struct data);
// gpuErrorCheck(cudaMalloc((void **)&d_x, size));
// x.v = 3;
// x.dim = 2;
// gpuErrorCheck(cudaMemcpy(x.arr, (void *)&arr, sizeof(float) * 3, cudaMemcpyHostToDevice));
//
// // Create the minimizer with a popsize of 192, 50 generations, Dimensions = 2, CR = 0.9, F = 2
// DifferentialEvolution minimizer(192,50, 2, 0.9, 0.5, minBounds, maxBounds);
//
// gpuErrorCheck(cudaMemcpy(d_x, (void *)&x, sizeof(struct data), cudaMemcpyHostToDevice));
//
// // get the result from the minimizer
// std::vector<float> result = minimizer.fmin(d_x);
//

#ifndef DifferentialEvolution_hpp
#define DifferentialEvolution_hpp

#include <stdio.h>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "DifferentialEvolutionGPU.h"

///// IMPORTANT ////////
// This is a convienent place to put any structs that could be used to send
// data to the cost. Here is the struct for the example of passing arguements.
// I recommend you keep it here.
struct data {
    float *arr;
    float v;
    int dim;
};

class DifferentialEvolution {
private:
    float *d_target1;
    float *d_target2;
    float *d_cost;
    float *d_mutant;
    float *d_trial;
    
    
    float *d_min;
    float *d_max;
    float *h_cost;
    
    void *d_randStates;
    
    int popSize;
    int dim;
    
    int CR;
    int numGenerations;
    float *F;
    
public:
    
    // Constructor for DifferentialEvolution
    //
    // @param PopulationSize - the number of agents the DE solver uses.
    // @param NumGenerations - the number of generation the differential evolution solver uses.
    // @param Dimensions - the number of dimesnions for the solution.
    // @param crossoverConstant - the number of mutants allowed pass each generation CR in
    //              literature given in the range [0,1]
    // @param mutantConstant - the scale on mutant changes (F in literature) given [0,2]
    //              default = 0.5
    // @param func - the cost function to minimize.
    DifferentialEvolution(int PopulationSize, int NumGenerations, int Dimensions,
                float crossoverConstant, float *mutantConstant,
                float *minBounds, float *maxBounds);
    
    // fmin
    // wrapper to the cuda function C function for differential evolution.
    // @param args - this a pointer to arguments for the cost function.
    //      This MUST point to device memory or NULL.
    //
    // @return the best set of parameters
    std::vector<float> fmin(void *args);
    float getBestCost() const;
};

#endif /* DifferentialEvolution_hpp */
