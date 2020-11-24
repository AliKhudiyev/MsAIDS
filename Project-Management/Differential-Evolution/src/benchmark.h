
#pragma once

#include <cmath>
#include <vector>
#include "../include/expreval.h"

namespace Function
{   
    ExprEval::Function function;

    double _calculate_(const std::vector<double>& x){
        std::initializer_list<double> args;
        return function(x);
    }

    namespace Benchmark
    {
        double ackley(const std::vector<double>& x){
            double a=20, b=0.2, c=2*M_PI;
            double d = (double)x.size();
            double result[2] = { 0, 0 };

            for(size_t i=0; i<x.size(); ++i){
                result[0] += pow(x[i], 2) / d;
                result[1] += cos(c*x[i]) / d;
            }

            return -a*exp(-b*sqrt(result[0])) - exp(result[1]) + a + exp(1);
        }

        double rastrigin(const std::vector<double>& x){
            double d = (double)x.size();
            double result = 0;

            for(size_t i=0; i<x.size(); ++i){
                result += (pow(x[i], 2) - 10*cos(2*M_PI*x[i]));
            }

            return 10*d + result;
        }

        double rosenbrock(const std::vector<double>& x){
            double d = (double)x.size();
            double result = 0;

            for(size_t i=0; i<x.size()-1; ++i){
                result += (100*pow(x[i+1]-pow(x[i], 2), 2) + pow(x[i]-1, 2));
            }

            return result;
        }

        double schwefel(const std::vector<double>& x){
            double d = (double)x.size();
            double result = 0;

            for(size_t i=0; i<x.size()-1; ++i){
                result += (x[i]*sin(sqrt(fabs(x[i]))));
            }

            return 418.9829*d - result;
        }
    } // namespace Benchmark
    
} // namespace Function
