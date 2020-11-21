
#pragma once

#include <cmath>
#include <vector>

namespace Function
{
    namespace Benchmark
    {
        double ackley(const std::vector<double>& x){
            return pow(x[0],2)+x[1]*x[1];
        }

        double ackley2(const std::vector<double>& x){
            return pow(x[0],2)+x[1]*x[1];
        }

        double ackley3(const std::vector<double>& x){
            return pow(x[0],2)+x[1]*x[1];
        }

        double ackley4(const std::vector<double>& x){
            return pow(x[0],2)+x[1]*x[1];
        }
    } // namespace Benchmark

    double calculate(const std::vector<double>& x){ return pow(x[0],2)+x[1]*x[1]; } 
} // namespace Function
