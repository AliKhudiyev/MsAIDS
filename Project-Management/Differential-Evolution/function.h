
#pragma once

#include <cmath>
#include <vector>

namespace Function
{
    namespace Benchmark
    {
        double ackley(const std::vector<double>& x){
            return 0;
        }

        double ackley2(const std::vector<double>& x){
            return 0;
        }

        double ackley3(const std::vector<double>& x){
            return 0;
        }

        double ackley4(const std::vector<double>& x){
            return 0;
        }
    } // namespace Benchmark

    double calculate(const std::vector<double>& x){ return pow(x[0]+1, 2)+pow(x[1], 2); } 
} // namespace Function
