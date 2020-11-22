
#pragma once

#include "benchmark.h"

namespace Function
{
    double (*calculate)(const std::vector<double>& x);
    double calculate_(const std::vector<double>& x){ return -pow(x[0], 2)-pow(x[1], 2); } 
} // namespace Function
