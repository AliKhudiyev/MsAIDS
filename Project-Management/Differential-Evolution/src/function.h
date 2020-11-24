
#pragma once

#include "benchmark.h"

namespace Function
{
    double (*calculate)(const std::vector<double>& x);
    double calculate_(const std::vector<double>& x){ return x[0]+1; } 
} // namespace Function
