#include <iostream>
#include <limits>
#include <cmath>
#include "include/solver.hpp"

using namespace std;

int main(int argc, char **argv)
{
	LP_PRINT_INFO;
	printf("\n= = = LP Solver = = =\n");

	lp::Variable a, b, c;
	lp::Solver solver;

	solver.add_variable(a);
	solver.add_variable(b);
	solver.add_variable(c);

	solver.set_objective({-3, -1, 1}, LP_MAXIMIZE);
	solver.subject_to({-3, -1, 5}, LP_GREATER_THAN_OR_EQUAL, 18);
	solver.subject_to({-1, -1, 2}, LP_GREATER_THAN_OR_EQUAL, 5);
	solver.subject_to({-1, 1, 1}, LP_LESS_THAN_OR_EQUAL, 6);

	auto status = solver.solve(/*true*/);

	if(status == LP_NONE){
		printf("Could not find any solution!\n");
		return -1;
	}
	
	auto solution = solver.get_solution();
	printf("=> Solution: ");
	for(size_t i=0; i<solution.size()-1; ++i){
		printf("%.3lf, ", solution[i]);
	}
	printf("\n=> Value: %.3lf\n", solution.back());

 	printf("\n\n= = = Knapsack Solver = = =\n");
	
 	using item_t = knapsack::item_t;
 	knapsack::items_t items {
        item_t(3, 4),
        item_t(5, 3),
        item_t(10, 9),
        item_t(2, 3),
        item_t(6, 8),
    };
    unsigned capacity = 11;
	auto ksolver = knapsack::Solver();

	auto result = ksolver.solve(items, capacity);
	printf("=> Solution: ");
	for(const auto& bit: result.first)
        cout << bit << ' ';
    printf("\n=> Value: %d\n", result.second);

	return 0;
}
