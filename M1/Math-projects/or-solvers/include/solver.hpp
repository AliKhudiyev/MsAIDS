
#pragma once

#include "variable.hpp"
#include "tableau.hpp"
#include <vector>
#include <map>
#include <initializer_list>

#define LP_LESS_THAN 0
#define LP_LESS_THAN_OR_EQUAL 1
#define LP_GREATER_THAN 2
#define LP_GREATER_THAN_OR_EQUAL 3
#define LP_EQUAL 4

#define LP_MAXIMIZE 0
#define LP_MINIMIZE 1

#define LP_OPTIMAL 0
#define LP_FEASIBLE 1
#define LP_NONE 2

namespace lp
{
	using namespace utils;
	using status_t = char;

	class Solver
	{
	private:
		size_t m_basisVariableCount = 0;
		std::vector<Variable> m_variables; // includes original and slack/surplus variables
		std::vector<double> m_objective;
		std::vector<std::vector<double>> m_constraints;
		std::vector<char> m_operators;
		bool m_goal;

		std::vector<size_t> m_artificialVariables;
		std::vector<double> m_transforms;
		std::vector<std::vector<size_t>> m_constraintIndices; // To hold indices of the originally given constraints

	public:
		Solver() = default;
		~Solver() {}

		void add_variable(const Variable &variable)
		{
			m_variables.push_back(variable);
			m_basisVariableCount += 1;

			/*
      variable.satisfy(0.0, variable.upper_boundary()-variable.lower_boundary());
      if(variable.upper_boundary() < LP_INFINITY-1.0){
	// add another constraint: 
      }
      m_transforms.push_back();
      */
		}
		void set_objective(const std::vector<double> &coefficients, bool goal)
		{
			m_objective = coefficients;
			if (goal == LP_MINIMIZE)
			{
				for (auto &coef : m_objective)
				{
					coef *= -1.0;
				}
			}
			m_goal = LP_MAXIMIZE; // To be removed
		}
		void set_objective(const std::initializer_list<double> &coefficients, bool goal)
		{
			std::vector<double> coefs;
			for (auto it = coefficients.begin(); it != coefficients.end(); ++it)
			{
				coefs.push_back(*it);
			}
			set_objective(coefs, goal);
		}

		void subject_to(const std::vector<double> &coefficients, char operator_, double number)
		{
			auto constraint = coefficients;
			constraint.push_back(number);

			if (operator_ == LP_GREATER_THAN_OR_EQUAL || operator_ == LP_GREATER_THAN)
			{
				for (auto &num : constraint)
				{
					num *= -1.0;
				}
			}

			m_constraints.push_back(constraint);
			m_operators.push_back(LP_LESS_THAN_OR_EQUAL); // To be removed

			if (constraint.back() < 0)
			{
				m_artificialVariables.push_back(m_constraints.size() - 1);
			}

			auto slack = Variable();
			m_variables.push_back(slack);

			if (operator_ == LP_EQUAL)
			{
				auto constraint2 = coefficients;
				for (auto &num : constraint2)
				{
					num *= -1.0;
				}
				subject_to(constraint2, LP_LESS_THAN_OR_EQUAL, -number);
			}
		}
		void subject_to(const std::initializer_list<double> &coefficients, char operator_, double number)
		{
			std::vector<double> coefficients_;
			for (auto it = coefficients.begin(); it != coefficients.end(); ++it)
			{
				coefficients_.push_back(*it);
			}
			subject_to(coefficients_, operator_, number);
		}
		void remove(const std::vector<double> &coefficients, char operator_, double number)
		{
			// TO DO
		}
		void remove(size_t index)
		{
			if (index == -1)
			{
				for (size_t i = m_basisVariableCount; i < m_variables.size(); ++i)
					m_variables.pop_back();
				m_constraints.clear();
			}
			else if (index < m_constraints.size())
				m_constraints.erase(m_constraints.begin() + index);
		}

		status_t solve(bool verbose=false)
		{
			/* --- */
			Tableau tableau(m_constraints.size() + 2, m_variables.size() + m_artificialVariables.size() + 1);
			std::vector<std::vector<double>> constraints;
			size_t slack_index = m_basisVariableCount, index = 0;

			for (size_t i = 0; i < m_objective.size(); ++i)
			{
				tableau.at(0, i) = m_objective[i];
			}
			for (size_t i = 0; i < m_variables.size() + 1 - m_objective.size(); ++i)
			{
				tableau.at(0, m_objective.size() + i) = 0.0;
			}

			for (size_t i = 0; i < m_constraints.size(); ++i)
			{
				// std::vector<double> constraint;
				for (size_t j = 0; j < m_constraints[i].size() - 1; ++j)
				{
					// constraint.push_back(m_constraints[i][j]);
					tableau.at(i, j) = m_constraints[i][j];
				}
				for (size_t j = 0; j < m_variables.size() - m_basisVariableCount; ++j)
				{
					// constraint.push_back(0.0);
					tableau.at(i, m_basisVariableCount + j) = 0.0;
				}

				// constraint[slack_index] = 1.0;
				tableau.at(i, slack_index++) = 1.0;

				// constraint.push_back(m_constraints[i].back());
				tableau.at(i, m_variables.size() + m_artificialVariables.size()) = m_constraints[i].back();
			}

			std::vector<double> artificial_objective;
			for (size_t i = 0; i < m_variables.size() + m_artificialVariables.size() + 1; ++i)
				artificial_objective.push_back(0.0);

			for (size_t i = 0; i < m_artificialVariables.size(); ++i)
			{
				tableau.at(m_artificialVariables[i], m_variables.size() + m_artificialVariables[i] - 1) = -1.0;
				tableau.mult(-1.0, m_artificialVariables[i]);

				for (size_t j = 0; j < m_constraints[m_artificialVariables[i]].size(); ++j)
				{
					artificial_objective[j] += tableau.at(m_artificialVariables[i], j);
				}
				artificial_objective[m_basisVariableCount + m_artificialVariables[i]] = tableau.at(m_artificialVariables[i], m_basisVariableCount + m_artificialVariables[i]);
				artificial_objective[m_variables.size() + m_artificialVariables.size()] += tableau.at(m_artificialVariables[i], tableau.width() - 1);
			}

			// printf(" dbg objective/constraint size: %zu, %zu\n", m_objective.size(), m_constraints[0].size());
			for (size_t i = 0; i < m_objective.size(); ++i)
			{
				// printf("%lf :\n", m_objective[i]);
				tableau.at(m_constraints.size(), i) = m_objective[i];
				tableau.at(m_constraints.size() + 1, i) = artificial_objective[i];
			}
			for (size_t i = m_objective.size(); i < m_variables.size(); ++i)
			{
				tableau.at(m_constraints.size() + 1, i) = artificial_objective[i];
			}
			tableau.at(m_constraints.size() + 1, tableau.width() - 1) = artificial_objective.back();
			/* --- */

			size_t n_phase = m_artificialVariables.size() ? 2 : 1;
			size_t offset = n_phase == 1 ? 0 : 1;
			status_t status = LP_NONE;
			for (size_t phase = 0; phase < n_phase; ++phase, offset = 0)
			{
				index = 0;
				if(verbose)
					std::cout << "Tableau:\n" << tableau << "\n";

				do
				{
					// TO DO
					size_t row_index, col_index;

					// Find column index for pivot vector
					col_index = tableau.get_index(/*index:*/ m_constraints.size() + offset, /*from_row:*/ true);
					if (col_index == -1)
					{
						if(verbose)
							printf(" -> (column)Stop! [info: cannot be further optimized]\n");
						status = LP_OPTIMAL;
						break;
					}
					if(verbose)
						printf(" dbg pivot column: %zu\n", col_index);
					// Find row index for pivot element
					row_index = tableau.get_index(col_index, false);
					if (row_index == -1)
					{
						if(verbose)
							printf(" -> (row)Stop! [info: cannot select any pivot element]\n");
						break;
					}
					if(verbose)
						printf(" dbg pivot row: %zu; element: %lf\n", row_index, tableau.at(row_index, col_index));

					// Processing with pivot
					double pivot = tableau.at(row_index, col_index);

					tableau.mult(1.0 / pivot, row_index);
					for (size_t i = 0; i < m_constraints.size() + 2; ++i)
					{
						if (i == row_index)
							continue;
						tableau.add(i, row_index, i, -tableau.at(i, col_index));
					}

					if(verbose)
						std::cout << "Tableau:\n" << tableau << "\n";
				} while (index++ < 10);
				if(verbose)
					printf(" dbg stopped @%zu iteration...\n", index);
			}
			
			for(size_t i=0; i<m_variables.size(); ++i){
				m_variables[i].set(0);
				int index = -1;
				const double epsilon = 1e-9;
				double tmp = 0;
				for(size_t j=0; j<tableau.height()-2; ++j){
					tmp += tableau.at(j, i);
					if(fabs(tmp) > epsilon && fabs(tmp - 1) > epsilon){
						index = -1;
						break;
					} else if(fabs(tmp - 1) <= epsilon && index == -1){
						index = j;
					}
				}
				if(index != -1) m_variables[i].set(tableau.at(index, tableau.width()-1));
			}
			return status;
		}
		std::vector<double> get_solution() const
		{
			std::vector<double> vec;
			for (const auto& variable : m_variables)
			{
				vec.push_back(variable.get());
			}
			double result = 0;
			for(size_t i=0; i<m_objective.size(); ++i)
				result += m_objective[i] * vec[i];
			vec.push_back(result);
			return vec;
		}
	};

}

namespace knapsack
{
	using item_t = std::pair<unsigned /*value*/, unsigned /*weight*/>;
    using items_t = std::vector<item_t>;
    using stat_t = item_t;
    using solution_t = std::vector<bool>;
    using result_t = std::pair<solution_t /*solution*/, unsigned /*value*/>;

    #define value(item) item.first
    #define weight(item) item.second

    static unsigned call_count = 0;
    static solution_t solution;

    class Solver{
        private:
        items_t m_items;
        unsigned m_capacity;

        public:
        result_t solve(const std::vector<item_t>& items, unsigned capacity){
            result_t result;
            result.second = 0;
            solution.resize(items.size());
            solution_t tmp = solution;

            _solve(items, capacity, stat_t(0, 0), 0, tmp, result.second);
            result.first = solution;
            return result;
        }

        private:
        void _solve(const std::vector<item_t>& items, unsigned capacity, stat_t status, unsigned level, std::vector<bool> solution_, unsigned& max_value){
            ++call_count;
            if(level == items.size()){
                return ;
            }

            unsigned current_value = value(status)+value(items[level]);
            unsigned current_weight = weight(status) + weight(items[level]);

            if(current_weight <= capacity){
                solution_[level] = true;
                if(current_value > max_value){
                    max_value = current_value;
                    solution = solution_;
                }
                _solve(items, capacity, stat_t(current_value, current_weight), level+1, solution_, max_value);
            }
            solution_[level] = false;
            _solve(items, capacity, status, level+1, solution_, max_value);
        }
    };
} // namespace knapsack

