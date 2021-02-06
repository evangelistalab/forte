/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef _sparse_operator_h_
#define _sparse_operator_h_

#include <vector>
#include <unordered_map>

#include "sparse_ci/determinant.h"
#include "sparse_ci/sq_operator.h"

namespace forte {

class ActiveSpaceIntegrals;

/// This function converts a string to a single operator
std::vector<SQOperator> string_to_op_term(const std::string& str);

/**
 * @brief The SparseOperator class
 * Base class for second quantized operators.
 *
 * Each term is represented as a linear combination of second quantized
 * operator strings times a coefficient.
 *
 * For example:
 *   0.1 * [2a+ 0a-] - 0.5 * [2a+ 0a-] + ...
 *       Term 0            Term 1
 */
class SparseOperator {
  public:
    SparseOperator(bool antihermitian = false) { antihermitian_ = antihermitian; }
    /// add a term to this operator (python-friendly version) of the form
    ///
    ///     coefficient * [... q_2 q_1 q_0]
    ///
    /// where q_0, q_1, ... are second quantized operators. These operators are
    /// passed as a list of tuples of the form
    ///
    ///     [(creation_0, alpha_0, orb_0), (creation_1, alpha_1, orb_1), ...]
    ///
    /// where the indices are defined as
    ///
    ///     creation_i  : bool (true = creation, false = annihilation)
    ///     alpha_i     : bool (true = alpha, false = beta)
    ///     orb_i       : int  (the index of the mo)
    void add_term(const std::vector<std::tuple<bool, bool, int>>& op_list,
                  double coefficient = 0.0);
    /// add a term to this operator
    void add_term(const SQOperator& sqop);
    /// add a term to this operator
    void add_term_from_str(std::string str, double value);
    /// remove the last term from this operator
    void pop_term();
    /// @return a term
    const SQOperator& get_term(size_t n) const;
    /// @return the number of terms
    size_t nterms() const { return op_list_.size(); }
    /// set the value of the coefficients
    std::vector<double> coefficients();
    /// set the value of the coefficients
    void set_coefficients(std::vector<double>& values);
    /// set the value of one coefficient
    void set_coefficient(double value, size_t n) { op_list_[n].set_factor(value); }

    bool is_antihermitian() const { return antihermitian_; }
    const std::vector<SQOperator>& op_list() const { return op_list_; }
    std::vector<std::string> str() const;
    std::string latex() const;

    static std::vector<std::pair<std::string, double>> timing();
    static void reset_timing();

  private:
    ///
    bool antihermitian_ = false;
    /// a vector of SQOperator objects
    std::vector<SQOperator> op_list_;
};

} // namespace forte

#endif // _sparse_operator_h_
