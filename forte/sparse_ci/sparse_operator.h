/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "sparse_ci/sq_operator.h"

namespace forte {

class ActiveSpaceIntegrals;

/**
 * @brief The SparseOperator class
 * Base class for second quantized operators.
 *
 * An operator is a linear combination of terms, where each term is a numerical factor
 * times a product of second quantized operators (a SQOperator object)
 *
 * For example:
 *   0.1 * [2a+ 0a-] - 0.5 * [2a+ 0a-] + ...
 *       Term 0            Term 1
 *  
 *        This class stores operators in each term in the following canonical form
 *            a+_p1 a+_p2 ...  a+_P1 a+_P2 ...   ... a-_Q2 a-_Q1   ... a-_q2 a-_q1
 *            alpha creation   beta creation    beta annihilation  alpha annihilation
 * 
 *        with indices sorted as
 * 
 *            (p1 < p2 < ...) (P1 < P2 < ...)  (... > Q2 > Q1) (... > q2 > q1)
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
    /// 
    void add_term(const std::vector<std::tuple<bool, bool, int>>& op_list,
                  double coefficient = 0.0, bool allow_reordering = false);
    /// add a term to this operator
    void add_term(const SQOperator& sqop);
    /// add a term to this operator of the form
    ///
    ///     coefficient * [... q_2 q_1 q_0]
    ///
    /// where q_0, q_1, ... are second quantized operators. These operators are
    /// passed as string
    ///
    ///     '[... q_2 q_1 q_0]'
    ///
    /// where q_i = <orbital_i><spin_i><type_i> and the quantities in <> are
    ///
    ///     orbital_i: int
    ///     spin_i: 'a' (alpha) or 'b' (beta)
    ///     type_i: '+' (creation) or '-' (annihilation)
    ///
    /// For example, '[0a+ 1b+ 12b- 0a-]'
    ///
    /// @param str a string that defines the product of operators in the format [... q_2 q_1 q_0]
    /// @param coefficient a coefficient that multiplies the product of second quantized operators
    void add_term_from_str(std::string str, double coefficient, bool allow_reordering = false);
    /// remove the last term from this operator
    void pop_term();
    /// @return a term
    const SQOperator& term(size_t n) const;
    /// @return the number of terms
    size_t size() const { return op_list_.size(); }
    /// set the value of the coefficients
    std::vector<double> coefficients() const;
    /// set the value of the coefficients
    void set_coefficients(std::vector<double>& values);
    /// set the value of one coefficient
    void set_coefficient(size_t n, double value) { op_list_[n].set_coefficient(value); }
    /// is this operator antihermitian?
    bool is_antihermitian() const { return antihermitian_; }
    /// @return the list of operators
    const std::vector<SQOperator>& op_list() const { return op_list_; }
    /// @return a string representation of this operator
    std::vector<std::string> str() const;
    /// @return a latex representation of this operator
    std::string latex() const;

  private:
    /// is this an antihermitian operator?
    bool antihermitian_ = false;
    /// a vector of SQOperator objects
    std::vector<SQOperator> op_list_;
};

} // namespace forte

#endif // _sparse_operator_h_
