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

#ifndef _general_operator_h_
#define _general_operator_h_

#include <vector>
#include <unordered_map>

#include "sparse_ci/determinant.h"
#include "sparse_ci/sq_operator.h"

namespace forte {

class ActiveSpaceIntegrals;

/**
 * A data structure used to represent a second quantized operator string like
 *
 *  factor * ... op(2) op(1) op(0),
 *
 * like, for example
 *
 *  0.1 * a^\dagger_2a a^\dagger_3b a_1b a_0a
 *
 * This operator is stored as
 *
 *  (0.1, [(false,true,0),(false,false,1),(true,false,3),(true,true,2)])
 *
 * The data format is
 *
 *  (factor, [(creation_0, alpha_0, orb_0), (creation_1, alpha_1, orb_1), ...])
 *
 * where the operators are arranged as
 *
 * where
 *  creation_i  : bool (true = creation, false = annihilation)
 *  alpha_i     : bool (true = alpha, false = beta)
 *  orb_i       : int  (the index of the mo)
 *
 */
using op_t = std::pair<double, std::vector<std::tuple<bool, bool, int>>>;

/**
 * @brief The GeneralOperator class
 * Base class for generic second quantized operators.
 *
 * Each term is represented as a linear combination of second quantized
 * operator strings times a coefficient.
 *
 * For example:
 *   0.1 * {[2a+ 0a-] + 2.0 [2b+ 0b-]} - 0.5 * {2.0 [2a+ 0a-] + [2b+ 0b-]} + ...
 *              Term 0                                      Term 1
 */
class GeneralOperator {
  public:
    /// add a term to this operator (python-friendly version)
    /// the user has to pass a list of tuples of the form
    ///
    ///     [(creation_0, alpha_0, orb_0), (creation_1, alpha_1, orb_1), ...]
    ///
    /// where the indices are defined as
    /// creation_i  : bool (true = creation, false = annihilation)
    /// alpha_i     : bool (true = alpha, false = beta)
    /// orb_i       : int  (the index of the mo)
    void add_term(const std::vector<std::pair<double, op_tuple_t>>& op_list, double value = 0.0);
    /// add a term to this operator
    void add_term(const std::vector<SQOperator>& ops, double value = 0.0);
    /// add a term to this operator
    void add_term_from_str(std::string str, double value);
    /// remove the last term from this operator
    void pop_term();
    /// @return a term
    std::pair<std::vector<SQOperator>, double> get_term(size_t n);
    /// @return the number of terms
    size_t nterms() const { return coefficients_.size(); }

    /// @return the coefficients associated with each term
    const std::vector<double>& coefficients() const { return coefficients_; }
    /// set the value of the coefficients
    void set_coefficients(std::vector<double>& values) { coefficients_ = values; }
    /// set the value of one coefficient
    void set_coefficient(double value, size_t n) { coefficients_[n] = value; }

    const std::vector<std::pair<size_t, size_t>>& op_indices() const { return op_indices_; }
    const std::vector<SQOperator>& op_list() const { return op_list_; }
    std::vector<std::string> str() const;
    std::string latex() const;
    static std::vector<std::pair<std::string, double>> timing();
    static void reset_timing();

  private:
    /// coefficients associate with each operator
    std::vector<double> coefficients_;
    /// beginning and end of the SQOperator associated with a given amplitude
    std::vector<std::pair<size_t, size_t>> op_indices_;
    /// a vector of SQOperator objects
    std::vector<SQOperator> op_list_;
};

} // namespace forte

#endif // _general_operator_h_