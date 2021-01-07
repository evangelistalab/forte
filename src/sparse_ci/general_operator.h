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

namespace forte {

class ActiveSpaceIntegrals;

/**
 * @brief A class to represent a second quantized operator in normal ordered
 * form with respect to the true vacuum
 *
 *   a+_p1 a+_p2  ... a+_P1 a+_P2   ... a-_Q2 a-_Q1   ... a-_q2 a-_q1
 *   alpha creation  beta creation   alpha annihilation  beta annihilation
 *
 * The creation and annihilation operators are stored separately as bit arrays
 * using the Determinant class
 */
class SingleOperator {
  public:
    SingleOperator(double factor, const Determinant& cre, const Determinant& ann);
    double factor() const;
    const Determinant& cre() const;
    const Determinant& ann() const;

  private:
    double factor_;
    Determinant cre_;
    Determinant ann_;
};

/// This function converts a string to a single operator
std::vector<SingleOperator> string_to_op_term(const std::string& str);

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
    /// add a term to this operator
    void add_term(const std::vector<SingleOperator>& ops, double value = 0.0);
    /// add a term to this operator
    void add_term_from_str(std::string str, double value);
    /// remove the last term from this operator
    void pop_term();
    /// @return a term
    std::pair<std::vector<SingleOperator>, double> get_term(size_t n);
    /// @return the number of terms
    size_t nterms() const { return coefficients_.size(); }

    /// @return the coefficients associated with each term
    const std::vector<double>& coefficients() const { return coefficients_; }
    /// set the value of the coefficients
    void set_coefficients(std::vector<double>& values) { coefficients_ = values; }
    /// set the value of one coefficient
    void set_coefficient(double value, size_t n) { coefficients_[n] = value; }

    const std::vector<std::pair<size_t, size_t>>& op_indices() const { return op_indices_; }
    const std::vector<SingleOperator>& op_list() const { return op_list_; }
    std::vector<std::string> str();
    static std::vector<std::pair<std::string, double>> timing();
    static void reset_timing();

  private:
    /// coefficients associate with each operator
    std::vector<double> coefficients_;
    /// beginning and end of the SingleOperator associated with a given amplitude
    std::vector<std::pair<size_t, size_t>> op_indices_;
    /// a vector of SingleOperator objects
    std::vector<SingleOperator> op_list_;
};

class StateVector {
  public:
    StateVector();
    StateVector(const det_hash<double>& state_vec);
    det_hash<double>& map() { return state_vec_; }

    auto size() const { return state_vec_.size(); }
    void clear() { state_vec_.clear(); }
    auto find(const Determinant& d) const { return state_vec_.find(d); }

    auto begin() { return state_vec_.begin(); }
    auto end() { return state_vec_.end(); }

    auto begin() const { return state_vec_.begin(); }
    auto end() const { return state_vec_.end(); }

    double& operator[](const Determinant& d) { return state_vec_[d]; }

  private:
    det_hash<double> state_vec_;
};

StateVector apply_operator(GeneralOperator& gop, const StateVector& state);
StateVector apply_exp_ah_factorized(GeneralOperator& gop, const StateVector& state);

StateVector apply_operator_fast(GeneralOperator& gop, const StateVector& state0,
                                double screen_thresh = 1.0e-12);
StateVector apply_exp_operator_fast(GeneralOperator& gop, const StateVector& state0,
                                    double scaling_factor = 1.0, int maxk = 20,
                                    double screen_thresh = 1.0e-12);

StateVector apply_operator_fast2(GeneralOperator& gop, const StateVector& state0,
                                 double screen_thresh = 1.0e-12);
StateVector apply_exp_operator_fast2(GeneralOperator& gop, const StateVector& state0,
                                     double scaling_factor = 1.0, int maxk = 20,
                                     double screen_thresh = 1.0e-12);

StateVector apply_exp_ah_factorized_fast(GeneralOperator& gop, const StateVector& state0,
                                         bool inverse = false);
double energy_expectation_value(StateVector& left_state, StateVector& right_state,
                                std::shared_ptr<ActiveSpaceIntegrals> as_ints);
StateVector apply_number_projector(int na, int nb, StateVector& state);

StateVector apply_hamiltonian(std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                              const StateVector& state0, double screen_thresh = 1.0e-12);

/// Compute the projection  <state0 | op | ref>, for each operator op in gop
std::vector<double> get_projection(GeneralOperator& gop, const StateVector& ref,
                                   const StateVector& state0);

double overlap(StateVector& left_state, StateVector& right_state);

} // namespace forte

#endif // _general_operator_h_
