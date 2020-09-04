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

// used to represent a combination of:
//    creation  : bool (true = creation, false = annihilation)
//    alpha     : bool (true = alpha, false = beta)
//    orb       : int  (the index of the mo)
using op_t = std::pair<double, std::vector<std::tuple<bool, bool, int>>>;

/**
 * @brief The GeneralOperator class
 *        Base class for generic second quantized operators
 */
class GeneralOperator {
  public:
    std::pair<std::vector<SingleOperator>, double> get_operator(size_t n);
    void add_operator(const std::vector<op_t>& op_list, double value = 0.0);
    void add_operator2(const std::vector<SingleOperator>& ops, double value = 0.0);
    void pop_operator();
    size_t nops() const { return amplitudes_.size(); }
    const std::vector<double>& amplitudes() const { return amplitudes_; }
    void set_amplitudes(std::vector<double>& amplitudes) { amplitudes_ = amplitudes; }
    void set_amplitude(double value, size_t n) { amplitudes_[n] = value; }
    const std::vector<std::pair<size_t, size_t>>& op_indices() const { return op_indices_; }
    const std::vector<SingleOperator>& op_list() const { return op_list_; }
    std::vector<std::string> str();
    static std::vector<std::pair<std::string, double>> timing();
    static void reset_timing();

  private:
    std::vector<double> amplitudes_;
    std::vector<std::pair<size_t, size_t>> op_indices_;
    std::vector<SingleOperator> op_list_;
};

det_hash<double> apply_operator(GeneralOperator& gop, const det_hash<double>& state);
det_hash<double> apply_exp_ah_factorized(GeneralOperator& gop, const det_hash<double>& state);

det_hash<double> apply_operator_fast(GeneralOperator& gop, const det_hash<double>& state0,
                                     double screen_thresh = 1.0e-12);
det_hash<double> apply_exp_operator_fast(GeneralOperator& gop, const det_hash<double>& state0,
                                         double scaling_factor = 1.0, int maxk = 20,
                                         double screen_thresh = 1.0e-12);

det_hash<double> apply_operator_fast2(GeneralOperator& gop, const det_hash<double>& state0,
                                      double screen_thresh = 1.0e-12);
det_hash<double> apply_exp_operator_fast2(GeneralOperator& gop, const det_hash<double>& state0,
                                          double scaling_factor = 1.0, int maxk = 20,
                                          double screen_thresh = 1.0e-12);

det_hash<double> apply_exp_ah_factorized_fast(GeneralOperator& gop, const det_hash<double>& state0,
                                              bool inverse = false);
double energy_expectation_value(det_hash<double>& left_state, det_hash<double>& right_state,
                                std::shared_ptr<ActiveSpaceIntegrals> as_ints);
det_hash<double> apply_number_projector(int na, int nb, det_hash<double>& state);

det_hash<double> apply_hamiltonian(std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                                   const det_hash<double>& state0, double screen_thresh = 1.0e-12);

/// Compute the projection  <state0 | op | ref>, for each operator op in gop
std::vector<double> get_projection(GeneralOperator& gop, const det_hash<double>& ref,
                                   const det_hash<double>& state0);

double overlap(det_hash<double>& left_state, det_hash<double>& right_state);

} // namespace forte

#endif // _general_operator_h_
