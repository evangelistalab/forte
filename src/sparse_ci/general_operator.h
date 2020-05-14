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

using op_t = std::pair<double, std::vector<std::tuple<bool, bool, int>>>;
using py_op_t = std::pair<double, std::vector<std::tuple<bool, bool, int>>>;

/**
 * @brief The SigmaVector class
 *        Base class for a sigma vector object.
 */
class GeneralOperator {
  public:
    void add_operator(std::vector<op_t> op_list);
    std::vector<double>& amplitudes() { return amplitudes_; }
    void set_amplitudes(std::vector<double>& amplitudes) { amplitudes_ = amplitudes; }
    std::vector<std::pair<size_t, size_t>>& op_indices() { return op_indices_; }
    std::vector<op_t>& op_list() { return op_list_; }

  private:
    std::vector<double> amplitudes_;
    std::vector<std::pair<size_t, size_t>> op_indices_;
    std::vector<op_t> op_list_;
};

det_hash<double> apply_general_operator(GeneralOperator& gop, det_hash<double>& state);
det_hash<double> apply_general_operator_spin(GeneralOperator& gop, det_hash<double>& state);
det_hash<double> apply_exp_general_operator(GeneralOperator& gop, det_hash<double> state, int maxn);
det_hash<double> apply_exp_general_operator_matrix(GeneralOperator& gop, det_hash<double> state,
                                                   int norbs, int maxn);
det_hash<double> apply_exp_general_operator_spin(GeneralOperator& gop, det_hash<double> state,

                                                 int maxn);
det_hash<double> apply_number_projector(int na, int nb, det_hash<double>& state);
double overlap(det_hash<double>& left_state, det_hash<double>& right_state);

double energy_expectation_value(det_hash<double>& left_state, det_hash<double>& right_state,
                                std::shared_ptr<ActiveSpaceIntegrals> as_ints);

} // namespace forte

#endif // _general_operator_h_
