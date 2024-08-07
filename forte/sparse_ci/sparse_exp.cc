/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <algorithm>
#include <numeric>
#include <cmath>

#include "sparse_ci/sparse_exp.h"

namespace forte {

size_t num_attempts_ = 0;
size_t num_success_ = 0;

SparseExp::SparseExp(int maxk, double screen_thresh) : maxk_(maxk), screen_thresh_(screen_thresh) {}

SparseState SparseExp::apply_op(const SparseOperator& sop, const SparseState& state,
                                double scaling_factor) {
    return apply_exp_operator(OperatorType::Excitation, sop, state, scaling_factor);
}

SparseState SparseExp::apply_op(const SparseOperatorList& sop_list, const SparseState& state,
                                double scaling_factor) {
    SparseOperator sop = sop_list.to_operator();
    return apply_op(sop, state, scaling_factor);
}

SparseState SparseExp::apply_antiherm(const SparseOperator& sop, const SparseState& state,
                                      double scaling_factor) {
    return apply_exp_operator(OperatorType::Antihermitian, sop, state, scaling_factor);
}

SparseState SparseExp::apply_antiherm(const SparseOperatorList& sop_list, const SparseState& state,
                                      double scaling_factor) {
    SparseOperator sop = sop_list.to_operator();
    return apply_antiherm(sop, state, scaling_factor);
}

SparseState SparseExp::apply_exp_operator(OperatorType op_type, const SparseOperator& sop,
                                          const SparseState& state, double scaling_factor) {
    SparseState exp_state(state);
    SparseState old_terms(state);
    SparseState new_terms;

    for (int k = 1; k <= maxk_; k++) {
        old_terms *= scaling_factor / static_cast<double>(k);
        if (op_type == OperatorType::Excitation) {
            new_terms = apply_operator_lin(sop, old_terms, screen_thresh_);
        } else if (op_type == OperatorType::Antihermitian) {
            new_terms = apply_operator_antiherm(sop, old_terms, screen_thresh_);
        }
        double norm = 0.0;
        double inf_norm = 0.0;
        exp_state += new_terms;
        for (const auto& [det, c] : new_terms) {
            norm += std::pow(std::abs(c), 2.0);
            inf_norm = std::max(inf_norm, std::abs(c));
        }
        norm = std::sqrt(norm);
        if (inf_norm < screen_thresh_) {
            break;
        }
        old_terms = new_terms;
    }
    return exp_state;
}

} // namespace forte
