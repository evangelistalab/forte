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

#include "general_operator.h"

namespace forte {

void GeneralOperator::add_operator(std::vector<op_t> op_list) {
    amplitudes_.push_back(0.0);
    size_t start = op_list_.size();
    size_t end = start + op_list.size();
    op_indices_.push_back(std::make_pair(start, end));
    op_list_.insert(op_list_.end(), op_list.begin(), op_list.end());
}

double apply_operator(Determinant& d, const std::vector<std::tuple<bool, bool, int>>& sqops) {
    double factor = 1.0;
    for (const auto& sqop : sqops) {
        bool creator = std::get<0>(sqop);
        bool spin = std::get<1>(sqop);
        int mo = std::get<2>(sqop);
        if (creator) {
            if (spin) {
                factor *= d.create_alfa_bit(mo);
            } else {
                factor *= d.create_beta_bit(mo);
            }
        } else {
            if (spin) {
                factor *= d.destroy_alfa_bit(mo);
            } else {
                factor *= d.destroy_beta_bit(mo);
            }
        }
    }
    return factor;
}

det_hash<double> apply_general_operator(GeneralOperator& gop, det_hash<double>& state) {
    det_hash<double> new_state;
    const auto& amplitudes = gop.amplitudes();
    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();
    size_t nops = amplitudes.size();
    Determinant d;
    for (const auto& det_c : state) {
        double c = det_c.second;
        for (size_t n = 0; n < nops; n++) {
            size_t begin = op_indices[n].first;
            size_t end = op_indices[n].second;
            for (size_t j = begin; j < end; j++) {
                d = det_c.first;
                double factor = apply_operator(d, op_list[j].second);
                if (factor != 0.0) {
                    new_state[d] += amplitudes[n] * op_list[j].first * factor * c;
                }
            }
        }
    }
    return new_state;
}

det_hash<double> apply_exp_general_operator(GeneralOperator& gop, det_hash<double> state,
                                            int maxn) {
    det_hash<double> exp_state = state;
    double factor = 1.0;
    for (int n = 1; n <= maxn; n++) {
        factor = factor / static_cast<double>(n);
        det_hash<double> new_state = apply_general_operator(gop, state);
        for (const auto& det_c : new_state) {
            exp_state[det_c.first] += factor * det_c.second;
        }
        state = new_state;
    }
    return exp_state;
}
} // namespace forte
