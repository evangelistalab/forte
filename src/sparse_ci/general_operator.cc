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

#include "integrals/active_space_integrals.h"

#include "general_operator.h"
#include "determinant_hashvector.h"

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

det_hash<double> apply_general_operator_spin(GeneralOperator& gop, det_hash<double>& state) {
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
                // in the spin case we ignore the phase factor
                double factor = apply_operator(d, op_list[j].second);
                if (factor != 0.0) {
                    new_state[d] += amplitudes[n] * op_list[j].first * c;
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

det_hash<double> apply_exp_general_operator_spin(GeneralOperator& gop, det_hash<double> state,
                                                 int maxn) {
    det_hash<double> exp_state = state;
    double factor = 1.0;
    for (int n = 1; n <= maxn; n++) {
        factor = factor / static_cast<double>(n);
        det_hash<double> new_state = apply_general_operator_spin(gop, state);
        for (const auto& det_c : new_state) {
            exp_state[det_c.first] += factor * det_c.second;
        }
        state = new_state;
    }
    return exp_state;
}

std::vector<std::tuple<size_t, size_t, double>> compute_matrix_repr(GeneralOperator& gop,
                                                                    DeterminantHashVec& det_idx) {
    std::vector<std::tuple<size_t, size_t, double>> m;
    size_t dim = det_idx.size();

    const auto& amplitudes = gop.amplitudes();
    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();
    size_t nops = amplitudes.size();
    Determinant d;
    for (size_t I = 0; I < dim; I++) {
        for (size_t n = 0; n < nops; n++) {
            size_t begin = op_indices[n].first;
            size_t end = op_indices[n].second;
            for (size_t j = begin; j < end; j++) {
                d = det_idx.get_det(I);
                // in the spin case we ignore the phase factor
                double factor = apply_operator(d, op_list[j].second);
                if (factor != 0.0) {
                    size_t J = det_idx.get_idx(d);
                    m.push_back(std::make_tuple(I, J, factor * amplitudes[n] * op_list[j].first));
                }
            }
        }
    }
    return m;
}

void apply_matrix_repr(const std::vector<double>& in, std::vector<double>& out,
                       std::vector<std::tuple<size_t, size_t, double>> m) {
    std::fill(out.begin(), out.end(), 0.0);
    size_t I, J;
    double f;
    for (const auto& IJf : m) {
        std::tie(I, J, f) = IJf;
        out[J] += f * in[I];
    }
}

det_hash<double> apply_exp_general_operator_matrix(GeneralOperator& gop, det_hash<double> state,
                                                   int norbs, int maxn) {
    size_t dim = std::pow(2, 2 * norbs);
    std::vector<double> C(dim);

    Determinant d;
    DeterminantHashVec det_idx;
    for (size_t I = 0; I < dim; I++) {
        for (size_t i = 0; i < norbs; i++) {
            bool bi = I & (1 << i);
            d.set_alfa_bit(i, bi);
        }
        for (size_t i = 0; i < norbs; i++) {
            bool bi = I & (1 << (norbs + i));
            d.set_beta_bit(i, bi);
        }
        det_idx.add(d);
    }
    for (const auto& det_c : state) {
        size_t idx = det_idx.get_idx(det_c.first);
        C[idx] = det_c.second;
    }
    //    size_t dim = state.size();
    //    size_t k = 0;
    std::vector<std::tuple<size_t, size_t, double>> matrix_repr = compute_matrix_repr(gop, det_idx);

    std::vector<double> exp_C = C;
    std::vector<double> new_C = C;
    double factor = 1.0;
    for (int n = 1; n <= maxn; n++) {
        factor = factor / static_cast<double>(n);
        apply_matrix_repr(C, new_C, matrix_repr);
        for (size_t I = 0; I < dim; I++) {
            exp_C[I] += factor * new_C[I];
        }
        C = new_C;
    }
    det_hash<double> exp_state;
    for (size_t I = 0; I < dim; I++) {
        exp_state[det_idx.get_det(I)] = exp_C[I];
    }
    return exp_state;
}

double energy_expectation_value(det_hash<double>& left_state, det_hash<double>& right_state,
                                std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    /// Return nuclear repulsion energy
    double E_0 = as_ints->nuclear_repulsion_energy() + as_ints->frozen_core_energy() +
                 as_ints->scalar_energy();
    double E = 0.0;
    Determinant t_d;
    for (const auto& det_c_l : left_state) {
        const auto& d_l = det_c_l.first;
        for (const auto& det_c_r : right_state) {
            const auto& d_r = det_c_r.first;
            t_d = d_l ^ d_r;
            int ndiff = t_d.count();
            if (ndiff == 0) {
                E += (E_0 + as_ints->slater_rules(d_l, d_r)) * det_c_l.second * det_c_r.second;
            } else if (ndiff <= 4) {
                E += as_ints->slater_rules(d_l, d_r) * det_c_l.second * det_c_r.second;
            }
        }
    }
    return E;
}

det_hash<double> apply_number_projector(int na, int nb, det_hash<double>& state) {
    det_hash<double> new_state;
    for (const auto& det_c : state) {
        if ((det_c.first.count_alfa() == na) and (det_c.first.count_beta() == nb) and
            (std::fabs(det_c.second) > 1.0e-12)) {
            new_state[det_c.first] = det_c.second;
        }
    }
    return new_state;
}

double overlap(det_hash<double>& left_state, det_hash<double>& right_state) {
    double overlap = 0.0;
    for (const auto& det_c_r : right_state) {
        auto it = left_state.find(det_c_r.first);
        if (it != left_state.end()) {
            overlap += it->second * det_c_r.second;
        }
    }
    return overlap;
}

} // namespace forte
