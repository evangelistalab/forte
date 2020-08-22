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

#include <algorithm>
#include <numeric>

#include "integrals/active_space_integrals.h"
#include "helpers/combinatorial.h"

#include "general_operator.h"
#include "determinant_hashvector.h"

namespace forte {

std::tuple<bool, bool, int> flip_spin(const std::tuple<bool, bool, int>& t) {
    return std::make_tuple(std::get<0>(t), not std::get<1>(t), std::get<2>(t));
}

// Enforce the order  a+ b+ b- a-
bool compare_ops(const std::tuple<bool, bool, int>& lhs, const std::tuple<bool, bool, int>& rhs) {
    const auto& l_cre = std::get<0>(lhs);
    const auto& r_cre = std::get<0>(rhs);
    if ((l_cre == true) and (r_cre == true)) {
        return flip_spin(lhs) > flip_spin(rhs);
    }
    return flip_spin(lhs) < flip_spin(rhs);
}

void GeneralOperator::add_operator(const std::vector<op_t>& op_list, double value) {
    amplitudes_.push_back(value);
    size_t start = op_list_.size();
    size_t end = start + op_list.size();
    op_indices_.push_back(std::make_pair(start, end));
    // transform each term in the input into a SingleOperator object
    for (const op_t& op : op_list) {
        // Form a single operator object
        SingleOperator sop;

        std::vector<std::tuple<bool, bool, int>> creation_alpha_orb_vec = op.second;

        // We first sort the operators so that they are ordered in the following way
        // (alpha cre. ascending) (beta cre. ascending) (beta ann. descending) (alpha ann.
        // descending) and keep track of the sign. We sort the operators using a set of auxiliary
        // indices so that we can keep track of the permutation of the operators and their sign
        std::vector<size_t> idx(creation_alpha_orb_vec.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&creation_alpha_orb_vec](size_t i1, size_t i2) {
            return compare_ops(creation_alpha_orb_vec[i1], creation_alpha_orb_vec[i2]);
        });
        auto parity = permutation_parity(idx);

        // set the factor including the parity of the permutation
        sop.factor = op.first * (1.0 - 2.0 * parity);

        // set the bitarray part of the operator (the order does not matter)
        for (auto creation_alpha_orb : creation_alpha_orb_vec) {
            bool creation = std::get<0>(creation_alpha_orb);
            bool alpha = std::get<1>(creation_alpha_orb);
            int orb = std::get<2>(creation_alpha_orb);
            if (creation) {
                if (alpha) {
                    sop.cre.set_alfa_bit(orb, true);
                } else {
                    sop.cre.set_beta_bit(orb, true);
                }
            } else {
                if (alpha) {
                    sop.ann.set_alfa_bit(orb, true);
                } else {
                    sop.ann.set_beta_bit(orb, true);
                }
            }
        }
        op_list_.push_back(sop);
    }
}

std::vector<std::string> GeneralOperator::str() {
    std::vector<std::string> result;
    size_t nops = amplitudes_.size();
    for (size_t n = 0; n < nops; n++) {
        std::string s = std::to_string(amplitudes_[n]) + " * ( ";
        size_t begin = op_indices_[n].first;
        size_t end = op_indices_[n].second;
        for (size_t j = begin; j < end; j++) {
            const double factor = op_list_[j].factor;
            const auto& ann = op_list_[j].ann;
            const auto& cre = op_list_[j].cre;
            s += (j == begin ? "" : " + ") + std::to_string(factor) + " * [ ";
            auto acre = cre.get_alfa_occ(cre.norb());
            auto bcre = cre.get_beta_occ(cre.norb());
            auto aann = ann.get_alfa_occ(ann.norb());
            auto bann = ann.get_beta_occ(ann.norb());
            std::reverse(aann.begin(), aann.end());
            std::reverse(bann.begin(), bann.end());
            for (auto p : acre) {
                s += std::to_string(p) + "a+ ";
            }
            for (auto p : bcre) {
                s += std::to_string(p) + "b+ ";
            }
            for (auto p : bann) {
                s += std::to_string(p) + "b- ";
            }
            for (auto p : aann) {
                s += std::to_string(p) + "a- ";
            }
            s += "]";
        }
        s += " )";
        result.push_back(s);
    }
    return result;
}

det_hash<double> apply_operator(GeneralOperator& gop, const det_hash<double>& state) {
    det_hash<double> new_state;
    const auto& amplitudes = gop.amplitudes();
    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();
    size_t nops = amplitudes.size();
    Determinant d;
    for (const auto& det_c : state) {
        const double c = det_c.second;
        for (size_t n = 0; n < nops; n++) {
            size_t begin = op_indices[n].first;
            size_t end = op_indices[n].second;
            for (size_t j = begin; j < end; j++) {
                d = det_c.first;
                double sign = apply_op(d, op_list[j].cre, op_list[j].ann);
                if (sign != 0.0) {
                    new_state[d] += amplitudes[n] * op_list[j].factor * sign * c;
                }
            }
        }
    }
    return new_state;
}

det_hash<double> apply_lin_op(det_hash<double> state, size_t n, const GeneralOperator& gop) {
    det_hash<double> new_state;

    const auto& amplitudes = gop.amplitudes();
    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();
    const size_t begin = op_indices[n].first;
    const size_t end = op_indices[n].second;
    Determinant d;
    for (size_t j = begin; j < end; j++) {
        for (const auto& det_c : state) {
            const double c = det_c.second;
            d = det_c.first;
            const double sign = apply_op(d, op_list[j].cre, op_list[j].ann);
            if (sign != 0.0) {
                new_state[d] += amplitudes[n] * op_list[j].factor * sign * c;
            }
        }
    }
    return new_state;
}

det_hash<double> apply_exp_op(const Determinant& d, size_t n, const GeneralOperator& gop) {
    det_hash<double> state;
    state[d] = 1.0;
    det_hash<double> exp_state = state;
    double factor = 1.0;
    int maxk = 16;
    for (int k = 1; k <= maxk; k++) {
        factor = factor / static_cast<double>(k);
        det_hash<double> new_state = apply_lin_op(state, n, gop);
        if (new_state.size() == 0)
            break;
        for (const auto& det_c : new_state) {
            exp_state[det_c.first] += factor * det_c.second;
        }
        state = new_state;
    }
    return exp_state;
}

det_hash<double> apply_exp_ah_factorized(GeneralOperator& gop, const det_hash<double>& state0) {
    det_hash<double> state(state0);
    det_hash<double> new_state;
    size_t nops = gop.nops();
    Determinant d;
    for (size_t n = 0; n < nops; n++) {
        new_state.clear();
        for (const auto& det_c : state) {
            const double c = det_c.second;
            d = det_c.first;
            det_hash<double> terms = apply_exp_op(d, n, gop);
            for (const auto& d_c : terms) {
                new_state[d_c.first] += d_c.second * c;
            }
        }
        state = new_state;
    }
    return new_state;
}

#define DEBUG_EXP_ALGORITHM 0

void apply_exp_op_fast(const Determinant& d, Determinant& new_d, const Determinant& cre,
                       const Determinant& ann, double amp, double c, det_hash<double>& new_terms) {
#if DEBUG_EXP_ALGORITHM
    std::cout << "\Applying: " << amp << "\n"
              << str(cre, 16) << "+\n"
              << str(ann, 16) << "-\n"
              << str(d, 16) << std::endl;
#endif
    new_d = d;
    double f = apply_op(new_d, cre, ann) * amp;
    // this is to deal with number operators (should be removed)
    if (d != new_d) {
        new_terms[d] += c * (std::cos(f) - 1.0);
        new_terms[new_d] += c * std::sin(f);
    }
}

det_hash<double> apply_exp_ah_factorized_fast(GeneralOperator& gop,
                                              const det_hash<double>& state0) {
    const auto& amplitudes = gop.amplitudes();
    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();

    // initialize a state object
    det_hash<double> state(state0);
    det_hash<double> new_terms;
    for (size_t n = 0, nops = gop.nops(); n < nops; n++) {
        // zero the new terms
        new_terms.clear();

        const size_t begin = op_indices[n].first;
        const SingleOperator& op = op_list[begin];
        const Determinant ucre = op.cre - op.ann;
        const Determinant uann = op.ann - op.cre;
        const double amp = amplitudes[n];
        Determinant new_d;
        // loop over all determinants
        for (const auto& det_c : state) {
            const Determinant& d = det_c.first;

            // test if we can apply this operator to this determinant
#if DEBUG_EXP_ALGORITHM
            std::cout << "\nOperation\n"
                      << str(op.cre, 16) << "+\n"
                      << str(op.ann, 16) << "-\n"
                      << str(d, 16) << std::endl;
#endif

#if DEBUG_EXP_ALGORITHM
            std::cout << "Testing (cre)(ann) sequence" << std::endl;
            std::cout << "Can annihilate: " << (d.fast_a_and_b_equal_b(op.ann) ? "True" : "False")
                      << std::endl;
            std::cout << "Can create:     " << (ucre.fast_a_and_b_eq_zero(d) ? "True" : "False")
                      << std::endl;
#endif
            if (d.fast_a_and_b_equal_b(op.ann) and d.fast_a_and_b_eq_zero(ucre)) {
#if DEBUG_EXP_ALGORITHM
                std::cout << "Applying the (cre)(ann) sequence!" << std::endl;
#endif
                const double c = det_c.second;
                apply_exp_op_fast(d, new_d, op.cre, op.ann, amp, c, new_terms);
            } else if (d.fast_a_and_b_equal_b(op.cre) and d.fast_a_and_b_eq_zero(uann)) {
#if DEBUG_EXP_ALGORITHM
                std::cout << "Applying the (ann)(cre) sequence!" << std::endl;
#endif
                const double c = det_c.second;
                apply_exp_op_fast(d, new_d, op.ann, op.cre, -amp, c, new_terms);
            }
        }
        for (const auto& d_c : new_terms) {
            state[d_c.first] += d_c.second;
        }
    }

    //            const double c = det_c.second;
    //            det_hash<double> terms = apply_exp_op(d, n, gop);
    //            for (const auto& d_c : terms) {
    //                new_state[d_c.first] += d_c.second * c;
    //            }
    //    det_hash<double> new_state;
    //    Determinant d;
    //        new_state.clear();
    //        for (const auto& det_c : state) {
    //            const double c = det_c.second;
    //            d = det_c.first;
    //            det_hash<double> terms = apply_exp_op(d, n, gop);
    //            for (const auto& d_c : terms) {
    //                new_state[d_c.first] += d_c.second * c;
    //            }
    //        }
    //        state = new_state;
    //    }
    return state;
}

// det_hash<double> apply_exp_ah_factorized_fast(GeneralOperator& gop,
//                                              const det_hash<double>& state0) {
//    det_hash<double> state(state0);
//    det_hash<double> new_state;
//    size_t nops = gop.nops();
//    Determinant d;
//    for (size_t n = 0; n < nops; n++) {
//        new_state.clear();
//        for (const auto& det_c : state) {
//            const double c = det_c.second;
//            d = det_c.first;
//            det_hash<double> terms = apply_exp_op(d, n, gop);
//            for (const auto& d_c : terms) {
//                new_state[d_c.first] += d_c.second * c;
//            }
//        }
//        state = new_state;
//    }
//    return new_state;
//}

// det_hash<double> apply_general_operator_exp_factorized(GeneralOperator& gop,
//                                                       const det_hash<double>& state) {
//    det_hash<double> new_state;
//    const auto& amplitudes = gop.amplitudes();
//    const auto& op_indices = gop.op_indices();
//    const auto& op_list = gop.op_list();
//    size_t nops = amplitudes.size();
//    Determinant d;
//    for (const auto& det_c : state) {
//        double c = det_c.second;
//        for (size_t n = 0; n < nops; n++) {
//            size_t begin = op_indices[n].first;
//            size_t end = op_indices[n].second;
//            for (size_t j = begin; j < end; j++) {
//                d = det_c.first;
//                double factor = apply_operator(d, op_list[j]);
//                if (factor != 0.0) {
//                    new_state[d] += amplitudes[n] * op_list[j].factor * factor * c;
//                }
//            }
//        }
//    }
//    return new_state;
//}

// det_hash<double> apply_exp_general_operator(GeneralOperator& gop, det_hash<double> state,
//                                            int maxn) {
//    det_hash<double> exp_state = state;
//    double factor = 1.0;
//    for (int n = 1; n <= maxn; n++) {
//        factor = factor / static_cast<double>(n);
//        det_hash<double> new_state = apply_general_operator(gop, state);
//        for (const auto& det_c : new_state) {
//            exp_state[det_c.first] += factor * det_c.second;
//        }
//        state = new_state;
//    }
//    return exp_state;
//}

// std::vector<std::tuple<size_t, size_t, double>> compute_matrix_repr(GeneralOperator& gop,
//                                                                    DeterminantHashVec&
//                                                                    det_idx) {
//    std::vector<std::tuple<size_t, size_t, double>> m;
//    size_t dim = det_idx.size();

//    const auto& amplitudes = gop.amplitudes();
//    const auto& op_indices = gop.op_indices();
//    const auto& op_list = gop.op_list();
//    size_t nops = amplitudes.size();
//    Determinant d;
//    for (size_t I = 0; I < dim; I++) {
//        for (size_t n = 0; n < nops; n++) {
//            size_t begin = op_indices[n].first;
//            size_t end = op_indices[n].second;
//            for (size_t j = begin; j < end; j++) {
//                d = det_idx.get_det(I);
//                // in the spin case we ignore the phase factor
//                double factor = apply_operator(d, op_list[j].second);
//                if (factor != 0.0) {
//                    size_t J = det_idx.get_idx(d);
//                    m.push_back(std::make_tuple(I, J, factor * amplitudes[n] *
//                    op_list[j].first));
//                }
//            }
//        }
//    }
//    return m;
//}

// void apply_matrix_repr(const std::vector<double>& in, std::vector<double>& out,
//                       std::vector<std::tuple<size_t, size_t, double>> m) {
//    std::fill(out.begin(), out.end(), 0.0);
//    size_t I, J;
//    double f;
//    for (const auto& IJf : m) {
//        std::tie(I, J, f) = IJf;
//        out[J] += f * in[I];
//    }
//}

// det_hash<double> apply_exp_general_operator_matrix(GeneralOperator& gop, det_hash<double>
// state,
//                                                   int norbs, int maxn) {
//    size_t dim = std::pow(2, 2 * norbs);
//    std::vector<double> C(dim);

//    Determinant d;
//    DeterminantHashVec det_idx;
//    for (size_t I = 0; I < dim; I++) {
//        for (size_t i = 0; i < norbs; i++) {
//            bool bi = I & (1 << i);
//            d.set_alfa_bit(i, bi);
//        }
//        for (size_t i = 0; i < norbs; i++) {
//            bool bi = I & (1 << (norbs + i));
//            d.set_beta_bit(i, bi);
//        }
//        det_idx.add(d);
//    }
//    for (const auto& det_c : state) {
//        size_t idx = det_idx.get_idx(det_c.first);
//        C[idx] = det_c.second;
//    }
//    //    size_t dim = state.size();
//    //    size_t k = 0;
//    std::vector<std::tuple<size_t, size_t, double>> matrix_repr = compute_matrix_repr(gop,
//    det_idx);

//    std::vector<double> exp_C = C;
//    std::vector<double> new_C = C;
//    double factor = 1.0;
//    for (int n = 1; n <= maxn; n++) {
//        factor = factor / static_cast<double>(n);
//        apply_matrix_repr(C, new_C, matrix_repr);
//        for (size_t I = 0; I < dim; I++) {
//            exp_C[I] += factor * new_C[I];
//        }
//        C = new_C;
//    }
//    det_hash<double> exp_state;
//    for (size_t I = 0; I < dim; I++) {
//        exp_state[det_idx.get_det(I)] = exp_C[I];
//    }
//    return exp_state;
//}

// double energy_expectation_value(det_hash<double>& left_state, det_hash<double>& right_state,
//                                std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
//    /// Return nuclear repulsion energy
//    double E_0 = as_ints->nuclear_repulsion_energy() + as_ints->frozen_core_energy() +
//                 as_ints->scalar_energy();
//    double E = 0.0;
//    Determinant t_d;
//    for (const auto& det_c_l : left_state) {
//        const auto& d_l = det_c_l.first;
//        for (const auto& det_c_r : right_state) {
//            const auto& d_r = det_c_r.first;
//            t_d = d_l ^ d_r;
//            int ndiff = t_d.count();
//            if (ndiff == 0) {
//                E += (E_0 + as_ints->slater_rules(d_l, d_r)) * det_c_l.second *
//                det_c_r.second;
//            } else if (ndiff <= 4) {
//                E += as_ints->slater_rules(d_l, d_r) * det_c_l.second * det_c_r.second;
//            }
//        }
//    }
//    return E;
//}

// det_hash<double> apply_number_projector(int na, int nb, det_hash<double>& state) {
//    det_hash<double> new_state;
//    for (const auto& det_c : state) {
//        if ((det_c.first.count_alfa() == na) and (det_c.first.count_beta() == nb) and
//            (std::fabs(det_c.second) > 1.0e-12)) {
//            new_state[det_c.first] = det_c.second;
//        }
//    }
//    return new_state;
//}

// double overlap(det_hash<double>& left_state, det_hash<double>& right_state) {
//    double overlap = 0.0;
//    for (const auto& det_c_r : right_state) {
//        auto it = left_state.find(det_c_r.first);
//        if (it != left_state.end()) {
//            overlap += it->second * det_c_r.second;
//        }
//    }
//    return overlap;
//}

} // namespace forte
