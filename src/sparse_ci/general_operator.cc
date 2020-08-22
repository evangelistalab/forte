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
#include "helpers/timer.h"

#include "general_operator.h"
#include "determinant_hashvector.h"

double time_apply_exp_ah_factorized_fast = 0.0;
double time_energy_expectation_value = 0.0;
double time_apply_operator_fast = 0.0;
double time_apply_exp_operator_fast = 0.0;

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

SingleOperator op_t_to_SingleOperator(const op_t& op) {
    // Form a single operator object
    SingleOperator sop;

    const std::vector<std::tuple<bool, bool, int>>& creation_alpha_orb_vec = op.second;

    // set the factor including the parity of the permutation
    sop.factor = op.first;

    bool is_sorted =
        std::is_sorted(creation_alpha_orb_vec.begin(), creation_alpha_orb_vec.end(), compare_ops);

    if (not is_sorted) {
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
        sop.factor *= 1.0 - 2.0 * parity;
    }

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
    return sop;
}

void GeneralOperator::add_operator(const std::vector<op_t>& op_list, double value) {
    amplitudes_.push_back(value);
    size_t start = op_list_.size();
    size_t end = start + op_list.size();
    op_indices_.push_back(std::make_pair(start, end));
    // transform each term in the input into a SingleOperator object
    for (const op_t& op : op_list) {
        // Form a single operator object
        op_list_.push_back(op_t_to_SingleOperator(op));
    }
}

void GeneralOperator::add_operator2(const std::vector<SingleOperator>& ops, double value) {
    amplitudes_.push_back(value);
    size_t start = op_list_.size();
    size_t end = start + ops.size();
    op_indices_.push_back(std::make_pair(start, end));
    // transform each term in the input into a SingleOperator object
    for (const SingleOperator& op : ops) {
        op_list_.push_back(op);
    }
}

void GeneralOperator::pop_operator() {
    if (nops() > 0) {
        amplitudes_.pop_back();
        auto start_end = op_indices_.back();
        op_indices_.pop_back();
        size_t start = start_end.first;
        size_t end = start_end.second;
        for (; start < end; ++start) {
            op_list_.pop_back();
        }
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

std::vector<std::pair<std::string, double>> GeneralOperator::timing() {
    std::vector<std::pair<std::string, double>> t;
    t.push_back(
        std::make_pair("time_apply_exp_ah_factorized_fast", time_apply_exp_ah_factorized_fast));
    t.push_back(std::make_pair("time_energy_expectation_value", time_energy_expectation_value));
    t.push_back(std::make_pair("time_apply_operator_fast", time_apply_operator_fast));
    t.push_back(std::make_pair("time_apply_exp_operator_fast", time_apply_exp_operator_fast));
    return t;
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

det_hash<double> apply_operator_fast(GeneralOperator& gop, const det_hash<double>& state0) {
    local_timer t;
    const auto& amplitudes = gop.amplitudes();
    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();

    det_hash<double> new_terms;

    for (size_t n = 0, nops = gop.nops(); n < nops; n++) {
        const size_t begin = op_indices[n].first;
        const size_t end = op_indices[n].second;

        for (size_t j = begin; j < end; j++) {
            const SingleOperator& op = op_list[j];
            const Determinant ucre = op.cre - op.ann;
            const double tau = amplitudes[n] * op.factor;
            Determinant new_d;
            // loop over all determinants
            for (const auto& det_c : state0) {
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
                std::cout << "Can annihilate: "
                          << (d.fast_a_and_b_equal_b(op.ann) ? "True" : "False") << std::endl;
                std::cout << "Can create:     " << (ucre.fast_a_and_b_eq_zero(d) ? "True" : "False")
                          << std::endl;
#endif
                if (d.fast_a_and_b_equal_b(op.ann) and d.fast_a_and_b_eq_zero(ucre)) {
#if DEBUG_EXP_ALGORITHM
                    std::cout << "Applying the (cre)(ann) sequence!" << std::endl;
#endif
                    const double c = det_c.second;
                    new_d = d;
                    double value = apply_op(new_d, op.cre, op.ann) * tau * c;
                    if (std::fabs(value) > 1.0e-12) {
                        new_terms[new_d] += value;
                    }
                }
            }
        }
    }
    time_apply_operator_fast += t.get();
    return new_terms;
}

det_hash<double> apply_exp_operator_fast(GeneralOperator& gop, const det_hash<double>& state0,
                                         double scaling_factor) {
    local_timer t;
    det_hash<double> exp_state(state0);
    det_hash<double> state(state0);
    double factor = 1.0;
    int maxk = 20;
    for (int k = 1; k <= maxk; k++) {
        factor *= scaling_factor / static_cast<double>(k);
        det_hash<double> new_terms = apply_operator_fast(gop, state);
        double norm = 0.0;
        for (const auto& det_c : new_terms) {
            exp_state[det_c.first] += factor * det_c.second;
            norm += std::pow(factor * det_c.second, 2.0);
        }
        if (std::sqrt(norm) < 1.0e-12)
            break;
        state = new_terms;
    }
    time_apply_exp_operator_fast + t.get();
    return exp_state;
}

#define DEBUG_EXP_ALGORITHM 0
void apply_exp_op_fast(const Determinant& d, Determinant& new_d, const Determinant& cre,
                       const Determinant& ann, double amp, double c, det_hash<double>& new_terms) {
#if DEBUG_EXP_ALGORITHM
    std::cout << "\nApplying: " << amp << "\n"
              << str(cre, 16) << "+\n"
              << str(ann, 16) << "-\n"
              << str(d, 16) << std::endl;
#endif
    new_d = d;
    const double f = apply_op(new_d, cre, ann) * amp;
    // this is to deal with number operators (should be removed)
    if (d != new_d) {
        new_terms[d] += c * (std::cos(f) - 1.0);
        new_terms[new_d] += c * std::sin(f);
#if DEBUG_EXP_ALGORITHM
        std::cout << "\nf: " << f << std::endl;
#endif
    }
}

det_hash<double> apply_exp_ah_factorized_fast(GeneralOperator& gop,
                                              const det_hash<double>& state0) {
    local_timer t;
    const auto& amplitudes = gop.amplitudes();
    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();

    // initialize a state object
    det_hash<double> state(state0);
    det_hash<double> new_terms;

    //    // create a vector of determinants (for fast comparison)
    //    std::vector<Determinant> dets;
    //    for (const auto& det_c : state) {
    //        dets.push_back(det_c.first);
    //    }

    for (size_t n = 0, nops = gop.nops(); n < nops; n++) {
        // zero the new terms
        new_terms.clear();

        const size_t begin = op_indices[n].first;
        const SingleOperator& op = op_list[begin];
        const Determinant ucre = op.cre - op.ann;
        const Determinant uann = op.ann - op.cre;
        const double tau = amplitudes[n] * op.factor;
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
                apply_exp_op_fast(d, new_d, op.cre, op.ann, tau, c, new_terms);
            } else if (d.fast_a_and_b_equal_b(op.cre) and d.fast_a_and_b_eq_zero(uann)) {
#if DEBUG_EXP_ALGORITHM
                std::cout << "Applying the (ann)(cre) sequence!" << std::endl;
#endif
                const double c = det_c.second;
                apply_exp_op_fast(d, new_d, op.ann, op.cre, -tau, c, new_terms);
            }
        }
        for (const auto& d_c : new_terms) {
            if (std::fabs(d_c.second) > 1.0e-12) {
                state[d_c.first] += d_c.second;
            }
        }
    }
    time_apply_exp_ah_factorized_fast += t.get();
    return state;
}

double energy_expectation_value(det_hash<double>& left_state, det_hash<double>& right_state,
                                std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    local_timer t;
    /// Return nuclear repulsion energy
    double E_0 = as_ints->nuclear_repulsion_energy() + as_ints->frozen_core_energy() +
                 as_ints->scalar_energy();
    double E = 0.0;
    Determinant t_d;
    for (const auto& det_c_l : left_state) {
        const auto& d_l = det_c_l.first;
        for (const auto& det_c_r : right_state) {
            const auto& d_r = det_c_r.first;
            int ndiff = d_l.fast_a_xor_b_count(d_r);
            if (ndiff == 0) {
                E += (E_0 + as_ints->slater_rules(d_l, d_r)) * det_c_l.second * det_c_r.second;
            } else if (ndiff <= 4) {
                E += as_ints->slater_rules(d_l, d_r) * det_c_l.second * det_c_r.second;
            }
        }
    }

    time_energy_expectation_value += t.get();

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

// std::vector<SingleOperator> to_gen_op(const std::string& s)
//{
//    std::vector<SingleOperator> terms;
//    // '<something>[1b+ 0b+] +-<something>[1b+ 0b+]'
//    std::regex
//    word_regex("'\\s?([\\+\\-])?\\s*(\\d*\\.?\\d*)?\\s*\\*?\\s*(\\[[0-9ab\\+\\-\\s]*\\])'");

//    smatch res;
//    string str = "first second third forth";

//    while (regex_search(str, res, exp)) {
//        cout << res[0] << endl;
//        str = res.suffix();
//    }

//    m = re.findall(match_op,str)
//    if m:
//        for group in m:
//            sign = parse_sign(group[0])
//            factor = parse_factor(group[1])
//            ops = parse_ops(group[2])
//            terms.append((sign * factor,ops))
//    return terms

//    }

////    def parse_sign(s):
////    if s == '' or s == '+':
////        return 1.0
////    if s == '-':
////        return -1.0
////    print(f'There was an error parsing the sign {s}')

////def parse_factor(s):
////    if s == '':
////        return 1.0
////    return(float(s))

////def parse_ops(s):
////    ops = []
////    # we reverse the operator order
////    for op in s[1:-1].split(' ')[::-1]:
////        creation = True if op[-1] == '+' else False
////        alpha = True if op[-2] == 'a' else False
////        orb = int(op[0:-2])
////        ops.append((creation,alpha,orb))
////    return ops

//}

} // namespace forte
