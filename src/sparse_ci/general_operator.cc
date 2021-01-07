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
#include <regex>

#include "integrals/active_space_integrals.h"
#include "helpers/combinatorial.h"
#include "helpers/timer.h"
#include "helpers/string_algorithms.h"

#include "general_operator.h"
#include "determinant_hashvector.h"

double time_apply_exp_ah_factorized_fast = 0.0;
double time_energy_expectation_value = 0.0;
double time_apply_operator_fast = 0.0;
double time_apply_exp_operator_fast = 0.0;
double time_apply_operator_fast2 = 0.0;
double time_apply_exp_operator_fast2 = 0.0;
double time_apply_hamiltonian = 0.0;
double time_get_projection = 0.0;
size_t ops_apply_hamiltonian = 0;
size_t ops_hash_push = 0;
size_t ops_screen = 0;
size_t ops_det_visit = 0;

namespace forte {

SingleOperator::SingleOperator(double factor, const Determinant& cre, const Determinant& ann)
    : factor_(factor), cre_(cre), ann_(ann) {}

double SingleOperator::factor() const { return factor_; }
const Determinant& SingleOperator::cre() const { return cre_; }
const Determinant& SingleOperator::ann() const { return ann_; }

std::tuple<bool, bool, int> flip_spin(const std::tuple<bool, bool, int>& t) {
    return std::make_tuple(std::get<0>(t), not std::get<1>(t), std::get<2>(t));
}

// a comparison function used to sort second quantized operators in the order
//  alpha+ beta+ beta- alpha-
bool compare_ops(const std::tuple<bool, bool, int>& lhs, const std::tuple<bool, bool, int>& rhs) {
    const auto& l_cre = std::get<0>(lhs);
    const auto& r_cre = std::get<0>(rhs);
    if ((l_cre == true) and (r_cre == true)) {
        return flip_spin(lhs) > flip_spin(rhs);
    }
    return flip_spin(lhs) < flip_spin(rhs);
}

SingleOperator op_t_to_SingleOperator(const op_t& op) {
    const std::vector<std::tuple<bool, bool, int>>& creation_alpha_orb_vec = op.second;

    Determinant cre, ann;
    // set the factor including the parity of the permutation
    double factor = op.first;

    bool is_sorted =
        std::is_sorted(creation_alpha_orb_vec.begin(), creation_alpha_orb_vec.end(), compare_ops);

    // if not sorted, compute the permutation factor
    if (not is_sorted) {
        // We first sort the operators so that they are ordered in the following way
        // [last](alpha cre. ascending) (beta cre. ascending) (beta ann. descending) (alpha ann.
        // descending)[first] and keep track of the sign. We sort the operators using a set of
        // auxiliary indices so that we can keep track of the permutation of the operators and their
        // sign
        std::vector<size_t> idx(creation_alpha_orb_vec.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&creation_alpha_orb_vec](size_t i1, size_t i2) {
            return compare_ops(creation_alpha_orb_vec[i1], creation_alpha_orb_vec[i2]);
        });
        auto parity = permutation_parity(idx);
        // set the factor including the parity of the permutation
        factor *= 1.0 - 2.0 * parity;
    }

    // set the bitarray part of the operator (the order does not matter)
    for (auto creation_alpha_orb : creation_alpha_orb_vec) {
        bool creation = std::get<0>(creation_alpha_orb);
        bool alpha = std::get<1>(creation_alpha_orb);
        int orb = std::get<2>(creation_alpha_orb);
        if (creation) {
            if (alpha) {
                cre.set_alfa_bit(orb, true);
            } else {
                cre.set_beta_bit(orb, true);
            }
        } else {
            if (alpha) {
                ann.set_alfa_bit(orb, true);
            } else {
                ann.set_beta_bit(orb, true);
            }
        }
    }
    return SingleOperator(factor, cre, ann);
}

double parse_sign(const std::string& s) {
    if (s == "-") {
        return -1.0;
    }
    return 1.0;
}

double parse_factor(const std::string& s) {
    if (s == "") {
        return 1.0;
    }
    return stod(s);
}

std::vector<std::tuple<bool, bool, int>> parse_ops(const std::string& s) {
    // reverse the operator order
    auto clean_s = s.substr(1, s.size() - 2);

    auto ops_str = split_string(clean_s, " ");
    std::reverse(ops_str.begin(), ops_str.end());

    std::vector<std::tuple<bool, bool, int>> ops_tuple;
    for (auto op_str : ops_str) {
        size_t len = op_str.size();
        bool creation = op_str[len - 1] == '+' ? true : false;
        bool alpha = op_str[len - 2] == 'a' ? true : false;
        int orb = stoi(op_str.substr(0, len - 2));
        ops_tuple.push_back(std::make_tuple(creation, alpha, orb));
    }
    return ops_tuple;
}
// void GeneralOperator::add_operator(const std::vector<op_t>& op_list, double value) {
//    coefficients_.push_back(value);
//    size_t start = op_list_.size();
//    size_t end = start + op_list.size();
//    op_indices_.push_back(std::make_pair(start, end));
//    // transform each term in the input into a SingleOperator object
//    for (const op_t& op : op_list) {
//        // Form a single operator object
//        op_list_.push_back(op_t_to_SingleOperator(op));
//    }
//}

void GeneralOperator::add_term(const std::vector<SingleOperator>& ops, double value) {
    coefficients_.push_back(value);
    size_t start = op_list_.size();
    size_t end = start + ops.size();
    op_indices_.push_back(std::make_pair(start, end));
    // transform each term in the input into a SingleOperator object
    for (const SingleOperator& op : ops) {
        op_list_.push_back(op);
    }
}

void GeneralOperator::add_term_from_str(std::string str, double value) {
    std::vector<SingleOperator> ops;

    // the regex to parse the entries
    std::regex re("\\s?([\\+\\-])?\\s*(\\d*\\.?\\d*)?\\s*\\*?\\s*(\\[[0-9ab\\+\\-\\s]*\\])");
    // the match object
    std::smatch m;

    // here we match all the terms that look like +/- factor [<orb><a/b><+/-> ...]
    // in the middle of this code we parse the operator part and store it as a
    // std::vector<std::tuple<bool, bool, int>>  (in parsed_ops)
    // then we call op_t_to_SingleOperator to get a SingleOperator object
    while (std::regex_search(str, m, re)) {
        if (m.ready()) {
            double sign = parse_sign(m[1]);
            double factor = parse_factor(m[2]);
            auto op = parse_ops(m[3]);
            op_t parsed_ops = std::make_pair(sign * factor, op);
            ops.push_back(op_t_to_SingleOperator(parsed_ops));
        }
        str = m.suffix().str();
    }
    add_term(ops, value);
}

std::pair<std::vector<SingleOperator>, double> GeneralOperator::get_term(size_t n) {
    size_t begin = op_indices_[n].first;
    size_t end = op_indices_[n].second;
    std::vector<SingleOperator> ops(op_list_.begin() + begin, op_list_.begin() + end);
    return std::make_pair(ops, coefficients_[n]);
}

void GeneralOperator::pop_term() {
    if (nterms() > 0) {
        coefficients_.pop_back();
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
    size_t nterms = coefficients_.size();
    for (size_t n = 0; n < nterms; n++) {
        std::string s = std::to_string(coefficients_[n]) + " * ( ";
        size_t begin = op_indices_[n].first;
        size_t end = op_indices_[n].second;
        for (size_t j = begin; j < end; j++) {
            const double factor = op_list_[j].factor();
            const auto& ann = op_list_[j].ann();
            const auto& cre = op_list_[j].cre();
            if (j != begin) {
                s += (factor < 0.0) ? " " : " +";
            }
            s += std::to_string(factor) + " * [ ";
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
    t.push_back(std::make_pair("time_apply_hamiltonian", time_apply_hamiltonian));
    t.push_back(
        std::make_pair("ops_apply_hamiltonian", static_cast<double>(ops_apply_hamiltonian)));
    t.push_back({"ops_hash_push", static_cast<double>(ops_hash_push)});
    t.push_back({"ops_screen", static_cast<double>(ops_screen)});
    t.push_back({"ops_det_visit", static_cast<double>(ops_det_visit)});
    t.push_back({"time_get_projection", static_cast<double>(time_get_projection)});
    t.push_back({"time_apply_operator_fast2", time_apply_operator_fast2});
    t.push_back({"time_apply_exp_operator_fast2", time_apply_exp_operator_fast2});

    return t;
}

void GeneralOperator::reset_timing() {
    time_apply_exp_ah_factorized_fast = 0.0;
    time_energy_expectation_value = 0.0;
    time_apply_operator_fast = 0.0;
    time_apply_exp_operator_fast = 0.0;
    time_apply_operator_fast2 = 0.0;
    time_apply_exp_operator_fast2 = 0.0;
    time_apply_hamiltonian = 0.0;
    time_get_projection = 0.0;
    ops_apply_hamiltonian = 0;
    ops_hash_push = 0;
    ops_screen = 0;
    ops_det_visit = 0;
}

StateVector::StateVector() { std::cout << "Created a StateVector object" << std::endl; }

StateVector::StateVector(const det_hash<double>& state_vec) : state_vec_(state_vec) {}

StateVector apply_operator(GeneralOperator& gop, const StateVector& state) {
    StateVector new_state;
    const auto& amplitudes = gop.coefficients();
    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();
    size_t nterms = amplitudes.size();
    Determinant d;
    for (const auto& det_c : state) {
        const double c = det_c.second;
        for (size_t n = 0; n < nterms; n++) {
            size_t begin = op_indices[n].first;
            size_t end = op_indices[n].second;
            for (size_t j = begin; j < end; j++) {
                d = det_c.first;
                double sign = apply_op(d, op_list[j].cre(), op_list[j].ann());
                if (sign != 0.0) {
                    new_state[d] += amplitudes[n] * op_list[j].factor() * sign * c;
                }
            }
        }
    }
    return new_state;
}

StateVector apply_lin_op(StateVector state, size_t n, const GeneralOperator& gop) {
    StateVector new_state;

    const auto& amplitudes = gop.coefficients();
    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();
    const size_t begin = op_indices[n].first;
    const size_t end = op_indices[n].second;
    Determinant d;
    for (size_t j = begin; j < end; j++) {
        for (const auto& det_c : state) {
            const double c = det_c.second;
            d = det_c.first;
            const double sign = apply_op(d, op_list[j].cre(), op_list[j].ann());
            if (sign != 0.0) {
                new_state[d] += amplitudes[n] * op_list[j].factor() * sign * c;
            }
        }
    }
    return new_state;
}

StateVector apply_exp_op(const Determinant& d, size_t n, const GeneralOperator& gop) {
    StateVector state;
    state[d] = 1.0;
    StateVector exp_state = state;
    double factor = 1.0;
    int maxk = 16;
    for (int k = 1; k <= maxk; k++) {
        factor = factor / static_cast<double>(k);
        StateVector new_state = apply_lin_op(state, n, gop);
        if (new_state.size() == 0)
            break;
        for (const auto& det_c : new_state) {
            exp_state[det_c.first] += factor * det_c.second;
        }
        state = new_state;
    }
    return exp_state;
}

StateVector apply_exp_ah_factorized(GeneralOperator& gop, const StateVector& state0) {
    StateVector state(state0);
    StateVector new_state;
    size_t nterms = gop.nterms();
    Determinant d;
    for (size_t n = 0; n < nterms; n++) {
        new_state.clear();
        for (const auto& det_c : state) {
            const double c = det_c.second;
            d = det_c.first;
            StateVector terms = apply_exp_op(d, n, gop);
            for (const auto& d_c : terms) {
                new_state[d_c.first] += d_c.second * c;
            }
        }
        state = new_state;
    }
    return new_state;
}

StateVector apply_operator_fast(GeneralOperator& gop, const StateVector& state0,
                                     double screen_thresh) {
    local_timer t;
    const auto& amplitudes = gop.coefficients();
    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();

    StateVector new_terms;

    // loop over all the operators
    for (size_t n = 0, nterms = gop.nterms(); n < nterms; n++) {
        const size_t begin = op_indices[n].first;
        const size_t end = op_indices[n].second;
        if (amplitudes[n] == 0.0)
            continue;
        for (size_t j = begin; j < end; j++) {
            const SingleOperator& op = op_list[j];
            // create a mask for screening determinants according to the creation operators
            // this mask looks only at creation operators that are not preceeded by annihilation
            // operators
            const Determinant ucre = op.cre() - op.ann();
            const double tau = amplitudes[n] * op.factor();
            Determinant new_d;
            // loop over all determinants
            for (const auto& det_c : state0) {
                const Determinant& d = det_c.first;
                const double c = det_c.second;
                // test if we can apply this operator to this determinant
#if DEBUG_EXP_ALGORITHM
                std::cout << "\nOperation\n"
                          << str(op.cre, 16) << "+\n"
                          << str(op.ann, 16) << "-\n"
                          << str(d, 16) << std::endl;
                std::cout << "Testing (cre)(ann) sequence" << std::endl;
                std::cout << "Can annihilate: "
                          << (d.fast_a_and_b_equal_b(op.ann) ? "True" : "False") << std::endl;
                std::cout << "Can create:     " << (ucre.fast_a_and_b_eq_zero(d) ? "True" : "False")
                          << std::endl;
#endif
#if DEBUG_EXP_ALGORITHM
                std::cout << "Applying the (cre)(ann) sequence!" << std::endl;
#endif
                // screen according to the product tau * c
                if (std::fabs(tau * c) > screen_thresh) {
                    // check if this operator can be applied
                    if (d.fast_a_and_b_equal_b(op.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                        new_d = d;
                        double value = apply_op_safe(new_d, op.cre(), op.ann()) * tau * c;
                        new_terms[new_d] += value;
                    }
                }
            }
        }
    }
    time_apply_operator_fast += t.get();
    return new_terms;
}

StateVector apply_operator_fast2(GeneralOperator& gop, const StateVector& state0,
                                      double screen_thresh) {
    // make a copy of the state
    std::vector<std::tuple<double, double, Determinant>> state_sorted(state0.size());
    size_t k = 0;
    for (const auto& det_c : state0) {
        const Determinant& d = det_c.first;
        const double c = det_c.second;
        state_sorted[k] = std::make_tuple(std::fabs(c), c, d);
        ++k;
    }
    std::sort(state_sorted.rbegin(), state_sorted.rend());

    local_timer t;
    const auto& amplitudes = gop.coefficients();
    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();

    StateVector new_terms;

    Determinant d;
    double c;
    double absc;

    // loop over all the operators
    for (size_t n = 0, nterms = gop.nterms(); n < nterms; n++) {
        const size_t begin = op_indices[n].first;
        const size_t end = op_indices[n].second;
        if (amplitudes[n] == 0.0)
            continue;
        for (size_t j = begin; j < end; j++) {
            const SingleOperator& op = op_list[j];
            // create a mask for screening determinants according to the creation operators
            // this mask looks only at creation operators that are not preceeded by annihilation
            // operators
            const Determinant ucre = op.cre() - op.ann();
            const double tau = amplitudes[n] * op.factor();
            // loop over all determinants
            for (const auto& absc_c_det : state_sorted) {
                std::tie(absc, c, d) = absc_c_det;
                // screen according to the product tau * c
                if (std::fabs(tau * c) > screen_thresh) {
                    // check if this operator can be applied
                    if (d.fast_a_and_b_equal_b(op.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                        double value = apply_op_safe(d, op.cre(), op.ann()) * tau * c;
                        new_terms[d] += value;
                        ++ops_hash_push;
                    }
                    ++ops_screen;
                } else {
                    break;
                }
                ++ops_det_visit;
            }
        }
    }
    time_apply_operator_fast2 += t.get();
    return new_terms;
}

StateVector apply_exp_operator_fast(GeneralOperator& gop, const StateVector& state0,
                                         double scaling_factor, int maxk, double screen_thresh) {
    double convergence_threshold_ = screen_thresh;

    local_timer t;
    StateVector exp_state(state0);
    StateVector state(state0);
    double factor = 1.0;
    for (int k = 1; k <= maxk; k++) {
        factor *= scaling_factor / static_cast<double>(k);
        StateVector new_terms = apply_operator_fast(gop, state, screen_thresh);
        double norm = 0.0;
        double inf_norm = 0.0;
        for (const auto& det_c : new_terms) {
            exp_state[det_c.first] += factor * det_c.second;
            norm += std::pow(factor * det_c.second, 2.0);
            inf_norm = std::max(inf_norm, std::fabs(factor * det_c.second));
        }
        if (inf_norm < convergence_threshold_)
            break;
        state = new_terms;
    }
    time_apply_exp_operator_fast += t.get();
    return exp_state;
}

StateVector apply_exp_operator_fast2(GeneralOperator& gop, const StateVector& state0,
                                          double scaling_factor, int maxk, double screen_thresh) {
    double convergence_threshold_ = screen_thresh;

    local_timer t;
    StateVector exp_state(state0);
    StateVector state(state0);
    double factor = 1.0;
    for (int k = 1; k <= maxk; k++) {
        factor *= scaling_factor / static_cast<double>(k);
        StateVector new_terms = apply_operator_fast2(gop, state, screen_thresh);
        double norm = 0.0;
        double inf_norm = 0.0;
        for (const auto& det_c : new_terms) {
            exp_state[det_c.first] += factor * det_c.second;
            norm += std::pow(factor * det_c.second, 2.0);
            inf_norm = std::max(inf_norm, std::fabs(factor * det_c.second));
        }
        if (inf_norm < convergence_threshold_)
            break;
        state = new_terms;
    }
    time_apply_exp_operator_fast2 += t.get();
    return exp_state;
}

#define DEBUG_EXP_ALGORITHM 0
void apply_exp_op_fast(const Determinant& d, Determinant& new_d, const Determinant& cre,
                       const Determinant& ann, double amp, double c, StateVector& new_terms) {
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

StateVector apply_exp_ah_factorized_fast(GeneralOperator& gop, const StateVector& state0,
                                              bool inverse) {
    local_timer t;
    const auto& amplitudes = gop.coefficients();
    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();

    // initialize a state object
    StateVector state(state0);
    StateVector new_terms;

    //    // create a vector of determinants (for fast comparison)
    //    std::vector<Determinant> dets;
    //    for (const auto& det_c : state) {
    //        dets.push_back(det_c.first);
    //    }

    for (size_t m = 0, nterms = gop.nterms(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;

        // zero the new terms
        new_terms.clear();

        const size_t begin = op_indices[n].first;
        const SingleOperator& op = op_list[begin];
        const Determinant ucre = op.cre() - op.ann();
        const Determinant uann = op.ann() - op.cre();
        const double tau = (inverse ? -1.0 : 1.0) * amplitudes[n] * op.factor();
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
            if (d.fast_a_and_b_equal_b(op.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
#if DEBUG_EXP_ALGORITHM
                std::cout << "Applying the (cre)(ann) sequence!" << std::endl;
#endif
                const double c = det_c.second;
                apply_exp_op_fast(d, new_d, op.cre(), op.ann(), tau, c, new_terms);
            } else if (d.fast_a_and_b_equal_b(op.cre()) and d.fast_a_and_b_eq_zero(uann)) {
#if DEBUG_EXP_ALGORITHM
                std::cout << "Applying the (ann)(cre) sequence!" << std::endl;
#endif
                const double c = det_c.second;
                apply_exp_op_fast(d, new_d, op.ann(), op.cre(), -tau, c, new_terms);
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

StateVector apply_hamiltonian(std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                                   const StateVector& state0, double screen_thresh) {
    local_timer t;

    // initialize a state object
    StateVector state;

    size_t nmo = as_ints->nmo();

    auto mo_symmetry = as_ints->active_mo_symmetry();

    Determinant new_det;

    //    std::vector<int> aocc(nmo);
    //    std::vector<int> bocc(nmo);
    //    std::vector<int> avir(nmo);
    //    std::vector<int> bvir(nmo);

    for (const auto& det_c : state0) {
        const Determinant& det = det_c.first;
        const double c = det_c.second;

        //        det.get_alfa_occ(aocc, nmo);
        //        det.get_beta_occ(bocc, nmo);
        //        det.get_alfa_vir(avir, nmo);
        //        det.get_beta_vir(bvir, nmo);

        std::vector<int> aocc = det.get_alfa_occ(nmo);
        std::vector<int> bocc = det.get_beta_occ(nmo);
        std::vector<int> avir = det.get_alfa_vir(nmo);
        std::vector<int> bvir = det.get_beta_vir(nmo);

        size_t noalpha = aocc.size();
        size_t nobeta = bocc.size();
        size_t nvalpha = avir.size();
        size_t nvbeta = bvir.size();

        double E_0 = as_ints->nuclear_repulsion_energy() + as_ints->scalar_energy();

        state[det] += (E_0 + as_ints->slater_rules(det, det)) * c;
        // aa singles
        for (size_t i : aocc) {
            for (size_t a : avir) {
                if ((mo_symmetry[i] ^ mo_symmetry[a]) == 0) {
                    double DHIJ = as_ints->slater_rules_single_alpha(det, i, a) * c;
                    if (std::abs(DHIJ) >= screen_thresh) {
                        new_det = det;
                        new_det.set_alfa_bit(i, false);
                        new_det.set_alfa_bit(a, true);
                        state[new_det] += DHIJ;
                        ops_apply_hamiltonian++;
                    }
                }
            }
        }
        // bb singles
        for (size_t i : bocc) {
            for (size_t a : bvir) {
                if ((mo_symmetry[i] ^ mo_symmetry[a]) == 0) {
                    double DHIJ = as_ints->slater_rules_single_beta(det, i, a) * c;
                    if (std::abs(DHIJ) >= screen_thresh) {
                        new_det = det;
                        new_det.set_beta_bit(i, false);
                        new_det.set_beta_bit(a, true);
                        state[new_det] += DHIJ;
                        ops_apply_hamiltonian++;
                    }
                }
            }
        }
        // Generate aa excitations
        for (size_t ii = 0; ii < noalpha; ++ii) {
            size_t i = aocc[ii];
            for (size_t jj = ii + 1; jj < noalpha; ++jj) {
                size_t j = aocc[jj];
                for (size_t aa = 0; aa < nvalpha; ++aa) {
                    size_t a = avir[aa];
                    for (size_t bb = aa + 1; bb < nvalpha; ++bb) {
                        size_t b = avir[bb];
                        if ((mo_symmetry[i] ^ mo_symmetry[j] ^ mo_symmetry[a] ^ mo_symmetry[b]) ==
                            0) {
                            double DHIJ = as_ints->tei_aa(i, j, a, b) * c;
                            if (std::abs(DHIJ) >= screen_thresh) {
                                new_det = det;
                                DHIJ *= new_det.double_excitation_aa(i, j, a, b);
                                state[new_det] += DHIJ;
                                ops_apply_hamiltonian++;
                            }
                        }
                    }
                }
            }
        }
        // Generate ab excitations
        for (size_t i : aocc) {
            for (size_t j : bocc) {
                for (size_t a : avir) {
                    for (size_t b : bvir) {
                        if ((mo_symmetry[i] ^ mo_symmetry[j] ^ mo_symmetry[a] ^ mo_symmetry[b]) ==
                            0) {
                            double DHIJ = as_ints->tei_ab(i, j, a, b) * c;
                            if (std::abs(DHIJ) >= screen_thresh) {
                                new_det = det;
                                DHIJ *= new_det.double_excitation_ab(i, j, a, b);
                                state[new_det] += DHIJ;
                                ops_apply_hamiltonian++;
                            }
                        }
                    }
                }
            }
        }
        // Generate bb excitations
        for (size_t ii = 0; ii < nobeta; ++ii) {
            size_t i = bocc[ii];
            for (size_t jj = ii + 1; jj < nobeta; ++jj) {
                size_t j = bocc[jj];
                for (size_t aa = 0; aa < nvbeta; ++aa) {
                    size_t a = bvir[aa];
                    for (size_t bb = aa + 1; bb < nvbeta; ++bb) {
                        size_t b = bvir[bb];
                        if ((mo_symmetry[i] ^ mo_symmetry[j] ^ mo_symmetry[a] ^ mo_symmetry[b]) ==
                            0) {
                            double DHIJ = as_ints->tei_bb(i, j, a, b) * c;
                            if (std::abs(DHIJ) >= screen_thresh) {
                                new_det = det;
                                DHIJ *= new_det.double_excitation_bb(i, j, a, b);
                                state[new_det] += DHIJ;
                                ops_apply_hamiltonian++;
                            }
                        }
                    }
                }
            }
        }
    }
    time_apply_hamiltonian += t.get();
    return state;
}

std::vector<double> get_projection(GeneralOperator& gop, const StateVector& ref,
                                   const StateVector& state0) {
    local_timer t;
    std::vector<double> proj(gop.nterms(), 0.0);

    const auto& op_indices = gop.op_indices();
    const auto& op_list = gop.op_list();

    Determinant d;

    // loop over all the operators
    for (size_t n = 0, nterms = gop.nterms(); n < nterms; n++) {
        double value = 0.0;

        // apply the operator op_n
        const size_t begin = op_indices[n].first;
        const size_t end = op_indices[n].second;
        for (size_t j = begin; j < end; j++) {
            for (const auto& det_c : ref) {
                const double c = det_c.second;
                d = det_c.first;
                const double sign = apply_op(d, op_list[j].cre(), op_list[j].ann());
                if (sign != 0.0) {
                    auto search = state0.find(d);
                    if (search != state0.end()) {
                        value += op_list[j].factor() * sign * c * search->second;
                    }
                }
            }
        }
        proj[n] = value;
    }
    time_get_projection += t.get();

    return proj;
}

double energy_expectation_value(StateVector& left_state, StateVector& right_state,
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

StateVector apply_number_projector(int na, int nb, StateVector& state) {
    StateVector new_state;
    for (const auto& det_c : state) {
        if ((det_c.first.count_alfa() == na) and (det_c.first.count_beta() == nb) and
            (std::fabs(det_c.second) > 1.0e-12)) {
            new_state[det_c.first] = det_c.second;
        }
    }
    return new_state;
}

double overlap(StateVector& left_state, StateVector& right_state) {
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
