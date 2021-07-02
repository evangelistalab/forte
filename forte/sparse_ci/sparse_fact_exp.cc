/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <cmath>

#include "sparse_ci/sparse_fact_exp.h"

namespace forte {

SparseFactExp::SparseFactExp(bool phaseless) : phaseless_(phaseless) {}

StateVector SparseFactExp::compute(const SparseOperator& sop, const StateVector& state,
                                   const std::string& algorithm, bool inverse,
                                   double screen_thresh) {
    local_timer t;
    StateVector result;
    if (algorithm == "onthefly") {
        if (sop.is_antihermitian()) {
            result = compute_on_the_fly_antihermitian(sop, state, inverse, screen_thresh);
        } else {
            result = compute_on_the_fly_excitation(sop, state, inverse, screen_thresh);
        }
    } else {
        if (sop.is_antihermitian()) {
            result = compute_cached(sop, state, inverse, screen_thresh);
        } else {
            result = compute_on_the_fly_excitation(sop, state, inverse, screen_thresh);                                     
        }
    }
    timings_["total"] += t.get();
    return result;
}

StateVector SparseFactExp::compute_cached(const SparseOperator& sop, const StateVector& state,
                                          bool inverse, double screen_thresh) {
    for (const auto& det_c : state) {
        const Determinant& det = det_c.first;
        exp_hash_.add(det);
    }

    // compute the couplings
    if (inverse) {
        if (not initialized_inverse_) {
            compute_couplings(sop, state, inverse);
            initialized_inverse_ = true;
        }
    } else {
        if (not initialized_) {
            compute_couplings(sop, state, inverse);
            initialized_ = true;
        }
    }
    return compute_exp(sop, state, inverse, screen_thresh);
}

void SparseFactExp::compute_couplings(const SparseOperator& sop, const StateVector& state0,
                                      bool inverse) {
    local_timer t;
    const auto& op_list = sop.op_list();

    // initialize a state object
    StateVector state(state0);
    StateVector new_terms;

    // loop over all operators
    for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;

        std::vector<std::tuple<size_t, size_t, double>> d_couplings;

        // zero the new terms
        new_terms.clear();

        const SQOperator& sqop = op_list[n];
        const Determinant ucre = sqop.cre() - sqop.ann();
        const Determinant uann = sqop.ann() - sqop.cre();
        const double sign = inverse ? -1.0 : 1.0;
        Determinant new_d;
        // loop over all determinants
        if (phaseless_) {
            for (const auto& det_c : state) {
                const Determinant& d = det_c.first;

                // test if we can apply this operator to this determinant
                if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                    new_d = d;
                    // we ignore the phase here
                    apply_op(new_d, sqop.cre(), sqop.ann());
                    size_t d_idx = exp_hash_.add(d);
                    size_t new_d_idx = exp_hash_.add(new_d);
                    d_couplings.emplace_back(d_idx, new_d_idx, sign);
                    new_terms[new_d] += 1.0;
                } else if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(uann)) {
                    new_d = d;
                    // we ignore the phase here
                    apply_op(new_d, sqop.ann(), sqop.cre());
                    size_t d_idx = exp_hash_.add(d);
                    size_t new_d_idx = exp_hash_.add(new_d);
                    d_couplings.emplace_back(d_idx, new_d_idx, -sign);
                    new_terms[new_d] += 1.0;
                }
            }
        } else {
            for (const auto& det_c : state) {
                const Determinant& d = det_c.first;

                // test if we can apply this operator to this determinant
                if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                    new_d = d;
                    double f = sign * apply_op(new_d, sqop.cre(), sqop.ann());
                    size_t d_idx = exp_hash_.add(d);
                    size_t new_d_idx = exp_hash_.add(new_d);
                    d_couplings.emplace_back(d_idx, new_d_idx, f);
                    new_terms[new_d] += 1.0;
                } else if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(uann)) {
                    new_d = d;
                    double f = -sign * apply_op(new_d, sqop.ann(), sqop.cre());
                    size_t d_idx = exp_hash_.add(d);
                    size_t new_d_idx = exp_hash_.add(new_d);
                    d_couplings.emplace_back(d_idx, new_d_idx, f);
                    new_terms[new_d] += 1.0;
                }
            }
        }
        for (const auto& d_c : new_terms) {
            state[d_c.first] = 1.0;
        }
        if (inverse) {
            inverse_couplings_.push_back(d_couplings);
        } else {
            couplings_.push_back(d_couplings);
        }
    }
    timings_["total"] += t.get();
    timings_["couplings"] += t.get();
}

StateVector SparseFactExp::compute_exp(const SparseOperator& sop, const StateVector& state0,
                                       bool inverse, double screen_thresh) {
    local_timer t;

    // create and fill in the state vector
    std::vector<double> state_c(exp_hash_.size(), 0.0);

    // temporary space to store new elements
    std::vector<std::pair<size_t, double>> new_terms(100);

    for (const auto& det_c : state0) {
        const Determinant& d = det_c.first;
        double c = det_c.second;
        size_t d_idx = exp_hash_.get_idx(d);
        state_c[d_idx] = c;
    }

    // loop over all operators
    for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;

        double amp = sop.term(n).coefficient();

        const std::vector<std::tuple<size_t, size_t, double>>& d_couplings =
            inverse ? inverse_couplings_[m] : couplings_[m];

        // zero the new terms
        size_t k = 0;
        const size_t vec_size = new_terms.size();
        for (const auto& coupling : d_couplings) {
            const size_t d_idx = std::get<0>(coupling);
            const size_t new_d_idx = std::get<1>(coupling);
            const double f = amp * std::get<2>(coupling);
            const double c = state_c[d_idx];
            // do not apply this operator to this determinant if we expect the new determinant
            // to have an amplitude less than screen_thresh
            // (here we use the approximation sin(x) ~ x, for x small)
            if (std::fabs(f * c) > screen_thresh) {
                if (k < vec_size) {
                    new_terms[k] = std::make_pair(d_idx, c * (std::cos(f) - 1.0));
                    new_terms[k + 1] = std::make_pair(new_d_idx, c * std::sin(f));
                } else {
                    new_terms.push_back(std::make_pair(d_idx, c * (std::cos(f) - 1.0)));
                    new_terms.push_back(std::make_pair(new_d_idx, c * std::sin(f)));
                }
                k += 2;
            }
        }
        for (size_t j = 0; j < k; j++) {
            state_c[new_terms[j].first] += new_terms[j].second;
        }
    }
    StateVector state;
    for (size_t idx = 0, maxidx = exp_hash_.size(); idx < maxidx; idx++) {
        const Determinant& d = exp_hash_.get_det(idx);
        state[d] = state_c[idx];
    }
    timings_["total"] += t.get();
    timings_["exp"] += t.get();
    return state;
}

void SparseFactExp::apply_exp_op_fast(const Determinant& d, Determinant& new_d,
                                      const Determinant& cre, const Determinant& ann, double amp,
                                      double c, StateVector& new_terms) {
    new_d = d;
    const double f = apply_op(new_d, cre, ann) * amp;
    // this is to deal with number operators (should be removed)
    if (d != new_d) {
        new_terms[d] += c * (std::cos(f) - 1.0);
        new_terms[new_d] += c * std::sin(f);
    }
}

StateVector SparseFactExp::compute_on_the_fly_antihermitian(const SparseOperator& sop,
                                                            const StateVector& state0, bool inverse,
                                                            double screen_thresh) {
    local_timer t;
    const auto& op_list = sop.op_list();

    // initialize a state object
    StateVector state(state0);
    StateVector new_terms;

    for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;

        // zero the new terms
        new_terms.clear();

        const SQOperator& sqop = op_list[n];
        const Determinant ucre = sqop.cre() - sqop.ann();
        const Determinant uann = sqop.ann() - sqop.cre();
        const double tau = (inverse ? -1.0 : 1.0) * sqop.coefficient();
        Determinant new_d;
        // loop over all determinants
        for (const auto& det_c : state) {
            const Determinant& d = det_c.first;
            // do not apply this operator to this determinant if we expect the new determinant
            // to have an amplitude less than screen_thresh
            // (here we use the approximation sin(x) ~ x, for x small)
            if (std::fabs(det_c.second * tau) > screen_thresh) {
                // test if we can apply this operator to this determinant
                if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                    const double c = det_c.second;
                    apply_exp_op_fast(d, new_d, sqop.cre(), sqop.ann(), tau, c, new_terms);
                } else if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(uann)) {
                    const double c = det_c.second;
                    apply_exp_op_fast(d, new_d, sqop.ann(), sqop.cre(), -tau, c, new_terms);
                }
            }
        }
        for (const auto& d_c : new_terms) {
            state[d_c.first] += d_c.second;
        }
    }
    timings_["on_the_fly"] += t.get();
    return state;
}

StateVector SparseFactExp::compute_on_the_fly_excitation(const SparseOperator& sop,
                                                         const StateVector& state0, bool inverse,
                                                         double screen_thresh) {
    local_timer t;
    const auto& op_list = sop.op_list();

    // initialize a state object
    StateVector state(state0);
    StateVector new_terms;
    // const auto& amps = sop.coefficients();
    // std::vector<std::pair<double, size_t>> sorted_amps(sop.size());
    // for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
    //     sorted_amps[m] = std::make_pair(std::fabs(amps[m]), m);
    // }
    // sort(begin(sorted_amps), end(sorted_amps),
    //      [](const auto& a, const auto& b) { return a.first > b.first; });

    for (size_t n = 0, nterms = sop.size(); n < nterms; n++) {
        // zero the new terms
        new_terms.clear();

        const SQOperator& sqop = op_list[n];
        const Determinant ucre = sqop.cre() - sqop.ann();
        const double tau = (inverse ? -1.0 : 1.0) * sqop.coefficient();
        Determinant new_d;
        // loop over all determinants
        for (const auto& det_c : state) {
            const Determinant& d = det_c.first;
            // do not apply this operator to this determinant if we expect the new determinant
            // to have an amplitude less than screen_thresh
            // test if we can apply this operator to this determinant
            if ((std::fabs(det_c.second * tau) > screen_thresh) and
                d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                new_d = d;
                const double f = apply_op(new_d, sqop.cre(), sqop.ann()) * tau * det_c.second;
                new_terms[new_d] += f;
            }
        }
        for (const auto& d_c : new_terms) {
            state[d_c.first] += d_c.second;
        }
    }
    timings_["on_the_fly"] += t.get();
    return state;
}

std::map<std::string, double> SparseFactExp::timings() const { return timings_; }

} // namespace forte
