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

#include <cmath>

#include "helpers/memory.h"
#include "sparse_ci/sparse_fact_exp.h"

namespace forte {

SparseFactExp::SparseFactExp(double screen_thresh) : screen_thresh_(screen_thresh) {}

SparseState SparseFactExp::apply_op(const SparseOperatorList& sop, const SparseState& state,
                                    bool inverse) {
    // initialize a state object
    SparseState result(state);

    // temporary space to store new elements
    Buffer<std::pair<Determinant, double>> new_terms;

    Determinant new_det;
    for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;
        const auto& [sqop, coefficient] = sop(n);
        if (not sqop.is_nilpotent()) {
            std::string msg =
                "compute_on_the_fly_excitation is implemented only for nilpotent operators."
                "Operator " +
                sqop.str() + " is not nilpotent";
            throw std::runtime_error(msg);
        }
        const Determinant ucre = sqop.cre() - sqop.ann();
        const Determinant sign_mask = compute_sign_mask(sqop.ann(), sqop.cre());
        const double t = (inverse ? -1.0 : 1.0) * coefficient;
        const auto screen_thresh_div_t = screen_thresh_ / std::abs(t);
        // loop over all determinants
        for (const auto& [det, c] : result) {
            // do not apply this operator to this determinant if we expect the new determinant
            // to have an amplitude less than screen_thresh
            // test if we can apply this operator to this determinant
            if ((std::abs(c) > screen_thresh_div_t) and
                det.fast_can_apply_operator(sqop.ann(), ucre)) {
                const auto sign =
                    faster_apply_operator_to_det(det, new_det, sqop.cre(), sqop.ann(), sign_mask);
                new_terms.push_back(std::make_pair(new_det, c * t * sign));
            }
        }
        for (const auto& [det, c] : new_terms) {
            result[det] += c;
        }

        // reset the buffer
        new_terms.reset();
    }
    return result;
}

SparseState SparseFactExp::apply_antiherm(const SparseOperatorList& sop, const SparseState& state,
                                          bool inverse) {

    // initialize a state object
    SparseState result(state);
    Buffer<std::pair<Determinant, double>> new_terms;

    Determinant new_det;
    for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;

        const auto& [sqop, coefficient] = sop(n);
        if (not sqop.is_nilpotent()) {
            std::string msg =
                "compute_on_the_fly_antihermitian is implemented only for nilpotent operators."
                "Operator " +
                sqop.str() + " is not nilpotent";
            throw std::runtime_error(msg);
        }
        const Determinant ucre = sqop.cre() - sqop.ann();
        const Determinant uann = sqop.ann() - sqop.cre();
        const Determinant sign_mask = compute_sign_mask(sqop.ann(), sqop.cre());
        const double t = (inverse ? -1.0 : 1.0) * coefficient;
        const auto screen_thresh_div_t = screen_thresh_ / std::abs(t);
        // loop over all determinants
        for (const auto& [det, c] : result) {
            // do not apply this operator to this determinant if we expect the new determinant
            // to have an amplitude less than screen_thresh
            // (here we use the approximation sin(x) ~ x, for x small)
            if (std::fabs(c) > screen_thresh_div_t) {
                if (det.fast_can_apply_operator(sqop.ann(), ucre)) {
                    const auto theta = t * faster_apply_operator_to_det(det, new_det, sqop.cre(),
                                                                        sqop.ann(), sign_mask);
                    new_terms.emplace_back(det, c * (std::cos(theta) - 1.0));
                    new_terms.emplace_back(new_det, c * std::sin(theta));

                } else if (det.fast_can_apply_operator(sqop.cre(), uann)) {
                    const auto theta = -t * faster_apply_operator_to_det(det, new_det, sqop.ann(),
                                                                         sqop.cre(), sign_mask);
                    new_terms.emplace_back(det, c * (std::cos(theta) - 1.0));
                    new_terms.emplace_back(new_det, c * std::sin(theta));
                }
            }
        }
        for (const auto& [det, c] : new_terms) {
            result[det] += c;
        }

        // reset the buffer
        new_terms.reset();
    }
    return result;
}

} // namespace forte

/*

std::map<std::string, double> SparseFactExp::timings() const { return timings_; }
    /// Are the coupling initialized?
    bool initialized_ = false;
    /// Are the inverse couplings initialized?
    bool initialized_inverse_ = false;
    /// A map to store the determinants generated by the exponential
    DeterminantHashVec exp_hash_;
    /// A vector of determinant couplings
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> couplings_;
    /// A vector of determinant couplings used when applying the inverse exponential
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> inverse_couplings_;
    /// A map that stores timing information
    std::map<std::string, double> timings_;

SparseState SparseFactExp::compute_cached(const SparseOperatorList& sop, const SparseState& state,
                                          bool inverse) {
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
    return compute_exp(sop, state, inverse);
}

void SparseFactExp::compute_couplings(const SparseOperatorList& sop, const SparseState& state0,
                                      bool inverse) {
    local_timer t;
    // const auto& op_map = sop.op_map();

    // initialize a state object
    SparseState state(state0);
    SparseState new_terms;

    // loop over all operators
    for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;

        std::vector<std::tuple<size_t, size_t, double>> d_couplings;

        // zero the new terms
        new_terms.clear();

        const auto& [sqop, _] = sop(n);

        const Determinant ucre = sqop.cre() - sqop.ann();
        const Determinant uann = sqop.ann() - sqop.cre();
        const double sign = inverse ? -1.0 : 1.0;
        Determinant new_d;
        // loop over all determinants
        for (const auto& det_c : state) {
            const Determinant& d = det_c.first;

            // test if we can apply this operator to this determinant
            if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                new_d = d;
                double f = sign * apply_operator_to_det(new_d, sqop.cre(), sqop.ann());
                size_t d_idx = exp_hash_.add(d);
                size_t new_d_idx = exp_hash_.add(new_d);
                d_couplings.emplace_back(d_idx, new_d_idx, f);
                new_terms[new_d] += 1.0;
            } else if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(uann)) {
                new_d = d;
                double f = -sign * apply_operator_to_det(new_d, sqop.ann(), sqop.cre());
                size_t d_idx = exp_hash_.add(d);
                size_t new_d_idx = exp_hash_.add(new_d);
                d_couplings.emplace_back(d_idx, new_d_idx, f);
                new_terms[new_d] += 1.0;
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

SparseState SparseFactExp::compute_exp(const SparseOperatorList& sop, const SparseState& state0,
                                       bool inverse) {

    local_timer t;

    if (sop.size() == 0) {
        return state0;
    }

    // create and fill in the state vector
    std::vector<double> state_c(exp_hash_.size(), 0.0);

    // temporary space to store new elements
    std::vector<std::pair<size_t, double>> new_terms(100);

    for (const auto& [d, c] : state0) {
        size_t d_idx = exp_hash_.get_idx(d);
        state_c[d_idx] = c;
    }

    // loop over all operators
    for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;

        double amp = sop[n];

        const std::vector<std::tuple<size_t, size_t, double>>& d_couplings =
            inverse ? inverse_couplings_[m] : couplings_[m];

        // zero the new terms
        size_t k = 0;
        const size_t vec_size = new_terms.size();
        for (const auto& coupling : d_couplings) {
            const size_t d_idx = std::get<0>(coupling);
            const size_t new_d_idx = std::get<1>(coupling);
            // special case of number operator
            if (d_idx == new_d_idx)
                continue;
            const double f = amp * std::get<2>(coupling);
            const double c = state_c[d_idx];
            // do not apply this operator to this determinant if we expect the new determinant
            // to have an amplitude less than screen_thresh
            // (here we use the approximation sin(x) ~ x, for x small)
            if (std::fabs(f * c) > screen_thresh_) {
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
    SparseState state;
    for (size_t idx = 0, maxidx = exp_hash_.size(); idx < maxidx; idx++) {
        const Determinant& d = exp_hash_.get_det(idx);
        state[d] = state_c[idx];
    }
    timings_["total"] += t.get();
    timings_["exp"] += t.get();
    return state;
}
*/