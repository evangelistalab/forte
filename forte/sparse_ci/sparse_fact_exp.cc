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

    // temporary space to store new elements. This avoids reallocation
    Buffer<std::pair<Determinant, sparse_scalar_t>> new_terms;

    Determinant new_det;
    Determinant sign_mask;
    Determinant idx;
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
        compute_sign_mask(sqop.cre(), sqop.ann(), sign_mask, idx);
        const auto t = (inverse ? -1.0 : 1.0) * coefficient;
        const auto screen_thresh_div_t = screen_thresh_ / std::abs(t);
        // loop over all determinants
        for (const auto& [det, c] : result) {
            // do not apply this operator to this determinant if we expect the new determinant
            // to have an amplitude less than screen_thresh
            // test if we can apply this operator to this determinant
            if ((std::abs(c) > screen_thresh_div_t) and
                det.faster_can_apply_operator(sqop.cre(), sqop.ann())) {
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
    Buffer<std::pair<Determinant, sparse_scalar_t>> new_terms;

    Determinant new_det;
    Determinant sign_mask;
    Determinant idx;
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
        compute_sign_mask(sqop.cre(), sqop.ann(), sign_mask, idx);
        const auto t = (inverse ? -1.0 : 1.0) * coefficient;
        const auto screen_thresh_div_t = screen_thresh_ / std::abs(t);
        // loop over all determinants
        for (const auto& [det, c] : result) {
            // do not apply this operator to this determinant if we expect the new determinant
            // to have an amplitude less than screen_thresh
            // (here we use the approximation sin(x) ~ x, for x small)
            if (std::abs(c) > screen_thresh_div_t) {
                if (det.faster_can_apply_operator(sqop.cre(), sqop.ann())) {
                    const auto theta = t * faster_apply_operator_to_det(det, new_det, sqop.cre(),
                                                                        sqop.ann(), sign_mask);
                    new_terms.emplace_back(det, c * (std::cos(theta) - 1.0));
                    new_terms.emplace_back(new_det, c * std::sin(theta));
                } else if (det.faster_can_apply_operator(sqop.ann(), sqop.cre())) {
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