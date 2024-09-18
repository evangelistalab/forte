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

#include "helpers/combinatorial.h"
#include "helpers/string_algorithms.h"

#include "sparse_ci/sparse.h"
#include "sparse_ci/sq_operator_string.h"
#include "sparse_ci/sparse_operator.h"

namespace forte {

void generate_wick_contractions2(SQOperatorString& lhs, SQOperatorString& rhs,
                                 SparseOperator& result, const sparse_scalar_t sign);

void process_cre2(SQOperatorString& lhs, SQOperatorString& rhs, SparseOperator& result,
                  const sparse_scalar_t sign) {
    //    Left   |   Right
    // cre | ann | cre | ann |
    //  ^     ^     ^
    //  3     2     1
    //
    // 1. take the first creation operator on the right
    // 2. contract it with the corresponding annihilation operator on the left
    // 3. move the operator to the left creation part

    // find the first right alpha creation operator to move
    const auto i = rhs.cre().find_first_one();
    // remove the operator from the right
    // if a corresponding left annihilation operator exists, permute the operators and
    // introduce a contraction
    if (lhs.ann().get_bit(i) == true) {
        const auto coefficient = lhs.ann().slater_sign(i);
        lhs.ann_mod().set_bit(i, false);
        rhs.cre_mod().set_bit(i, false);
        generate_wick_contractions2(lhs, rhs, result, sign * coefficient);
        lhs.ann_mod().set_bit(i, true);
        rhs.cre_mod().set_bit(i, true);
    }
    // if the left creation operator does not exist, move the operator in place
    // otherwise we get a collision and the operator is removed
    if (lhs.cre().get_bit(i) == false) {
        const auto coefficient =
            (lhs.count()) % 2 == 0 ? lhs.cre().slater_sign(i) : -lhs.cre().slater_sign(i);
        lhs.cre_mod().set_bit(i, true);
        rhs.cre_mod().set_bit(i, false);
        generate_wick_contractions2(lhs, rhs, result, sign * coefficient);
        lhs.cre_mod().set_bit(i, false);
        rhs.cre_mod().set_bit(i, true);
    }
}

void process_ann2(SQOperatorString& lhs, SQOperatorString& rhs, SparseOperator& result,
                  const sparse_scalar_t sign) {
    // Here we assume that the operators are in the canonical form
    // and that the right alpha and beta creation have been alredy removed
    //    Left   |   Right
    // cre | ann | ann |
    //        ^     ^
    //        2     1
    //
    // 1. take the last annihilation operator on the right
    // 2. move the operator to the left annihilation part

    // find the last right annihilation operator to move
    const auto i = rhs.ann().find_last_one();
    // if the left annihilation operator does not exist, move the operator in place
    // otherwise we get a collision and the operator is removed
    if (lhs.ann().get_bit(i) == false) {
        const auto coefficient = lhs.ann().slater_sign(i);
        lhs.ann_mod().set_bit(i, true);
        rhs.ann_mod().set_bit(i, false);
        generate_wick_contractions2(lhs, rhs, result, sign * coefficient);
        lhs.ann_mod().set_bit(i, false);
        rhs.ann_mod().set_bit(i, true);
    }
}

void generate_wick_contractions2(SQOperatorString& lhs, SQOperatorString& rhs,
                                 SparseOperator& result, sparse_scalar_t sign) {
    // if there are no operators on the right then we return
    if (rhs.count() == 0) {
        result.add(lhs, sign);
        return;
    }
    if (rhs.cre().count() > 0) {
        process_cre2(lhs, rhs, result, sign);
    } else if (rhs.ann().count() > 0) {
        process_ann2(lhs, rhs, result, sign);
    }
}

void new_product2(SparseOperator& result, SQOperatorString lhs, SQOperatorString rhs,
                  sparse_scalar_t factor) {
    generate_wick_contractions2(lhs, rhs, result, factor);
}
} // namespace forte