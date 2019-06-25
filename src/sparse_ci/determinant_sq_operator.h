/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _determinant_sq_operator_
#define _determinant_sq_operator_

#include <array>
#include <tuple>
#include <vector>

namespace forte {

/**
 * @brief A class to store a second quantized operator
 */

class DeterminantSQOperator {
    static constexpr int maxex = 8;

    using op_idx_t =
        std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>>;

  public:
    /// Default constructor
    DeterminantSQOperator();

    void add_operator(const std::vector<int>& a_ann, const std::vector<int>& a_cre,
                      const std::vector<int>& b_ann, const std::vector<int>& b_cre, double value);

  private:
    std::vector<op_idx_t> operators_;
    std::vector<double> values_;
};

} // namespace forte

#endif // _determinant_sq_operator_
