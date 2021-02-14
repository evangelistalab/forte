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

#ifndef _sparse_fact_exp_h_
#define _sparse_fact_exp_h_

#include "integrals/active_space_integrals.h"
#include "helpers/timer.h"

#include "sparse_ci/sparse_state_vector.h"
#include "sparse_ci/sparse_operator.h"

#include "sparse_ci/determinant_hashvector.h"

namespace forte {

class SparseFactExp {
  public:
    SparseFactExp(bool phaseless = false);
    StateVector compute(const SparseOperator& sop, const StateVector& state, bool inverse,
                        double screen_thresh);
    StateVector compute_on_the_fly(const SparseOperator& sop, const StateVector& state, bool inverse,
                                   double screen_thresh);
    std::map<std::string, double> time() const;

  private:
    void apply_exp_op_fast(const Determinant& d, Determinant& new_d, const Determinant& cre,
                           const Determinant& ann, double amp, double c, StateVector& new_terms);
    void compute_couplings(const SparseOperator& sop, const StateVector& state0, bool inverse);
    StateVector compute_exp(const SparseOperator& sop, const StateVector& state0, bool inverse,
                            double screen_thresh);

    bool phaseless_;
    bool initialized_ = false;
    bool initialized_inverse_ = false;
    double time_ = 0.0;
    double couplings_time_ = 0.0;
    double exp_time_ = 0.0;
    double on_the_fly_time_ = 0.0;
    DeterminantHashVec exp_hash_;

    std::vector<std::vector<std::tuple<size_t, size_t, double>>> couplings_;
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> inverse_couplings_;
};

} // namespace forte

#endif // _sparse_fact_exp_h_
