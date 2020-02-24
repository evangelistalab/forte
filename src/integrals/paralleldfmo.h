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

#include "psi4/libmints/basisset.h"
#include "psi4/psi4-dec.h"

namespace forte {

class ParallelDFMO {
  public:
    ParallelDFMO(std::shared_ptr<psi::BasisSet> primary, std::shared_ptr<psi::BasisSet> auxiliary);
    void set_C(std::shared_ptr<psi::Matrix> C) { Ca_ = C; }
    void compute_integrals();
    int Q_PQ() { return GA_Q_PQ_; }

  protected:
    std::shared_ptr<psi::Matrix> Ca_;
    /// (A | Q)^{-1/2}
    void J_one_half();
    /// Compute (A|mn) integrals (distribute via mn indices)
    void transform_integrals();
    /// (A | pq) (A | Q)^{-1/2}

    std::shared_ptr<psi::BasisSet> primary_;
    std::shared_ptr<psi::BasisSet> auxiliary_;

    /// Distributed DF (Q | pq) integrals
    int GA_Q_PQ_;
    /// GA for J^{-1/2}
    int GA_J_onehalf_;

    size_t memory_;
    size_t nmo_;
};
}
}
