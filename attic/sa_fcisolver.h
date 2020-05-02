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

#ifndef SA_FCISOLVER_H
#define SA_FCISOLVER_H


#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/wavefunction.h"

#include "base_classes/reference.h"
#include "fci/fci_solver.h"
#include "fci/fci_vector.h"

namespace forte {

/// SA_FCISolver seeks to call multiple instances of CAS-CI and combine all the
/// RDMS and average them
class SA_FCISolver {
  public:
    SA_FCISolver(std::shared_ptr<ForteOptions> options, std::shared_ptr<psi::Wavefunction> wfn);

    /// E_{sa-casscf} = gamma_{avg} h_{pq} + Gamma_{avg} g_{pqrs}
    double compute_energy();

    Reference reference() { return sa_ref_; }

    void set_integral_pointer(std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
        fci_ints_ = fci_ints;
    }

    void set_mo_space_info(std::shared_ptr<MOSpaceInfo> mo_space_info) {
        mo_space_info_ = mo_space_info;
    }

    void set_integrals(std::shared_ptr<ForteIntegrals> ints) { ints_ = ints; }

    std::vector<std::shared_ptr<FCIVector>> StateAveragedCISolution() { return SA_C_; }

  private:
    /// Options from Psi4
    psi::Options options_;
    /// The wavefunction object of Psi4
    std::shared_ptr<psi::Wavefunction> wfn_;
    /// Integral objects (same for all SA computations)
    std::shared_ptr<ForteIntegrals> ints_;
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints_;
    /// MO space information of FORTE
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// A vector of the averaged FCI solutions
    std::vector<std::shared_ptr<FCIVector>> SA_C_;

    /// The vector that contains states and weights information
    std::vector<std::tuple<int, int, int, std::vector<double>>> parsed_options_;

    /// The total number of states to be averaged
    int nstates_;

    /// The reference object in FORTE
    Reference sa_ref_;

    /// Read options and fill in parsed_options_
    void read_options();
};
} // namespace forte

#endif // SA_FCISOLVER_H
