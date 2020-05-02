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

#ifndef _sa_mrpt3_h_
#define _sa_mrpt3_h_

#include "sa_dsrgpt.h"

using namespace ambit;

namespace forte {

class SA_MRPT3 : public SA_DSRGPT {
  public:
    /**
     * SA_MRPT3 Constructor
     * @param scf_info The SCFInfo
     * @param options The ForteOption
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    SA_MRPT3(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
             std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Compute the corr_level energy with fixed reference
    virtual double compute_energy();

  protected:
    /// Start-up function called in the constructor
    void startup();

    /// Fill up integrals
    void build_ints();

    /// Initialize amplitude tensors
    void init_amps();

    /// Three-index integrals
    ambit::BlockedTensor B_;
    /// One-body temp tensor ([[H0th,A1st],A1st] or 1st-order amplitudes)
    ambit::BlockedTensor O1_;
    /// Two-body temp tensor ([[H0th,A1st],A1st] or 1st-order amplitudes)
    ambit::BlockedTensor O2_;

    /// Check memory
    void check_memory();

    /// 2nd-order energy and transformed Hamiltonian
    double compute_energy_pt2();
    /// 3rd-order energy contribution 1.0 / 12.0 * [[[H0th,A1st],A1st],A1st] (done before pt2)
    double compute_energy_pt3_1();
    /// 3rd-order energy contribution 0.5 * [H1st + Hbar1st,A2nd]
    double compute_energy_pt3_2();
    /// 3rd-order energy contribution 0.5 * [Hbar2nd,A1st]
    double compute_energy_pt3_3();
};
} // namespace forte

#endif // _sa_mrpt3_h_
