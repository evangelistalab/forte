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

#ifndef _sa_dsrgpt_h_
#define _sa_dsrgpt_h_

#include "sadsrg.h"

using namespace ambit;

namespace forte {

class SA_DSRGPT : public SADSRG {
  public:
    /**
     * SA_DSRGPT Constructor
     * @param scf_info The SCFInfo
     * @param options The ForteOption
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    SA_DSRGPT(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
              std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Compute the corr_level energy with fixed reference
    virtual double compute_energy() = 0;

  protected:
    /// Read options
    void read_options();

    /// Print a summary of the options
    void print_options();

    /// Form Hbar or not?
    bool form_Hbar_;

    /// How to consider internal amplitudes
    std::string internal_amp_;
    /// Include which part of internal amplitudes?
    std::string internal_amp_select_;

    /// Two-electron integral
    ambit::BlockedTensor V_;
    /// Generalized Fock matrix
    ambit::BlockedTensor F_;
    /// Zeroth-order Hamiltonian (diagonal blocks of bare Fock)
    ambit::BlockedTensor F0th_;
    /// Generalized Fock matrix (bare off-diagonal blocks)
    ambit::BlockedTensor F1st_;
    /// Single excitation amplitude
    ambit::BlockedTensor T1_;
    /// Double excitation amplitude
    ambit::BlockedTensor T2_;
    /// Double excitation amplitude (2 * J - K)
    ambit::BlockedTensor S2_;

    /// Initialize Fock
    void init_fock();

    /// Compute 1st-order T2 amplitudes
    void compute_t2_full();

    /// Compute 1st-order T1 amplitudes
    void compute_t1();

    /// Renormalize integrals
    /// if add == True: scale by 1 + exp(-s * D^2); else: scale by exp(-s * D^2)
    void renormalize_integrals(bool add);
};
} // namespace forte

#endif // _sa_dsrgpt_h_
