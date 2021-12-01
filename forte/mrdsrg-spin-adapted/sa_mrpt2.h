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

#ifndef _sa_mrpt2_h_
#define _sa_mrpt2_h_

#include "sa_dsrgpt.h"

using namespace ambit;

namespace forte {

class SA_MRPT2 : public SA_DSRGPT {
  public:
    /**
     * SA_MRPT2 Constructor
     * @param scf_info The SCFInfo
     * @param options The ForteOption
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    SA_MRPT2(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
             std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Compute the corr_level energy with fixed reference
    double compute_energy() override;

  protected:
    /// Start-up function called in the constructor
    void startup();

    /// Fill up integrals
    void build_ints();
    /// Build minimal blocks of V from 3-index B
    void build_minimal_V();

    /// Initialize amplitude tensors
    void init_amps();

    /// Check memory
    void check_memory();

    /// Memory requirements for the three batched energy terms
    std::map<std::string, size_t> mem_batched_;

    /// Compute 1st-order T2 amplitudes
    void compute_t2();
    /// Compute 1st-order T2 amplitudes with at least two active indices
    void compute_t2_df_minimal();

    /// Energy contribution from CCVV block
    double E_V_T2_CCVV();
    /// Energy contribution from CAVV block
    double E_V_T2_CAVV();
    /// Energy contribution from CCAV block
    double E_V_T2_CCAV();

    /// Energy contribution from CCVV block using DF integrals
    double compute_Hbar0_CCVV_DF();
    /// Energy contribution from CCVV block using DiskDF integrals
    double compute_Hbar0_CCVV_diskDF();

    /// Compute DSRG-transformed Hamiltonian
    void compute_hbar();

    /// Compute Hbar1 from core contraction, renormalize V if Vr is true
    void compute_Hbar1C_DF(ambit::Tensor& Hbar1, bool Vr = true);
    /// Compute Hbar1 from virtual contraction, renormalize V if Vr is true
    void compute_Hbar1V_DF(ambit::Tensor& Hbar1, bool Vr = true);
    /// Compute Hbar1 from core contraction, renormalize V if Vr is true
    void compute_Hbar1C_diskDF(ambit::Tensor& Hbar1, bool Vr = true);
    /// Compute Hbar1 from virtual contraction, renormalize V if Vr is true
    void compute_Hbar1V_diskDF(ambit::Tensor& Hbar1, bool Vr = true);

    /// C1 = [Vr, T2] CAVV from compute_Hbar1V_diskDF
    ambit::Tensor C1_VT2_CAVV_;
    /// C1 = [Vr, T2] CCAV from compute_Hbar1C_diskDF
    ambit::Tensor C1_VT2_CCAV_;

    /// Return a vector of empty ambit Tensor objects
    std::vector<ambit::Tensor> init_tensor_vecs(int number_of_tensors);

    /// Separate indices into batches of indices
    std::vector<std::vector<size_t>> split_indices_to_batches(const std::vector<size_t>& indices,
                                                              size_t max_size);
};
} // namespace forte

#endif // _sa_mrpt2_h_
