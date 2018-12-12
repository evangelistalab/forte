/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _cc_h_
#define _cc_h_

#include <cmath>
#include "boost/assign.hpp"

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.h"
#include "ambit/blocked_tensor.h"

#include "integrals/integrals.h"
#include "reference.h"
#include "blockedtensorfactory.h"
#include "./mrdsrg-helper/dsrg_source.h"
#include "./mrdsrg-helper/dsrg_time.h"
#include "sparse_ci/determinant.h"

using namespace ambit;
namespace psi {
namespace forte {

class CC : public Wavefunction {
  public:
    /**
     * CC Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    CC(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
       std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~CC();

    /// Compute the corr_level energy with fixed reference
    double compute_energy();

  protected:
    // => Class initialization and termination <= //

    /// Reference single reference determinant energy
    double E_ref_;

    /// Start-up function called in the constructor
    void startup();
    /// Clean-up function called in the destructor

    /// Max iteration for amplitude update.
    int maxiter_;
    /// Convergence thresholds
    double e_convergence_;
    double r_convergence_;

    /// The frozen-core energy
    double frozen_core_energy_;

    /// The molecular integrals
    std::shared_ptr<ForteIntegrals> ints_;

    /// The integral type
    std::string ints_type_;
    /// If ERI density fitted or Cholesky decomposed
    bool eri_df_;

    /// MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// List of alpha occupied MOs
    std::vector<size_t> aocc_mos_;
    /// List of alpha virtual MOs
    std::vector<size_t> avir_mos_;
    /// List of beta occupied MOs
    std::vector<size_t> bocc_mos_;
    /// List of beta virtual MOs
    std::vector<size_t> bvir_mos_;
    /// List of auxiliary MOs when DF/CD
    std::vector<size_t> aux_mos_;

    /// Alpha occupied label
    std::string aocc_label_;
    /// Alpha virtual label
    std::string avir_label_;
    /// Beta occupied label
    std::string bocc_label_;
    /// Beta virtual label
    std::string bvir_label_;
    /// Auxillary basis label
    std::string aux_label_;

    /// Fill up integrals
    void build_ints();
    /// Build Fock matrix and diagonal Fock matrix elements
    void build_fock(BlockedTensor& H, BlockedTensor& V);
    void build_fock_df(BlockedTensor& H, BlockedTensor& B);
    /// Fill the tensor T with three-index DF or CD integrals
    void fill_three_index_ints(ambit::BlockedTensor T);

    /// Compute the denominator tensors
    void compute_denominators();
    /// Initialize T with mp2 amplitudes
    void initial_mp2_t();
    /// Compute the effective two-particle excitation operators
    void compute_effective_tau();
    /// Compute Stanton intermediate tensors
    void compute_intermediates();
    /// Update T amplitudes
    void update_t();
    /// Compute CC correlation energy by T amplitudes.
    double cc_energy();

    /// Kevin's Tensor Wrapper
    std::shared_ptr<BlockedTensorFactory> BTF_;
    /// Tensor type for AMBIT
    TensorType tensor_type_;

    /// One-electron integral
    ambit::BlockedTensor H_;
    /// Two-electron integral
    ambit::BlockedTensor V_;
    /// Three-index integrals
    ambit::BlockedTensor B_;
    /// Generalized Fock matrix
    ambit::BlockedTensor F_;
    /// Single excitation amplitude
    ambit::BlockedTensor T1_;
    /// Double excitation amplitude
    ambit::BlockedTensor T2_;
    /// Effective two-particle operator
    ambit::BlockedTensor tilde_tau_;
    ambit::BlockedTensor tau_;
    /// Denominators
    ambit::BlockedTensor D1_;
    ambit::BlockedTensor D2_;
    /// Intermediates
    ambit::BlockedTensor W1_;
    ambit::BlockedTensor W2_;
    /// Difference of consecutive singles
    ambit::BlockedTensor DT1_;
    /// Difference of consecutive doubles
    ambit::BlockedTensor DT2_;

    /// Diagonal elements of Fock matrices
    std::vector<double> Fa_;
    std::vector<double> Fb_;
};
}
}
#endif // _cc_h_
