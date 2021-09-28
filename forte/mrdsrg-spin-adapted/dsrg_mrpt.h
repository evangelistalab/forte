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

#ifndef _dsrg_mrpt_h_
#define _dsrg_mrpt_h_

#include <cmath>


#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.h"
#include "ambit/blocked_tensor.h"

#include "base_classes/rdms.h"
#include "base_classes/dynamic_correlation_solver.h"
#include "integrals/integrals.h"
#include "mrdsrg-helper/dsrg_source.h"
#include "mrdsrg-helper/dsrg_time.h"
#include "base_classes/mo_space_info.h"
#include "helpers/helpers.h"

using namespace ambit;

namespace forte {

class DSRG_MRPT : public DynamicCorrelationSolver {
  public:
    /**
     * DSRG-MRPT Constructor
     * @param rdms The reference reduced density matrices
     * @param scf_info The SCF info
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    DSRG_MRPT(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
              std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~DSRG_MRPT();

    /// Compute energy with fixed reference
    double compute_energy();

    /// DSRG transformed Hamiltonian (not implemented)
    std::shared_ptr<ActiveSpaceIntegrals> compute_Heff_actv();

  protected:
    // => Class initialization and termination <= //

    /// Start-up function called in the constructor
    void startup();
    /// Clean-up function called in the destructor
    void cleanup();

    /// Read options
    void read_options();

    /// Print levels
    int print_;

    /// The energy of the rdms
    double Eref_;

    /// The frozen-core energy
    double frozen_core_energy_;

    /// Are orbitals semi-canonicalized?
    bool semi_canonical_;

    /// List of core MOs
    std::vector<size_t> core_mos_;
    /// List of active MOs
    std::vector<size_t> actv_mos_;
    /// List of virtual MOs
    std::vector<size_t> virt_mos_;

    /// Core label
    std::string core_label_;
    /// Active label
    std::string actv_label_;
    /// Virtual label
    std::string virt_label_;

    /// Map from space label to list of MOs
    std::map<char, std::vector<size_t>> label_to_spacemo_;

    /// Fill up integrals
    void build_ints();
    /// Fill up density matrix and density cumulants
    void build_density();
    /// Build Fock matrix and diagonal Fock matrix elements
    void build_fock();

    /// Check if orbitals are semi-canonicalized
    bool check_semicanonical();
    /// Test spin adaptation of density matrix and density cumulants
    void test_density();
    /// Estimate peak memory
    void test_memory(const size_t& c, const size_t& a, const size_t& v);
    /// Number of batches for CCVV term if not enough memory
    int nbatch_;

    // => DSRG related <= //

    /// Correlation level: PT2 or PT3
    std::string corr_lv_;

    /// The flow parameter
    double s_;

    /// Source operator
    std::string source_;
    /// The dsrg source operator
    std::shared_ptr<DSRG_SOURCE> dsrg_source_;
    /// Core-core-virtual-virtual source
    std::string ccvv_source_;

    /// Smaller than which we will do Taylor expansion
    double taylor_threshold_;

    /// Tensor type for AMBIT
    TensorType tensor_type_;

    /// One-electron integral
    ambit::BlockedTensor H_;
    /// Two-electron integral
    ambit::BlockedTensor V_;
    /// Generalized Fock matrix
    ambit::BlockedTensor F_;
    /// Spin-summed one-particle density matrix
    ambit::BlockedTensor L1_;
    /// Spin-summed one-hole density matrix
    ambit::BlockedTensor Eta1_;
    /// Spin-summed two-body denisty cumulant
    ambit::BlockedTensor L2_;
    /// Spin-summed three-body density cumulant
    ambit::BlockedTensor L3_;
    /// Single excitation amplitude
    ambit::BlockedTensor T1_;
    /// Double excitation amplitude
    ambit::BlockedTensor T2_;

    /// Diagonal elements of Fock matrices
    std::vector<double> Fdiag_;

    /// RMS of T2
    double T2rms_;
    /// Norm of T2
    double T2norm_;
    /// Signed max of T2
    double T2max_;
    /// Check T2 and store the largest amplitudes
    void check_t2(BlockedTensor& T2);

    /// RMS of T1
    double T1rms_;
    /// Norm of T1
    double T1norm_;
    /// Signed max of T1
    double T1max_;
    /// Check T1 and store the largest amplitudes
    void check_t1(BlockedTensor& T1);

    /// Number of amplitudes will be printed in amplitude summary
    size_t ntamp_;
    /// Threshold for amplitudes considered as intruders
    double intruder_tamp_;
    /// List of large amplitudes
    std::vector<std::pair<std::vector<size_t>, double>> lt1_;
    std::vector<std::pair<std::vector<size_t>, double>> lt2_;
    /// Analyze T1 and T2 amplitudes
    void analyze_amplitudes(BlockedTensor& T1, BlockedTensor& T2, std::string name = "");

    /// Compute DSRG-transformed Hamiltonian Hbar
    void compute_hbar();
    /// Zero-body Hbar
    double Hbar0_;
    /// One-body Hbar
    ambit::BlockedTensor Hbar1_;
    /// Two-body Hbar
    ambit::BlockedTensor Hbar2_;
    /// Temporary one-body Hamiltonian
    ambit::BlockedTensor O1_;
    ambit::BlockedTensor C1_;
    /// Temporary two-body Hamiltonian
    ambit::BlockedTensor O2_;
    ambit::BlockedTensor C2_;

    /// Norm of off-diagonal Hbar2
    double Hbar2od_norm(const std::vector<std::string>& blocks);
    /// Norm of off-diagonal Hbar1
    double Hbar1od_norm(const std::vector<std::string>& blocks);

    /// Timings for computing the commutators
    DSRG_TIME dsrg_time_;

    /// Compute zero-body term of commutator [H1, T1] * alpha
    void H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H1, T2] * alpha
    void H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T1] * alpha
    void H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T2] * alpha
    void H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0,
                  const bool& stored);

    /// Compute zero-body term of commutator [H2, T2] involving only L1
    void H2_T2_C0_L1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0,
                     const bool& stored);
    /// Compute zero-body term of commutator [H2, T2] involving only L2
    void H2_T2_C0_L2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T2] involving only L3
    void H2_T2_C0_L3(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0);

    /// Compute zero-body term of commutator [H2, T2] involving only L3,
    /// intermediate of aaaaaa
    double H2_T2_C0_L3_a(BlockedTensor& H2, BlockedTensor& T2);
    /// Compute zero-body term of commutator [H2, T2] involving only L3,
    /// intermediate of aaav
    double H2_T2_C0_L3_v(BlockedTensor& H2, BlockedTensor& T2);

    /// Compute zero-body term of commutator [V, T2] involving only L1 and ccvv
    /// of V
    double V_T2_C0_L1_ccvv(const std::vector<std::vector<size_t>>& small_core_mo);
    /// Compute zero-body term of commutator [V, T2] involving only L1 and cavv
    /// of V
    double V_T2_C0_L1_cavv(const std::vector<std::vector<size_t>>& small_core_mo);
    /// Compute zero-body term of commutator [V, T2] involving only L1 and ccav
    /// of V
    double V_T2_C0_L1_ccav();

    /// Compute one-body term of commutator [H1, T1] * alpha
    void H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H1, T2] * alpha
    void H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T1] * alpha
    void H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T2] * alpha
    void H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);

    /// Compute two-body term of commutator [H2, T1] * alpha
    void H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H1, T2] * alpha
    void H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, T2] * alpha
    void H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);

    /// Compute diagonal blocks labels of a one-body operator
    std::vector<std::string> d_one_labels();
    /// Compute off-diagonal blocks labels of a one-body operator
    std::vector<std::string> od_one_labels();
    /// Compute off-diagonal blocks labels of a two-body operator
    std::vector<std::string> od_two_labels();

    /// Scale BlockedTensor by denominator
    void BT_scaled_by_D(BlockedTensor& BT);
    /// Scale BlockedTensor by bare source plus one (R + 1)
    void BT_scaled_by_Rplus1(BlockedTensor& BT);
    /// Scale BlockedTensor by renormalized denominator
    void BT_scaled_by_RD(BlockedTensor& BT);

    // => DSRG-MRPT2 <= //

    /// Compute DSRG-MRPT2 energy
    double compute_energy_pt2();
    /// Fold in [H0th, A] effect to F1st for 2nd-order energy
    void renormalize_F_E2nd();
    /// Fold in [H0th, A] effect to V1st for 2nd-order energy
    void renormalize_V_E2nd();
    /// Compute first-order T amplitudes
    void compute_T_1st(BlockedTensor& V, BlockedTensor& T2, BlockedTensor& F, BlockedTensor& T1);
    /// Compute first-order T1
    void compute_T1_1st(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1);
    /// Compute first-order T2
    void compute_T2_1st(BlockedTensor& V, BlockedTensor& T2);

    // => Reference relaxation <= //

    /// Reference relaxation type
    std::string ref_relax_;
    /// Transfer integrals for FCI
    void transfer_integrals();
    /// Reset integrals to bare Hamiltonian
    void reset_ints(BlockedTensor& H, BlockedTensor& V);
    /// Diagonalize the diagonal blocks of the Fock matrix
    std::vector<std::vector<double>> diagonalize_Fock_diagblocks(BlockedTensor& U);
    /// Separate an 2D ambit::Tensor according to its irrep
    ambit::Tensor separate_tensor(ambit::Tensor& tens, const psi::Dimension& irrep, const int& h);
    /// Combine a separated 2D ambit::Tensor
    void combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h, const psi::Dimension& irrep,
                        const int& h);

    // => Useful printings <= //

    /// Print a summary of the options
    void print_options();
    /// Print papers need to be cited
    void print_citation();
    /// Print amplitudes summary
    void print_amp_summary(const std::string& name,
                           const std::vector<std::pair<std::vector<size_t>, double>>& list,
                           const double& norm, const size_t& number_nonzero);
    /// Print intruder analysis
    void print_intruder(const std::string& name,
                        const std::vector<std::pair<std::vector<size_t>, double>>& list);
    /// Check the max and norm of density
    void check_density(BlockedTensor& D, const std::string& name);
    /// Print the summary of 2- and 3-body density cumulant
    void print_cumulant_summary();
};
} // namespace forte

#endif // DSRG_MRPT_H
