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

#ifndef _dsrg_mrpt2_h_
#define _dsrg_mrpt2_h_

#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "ambit/blocked_tensor.h"

#include "../mini-boost/boost/assign.hpp"
#include "../integrals/integrals.h"
#include "../reference.h"
#include "../helpers.h"
#include "../blockedtensorfactory.h"
#include "../mrdsrg-helper/dsrg_time.h"
#include "../mrdsrg-helper/dsrg_source.h"
#include "../stl_bitset_determinant.h"
#include "../fci/fci_integrals.h"
#include "master_mrdsrg.h"

using namespace ambit;
namespace psi {
namespace forte {

class DSRG_MRPT2 : public MASTER_DSRG {
  public:
    /**
     * DSRG_MRPT2 Constructor
     * @param reference The reference object of FORTE
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    DSRG_MRPT2(Reference reference, SharedWavefunction ref_wfn, Options& options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    virtual ~DSRG_MRPT2();

    /// Compute the DSRG-MRPT2 energy
    virtual double compute_energy();

    /// Compute effective Hamiltonian
    virtual std::shared_ptr<FCIIntegrals> compute_Heff();

    /// Compute the DSRG-MRPT2 energy with relaxed reference (once)
    double compute_energy_relaxed();

    /// Compute the multi-state DSRG-MRPT2 energies
    double compute_energy_multi_state();

    /// Set CASCI eigen values and eigen vectors for state averaging
    void set_eigens(const std::vector<std::vector<std::pair<SharedVector, double>>>& eigens) {
        eigens_ = eigens;
    }

    /// Set determinants in the model space
    void set_p_spaces(const std::vector<std::vector<psi::forte::STLBitsetDeterminant>>& p_spaces) {
        p_spaces_ = p_spaces;
    }

    /// Set unitary matrix (in active space) from original to semicanonical
    void set_Uactv(ambit::Tensor& Ua, ambit::Tensor& Ub) {
        Uactv_ = BTF_->build(tensor_type_, "Uactv", spin_cases({"aa"}));
        Uactv_.block("aa")("pq") = Ua("pq");
        Uactv_.block("AA")("pq") = Ub("pq");
    }

    /// Ignore semi-canonical testing in DSRG-MRPT2
    void set_ignore_semicanonical(bool ignore) { ignore_semicanonical_ = ignore; }

    /// Set active active occupied MOs (relative to active)
    void set_actv_occ(std::vector<size_t> actv_occ) {
        actv_occ_mos_ = std::vector<size_t>(actv_occ);
    }
    /// Set active active unoccupied MOs (relative to active)
    void set_actv_uocc(std::vector<size_t> actv_uocc) {
        actv_uocc_mos_ = std::vector<size_t>(actv_uocc);
    }

    /// Compute de-normal-ordered amplitudes and return the scalar term
    double Tamp_deGNO();

    /// Return a BlockedTensor of T1 amplitudes
    ambit::BlockedTensor get_T1(const std::vector<std::string>& blocks);
    ambit::BlockedTensor get_T1() { return T1_; }
    /// Return a BlockedTensor of de-normal-ordered T1 amplitudes
    ambit::BlockedTensor get_T1deGNO(const std::vector<std::string>& blocks);
    ambit::BlockedTensor get_T1deGNO() { return T1eff_; }
    /// Return a BlockedTensor of T2 amplitudes
    ambit::BlockedTensor get_T2(const std::vector<std::string>& blocks);
    ambit::BlockedTensor get_T2() { return T2_; }

    /// Rotate orbital basis for amplitudes according to unitary matrix U
    /// @param U unitary matrix from FCI_MO (INCLUDES frozen orbitals)
    void rotate_amp(SharedMatrix Ua, SharedMatrix Ub, const bool& transpose = false,
                    const bool& t1eff = false);

  protected:
    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();
    /// Print a summary of the options
    void print_summary();
    //    /// Print levels
    //    int print_;

    //    /// Do multi-state computation?
    //    bool multi_state_;
    /// CASCI eigen values and eigen vectors for state averaging
    std::vector<std::vector<std::pair<SharedVector, double>>> eigens_;
    /// Determinants with different symmetries in the model space
    std::vector<std::vector<psi::forte::STLBitsetDeterminant>> p_spaces_;

    //    /// The reference object
    //    Reference reference_;
    //    /// The energy of the reference
    //    double Eref_;
    //    /// The frozen-core energy
    //    double Efrzc_;

    //    /// The molecular integrals required by MethodBase
    //    std::shared_ptr<ForteIntegrals> ints_;

    //    /// MO space info
    //    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    //    /// List of alpha core MOs
    //    std::vector<size_t> core_mos_;
    //    /// List of alpha active MOs
    //    std::vector<size_t> actv_mos_;
    //    /// List of alpha virtual MOs
    //    std::vector<size_t> virt_mos_;
    //    /// List of beta core MOs
    //    std::vector<size_t> core_mos_;
    //    /// List of beta active MOs
    //    std::vector<size_t> actv_mos_;
    //    /// List of beta virtual MOs
    //    std::vector<size_t> virt_mos_;

    /// List of active active occupied MOs (relative to active)
    std::vector<size_t> actv_occ_mos_;
    /// List of active active unoccupied MOs (relative to active)
    std::vector<size_t> actv_uocc_mos_;

    //    /// Alpha core label
    //    std::string acore_label_;
    //    /// Alpha active label
    //    std::string aactv_label_;
    //    /// Alpha virtual label
    //    std::string avirt_label_;
    //    /// Beta core label
    //    std::string bcore_label_;
    //    /// Beta active label
    //    std::string bactv_label_;
    //    /// Beta virtual label
    //    std::string bvirt_label_;

    //    /// Map from space label to list of MOs
    //    std::map<char, std::vector<size_t>> label_to_spacemo_;

    /// Fill up two-electron integrals
    void build_ints();
    /// Fill up density matrix and density cumulants
    void build_density();
    /// Build Fock matrix and diagonal Fock matrix elements
    void build_fock();
    /// Fill up one-electron integrals from FORTE integral class
    void build_oei();
    /// Build effective OEI: hbar_{pq} = h_{pq} + sum_{m} V_{pm,qm}
    void build_eff_oei();

    /// Are orbitals semi-canonicalized?
    bool semi_canonical_;
    /// Check if orbitals are semi-canonicalized
    bool check_semicanonical();
    /// Ignore semi-canonical testing
    bool ignore_semicanonical_ = false;
    /// Diagonal elements of Fock matrices
    std::vector<double> Fa_;
    std::vector<double> Fb_;

    // => DSRG related <= //

    //    /// The flow parameter
    //    double s_;
    //    /// Source operator
    //    std::string source_;
    //    /// The dsrg source operator
    //    std::shared_ptr<DSRG_SOURCE> dsrg_source_;
    //    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    //    double taylor_threshold_;

    /// Renormalize Fock matrix
    void renormalize_F();
    /// Renormalize two-electron integrals
    void renormalize_V();

    // => Tensors <= //

    //    /// Kevin's Tensor Wrapper
    //    std::shared_ptr<BlockedTensorFactory> BTF_;
    //    /// Tensor type for AMBIT
    //    TensorType tensor_type_;

    //    /// One-particle density matrix
    //    ambit::BlockedTensor Gamma1_;
    //    /// One-hole density matrix
    //    ambit::BlockedTensor Eta1_;
    //    /// Two-body denisty cumulant
    //    ambit::BlockedTensor Lambda2_;
    //    /// Three-body density cumulant
    //    ambit::BlockedTensor Lambda3_;

    /// One-eletron integral
    ambit::BlockedTensor Hoei_;
    /// Generalized Fock matrix (bare or renormalized)
    ambit::BlockedTensor F_;
    /// Two-electron integral (bare or renormalized)
    ambit::BlockedTensor V_;
    /// Single excitation amplitude
    ambit::BlockedTensor T1_;
    /// Effective single excitation amplitudes resulting from de-normal ordering
    ambit::BlockedTensor T1eff_;
    /// Double excitation amplitude
    ambit::BlockedTensor T2_;
    /// One-body transformed Hamiltonian (active only)
    ambit::BlockedTensor Hbar1_;
    /// Two-body transformed Hamiltonian (active only)
    ambit::BlockedTensor Hbar2_;
    /// Three-body transformed Hamiltonian (active only)
    ambit::BlockedTensor Hbar3_;

    /// Unitary matrix to block diagonal Fock
    ambit::BlockedTensor U_;
    /// Active orbital rotation from semicanonicalizor (set from outside)
    ambit::BlockedTensor Uactv_;

    // => Amplitude <= //

    /// Compute T2 amplitudes
    void compute_t2();
    /// Check T2 and store large amplitudes
    void check_t2();
    /// Norm of T2
    double T2norm_;
    /// Max (with sign) of T2
    double T2max_;

    /// Compute T1 amplitudes
    void compute_t1();
    /// Check T1 and store large amplitudes
    void check_t1();
    /// Norm of T1
    double T1norm_;
    /// Max (with sign) of T1
    double T1max_;

    /// Include internal amplitude?
    bool internal_amp_;
    /// Include which part of internal amplitudes?
    std::string internal_amp_select_;

    //    /// Number of amplitudes will be printed in amplitude summary
    //    int ntamp_;
    /// Print amplitudes summary
    void print_amp_summary(const std::string& name,
                           const std::vector<std::pair<std::vector<size_t>, double>>& list,
                           const double& norm, const size_t& number_nonzero);

    //    /// Threshold for amplitudes considered as intruders
    //    double intruder_tamp_;
    /// List of large amplitudes
    std::vector<std::pair<std::vector<size_t>, double>> lt1a_;
    std::vector<std::pair<std::vector<size_t>, double>> lt1b_;
    std::vector<std::pair<std::vector<size_t>, double>> lt2aa_;
    std::vector<std::pair<std::vector<size_t>, double>> lt2ab_;
    std::vector<std::pair<std::vector<size_t>, double>> lt2bb_;
    /// Print intruder analysis
    void print_intruder(const std::string& name,
                        const std::vector<std::pair<std::vector<size_t>, double>>& list);

    // => Energy terms <= //

    /// Compute reference energy
    double compute_ref();
    /// Compute DSRG-PT2 correlation energy - Group of functions to calculate
    /// individual pieces of energy
    double E_FT1();
    double E_VT1();
    double E_FT2();
    double E_VT2_2();
    double E_VT2_4PP();
    double E_VT2_4HH();
    double E_VT2_4PH();
    double E_VT2_6();

    // => Dipole related <= //

    /// Compute DSRG transformed dipole integrals
    void compute_pt2_dm();
    /// Compute DSRG transformed dipole integrals for a given direction
    void compute_pt2_dm_helper(BlockedTensor& M, double& Mbar0, BlockedTensor& Mbar1,
                               BlockedTensor& Mbar2);

    // => Reference relaxation <= //

    //    /// Relaxation type
    //    std::string relax_ref_;

    //    /// Timings for computing the commutators
    //    DSRG_TIME dsrg_time_;

    /// Compute zero-body Hbar truncated to 2nd-order
    double Hbar0_;
    /// Compute one-body term of commutator [H1, T1]
    void H1_T1_C1aa(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H1, T2]
    void H1_T2_C1aa(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T1]
    void H2_T1_C1aa(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T2]
    void H2_T2_C1aa(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);

    /// Compute two-body term of commutator [H2, T1]
    void H2_T1_C2aaaa(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H1, T2]
    void H1_T2_C2aaaa(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, T2]
    void H2_T2_C2aaaa(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);

    //    /// Compute zero-body term of commutator [H1, T1]
    //    void H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0);
    //    /// Compute zero-body term of commutator [H1, T2]
    //    void H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0);
    //    /// Compute zero-body term of commutator [H2, T1]
    //    void H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0);
    //    /// Compute zero-body term of commutator [H2, T2]
    //    void H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0);

    //    /// Compute one-body term of commutator [H1, T1]
    //    void H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor&
    //    C1);
    //    /// Compute one-body term of commutator [H1, T2]
    //    void H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor&
    //    C1);
    //    /// Compute one-body term of commutator [H2, T1]
    //    void H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor&
    //    C1);
    //    /// Compute one-body term of commutator [H2, T2]
    //    void H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor&
    //    C1);

    //    /// Compute two-body term of commutator [H2, T1]
    //    void H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor&
    //    C2);
    //    /// Compute two-body term of commutator [H1, T2]
    //    void H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor&
    //    C2);
    //    /// Compute two-body term of commutator [H2, T2]
    //    void H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor&
    //    C2);

    /// Compute three-body term of commutator [H2, T2]
    void H2_T2_C3aaaaaa(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                        BlockedTensor& C3);

    /// Transfer integrals for FCI
    void transfer_integrals();
    /// Diagonalize the diagonal blocks of the Fock matrix
    std::vector<std::vector<double>> diagonalize_Fock_diagblocks(BlockedTensor& U);
    /// Separate an 2D ambit::Tensor according to its irrep
    ambit::Tensor separate_tensor(ambit::Tensor& tens, const Dimension& irrep, const int& h);
    /// Combine a separated 2D ambit::Tensor
    void combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h, const Dimension& irrep,
                        const int& h);

    // => Multi-state energy <= //

    /// Compute multi-state energy in the state-average way
    std::vector<std::vector<double>> compute_energy_sa();
    /// Compute multi-state energy in the MS/XMS way
    std::vector<std::vector<double>> compute_energy_xms();
    /// XMS rotation for the reference states
    SharedMatrix xms_rotation(std::shared_ptr<FCIIntegrals> fci_ints,
                              std::vector<STLBitsetDeterminant>& p_space, SharedMatrix civecs,
                              const int& irrep);

    /// Build effective singles: T_{ia} -= T_{iu,av} * Gamma_{vu}
    void build_T1eff_deGNO();

    /// Compute density cumulants
    void compute_cumulants(std::shared_ptr<FCIIntegrals> fci_ints,
                           std::vector<psi::forte::STLBitsetDeterminant>& p_space,
                           SharedMatrix evecs, const int& root1, const int& root2,
                           const int& irrep);
    /// Compute denisty matrices and puts in Gamma1_, Lambda2_, and Lambda3_
    void compute_densities(std::shared_ptr<FCIIntegrals> fci_ints,
                           std::vector<STLBitsetDeterminant>& p_space, SharedMatrix evecs,
                           const int& root1, const int& root2, const int& irrep);

    /// Compute MS coupling <M|H|N>
    double compute_ms_1st_coupling(const std::string& name);
    /// Compute MS coupling <M|HT|N>
    double compute_ms_2nd_coupling(const std::string& name);

    /// Rotate RDMs computed by p_spaces_ so that they are in the same basis as amplitudes
    void rotate_1rdm(std::vector<double>& opdm_a, std::vector<double>& opdm_b);
    void rotate_2rdm(std::vector<double>& tpdm_aa, std::vector<double>& tpdm_ab,
                     std::vector<double>& tpdm_bb);
    void rotate_3rdm(std::vector<double>& tpdm_aaa, std::vector<double>& tpdm_aab,
                     std::vector<double>& tpdm_abb, std::vector<double>& tpdm_bbb);
};
}
} // End Namespaces

#endif // _dsrg_mrpt2_h_
