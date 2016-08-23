/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#ifndef _dsrg_mrpt2_h_
#define _dsrg_mrpt2_h_

#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include "mini-boost/boost/assign.hpp"

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>
#include <libmints/pointgrp.h>

#include "integrals.h"
#include "ambit/blocked_tensor.h"
#include "reference.h"
#include "helpers.h"
#include "blockedtensorfactory.h"
#include "dsrg_time.h"
#include "dsrg_source.h"
#include "stl_bitset_determinant.h"

using namespace ambit;
namespace psi{ namespace forte{

class DSRG_MRPT2 : public Wavefunction
{
public:
    /**
     * DSRG_MRPT2 Constructor
     * @param reference The reference object of FORTE
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    DSRG_MRPT2(Reference reference, SharedWavefunction ref_wfn, Options &options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~DSRG_MRPT2();

    /// Compute the DSRG-MRPT2 energy
    double compute_energy();

    /// Compute the DSRG-MRPT2 energy with relaxed reference (once)
    double compute_energy_relaxed();

    /// Compute the multi-state DSRG-MRPT2 energies
    double compute_energy_multi_state();

    /// Set CASCI eigen values and eigen vectors for state averaging
    void set_eigens(std::vector<std::vector<std::pair<SharedVector,double>>> eigens) {eigens_ = eigens;}

    /// Set determinants in the model space
    void set_p_space(std::vector<psi::forte::STLBitsetDeterminant> p_space) {p_space_ = p_space;}

    /// Ignore semi-canonical testing in DSRG-MRPT2
    void ignore_semicanonical(bool ignore) {ignore_semicanonical_ = ignore;}

    /// Set active active occupied MOs (relative to active)
    void set_actv_occ(std::vector<size_t> actv_occ) {actv_occ_mos_ = std::vector<size_t>(actv_occ);}
    /// Set active active unoccupied MOs (relative to active)
    void set_actv_uocc(std::vector<size_t> actv_uocc) {actv_uocc_mos_ = std::vector<size_t>(actv_uocc);}

protected:
    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();
    /// Print a summary of the options
    void print_summary();
    /// Print levels
    int print_;

    /// Multi-state or not
    bool multi_state_;
    /// CASCI eigen values and eigen vectors for state averaging
    std::vector<std::vector<std::pair<SharedVector,double>>> eigens_;
    /// Determinants in the model space
    std::vector<psi::forte::STLBitsetDeterminant> p_space_;

    /// The reference object
    Reference reference_;
    /// The energy of the reference
    double Eref_;
    /// The frozen-core energy
    double frozen_core_energy_;

    /// The molecular integrals required by MethodBase
    std::shared_ptr<ForteIntegrals>  ints_;

    /// MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// List of alpha core MOs
    std::vector<size_t> acore_mos_;
    /// List of alpha active MOs
    std::vector<size_t> aactv_mos_;
    /// List of alpha virtual MOs
    std::vector<size_t> avirt_mos_;
    /// List of beta core MOs
    std::vector<size_t> bcore_mos_;
    /// List of beta active MOs
    std::vector<size_t> bactv_mos_;
    /// List of beta virtual MOs
    std::vector<size_t> bvirt_mos_;

    /// List of active active occupied MOs (relative to active)
    std::vector<size_t> actv_occ_mos_;
    /// List of active active unoccupied MOs (relative to active)
    std::vector<size_t> actv_uocc_mos_;

    /// Alpha core label
    std::string acore_label_;
    /// Alpha active label
    std::string aactv_label_;
    /// Alpha virtual label
    std::string avirt_label_;
    /// Beta core label
    std::string bcore_label_;
    /// Beta active label
    std::string bactv_label_;
    /// Beta virtual label
    std::string bvirt_label_;

    /// Map from space label to list of MOs
    std::map<char, std::vector<size_t>> label_to_spacemo_;

    /// Fill up two-electron integrals
    void build_ints();
    /// Fill up density matrix and density cumulants
    void build_density();
    /// Build Fock matrix and diagonal Fock matrix elements
    void build_fock();

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

    /// The flow parameter
    double s_;
    /// Source operator
    std::string source_;
    /// The dsrg source operator
    std::shared_ptr<DSRG_SOURCE> dsrg_source_;
    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;

    /// Renormalize Fock matrix
    void renormalize_F();
    /// Renormalize two-electron integrals
    void renormalize_V();

    // => Tensors <= //

    /// Kevin's Tensor Wrapper
    std::shared_ptr<BlockedTensorFactory> BTF_;
    /// Tensor type for AMBIT
    TensorType tensor_type_;

    /// One-particle density matrix
    ambit::BlockedTensor Gamma1_;
    /// One-hole density matrix
    ambit::BlockedTensor Eta1_;
    /// Two-body denisty cumulant
    ambit::BlockedTensor Lambda2_;
    /// Three-body density cumulant
    ambit::BlockedTensor Lambda3_;

    /// Generalized Fock matrix (bare or renormalized)
    ambit::BlockedTensor F_;
    /// Two-electron integral (bare or renormalized)
    ambit::BlockedTensor V_;
    /// Single excitation amplitude
    ambit::BlockedTensor T1_;
    /// Double excitation amplitude
    ambit::BlockedTensor T2_;
    /// One-body transformed Hamiltonian (active only)
    ambit::BlockedTensor Hbar1_;
    /// Two-body transformed Hamiltonian (active only)
    ambit::BlockedTensor Hbar2_;

    /// Unitary matrix to block diagonal Fock
    ambit::BlockedTensor U_;


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

    /// Number of amplitudes will be printed in amplitude summary
    int ntamp_;
    /// Print amplitudes summary
    void print_amp_summary(const std::string& name,
                           const std::vector<std::pair<std::vector<size_t>, double>>& list, const double &norm,
                           const size_t& number_nonzero);

    /// Threshold for amplitudes considered as intruders
    double intruder_tamp_;
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

    /// Compute DSRG-PT2 correlation energy - Group of functions to calculate individual pieces of energy
    double E_FT1();
    double E_VT1();
    double E_FT2();
    double E_VT2_2();
    double E_VT2_4PP();
    double E_VT2_4HH();
    double E_VT2_4PH();
    double E_VT2_6();


    // => Reference relaxation <= //

    /// Relaxation type
    std::string relax_ref_;

    /// Timings for computing the commutators
    DSRG_TIME dsrg_time_;

    /// Compute zero-body Hbar truncated to 2nd-order
    double Hbar0_;
    /// Compute one-body term of commutator [H1, T1]
    void H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H1, T2]
    void H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T1]
    void H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T2]
    void H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);

    /// Compute two-body term of commutator [H2, T1]
    void H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H1, T2]
    void H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, T2]
    void H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);

    /// Transfer integrals for FCI
    void transfer_integrals();
    /// Diagonalize the diagonal blocks of the Fock matrix
    std::vector<std::vector<double>> diagonalize_Fock_diagblocks(BlockedTensor& U);
    /// Separate an 2D ambit::Tensor according to its irrep
    ambit::Tensor separate_tensor(ambit::Tensor& tens, const Dimension& irrep, const int& h);
    /// Combine a separated 2D ambit::Tensor
    void combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h, const Dimension& irrep, const int& h);
};

}} // End Namespaces

#endif // _dsrg_mrpt2_h_
