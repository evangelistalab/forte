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

#ifndef _three_dsrg_mrpt2_h_
#define _three_dsrg_mrpt2_h_

#include <fstream>
#include <string>
#include <vector>
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "ambit/blocked_tensor.h"

#include "../integrals/integrals.h"
#include "../reference.h"
#include "../blockedtensorfactory.h"
#include "../mrdsrg-helper/dsrg_source.h"
#include "../mrdsrg-helper/dsrg_time.h"

namespace psi {
namespace forte {

class THREE_DSRG_MRPT2 : public Wavefunction {
  public:
    /**
     * THREE_DSRG_MRPT2 Constructor
     * @param reference The reference object of FORTE
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    THREE_DSRG_MRPT2(Reference reference, SharedWavefunction ref_wfn, Options& options,
                     std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~THREE_DSRG_MRPT2();

    /// Compute the DSRG-MRPT2 energy
    double compute_energy();

    /// Allow the reference to relax
    void relax_reference_once();

    /// Ignore semi-canonical testing in DSRG-MRPT2
    void ignore_semicanonical(bool ignore) { ignore_semicanonical_ = ignore; }

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
    // => Class data <= //

    /// The reference object
    Reference reference_;

    /// The energy of the reference
    double Eref_;

    /// The frozen-core energy
    double frozen_core_energy_;

    /// Include internal amplitudes or not
    bool internal_amp_;

    /// The molecular integrals required by MethodBase
    std::shared_ptr<ForteIntegrals> ints_;
    /// The type of SCF reference
    std::string ref_type_;
    /// The number of corrleated MO
    size_t ncmo_ = 0;
    /// The number of auxiliary/cholesky basis functions
    size_t nthree_ = 0;

    /// The number of correlated orbitals per irrep (excluding frozen core and
    /// virtuals)
    Dimension ncmopi_;
    /// The number of restricted doubly occupied orbitals per irrep (core)
    Dimension rdoccpi_;
    /// The number of active orbitals per irrep (active)
    Dimension actvpi_;
    /// The number of restricted unoccupied orbitals per irrep (virtual)
    Dimension ruoccpi_;

    /// List of alpha core MOs
    std::vector<size_t> acore_mos_;
    size_t core_;
    /// List of alpha active MOs
    std::vector<size_t> aactv_mos_;
    size_t active_;
    /// List of alpha virtual MOs
    std::vector<size_t> avirt_mos_;
    size_t virtual_;

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

    /// List of eigenvalues for fock alpha
    std::vector<double> Fa_;
    /// List of eigenvalues for fock beta
    std::vector<double> Fb_;

    /// Map from all the MOs to the alpha core
    std::map<size_t, size_t> mos_to_acore_;
    /// Map from all the MOs to the alpha active
    std::map<size_t, size_t> mos_to_aactv_;
    /// Map from all the MOs to the alpha virtual
    std::map<size_t, size_t> mos_to_avirt_;

    /// Map from all the MOs to the beta core
    std::map<size_t, size_t> mos_to_bcore_;
    /// Map from all the MOs to the beta active
    std::map<size_t, size_t> mos_to_bactv_;
    /// Map from all the MOs to the beta virtual
    std::map<size_t, size_t> mos_to_bvirt_;

    /// The flow parameter
    double s_;
    /// Source operator
    std::string source_;
    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;
    /// The dsrg source operator
    std::shared_ptr<DSRG_SOURCE> dsrg_source_;

    // => Tensors <= //
    ambit::TensorType tensor_type_;
    ambit::BlockedTensor H_;
    ambit::BlockedTensor F_;
    ambit::BlockedTensor F_no_renorm_;
    ambit::BlockedTensor Gamma1_;
    ambit::BlockedTensor Eta1_;
    ambit::BlockedTensor Lambda2_;
    ambit::BlockedTensor Delta1_;
    ambit::BlockedTensor RDelta1_;
    ambit::BlockedTensor T1_;
    ambit::BlockedTensor T1eff_;
    // one-particle exponential for renormalized Fock matrix
    // These three are defined as member variables, but if integrals use DiskDF,
    // these are not to be computed for the entire code
    ambit::BlockedTensor RExp1_;
    ambit::BlockedTensor T2_;
    ambit::BlockedTensor V_;
    ambit::BlockedTensor ThreeIntegral_;
    ambit::BlockedTensor H0_;
    ambit::BlockedTensor Hbar1_;
    ambit::BlockedTensor Hbar2_;

    /// A vector of strings that avoids creating ccvv indices
    std::vector<std::string> no_hhpp_;

    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Compute frozen natural orbitals
    /// Called in the destructor
    void cleanup();
    void print_summary();

    double renormalized_denominator(double D);

    /// compute the minimal amount of T2 for each term
    /// The spaces correspond to all the blocks you want to use
    ambit::BlockedTensor compute_T2_minimal(const std::vector<std::string>& spaces);
    /// compute ASTEI from DF/CD integrals
    /// function will take the spaces for V and use that to create the blocks
    /// for B
    ambit::BlockedTensor compute_B_minimal(const std::vector<std::string>& Vspaces);

    /// Computes the t1 amplitudes for three different cases of spin (alpha all,
    /// beta all, and alpha beta)
    void compute_t1();
    /// If DF or Cholesky, this function is not used
    void compute_t2();
    void check_t1();
    double T1norm_;
    double T1max_;

    // Compute V and maybe renormalize
    ambit::BlockedTensor compute_V_minimal(const std::vector<std::string>&,
                                           bool renormalize = true);
    /// Renormalize Fock matrix and two-electron integral
    void renormalize_F();
    void renormalize_V();
    double renormalized_exp(double D) { return std::exp(-s_ * std::pow(D, 2.0)); }

    /// Compute DSRG-PT2 correlation energy - Group of functions to calculate
    /// individual pieces of energy
    double compute_ref();
    double E_FT1();
    double E_VT1();
    double E_FT2();
    double E_VT2_2();
    /// Compute hhva and acvv terms
    double E_VT2_2_one_active();
    /// Different algorithms for handling ccvv term
    /// Core -> builds everything in core.  Probably fastest
    double E_VT2_2_core();
    /// ambit -> Uses ambit library to perform contractions
    double E_VT2_2_ambit();
    /// fly_open-> Code Kevin wrote at first with open mp threading
    double E_VT2_2_fly_openmp();
    /// batch_core Reads only M*N (where M and N are size of batches)
    double E_VT2_2_batch_core();
    /// batch_core Reads only E*F (where M and N are size of virtual batches)
    double E_VT2_2_batch_virtual();
    /// Core MPI parallel algorithms (MPI -> distriubuted B)
    /// ga->distrubuted B with Global Arrays API
    /// rep->Broadcast B (debug version)
    double E_VT2_2_batch_core_mpi();
    double E_VT2_2_batch_core_ga();
    double E_VT2_2_batch_core_rep();
    double E_VT2_2_batch_virtual_mpi();
    double E_VT2_2_batch_virtual_ga();
    double E_VT2_2_batch_virtual_rep();
    double E_VT2_2_AO_Slow();
    double E_VT2_4PP();
    double E_VT2_4HH();
    double E_VT2_4PH();
    double E_VT2_6();

    /// Timings for computing the commutators
    DSRG_TIME dsrg_time_;

    /// Compute zero-body Hbar truncated to 2nd-order
    double Hbar0_ = 0.0;

    /// Compute one-body term of commutator [H1, T1]
    void H1_T1_C1(ambit::BlockedTensor& H1, ambit::BlockedTensor& T1, const double& alpha,
                  ambit::BlockedTensor& C1);
    /// Compute one-body term of commutator [H1, T2]
    void H1_T2_C1(ambit::BlockedTensor& H1, ambit::BlockedTensor& T2, const double& alpha,
                  ambit::BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T1]
    void H2_T1_C1(ambit::BlockedTensor& H2, ambit::BlockedTensor& T1, const double& alpha,
                  ambit::BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T2]
    void H2_T2_C1(ambit::BlockedTensor& H2, ambit::BlockedTensor& T2, const double& alpha,
                  ambit::BlockedTensor& C1);

    /// Compute two-body term of commutator [H2, T1]
    void H2_T1_C2(ambit::BlockedTensor& H2, ambit::BlockedTensor& T1, const double& alpha,
                  ambit::BlockedTensor& C2);
    /// Compute two-body term of commutator [H1, T2]
    void H1_T2_C2(ambit::BlockedTensor& H1, ambit::BlockedTensor& T2, const double& alpha,
                  ambit::BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, T2]
    void H2_T2_C2(ambit::BlockedTensor& H2, ambit::BlockedTensor& T2, const double& alpha,
                  ambit::BlockedTensor& C2);

    void de_normal_order();

    std::vector<double> relaxed_energy();

    /// Print levels
    int print_;
    /// Print detailed timings
    bool detail_time_ = false;

    /// This function will remove the indices that do not have at least one
    /// active index
    /// This function generates all possible MO spaces and spin components
    /// Param:  std::string is the lables - "cav"
    /// Will take a string like cav and generate all possible combinations of
    /// this
    /// for a four character string
    std::shared_ptr<BlockedTensorFactory> BTF_;

    /// The MOSpace object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    IntegralType integral_type_;

    /// Effective alpha one-electron integrals (used in denormal ordering)
    std::vector<double> aone_eff_;
    /// Effective beta one-electron integrals (used in denormal ordering)
    std::vector<double> bone_eff_;

    /// Are orbitals semi-canonicalized?
    bool semi_canonical_;
    /// Check if orbitals are semi-canonicalized
    bool check_semicanonical();
    /// Ignore semi-canonical testing
    bool ignore_semicanonical_ = false;
    /// Unitary matrix to block diagonal Fock
    ambit::BlockedTensor U_;
    /// Diagonalize the diagonal blocks of the Fock matrix
    std::vector<std::vector<double>> diagonalize_Fock_diagblocks(ambit::BlockedTensor& U);
    /// Separate an 2D ambit::Tensor according to its irrep
    ambit::Tensor separate_tensor(ambit::Tensor& tens, const Dimension& irrep, const int& h);
    /// Combine a separated 2D ambit::Tensor
    void combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h, const Dimension& irrep,
                        const int& h);

  private:
    // maximum number of threads
    int num_threads_;
    /// Do we have OpenMP?
    static bool have_omp_;
    /// Do we have MPI (actually use GA)
    static bool have_mpi_;
};
}
} // End Namespaces

#endif // _three_dsrg_mrpt2_h_
