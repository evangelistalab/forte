/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "master_mrdsrg.h"

namespace forte {

class THREE_DSRG_MRPT2 : public MASTER_DSRG {
  public:
    /**
     * @brief THREE_DSRG_MRPT2
     * @param rdms          the RDMs for the state we are computing
     * @param scf_info      information about orbitals
     * @param options       a Forte options object
     * @param ints          integrals
     * @param mo_space_info information about orbital spaces
     */
    THREE_DSRG_MRPT2(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    virtual ~THREE_DSRG_MRPT2();

    /// Compute the DSRG-MRPT2 energy
    virtual double compute_energy();

    /// Compute second-order effective Hamiltonian couplings
    /// <M|H + HA(N)|N> = Heff1 * TrD1 + Heff2 * TrD2 + Heff3 * TrD3 if CAS
    virtual void compute_Heff_2nd_coupling(double& H0, ambit::Tensor& H1a, ambit::Tensor& H1b,
                                           ambit::Tensor& H2aa, ambit::Tensor& H2ab,
                                           ambit::Tensor& H2bb, ambit::Tensor& H3aaa,
                                           ambit::Tensor& H3aab, ambit::Tensor& H3abb,
                                           ambit::Tensor& H3bbb);

    /// Return de-normal-ordered T1 amplitudes
    virtual ambit::BlockedTensor get_T1deGNO(double& T0deGNO);

    /// Return T2 amplitudes
    virtual ambit::BlockedTensor get_T2(const std::vector<std::string>& blocks);

    //    /// Compute de-normal-ordered amplitudes and return the scalar term
    //    double Tamp_deGNO();

    //    /// Return a BlockedTensor of T1 amplitudes
    //    ambit::BlockedTensor get_T1(const std::vector<std::string>& blocks);
    //    ambit::BlockedTensor get_T1() { return T1_; }
    //    /// Return a BlockedTensor of de-normal-ordered T1 amplitudes
    //    ambit::BlockedTensor get_T1deGNO(const std::vector<std::string>& blocks);
    //    ambit::BlockedTensor get_T1deGNO() { return T1eff_; }
    //    /// Return a BlockedTensor of T2 amplitudes
    //    ambit::BlockedTensor get_T2(const std::vector<std::string>& blocks);
    //    ambit::BlockedTensor get_T2() { return T2_; }

    /// Rotate orbital basis for amplitudes according to unitary matrix U
    /// @param U unitary matrix from FCI_MO (INCLUDES frozen orbitals)
    void rotate_amp(std::shared_ptr<psi::Matrix> Ua, std::shared_ptr<psi::Matrix> Ub,
                    const bool& transpose = false, const bool& t1eff = false);

    void set_Ufull(std::shared_ptr<psi::Matrix>& Ua, std::shared_ptr<psi::Matrix>& Ub);

  protected:
    // => Class data <= //

    /// Include internal amplitudes or not
    bool internal_amp_;
    /// Include which part of internal amplitudes?
    std::string internal_amp_select_;

    /// The type of SCF reference
    std::string ref_type_;
    /// The number of corrleated MO
    size_t ncmo_ = 0;
    /// The number of auxiliary/cholesky basis functions
    size_t nthree_ = 0;

    /// The number of correlated orbitals per irrep (excluding frozen core and
    /// virtuals)
    psi::Dimension ncmopi_;
    /// The number of restricted doubly occupied orbitals per irrep (core)
    psi::Dimension rdoccpi_;
    /// The number of active orbitals per irrep (active)
    psi::Dimension actvpi_;
    /// The number of restricted unoccupied orbitals per irrep (virtual)
    psi::Dimension ruoccpi_;

    /// Number of core orbitals
    size_t ncore_;
    /// Number of active orbitals
    size_t nactive_;
    /// Number of virutal orbitals
    size_t nvirtual_;

    /// List of eigenvalues for fock alpha
    std::vector<double> Fa_;
    /// List of eigenvalues for fock beta
    std::vector<double> Fb_;

    /// Semicanonical Transformation matrices
    std::shared_ptr<psi::Matrix> Ua_full_;
    std::shared_ptr<psi::Matrix> Ub_full_;

    // => Tensors <= //
    ambit::BlockedTensor H_;
    ambit::BlockedTensor F_;
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

    /// A vector of strings that avoids creating ccvv indices
    std::vector<std::string> no_hhpp_;

    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Compute frozen natural orbitals
    /// Called in the destructor
    void cleanup();
    void print_options_summary();

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

    void de_normal_order();

    /// Form Hbar for reference relaxation
    void form_Hbar();

    /// Compute Hbar1 from core contraction when doing DiskDF
    void compute_Hbar1C_diskDF(ambit::BlockedTensor& Hbar1, bool scaleV = true);
    /// Compute Hbar1 from virtual contraction when doing DiskDF
    void compute_Hbar1V_diskDF(ambit::BlockedTensor& Hbar1, bool scaleV = true);

    /// Print detailed timings
    bool detail_time_ = false;

    //    /// This function will remove the indices that do not have at least one
    //    /// active index
    //    /// This function generates all possible MO spaces and spin components
    //    /// Param:  std::string is the lables - "cav"
    //    /// Will take a string like cav and generate all possible combinations of
    //    /// this for a four character string
    //    std::shared_ptr<BlockedTensorFactory> BTF_;

    /// Integral type (DF, CD, DISKDF)
    IntegralType integral_type_;

    /// Effective alpha one-electron integrals (used in denormal ordering)
    std::vector<double> aone_eff_;
    /// Effective beta one-electron integrals (used in denormal ordering)
    std::vector<double> bone_eff_;

    /// Are orbitals semi-canonicalized?
    bool semi_canonical_;
    /// Unitary matrix to block diagonal Fock
    ambit::BlockedTensor U_;

    // => Dipole related <= //

    /// Compute DSRG transformed dipole integrals
    void print_dm_pt2();
    /// Compute DSRG transformed dipole integrals for a given direction
    void compute_dm1d_pt2(BlockedTensor& M, double& Mbar0, BlockedTensor& Mbar1,
                          BlockedTensor& Mbar2);

  private:
    // maximum number of threads
    int num_threads_;
    /// Do we have OpenMP?
    static bool have_omp_;
    /// Do we have MPI (actually use GA)
    static bool have_mpi_;
};
} // namespace forte

#endif // _three_dsrg_mrpt2_h_
