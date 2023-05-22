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

#ifndef _dsrg_mrpt2_h_
#define _dsrg_mrpt2_h_

#include <iostream>
#include <fstream>

#include "master_mrdsrg.h"

using namespace ambit;

namespace forte {

class DSRG_MRPT2 : public MASTER_DSRG {
  public:
    /**
     * DSRG_MRPT2 Constructor
     * @param reference The reference object of FORTE
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */
    DSRG_MRPT2(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
               std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
               std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    virtual ~DSRG_MRPT2();

    /// Compute the DSRG-MRPT2 energy
    virtual double compute_energy() override;

    /// Compute the DSRG-MRPT2 gradient
    virtual std::shared_ptr<psi::Matrix> compute_gradient() override;

    /// Compute second-order effective Hamiltonian couplings
    /// <M|H + HA(N)|N> = Heff1 * TrD1 + Heff2 * TrD2 + Heff3 * TrD3 if CAS
    virtual void compute_Heff_2nd_coupling(double& H0, ambit::Tensor& H1a, ambit::Tensor& H1b,
                                           ambit::Tensor& H2aa, ambit::Tensor& H2ab,
                                           ambit::Tensor& H2bb, ambit::Tensor& H3aaa,
                                           ambit::Tensor& H3aab, ambit::Tensor& H3abb,
                                           ambit::Tensor& H3bbb) override;

    /// Return de-normal-ordered T1 amplitudes
    virtual ambit::BlockedTensor get_T1deGNO(double& T0deGNO) override;

    /// Return T2 amplitudes
    virtual ambit::BlockedTensor get_T2(const std::vector<std::string>& blocks) override;
    virtual ambit::BlockedTensor get_T2() override { return T2_; }

    /// Return de-normal-ordered 1-body renormalized 1st-order Hamiltonian
    virtual ambit::BlockedTensor get_RH1deGNO() override;

    /// Return 2-body renormalized 1st-order Hamiltonian
    virtual ambit::BlockedTensor get_RH2() override { return V_; }

    /// Compute one-electron density of DSRG
    /// Important: T1 and T2 are de-normal-ordered!
    ambit::BlockedTensor compute_OE_density(BlockedTensor& T1, BlockedTensor& T2, BlockedTensor& D1,
                                            BlockedTensor& D2, BlockedTensor& D3,
                                            const bool& transition);

    /// Compute the DSRG-MRPT2 energy with relaxed reference (once)
    double compute_energy_relaxed();

    /// Compute the multi-state DSRG-MRPT2 energies
    double compute_energy_multi_state();

    /// Set CASCI eigen values and eigen vectors for state averaging
    void set_eigens(
        const std::vector<std::vector<std::pair<std::shared_ptr<psi::Vector>, double>>>& eigens) {
        eigens_ = eigens;
    }

    /// Set determinants in the model space
    void set_p_spaces(const std::vector<std::vector<forte::Determinant>>& p_spaces) {
        p_spaces_ = p_spaces;
    }

    //    /// Compute de-normal-ordered amplitudes and return the scalar term
    //    double Tamp_deGNO();

    //    /// Return a BlockedTensor of T1 amplitudes
    //    ambit::BlockedTensor get_T1(const std::vector<std::string>& blocks);
    //    ambit::BlockedTensor get_T1() { return T1_; }
    //    /// Return a BlockedTensor of de-normal-ordered T1 amplitudes
    //    ambit::BlockedTensor get_T1deGNOa(const std::vector<std::string>& blocks);
    //    ambit::BlockedTensor get_T1deGNO() { return T1eff_; }
    //    /// Return a BlockedTensor of T2 amplitudes
    //    ambit::BlockedTensor get_T2(const std::vector<std::string>& blocks);
    //    ambit::BlockedTensor get_T2() { return T2_; }

    /// Rotate orbital basis for amplitudes according to unitary matrix U
    /// @param U unitary matrix from FCI_MO (INCLUDES frozen orbitals)
    void rotate_amp(std::shared_ptr<psi::Matrix> Ua, std::shared_ptr<psi::Matrix> Ub,
                    const bool& transpose = false, const bool& t1eff = false);

  protected:
    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();
    /// Print a summary of the options
    void print_options_summary();

    /// CASCI eigen values and eigen vectors for state averaging
    std::vector<std::vector<std::pair<std::shared_ptr<psi::Vector>, double>>> eigens_;
    /// Determinants with different symmetries in the model space
    std::vector<std::vector<forte::Determinant>> p_spaces_;

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
    /// Diagonal elements of Fock matrices
    std::vector<double> Fa_;
    std::vector<double> Fb_;

    // => DSRG related <= //

    /// Renormalize Fock matrix
    void renormalize_F();
    /// Renormalize two-electron integrals
    void renormalize_V();

    // => Tensors <= //

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

    /// Unitary matrix to block diagonal Fock
    ambit::BlockedTensor U_;

    // NOTICE These are essential variables and functions for computing DSRG-MRPT2 gradient.
    // Some variables may be redundant thus need further elimination
    /// Set ambit tensor labels
    void set_ambit_space();
    /**
     * Initialize tensors.
     *
     * Density tensors, one- and two-electron integrals, Fock matrix, DSRG and CI tensors.
     */
    void set_tensor();
    /**
     * Initializing the two-body density Gamma2_.
     *
     * NOTICE: this function shall be revoked in the future.
     */
    void set_density();
    /**
     * Initializing the Fock F.
     *
     * NOTICE: this function shall be revoked in the future.
     */
    void set_active_fock();
    /**
     * Initializing the one-electron integral H.
     *
     * NOTICE: this function shall be revoked in the future.
     */
    void set_h();
    /**
     * Initializing the ERIs V.
     *
     * NOTICE: this function shall be revoked in the future.
     */
    void set_v();
    /// Set CI-relevant integrals
    void set_ci_ints();
    /**
     * Initialize global variables.
     *
     * MO indices list, dimension of different MOs, CI-related variables etc.
     */
    void set_global_variables();
    /// # of MOs
    size_t nmo;
    /// # of core MOs
    size_t ncore;
    /// # of virtual MOs
    size_t nvirt;
    /// # of active MOs
    size_t na;
    /// # of auxiliary orbitals
    size_t naux;
    /// # of irreps
    size_t nirrep;

    /// <[V, T2]> (C_2)^4
    ///     1/4 * V'["klcd"] * T["ijab"] * Gamma["ki"] * Gamma["lj"] * Eta["ac"] * Eta["bd"]
    const bool PT2_TERM = true;
    /// <[V, T2]> C_4 (C_2)^2 PP
    ///     1/8 * V'["cdxy"] * T["uvab"] * Eta["ac"] * Eta["bd"] * Lambda["xyuv"]
    const bool X1_TERM = true;
    /// <[V, T2]> C_4 (C_2)^2 HH
    ///     1/8 * V'["uvkl"] * T["ijxy"] * Gamma["ki"] * Gamma["lj"] * Lambda["xyuv"]
    const bool X2_TERM = true;
    /// <[V, T2]> C_4 (C_2)^2 PH
    ///     V'["vbjx"] * T["ayiu"] * Gamma["ji"] * Eta["ab"] * Lambda["xyuv"]
    const bool X3_TERM = true;
    /// <[V, T2]> C_6 C_2
    ///     1/4 * V'["uviz"] * T["iwxy"] * Lambda["xyzuvw"]
    ///   + 1/4 * V'["waxy"] * T["uvaz"] * Lambda["xyzuvw"]
    const bool X4_TERM = do_cu3_;
    /// <[F, T2]>
    ///     1/2 * F'["ex"] * T["uvey"] * Lambda["xyuv"]
    ///   - 1/2 * F'["vm"] * T["umxy"] * Lambda["xyuv"]
    const bool X5_TERM = true;
    /// <[V, T1]>
    ///     1/2 V'["evxy"] * T["ue"] * Lambda["xyuv"]
    ///   - 1/2 V'["uvmy"] * T["mx"] * Lambda["xyuv"]
    const bool X6_TERM = true;
    /// <[F, T1]>
    ///     F'["bj"] * T["ia"] * Gamma["ji"] * Eta["ab"]
    const bool X7_TERM = true;
    /// If the correlation contribution is considered
    const bool CORRELATION_TERM = true;
    /**
     * Initializing the DSRG-related auxiliary tensors.
     */
    void set_dsrg_tensor();
    /**
     * Write the energy-weighted density matrix into Psi4 Lagrangian_.
     */
    void write_lagrangian();
    /**
     * Write spin_dependent one-RDMs coefficients into Psi4 Da_ and Db_.
     *
     * We assume "Da == Db". This function needs be changed if such constraint is revoked.
     */
    void write_1rdm_spin_dependent();
    /**
     * Write spin_dependent two-RDMs coefficients using IWL.
     *
     * Coefficients in d2aa and d2bb need be multiplied with additional 1/2!
     * Specifically:
     * If you have v_aa as coefficients before 2-RDMs_alpha_alpha, v_bb before
     * 2-RDMs_beta_beta and v_bb before 2-RDMs_alpha_beta, you need to write
     * 0.5 * v_aa, 0.5 * v_bb and v_ab into the IWL file instead of using
     * the original coefficients v_aa, v_bb and v_ab.
     */
    void write_2rdm_spin_dependent();
    /**
     * Write the density terms contracted with (P|Q)^x and (pq|Q)^x, where P and Q are auxiliary
     * basis functions.
     */
    void write_df_rdm();
    /**
     * Backtransform the TPDM.
     */
    void tpdm_backtransform();
    /**
     * Initialize and solve Lagrange multipliers.
     *
     * Sigma: constraint of the one-body DSRG amplitude (T1) definition.
     * Xi:    constraint of the renormalized Fock matrix (F1) definition.
     * Tau:   constraint of the two-body DSRG amplitude (T2) definition.
     * Kappa: constraint of the renormalized ERIs (\tilde{V}) definition.
     * Z:     OPDM, constraint of the CASSCF reference.
     * W:     EWDM, constraint of the orthonormal overlap integral.
     */
    void set_multiplier();
    /**
     * Solve the Linear System Ax=b and yield Z using iterative methods.
     */
    void set_preconditioner(std::vector<double>& D);
    void gmres_solver(std::vector<double>& x_new);
    void solve_linear_iter();
    void z_vector_contraction(std::vector<double>&, std::vector<double>&);
    void pre_contract();
    /**
     * Solve the Linear System Ax=b and yield Z using direct methods.
     */
    void solve_z();
    /**
     * Initialize and solve the multiplier Sigma and Xi.
     *
     * Sigma: constraint of the one-body DSRG amplitude (T1) definition.
     * Solved directly.
     */
    void set_sigma_xi();
    /**
     * Initialize and solve the multiplier Tau.
     *
     * Tau: constraint of the two-body DSRG amplitude (T2) definition.
     * Solved directly.
     */
    void set_tau();
    /**
     * Initialize and solve the multiplier Kappa.
     *
     * Kappa: constraint of the renormalized ERIs (\tilde{V}) definition.
     * Solved directly.
     */
    void set_kappa();
    /**
     * Initialize and solve the multiplier Z (OPDM).
     *
     * Z: OPDM, constraint of the CASSCF reference.
     * The core-core, virtual-virtual blocks and diagonal entries of the active-active
     * blocks are solved directly based on other multipliers. The rest need be solved
     * Through iterative approaches. Currently, we use LAPACK as the solver.
     */
    void set_z();
    /**
     * Setting the b of the Linear System Ax=b.
     * Parameters: preidx, block_dim
     */
    void set_b(int dim, const std::map<string, int>& block_dim);
    /**
     * The diagonal core-core, virtual-virtual blocks
     * and the diagonal entries of the active-active block of the OPDM Z.
     */
    void set_z_diag();
    /**
     * Initialize and solve the multiplier W.
     *
     * W: EWDM, constraint of the orthonormal overlap integral.
     * Solved directly after all other multipliers are solved.
     */
    void set_w();
    // Set MO orbital partition info for solving the z-vector equation
    void set_zvec_moinfo();

    /// Size of determinants
    size_t ndets;
    /// List of core MOs (including frozen orbitals)
    std::vector<size_t> core_all_;
    /// List of active MOs (including frozen orbitals)
    std::vector<size_t> actv_all_;
    /// List of virtual MOs (including frozen orbitals)
    std::vector<size_t> virt_all_;

    // MO orbital partition info for solving the z-vector equation
    int dim;
    std::map<string, int> preidx;
    std::map<string, int> block_dim;

    /// List of relative core MOs
    std::vector<std::pair<unsigned long, unsigned long>> core_mos_relative;
    /// List of relative active MOs
    std::vector<std::pair<unsigned long, unsigned long>> actv_mos_relative;
    /// List of relative virtual MOs
    std::vector<std::pair<unsigned long, unsigned long>> virt_mos_relative;

    /// Dimension of different irreps
    psi::Dimension irrep_vec;
    /// Two-body denisty tensor
    ambit::BlockedTensor Gamma2_;
    // core Hamiltonian
    ambit::BlockedTensor H;
    // DF integrals
    ambit::BlockedTensor B;
    /// Lagrangian tensor
    ambit::BlockedTensor W;
    // DF metric J^(-1/2)
    ambit::BlockedTensor Jm12;
    // two-electron integrals
    ambit::BlockedTensor V;
    // Fock matrix
    ambit::BlockedTensor F;
    /// e^[-s*(Delta1)^2]
    ambit::BlockedTensor Eeps1;
    /// {1-e^[-s*(Delta1)^2]}/(Delta1)
    ambit::BlockedTensor Eeps1_m1;
    /// {1-e^[-s*(Delta1)^2]}/(Delta1)^2
    ambit::BlockedTensor Eeps1_m2;
    /// Delta1_a^i = \varepsilon_i - \varepsilon_a
    ambit::BlockedTensor Delta1;
    /// Delta1 * Gamma1_
    ambit::BlockedTensor DelGam1;
    /// Delta1 * Eeps1
    ambit::BlockedTensor DelEeps1;
    /// Identity matrix
    ambit::BlockedTensor I;
    /// Identity matrix for the CI part
    ambit::Tensor I_ci;
    /// a vector with all entries equal 1
    ambit::BlockedTensor one_vec;

    /*** Lagrange multipliers ***/
    /// multiplier related to T2 amplitudes
    ambit::BlockedTensor Tau2;
    /// multiplier related to modified ERIs
    ambit::BlockedTensor Kappa;
    /// multiplier related to modified one-body quantities
    /// sigma : one-body amplitudes
    /// xi : modified one-body integrals
    ambit::BlockedTensor sigma3_xi3;
    ambit::BlockedTensor sigma2_xi3;
    ambit::BlockedTensor sigma1_xi1_xi2;
    /// multiplier related to orbital response (symmetrized)
    ambit::BlockedTensor Z;
    /// unsymmetrized Z components
    ambit::BlockedTensor temp_z;
    /// multiplier related to normalized CI coefficients
    double Alpha;
    /// multiplier related to CI response
    ambit::Tensor x_ci;

    /// Linear system Ax=b
    std::vector<double> b;
    /// orbital contribution to the b
    ambit::BlockedTensor Z_b;
    /// V["pmqm"], V["pMqM"], V["PmQm"], V["PMQM"]
    ambit::BlockedTensor V_pmqm;
    /// CI coefficients
    ambit::Tensor ci;
    /// c_i < \phi_i | p+ q | \phi_j > x_j
    ambit::BlockedTensor Gamma1_tilde;
    /// c_i < \phi_i | p+ q+ s r | \phi_j > x_j
    ambit::BlockedTensor Gamma2_tilde;

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

    /// Print amplitudes summary
    void print_amp_summary(const std::string& name,
                           const std::vector<std::pair<std::vector<size_t>, double>>& list,
                           const double& norm, const size_t& number_nonzero);

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
    void print_dm_pt2();
    /// Compute DSRG transformed dipole integrals for a given direction
    void compute_dm1d_pt2(BlockedTensor& M, double& Mbar0, BlockedTensor& Mbar1,
                          BlockedTensor& Mbar2);
    /// Compute DSRG transformed dipole integrals for a given direction
    void compute_dm1d_pt2(BlockedTensor& M, double& Mbar0, BlockedTensor& Mbar1,
                          BlockedTensor& Mbar2, BlockedTensor& Mbar3);

    /// Compute DSRG-PT2 correction for unrelaxed density
    BlockedTensor compute_pt2_unrelaxed_opdm();

    // => Reference relaxation <= //

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

    /// Compute three-body term of commutator [H2, T2]
    void H2_T2_C3aaaaaa(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                        BlockedTensor& C3);

    //    /// Transfer integrals for FCI
    //    void transfer_integrals();
    //    /// Diagonalize the diagonal blocks of the Fock matrix
    //    std::vector<std::vector<double>> diagonalize_Fock_diagblocks(BlockedTensor& U);
    //    /// Separate an 2D ambit::Tensor according to its irrep
    //    ambit::Tensor separate_tensor(ambit::Tensor& tens, const psi::Dimension& irrep,
    //    const int& h);
    //    /// Combine a separated 2D ambit::Tensor
    //    void combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h,
    //    const psi::Dimension& irrep, const int& h);

    // => Multi-state energy <= //

    /// Compute multi-state energy in the state-average way
    std::vector<std::vector<double>> compute_energy_sa();
    /// Compute multi-state energy in the MS/XMS way
    std::vector<std::vector<double>> compute_energy_xms();
    /// XMS rotation for the reference states
    std::shared_ptr<psi::Matrix> xms_rotation(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                              std::vector<Determinant>& p_space,
                                              std::shared_ptr<psi::Matrix> civecs);

    /// Build effective singles: T_{ia} -= T_{iu,av} * Gamma_{vu}
    void build_T1eff_deGNO();

    /// Compute density cumulants
    void compute_cumulants(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                           std::vector<forte::Determinant>& p_space,
                           std::shared_ptr<psi::Matrix> evecs, const int& root1, const int& root2);
    /// Compute denisty matrices and puts in Gamma1_, Lambda2_, and Lambda3_
    void compute_rdms(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                      std::vector<Determinant>& p_space, std::shared_ptr<psi::Matrix> evecs,
                      const int& root1, const int& root2);

    /// Compute MS coupling <M|H|N>
    double compute_ms_1st_coupling(const std::string& name);
    /// Compute MS coupling <M|HT|N>
    double compute_ms_2nd_coupling(const std::string& name);

    /// Rotate RDMs computed by eigens_ (in original basis) to semicanonical basis
    /// so that they are in the same basis as amplitudes (in semicanonical basis)
    void rotate_1rdm(ambit::Tensor& L1a, ambit::Tensor& L1b);
    void rotate_2rdm(ambit::Tensor& L2aa, ambit::Tensor& L2ab, ambit::Tensor& L2bb);
    void rotate_3rdm(ambit::Tensor& L3aaa, ambit::Tensor& L3aab, ambit::Tensor& L3abb,
                     ambit::Tensor& L3bbb);
};
} // namespace forte

#endif // _dsrg_mrpt2_h_
