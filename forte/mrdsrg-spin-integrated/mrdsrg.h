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

#ifndef _mrdsrg_h_
#define _mrdsrg_h_

#include "master_mrdsrg.h"

using namespace ambit;

namespace psi {
class DIISManager;
}

namespace forte {

class MRDSRG : public MASTER_DSRG {
    friend class MRSRG_ODEInt;
    friend class MRSRG_Print;
    friend class SRGPT2_ODEInt;

  public:
    /**
     * MRDSRG Constructor
     * @param scf_info The SCFInfo
     * @param options The ForteOption
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    MRDSRG(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
           std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    virtual ~MRDSRG();

    /// Compute the corr_level energy with fixed reference
    virtual double compute_energy() override;

    /// Compute state-average MR-DSRG energy
    double compute_energy_sa();

    //    /// Set CASCI eigen values and eigen vectors for state averaging
    //    void set_eigens(std::vector<std::vector<std::pair<psi::SharedVector, double>>> eigens) {
    //        eigens_ = eigens;
    //    }

    //    /// Set determinants in the model space
    //    void set_p_spaces(std::vector<std::vector<forte::Determinant>> p_spaces) {
    //        p_spaces_ = p_spaces;
    //    }

  protected:
    // => Class initialization and termination <= //

    /// Start-up function called in the constructor
    void startup();
    /// Clean-up function called in the destructor
    void cleanup();

    /// Read options
    void read_options();

    /// DSRG transformation type
    std::string dsrg_trans_type_;

    /// Are orbitals semi-canonicalized?
    bool semi_canonical_;

    /// Is the H_bar evaluated sequentially?
    bool sequential_Hbar_;

    /// Omitting blocks with >= 3 virtual indices?
    bool nivo_;

    /// Read amplitudes from previous computations
    bool restart_amps_;

    /// Prefix for file name
    std::string restart_file_prefix_;

    /// Dump the converged amplitudes to disk
    void dump_amps_to_disk() override;

    //    /// CASCI eigen values and eigen vectors for state averaging
    //    std::vector<std::vector<std::pair<psi::SharedVector, double>>> eigens_;
    //    /// Determinants in the model space
    //    std::vector<std::vector<forte::Determinant>> p_spaces_;

    /// Fill up integrals
    void build_ints();
    /// Fill up density matrix and density cumulants
    void build_density();
    /// Build Fock matrix and diagonal Fock matrix elements
    void build_fock(BlockedTensor& H, BlockedTensor& V);
    /// Build Fock matrix and diagonal Fock matrix elements with density fitted B tensor
    void build_fock_df(BlockedTensor& H, BlockedTensor& B);

    // => DSRG related <= //

    /// Correlation level option
    std::string corrlv_string_;

    /// Correlation level
    enum class CORR_LV {
        LDSRG2,
        LDSRG2_P3,
        PT2,
        PT3,
        QDSRG2,
        QDSRG2_P3,
        LDSRG2_QC,
        LSRG2,
        SRG_PT2
    };
    std::map<std::string, CORR_LV> corrlevelmap = {{"LDSRG2", CORR_LV::LDSRG2},
                                                   {"LDSRG2_P3", CORR_LV::LDSRG2_P3},
                                                   {"PT2", CORR_LV::PT2},
                                                   {"PT3", CORR_LV::PT3},
                                                   {"LDSRG2_QC", CORR_LV::LDSRG2_QC},
                                                   {"QDSRG2", CORR_LV::QDSRG2},
                                                   {"QDSRG2_P3", CORR_LV::QDSRG2_P3},
                                                   {"LSRG2", CORR_LV::LSRG2},
                                                   {"SRG_PT2", CORR_LV::SRG_PT2}};

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
    /// Difference of consecutive singles
    ambit::BlockedTensor DT1_;
    /// Difference of consecutive doubles
    ambit::BlockedTensor DT2_;
    /// Unitary matrix to block diagonal Fock
    ambit::BlockedTensor U_;

    /// Diagonal elements of Fock matrices
    std::vector<double> Fa_;
    std::vector<double> Fb_;

    /// Automatic adjust the flow parameter
    enum class SMART_S { DSRG_S, MIN_DELTA1, MAX_DELTA1, DAVG_MIN_DELTA1, DAVG_MAX_DELTA1 };
    std::map<std::string, SMART_S> smartsmap = {{"DSRG_S", SMART_S::DSRG_S},
                                                {"MIN_DELTA1", SMART_S::MIN_DELTA1},
                                                {"DAVG_MIN_DELTA1", SMART_S::DAVG_MIN_DELTA1},
                                                {"MAX_DELTA1", SMART_S::MAX_DELTA1},
                                                {"DAVG_MAX_DELTA1", SMART_S::DAVG_MAX_DELTA1}};
    /// Automatic adjusting the flow parameter
    double make_s_smart();
    /// Algorithm to compute energy threshold according to Delta1
    double smart_s_min_delta1();
    double smart_s_max_delta1();
    double smart_s_davg_min_delta1();
    double smart_s_davg_max_delta1();

    /// Algorithm for computing amplitudes
    std::string T_algor_;
    /// Initial guess of T amplitudes
    void guess_t(BlockedTensor& V, BlockedTensor& T2, BlockedTensor& F, BlockedTensor& T1);
    /// Initial guess of T amplitudes with density fitted B tensor.
    void guess_t_df(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& F, BlockedTensor& T1);
    /// Update T amplitude in every iteration
    void update_t();
    /// Analyze T1 and T2 amplitudes
    void analyze_amplitudes(std::string name, BlockedTensor& T1, BlockedTensor& T2);

    /// RMS of T2
    double T2rms_ = 0.0;
    /// Norm of T2
    double T2norm_;
    double t2aa_norm_;
    double t2ab_norm_;
    double t2bb_norm_;
    /// Signed max of T2
    double T2max_;
    /// Check T2 and store the largest amplitudes
    void check_t2(BlockedTensor& T2);
    /// Initial guess of T2
    void guess_t2_std(BlockedTensor& V, BlockedTensor& T2);
    void guess_t2_noccvv(BlockedTensor& V, BlockedTensor& T2);
    /// Initial guess of T2 with density fitted B tensor.
    void guess_t2_std_df(BlockedTensor& V, BlockedTensor& T2);
    void guess_t2_noccvv_df(BlockedTensor& V, BlockedTensor& T2);
    /// Update T2 in every iteration
    void update_t2_std();
    void update_t2_noccvv();
    void update_t2_pt();
    /// Compute T2 amplitudes norms
    void compute_t2_norm();

    /// RMS of T1
    double T1rms_ = 0.0;
    /// Norm of T1
    double T1norm_;
    double t1a_norm_;
    double t1b_norm_;
    /// Signed max of T1
    double T1max_;
    /// Check T1 and store the largest amplitudes
    void check_t1(BlockedTensor& T1);
    /// Initial guess of T1
    void guess_t1_std(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1);
    void guess_t1_nocv(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1);
    /// Update T1 in every iteration
    void update_t1_std();
    void update_t1_nocv();
    /// Compute T1 amplitudes norms
    void compute_t1_norm();

    /// List of large amplitudes
    std::vector<std::pair<std::vector<size_t>, double>> lt1a_;
    std::vector<std::pair<std::vector<size_t>, double>> lt1b_;
    std::vector<std::pair<std::vector<size_t>, double>> lt2aa_;
    std::vector<std::pair<std::vector<size_t>, double>> lt2ab_;
    std::vector<std::pair<std::vector<size_t>, double>> lt2bb_;

    /// Compute DSRG-transformed Hamiltonian Hbar
    void compute_hbar();
    /// Compute DSRG-transformed Hamiltonian Hbar sequentially
    void compute_hbar_sequential();
    /// Compute DSRG-transformed Hamiltonian Hbar sequentially by orbital rotation
    void compute_hbar_sequential_rotation();
    /// Compute DSRG-transformed Hamiltonian Hbar truncated to quadratic nested
    /// commutator
    void compute_hbar_qc();

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

    /// Compute zero-body term of commutator [H2, T1]
    void H2_T1_C0_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T2] with density fitting
    void H2_T2_C0_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha, double& C0);
    /// Compute one-body term of commutator [H2, T1]
    void H2_T1_C1_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T2] with density fitting
    void H2_T2_C1_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);
    /// Compute two-body term of commutator [H2, T1]
    void H2_T1_C2_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, T2] with density fitting
    void H2_T2_C2_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);

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

    /// Copy T1 and T2 to a big vector for DIIS
    std::vector<double> copy_amp_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                                      BlockedTensor& T2, const std::vector<std::string>& label2);
    /// Compute number of elements of the big vector for DIIS
    size_t vector_size_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                            BlockedTensor& T2, const std::vector<std::string>& label2);
    /// Copy extrapolated values back to T1 and T2
    void return_amp_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                         BlockedTensor& T2, const std::vector<std::string>& label2,
                         const std::vector<double>& data);

    /// Shared pointer of DIISManager object from Psi4
    std::shared_ptr<psi::DIISManager> diis_manager_;
    /// Amplitudes pointers
    std::vector<double*> amp_ptrs_;
    /// Residual pointers
    std::vector<double*> res_ptrs_;
    /// Initialize DIISManager
    void diis_manager_init();
    /// Add entry for DIISManager
    void diis_manager_add_entry();
    /// Extrapolate for DIISManager
    void diis_manager_extrapolate();
    /// Clean up for pointers used for DIIS
    void diis_manager_cleanup();

    /// Add H2's Hermitian conjugate to itself, H2 need to contain gggg (or
    /// GGGG) block
    void tensor_add_HC_aa(BlockedTensor& H2, const bool& spin_alpha = true);
    /// Add H2's Hermitian conjugate to itself, H2 need to contain gGgG block
    void tensor_add_HC_ab(BlockedTensor& H2);

    /// Compute MR-LDSRG(2) truncated to quadratic commutator
    double compute_energy_ldsrg2_qc();
    /// Compute MR-LDSRG(2)
    double compute_energy_ldsrg2();

    /// Zeroth-order Hamiltonian
    ambit::BlockedTensor H0th_;
    /// DSRG-MRPT2 zeroth-order Hamiltonian
    std::string pt2_h0th_;
    /// Compute DSRG-MRPT2 energy
    double compute_energy_pt2();
    /// Compute DSRG-MRPT3 energy
    double compute_energy_pt3();

    /// Compute DSRG-MRPT2 energy using Fdiag as H0th
    std::vector<std::pair<std::string, double>> compute_energy_pt2_Fdiag();
    /// Compute DSRG-MRPT2 energy using Ffull as H0th
    std::vector<std::pair<std::string, double>> compute_energy_pt2_Ffull();
    /// Compute DSRG-MRPT2 energy using Fdiag_Vactv or Fdiag_Vdiag as H0th
    std::vector<std::pair<std::string, double>> compute_energy_pt2_FdiagV();

    // => MR-SRG <= //

    /// Compute MR-LSRG(2) energy;
    double compute_energy_lsrg2();
    /// Compute SRG-MRPT2 energy
    double compute_energy_srgpt2();
    /// Time spent for each step
    double srg_time_;

    /// Compute zero-body term of commutator [H1, G1]
    void H1_G1_C0(BlockedTensor& H1, BlockedTensor& G1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H1, G2]
    void H1_G2_C0(BlockedTensor& H1, BlockedTensor& G2, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, G2]
    void H2_G2_C0(BlockedTensor& H2, BlockedTensor& G2, const double& alpha, double& C0);

    /// Compute one-body term of commutator [H1, G1]
    void H1_G1_C1(BlockedTensor& H1, BlockedTensor& G1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H1, G2]
    void H1_G2_C1(BlockedTensor& H1, BlockedTensor& G2, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, G2]
    void H2_G2_C1(BlockedTensor& H2, BlockedTensor& G2, const double& alpha, BlockedTensor& C1);

    /// Compute two-body term of commutator [H1, G2]
    void H1_G2_C2(BlockedTensor& H1, BlockedTensor& G2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, G2]
    void H2_G2_C2(BlockedTensor& H2, BlockedTensor& G2, const double& alpha, BlockedTensor& C2);

    // => Useful printings <= //

    /// Print a summary of the options
    void print_options();
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

/// The type of container used to hold the state vector used by boost::odeint
using odeint_state_type = std::vector<double>;

/// The functor used for boost ODE integrator in MR-SRG.
class MRSRG_ODEInt {
  public:
    MRSRG_ODEInt(MRDSRG& mrdsrg_obj) : mrdsrg_obj_(mrdsrg_obj) {}
    void operator()(const odeint_state_type& x, odeint_state_type& dxdt, const double t);

  protected:
    MRDSRG& mrdsrg_obj_;
};

/// The functor used for boost ODE integrator in SRG-MRPT2.
class SRGPT2_ODEInt {
  public:
    SRGPT2_ODEInt(MRDSRG& mrdsrg_obj, std::string Hzero, bool relax_ref)
        : mrdsrg_obj_(mrdsrg_obj), relax_ref_(relax_ref), Hzero_(Hzero) {}
    void operator()(const odeint_state_type& x, odeint_state_type& dxdt, const double t);

  protected:
    MRDSRG& mrdsrg_obj_;
    bool relax_ref_;
    std::string Hzero_;
};

/// The functor used to print in each ODE integration step
class MRSRG_Print {
  public:
    MRSRG_Print(MRDSRG& mrdsrg_obj) : mrdsrg_obj_(mrdsrg_obj) {}
    void operator()(const odeint_state_type& x, const double t);
    std::vector<double> energies() { return energies_; }

  protected:
    MRDSRG& mrdsrg_obj_;
    std::vector<double> energies_;
};
} // namespace forte

#endif // _mrdsrg_h_
