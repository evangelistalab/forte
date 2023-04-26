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

#ifndef _sadsrg_h_
#define _sadsrg_h_

#include "ambit/blocked_tensor.h"

#include "base_classes/active_space_solver.h"
#include "base_classes/dynamic_correlation_solver.h"

#include "helpers/blockedtensorfactory.h"
#include "mrdsrg-helper/dsrg_mem.h"
#include "mrdsrg-helper/dsrg_source.h"
#include "mrdsrg-helper/dsrg_time.h"

using namespace ambit;

namespace forte {
class SADSRG : public DynamicCorrelationSolver {
  public:
    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    SADSRG(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
           std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
           std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    virtual ~SADSRG();

    /// Compute energy
    virtual double compute_energy() override = 0;

    /// Compute DSRG transformed Hamiltonian
    std::shared_ptr<ActiveSpaceIntegrals> compute_Heff_actv() override;

    /// Compute DSRG transformed multipole integrals
    std::shared_ptr<ActiveMultipoleIntegrals> compute_mp_eff_actv() override;

    /// Set unitary matrix (in active space) from original to semicanonical
    void set_Uactv(ambit::Tensor& U);

  protected:
    /// Startup function called in constructor
    void startup();

    /// Warnings <description, changes in this run, how to get rid of it>
    std::vector<std::tuple<const char*, const char*, const char*>> warnings_;

    // ==> settings from options <==

    /// Read options
    void read_options();

    /// The flow parameter
    double s_;
    /// Source operator
    std::string source_;
    /// Source operator for the core-core-virtual-virtual block
    std::string ccvv_source_;
    /// The dsrg source operator
    std::shared_ptr<DSRG_SOURCE> dsrg_source_;
    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;

    /// Compute contributions from 3 cumulant
    bool do_cu3_;
    /// Explicitly store 3 cumulants
    bool store_cu3_;
    /// Three-body density cumulant algorithm
    std::string L3_algorithm_;

    /// Multi-state computation if true
    bool multi_state_;
    /// Multi-state algorithm
    std::string multi_state_algorithm_;

    /// Number of amplitudes will be printed in amplitude summary
    size_t ntamp_;
    /// Threshold for amplitudes considered as intruders
    double intruder_tamp_;

    /// How to consider internal amplitudes
    std::string internal_amp_;
    /// Include which part of internal amplitudes?
    std::string internal_amp_select_;

    /// Relaxation type
    std::string relax_ref_;

    /// Timings for computing the commutators
    DSRG_TIME dsrg_time_;

    /// Active orbital rotation from semicanonicalizor (set from outside)
    ambit::BlockedTensor Uactv_;
    /// Rotate 1-body DSRG transformed integrals from semicanonical back to original
    void rotate_one_ints_to_original(BlockedTensor& H1);
    /// Rotate 2-body DSRG transformed integrals from semicanonical back to original
    void rotate_two_ints_to_original(BlockedTensor& H2);
    /// Rotate 3-body DSRG transformed integrals from semicanonical back to original
    void rotate_three_ints_to_original(BlockedTensor& H3);

    /// Max body level of DSRG transformed dipole
    int max_dipole_level_ = 0;
    /// Max body level of DSRG transformed quadrupole
    int max_quadrupole_level_ = 0;

    /// Compute DSRG transformed multipoles
    virtual void transform_one_body(const std::vector<ambit::BlockedTensor>&,
                                    const std::vector<int>&) {
        throw std::runtime_error("Please override!");
    };

    /// Number of threads
    int n_threads_;

    // ==> system memory related <==

    /// Total memory available set by the user
    size_t mem_sys_;
    /// Memory checker and printer
    DSRG_MEM dsrg_mem_;

    /// Check initial memory
    void check_init_memory();

    // ==> some common energies for all DSRG levels <==

    /// The energy of the reference
    double Eref_;

    /// Compute reference (MK vacuum) energy from ForteIntegral and Fock_
    double compute_reference_energy_from_ints();

    /// Compute reference (MK vacuum) energy
    double compute_reference_energy(BlockedTensor H, BlockedTensor F, BlockedTensor V);

    // ==> MO space info <==

    /// Read MO space info
    void read_MOSpaceInfo();

    /// List of core MOs
    std::vector<size_t> core_mos_;
    /// List of active MOs
    std::vector<size_t> actv_mos_;
    /// List of virtual MOs
    std::vector<size_t> virt_mos_;
    /// List of the symmetry of the active MOs
    std::vector<int> actv_mos_sym_;

    /// List of active active occupied MOs (relative to active)
    std::vector<size_t> actv_occ_mos_;
    /// List of active active unoccupied MOs (relative to active)
    std::vector<size_t> actv_uocc_mos_;

    /// List of auxiliary MOs when DF/CD
    std::vector<size_t> aux_mos_;

    // ==> Ambit tensor settings <==

    /// Set Ambit tensor labels
    void set_ambit_MOSpace();

    /// Kevin's Tensor Wrapper
    std::shared_ptr<BlockedTensorFactory> BTF_;
    /// Tensor type for Ambit
    ambit::TensorType tensor_type_;

    /// Core MO label
    std::string core_label_;
    /// Active MO label
    std::string actv_label_;
    /// Virtual MO label
    std::string virt_label_;

    /// Auxillary basis label
    std::string aux_label_;

    /// Map from space label to list of MOs
    std::map<char, std::vector<size_t>> label_to_spacemo_;

    /// Compute diagonal blocks labels of a one-body operator
    std::vector<std::string> diag_one_labels();
    /// Compute diagonal blocks labels of a two-body operator
    std::vector<std::string> diag_two_labels();
    /// Compute retaining excitation blocks labels of a two-body operator
    std::vector<std::string> re_two_labels();
    /// Compute off-diagonal blocks labels of a one-body operator
    std::vector<std::string> od_one_labels();
    std::vector<std::string> od_one_labels_hp();
    std::vector<std::string> od_one_labels_ph();
    /// Compute off-diagonal blocks labels of a two-body operator
    std::vector<std::string> od_two_labels();
    std::vector<std::string> od_two_labels_hhpp();
    std::vector<std::string> od_two_labels_pphh();
    /// Compute the blocks labels used in NIVO (number of virtual < 3)
    std::vector<std::string> nivo_labels();

    // ==> fill in densities from RDMs <==

    /// Initialize density cumulants
    void init_density();
    /// Fill in density cumulants from the RDMs
    void fill_density();

    /// One-particle density matrix
    ambit::BlockedTensor L1_;
    /// One-hole density matrix
    ambit::BlockedTensor Eta1_;
    /// Two-body density cumulant
    ambit::BlockedTensor L2_;
    /// Three-body density cumulant
    ambit::Tensor L3_;

    // ==> Fock matrix related <==

    /// Fock matrix
    ambit::BlockedTensor Fock_;
    /// Diagonal elements of Fock matrix
    std::vector<double> Fdiag_;

    /// Initialize Fock matrix
    void init_fock();
    /// Build Fock matrix from ForteIntegrals
    void build_fock_from_ints();
    /// Fill in diagonal elements of Fock matrix to Fdiag
    void fill_Fdiag(BlockedTensor& F, std::vector<double>& Fdiag);

    /// Check orbitals if semicanonical
    bool check_semi_orbs();
    /// Are orbitals semi-canonicalized?
    bool semi_canonical_;
    /// Checked results of each block of Fock matrix
    std::map<std::string, bool> semi_checked_results_;
    /// Unitary matrix to block diagonal Fock
    ambit::BlockedTensor U_;
    /**
     * @brief Rotate 3-index integrals to semicanonical basis and dump to disk.
     * @param blocks blocks to be transformed, e.g., cc, ca, cv, ...
     *
     * For example: Br["Qme"] = B["Qnf"] * U_["mn"] * U_["ef"];
     * Note that the auxiliary index is always the first one.
     */
    void canonicalize_B(const std::unordered_set<std::string>& blocks);
    /**
     * @brief Read canonicalized 3-index integrals from disk generated by canonicalize_B.
     * @param block the block of B to be read from disk
     * @param mos1_range the range of the 1st MO index (0-based, python-like slicing)
     * @param mos2_range the range of the 2nd MO index (0-based, python-like slicing)
     * @param order the order of the output Tensor, either be Qpq or pqQ
     * @return ambit::Tensor of the desired index order
     *
     * For example, assume "cv" block of B is available on disk and
     * there are nQ auxiliary basis functions.
     *
     * The mos1_range and mos2_range are for MO spaces block[0] and block[1], respectively.
     *
     * auto B = read_Bcanonical("cv", {1,2}, {0,10});
     * loads a slice of the "cv" block of B on disk, where the "c" index is the 2nd core MO
     * and the "v" indices contain the first 10 virtual MOs.
     * The output is a Tensor of dimension nQ x 1 x 10.
     *
     * auto C = read_Bcanonical("vc", {0,10}, {1,2});
     * will load a slice of the "cv" block of B on disk because "vc" block is unavailable.
     * The output tensor is identical to B with the last two indices permuted:
     * C("Qpq") = B("Qqp");
     *
     * auto D = read_Bcanonical("cv", {1,2}, {0,10}, pqQ);
     * outputs a Tensor of dimension 1 x 10 x nQ and it is equivalent to
     * D("pqQ") = B("Qpq");
     */
    ambit::Tensor read_Bcanonical(const std::string& block,
                                  const std::pair<size_t, size_t>& mos1_range,
                                  const std::pair<size_t, size_t>& mos2_range,
                                  ThreeIntsBlockOrder order = Qpq);
    /// File names for canonicalized 3-index integrals
    std::unordered_map<std::string, std::string> Bcan_files_;

    // ==> integrals <==

    /// Fill the tensor B with three-index DF or CD integrals
    void fill_three_index_ints(ambit::BlockedTensor B);

    /// Scalar of the DSRG transformed Hamiltonian
    double Hbar0_;
    /// DSRG transformed 1-body Hamiltonian (active only in DSRG-PT, but full in MRDSRG)
    ambit::BlockedTensor Hbar1_;
    /// DSRG transformed 2-body Hamiltonian (active only in DSRG-PT, but full in MRDSRG)
    ambit::BlockedTensor Hbar2_;
    /// DSRG transformed 3-body Hamiltonian (active only in DSRG-PT, but full in MRDSRG)
    ambit::BlockedTensor Hbar3_;

    /// Scalar of the DSRG transformed multipoles
    std::vector<double> Mbar0_;
    /// DSRG transformed 1-body multipoles
    std::vector<ambit::BlockedTensor> Mbar1_;
    /// DSRG transformed 2-body multipoles
    std::vector<ambit::BlockedTensor> Mbar2_;

    /**
     * De-normal-order a 1-body DSRG transformed integrals (active only)
     * This will change H0!!!
     */
    void deGNO_ints(const std::string& name, double& H0, BlockedTensor& H1);

    /**
     * De-normal-order a 2-body DSRG transformed integrals (active only)
     * This will change H0 and H1 !!!
     */
    void deGNO_ints(const std::string& name, double& H0, BlockedTensor& H1, BlockedTensor& H2);

    /**
     * De-normal-order a 3-body DSRG transformed integrals (active only)
     * This will change H0, H1, and H2 !!!
     */
    void deGNO_ints(const std::string& name, double& H0, BlockedTensor& H1, BlockedTensor& H2,
                    BlockedTensor& H3);

    /**
     * De-normal-order the T1 and T2 amplitudes and return the effective T1
     * T1eff = T1 - T2["ivau"] * D1["uv"]
     *
     * This assumes no internal amplitudes !!!
     */
    ambit::BlockedTensor deGNO_Tamp(BlockedTensor& T1, BlockedTensor& T2, BlockedTensor& D1);

    // ==> commutators <==

    /**
     * H1, C1, G1: a rank 2 tensor of all MOs in general
     * H2, C2, G2: a rank 4 tensor of all MOs in general
     * C3: a rank 6 tensor of all MOs in general
     * T1: a rank 2 tensor of hole-particle
     * T2: a rank 4 tensor of hole-hole-particle-particle
     * V: antisymmetrized 2-electron integrals
     * B: 3-index integrals from DF/CD
     */

    /// Compute zero-body term of commutator [H1, T1]
    double H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H1, T2]
    double H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T1]
    double H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T2], S2[ijab] = 2 * T[ijab] - T[ijba]
    std::vector<double> H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2,
                                 const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T2], T2 and S2 contain at least two active indices
    std::vector<double> H2_T2_C0_T2small(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2);

    /// Compute one-body term of commutator [H1, T1]
    void H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H1, T2]
    void H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T1]
    void H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T2], S2[ijab] = 2 * T[ijab] - T[ijba]
    void H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                  BlockedTensor& C1);

    /// Compute two-body term of commutator [H2, T1]
    void H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H1, T2]
    void H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, T2], S2[ijab] = 2 * T[ijab] - T[ijba]
    void H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                  BlockedTensor& C2);

    /// Compute zero-body term of commutator [V, T1], V is constructed from B (DF/CD)
    void V_T1_C0_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [V, T2], V is constructed from B (DF/CD)
    std::vector<double> V_T2_C0_DF(BlockedTensor& B, BlockedTensor& T1, BlockedTensor& S2,
                                   const double& alpha, double& C0);

    /// Compute one-body term of commutator [V, T1], V is constructed from B (DF/CD)
    void V_T1_C1_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [V, T2], V is constructed from B (DF/CD)
    void V_T2_C1_DF(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                    BlockedTensor& C1);

    /// Compute two-body term of commutator [V, T1], V is constructed from B (DF/CD)
    void V_T1_C2_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [V, T2], V is constructed from B (DF/CD)
    void V_T2_C2_DF(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                    BlockedTensor& C2);
    /// Compute two-body term of commutator [V, T2], exchange of particle-hole contraction
    void V_T2_C2_DF_PH_X(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2);

    /// Compute the active part of commutator C1 + C2 = alpha * [H1 + H2, A1 + A2]
    void H_A_Ca(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                BlockedTensor& S2, const double& alpha, BlockedTensor& C1, BlockedTensor& C2);
    /// Compute the active part of commutator C1 + C2 = alpha * [H1 + H2, A1 + A2]
    /// G2[pqrs] = 2 * H2[pqrs] - H2[pqsr], S2[ijab] = 2 * T2[ijab] - T2[ijba]
    void H_A_Ca_small(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& G2, BlockedTensor& T1,
                      BlockedTensor& T2, BlockedTensor& S2, const double& alpha, BlockedTensor& C1,
                      BlockedTensor& C2);
    /// Compute the active part of commutator C1 = [H2, T1 + T2] that uses G2
    void H2_T_C1a_smallG(BlockedTensor& G2, BlockedTensor& T1, BlockedTensor& T2,
                         BlockedTensor& C1);
    /// Compute the active part of commutator C1 = [H1, T1 + T2] that uses S2
    void H1_T_C1a_smallS(BlockedTensor& H1, BlockedTensor& T1, BlockedTensor& S2,
                         BlockedTensor& C1);
    /// Compute the active part of commutator C1 = [H2, T1 + T2] that uses S2
    void H2_T_C1a_smallS(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2,
                         BlockedTensor& C1);
    /// Compute the active part of commutator C2 = [H1 + H2, T1 + T2] that uses S2
    void H_T_C2a_smallS(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                        BlockedTensor& S2, BlockedTensor& C2);

    /// Compute the ph part of commutator C1 = [H1d, A1]
    void H1d_A1_C1ph(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute the ph part of commutator C1 = [H1d, A2]
    void H1d_A2_C1ph(BlockedTensor& H1, BlockedTensor& S2, const double& alpha, BlockedTensor& C1);
    /// Compute the pphh part of commutator C2 = [H1d, A2]
    void H1d_A2_C2pphh(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                       BlockedTensor& C2);
    /// Compute the small pphh part of commutator C2 = [H1d, A2]
    void H1d_A2_C2pphh_small(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                             BlockedTensor& C2);

    /// Compute the ph part of commutator C1 = [H1, A1]
    void H1_A1_C1ph(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute the ph part of commutator C1 = [H1, A2]
    void H1_A2_C1ph(BlockedTensor& H1, BlockedTensor& S2, const double& alpha, BlockedTensor& C1);

    // ==> miscellaneous <==

    /// File name prefix for checkpoint files
    std::string chk_filename_prefix_;

    /// Diagonalize the diagonal blocks of the Fock matrix
    std::vector<double> diagonalize_Fock_diagblocks(BlockedTensor& U);

    /// Print the summary of 2- and 3-body density cumulant
    void print_cumulant_summary();

    /// Print the contents with padding: <text> <padding with dots>
    void print_contents(const std::string& str, size_t size = 45);
    /// Print done and timing
    void print_done(double t, const std::string& done="Done");

    // ==> common amplitudes analysis and printing <==

    /// Prune internal amplitudes for T1
    void internal_amps_T1(BlockedTensor& T1);
    /// Prune internal amplitudes for T2
    void internal_amps_T2(BlockedTensor& T2);

    /// Check T1 and return the largest amplitudes
    std::vector<std::pair<std::vector<size_t>, double>> check_t1(BlockedTensor& T1);
    /// Check T2 and return the largest amplitudes
    std::vector<std::pair<std::vector<size_t>, double>> check_t2(BlockedTensor& T2);
    /// Analyze T1 and T2 amplitudes
    void analyze_amplitudes(std::string name, BlockedTensor& T1, BlockedTensor& T2);
    /// Print t1 amplitudes summary
    void print_t1_summary(const std::vector<std::pair<std::vector<size_t>, double>>& list,
                          const double& norm, const size_t& number_nonzero);
    /// Print t2 amplitudes summary
    void print_t2_summary(const std::vector<std::pair<std::vector<size_t>, double>>& list,
                          const double& norm, const size_t& number_nonzero);
    /// Print t1 intruder analysis
    void print_t1_intruder(const std::vector<std::pair<std::vector<size_t>, double>>& list);
    /// Print t2 intruder analysis
    void print_t2_intruder(const std::vector<std::pair<std::vector<size_t>, double>>& list);

    /// Comparison function used to sort pair<vector, double>
    static bool sort_pair_second_descend(const std::pair<std::vector<size_t>, double>& left,
                                         const std::pair<std::vector<size_t>, double>& right) {
        return std::fabs(left.second) > std::fabs(right.second);
    }
};
} // namespace forte

#endif // SADSRG_H
