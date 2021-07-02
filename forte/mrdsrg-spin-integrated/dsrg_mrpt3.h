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

#ifndef _dsrg_mrpt3_h_
#define _dsrg_mrpt3_h_

#include "master_mrdsrg.h"

using namespace ambit;

namespace forte {

class DSRG_MRPT3 : public MASTER_DSRG {
  public:
    /**
     * DSRG_MRPT3 Constructor
     * @param reference The reference object of FORTE
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    DSRG_MRPT3(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    virtual ~DSRG_MRPT3();

    /// Compute the DSRG-MRPT3 energy
    virtual double compute_energy();

    /// Compute the DSRG-MRPT3 energy with relaxed reference (once)
    double compute_energy_relaxed();

    /// Compute the state-averaged DSRG-MRPT3 energies
    double compute_energy_sa();

    /// Set CASCI eigen values and eigen vectors for state averaging
    void set_eigens(std::vector<std::vector<std::pair<psi::SharedVector, double>>> eigens) {
        eigens_ = eigens;
    }

    /// Set determinants in the model space
    void set_p_spaces(std::vector<std::vector<forte::Determinant>> p_spaces) {
        p_spaces_ = p_spaces;
    }

  protected:
    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();
    /// Print a summary of the options
    void print_options_summary();
    /// Profile printing for DF
    bool profile_print_;
    /// Time variable
    std::chrono::time_point<std::chrono::system_clock> start_, end_;
    std::time_t tt1_, tt2_;
    /// Compute elapsed time
    std::chrono::duration<double>
    compute_elapsed_time(std::chrono::time_point<std::chrono::system_clock> t1,
                         std::chrono::time_point<std::chrono::system_clock> t2) {
        return t2 - t1;
    }

    /// CASCI eigen values and eigen vectors for state averaging
    std::vector<std::vector<std::pair<psi::SharedVector, double>>> eigens_;
    /// Determinants in the model space
    std::vector<std::vector<forte::Determinant>> p_spaces_;

    /// Total memory left
    int64_t mem_total_;

    /// Fill up two-electron integrals
    void build_tei(BlockedTensor& V);
    /// Build Fock matrix and diagonal Fock matrix elements
    void build_fock_half();
    /// Build Fock matrix when two-electron integrals are fully stored
    void build_fock_full();

    /// Are orbitals semi-canonicalized?
    bool semi_canonical_;
    /// Diagonal elements of Fock matrices
    std::vector<double> Fa_;
    std::vector<double> Fb_;

    // => DSRG related <= //

    /// Effective alpha one-electron integrals (used in denormal ordering)
    std::vector<double> aone_eff_;
    /// Effective beta one-electron integrals (used in denormal ordering)
    std::vector<double> bone_eff_;

    /// Renormalize Fock matrix
    void renormalize_F(const bool& plusone = true);
    /// Renormalize two-electron integrals
    void renormalize_V(const bool& plusone = true);

    // => Tensors <= //

    /// Generalized Fock matrix (bare or renormalized)
    ambit::BlockedTensor F_;
    /// Zeroth-order Hamiltonian (bare diagonal blocks of Fock)
    ambit::BlockedTensor F0th_;
    /// Generalized Fock matrix (bare off-diagonal blocks)
    ambit::BlockedTensor F1st_;
    /// Two-electron integrals (bare or renormalized)
    ambit::BlockedTensor V_;
    /// Three-index integrals
    ambit::BlockedTensor B_;
    /// Single excitation amplitude
    ambit::BlockedTensor T1_;
    /// Double excitation amplitude
    ambit::BlockedTensor T2_;
    /// One-body temp tensor ([[H0th,A1st],A1st] or 1st-order amplitudes)
    ambit::BlockedTensor O1_;
    /// Two-body temp tensor ([[H0th,A1st],A1st] or 1st-order amplitudes)
    ambit::BlockedTensor O2_;

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

    /// compute 2nd-order energy and transformed Hamiltonian
    double compute_energy_pt2();
    /// compute 3rd-order energy contribution 1.0 / 12.0 * [[[H0th,A1st],A1st],A1st],
    /// computed before pt2
    double compute_energy_pt3_1();
    /// compute 3d-order energy contribution 1.0 / 2.0  * [H1st + Hbar1st,A2nd]
    double compute_energy_pt3_2();
    /// compute 3rd-order energy contribution 1.0 / 2.0  * [Hbar2nd,A1st]
    double compute_energy_pt3_3();

    /// Compute one-body term of commutator [V, T1], V is constructed from B (DF/CD)
    void V_T1_C1_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [V, T2], V is constructed from B (DF/CD)
    void V_T2_C1_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);
    /// Compute two-body term of commutator [V, T1], V is constructed from B (DF/CD)
    void V_T1_C2_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [V, T2], V is constructed from B (DF/CD)
    void V_T2_C2_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);

    /// Compute two-body term of commutator [V, T2], particle-particle
    /// contraction when "ab" in T2 are actives
    void V_T2_C2_DF_AA(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [V, T2] (batch), particle-particle
    /// contraction when "ab" in T2 are active and virtual
    void V_T2_C2_DF_AV(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [V, T2] (batch), particle-particle
    /// contraction when "ab" in T2 are virtuals
    void V_T2_C2_DF_VV(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [V, T2], particle-hole contraction
    /// (exchange part), contracted particle index is active
    void V_T2_C2_DF_AH_EX(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                          BlockedTensor& C2, const std::vector<std::vector<std::string>>& qs,
                          const std::vector<std::vector<std::string>>& jb);
    /// Compute two-body term of commutator [V, T2], particle-hole contraction
    /// (exchange part), contracted particle index is virtual
    void V_T2_C2_DF_VA_EX(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                          BlockedTensor& C2, const std::vector<std::string>& qs_lower,
                          const std::vector<std::string>& jb_lower);
    /// Compute two-body term of commutator [V, T2], particle-hole contraction
    /// (exchange part), contracted particle index is virtual
    void V_T2_C2_DF_VC_EX(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                          BlockedTensor& C2, const std::vector<std::string>& qs_lower,
                          const std::vector<std::string>& jb_lower);

    /// Compute two-body term of commutator [V, T2], particle-hole contraction
    /// (exchange part), contracted particle index is virtual
    //    void V_T2_C2_DF_VH_EX(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
    //                          BlockedTensor& C2, const std::vector<std::vector<std::string>>& qs,
    //                          const std::vector<std::vector<std::string>>& jb);
    // TODO: commented this out because it's never called

    /**
     * Get a sub block of tensor T
     * @param T The big tensor
     * @param P The map of <partitioned dimension index, its absolute indices in T>
     * @param name The name of the returned sub tensor
     * @return A sub tensor of T with the same dimension
     */
    //    ambit::Tensor sub_block(ambit::Tensor& T, const std::map<size_t, std::vector<size_t>>& P,
    //                            const std::string& name);

    /// Rotate RDMs computed by eigens_ (in original basis) to semicanonical basis
    /// so that they are in the same basis as amplitudes (in semicanonical basis)
    //    void rotate_1rdm(ambit::Tensor& L1a, ambit::Tensor& L1b);
    //    void rotate_2rdm(ambit::Tensor& L2aa, ambit::Tensor& L2ab, ambit::Tensor& L2bb);

    // => Dipole related <= //

    /// Compute DSRG transformed dipole integrals from 1st-order amplitudes for a given direction
    void compute_dm1d_pt3_1(BlockedTensor& M, double& Mbar0, double& Mbar0_pt2,
                            BlockedTensor& Mbar1, BlockedTensor& Mbar2);
    /// Compute DSRG transformed dipole integrals from 2nd-order amplitudes for a given direction
    void compute_dm1d_pt3_2(BlockedTensor& M, double& Mbar0, double& Mbar0_pt2,
                            BlockedTensor& Mbar1, BlockedTensor& Mbar2);
    /// Print unrelaxed dipole
    void print_dm_pt3();

    /// DSRG-MRPT2 transformed dipole scalar
    std::array<double, 3> Mbar0_pt2_;
    /// DSRG-MRPT2 (2nd-order complete) transformed dipole scalar
    std::array<double, 3> Mbar0_pt2c_;
};
} // namespace forte

#endif // _dsrg_mrpt3_h_
