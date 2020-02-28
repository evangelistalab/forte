/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _sa_mrdsrg_h_
#define _sa_mrdsrg_h_

#include <cmath>
#include <memory>

#include "psi4/libdiis/diismanager.h"

#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.h"
#include "ambit/blocked_tensor.h"

#include "integrals/integrals.h"
#include "base_classes/rdms.h"
#include "helpers/blockedtensorfactory.h"
#include "sparse_ci/determinant.h"
#include "sadsrg.h"

using namespace ambit;

namespace forte {

class SA_MRDSRG : public SADSRG {
  public:
    /**
     * SA_MRDSRG Constructor
     * @param scf_info The SCFInfo
     * @param options The ForteOption
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    SA_MRDSRG(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
              std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Compute the corr_level energy with fixed reference
    virtual double compute_energy();

  protected:
    // => Class initialization and termination <= //

    /// Start-up function called in the constructor
    void startup();

    /// Read options
    void read_options();

    /// Are orbitals semi-canonicalized?
    bool semi_canonical_;

    /// Is the H_bar evaluated sequentially?
    bool sequential_Hbar_;

    /// Omitting blocks with >= 3 virtual indices?
    bool nivo_;

    /// Fill up integrals
    void build_ints();

    // => DSRG related <= //

    /// Correlation level option
    std::string corrlv_string_;

    /// Correlation level
    enum class CORR_LV { LDSRG2, LDSRG2_QC };
    std::map<std::string, CORR_LV> corrlevelmap{{"LDSRG2", CORR_LV::LDSRG2},
                                                {"LDSRG2_QC", CORR_LV::LDSRG2_QC}};

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

    /// Initial guess of T amplitudes
    void guess_t(BlockedTensor& V, BlockedTensor& T2, BlockedTensor& F, BlockedTensor& T1);
    /// Initial guess of T amplitudes with density fitted B tensor.
    void guess_t_df(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& F, BlockedTensor& T1);
    /// Update T amplitude in every iteration
    void update_t();
    /// Analyze T1 and T2 amplitudes
    void analyze_amplitudes(std::string name, BlockedTensor& T1, BlockedTensor& T2);

    /// RMS of T2
    double T2rms_;
    /// Norm of T2
    double T2norm_;
    /// Signed max of T2
    double T2max_;
    /// Check T2 and store the largest amplitudes
    void check_t2(BlockedTensor& T2);
    /// Initial guess of T2
    void guess_t2_std(BlockedTensor& V, BlockedTensor& T2);
    void guess_t2_noccvv(BlockedTensor& V, BlockedTensor& T2);
    /// Update T2 in every iteration
    void update_t2_std();
    void update_t2_noccvv();

    /// RMS of T1
    double T1rms_;
    /// Norm of T1
    double T1norm_;
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

    /// Write T2 to a psi4 file
    void write_t2_file(BlockedTensor& T2);
    /// Read T2 from a psi4 file
    void read_t2_file(BlockedTensor& T2);
    /// Write T1 to a psi4 file
    void write_t1_file(BlockedTensor& T1);
    /// Read T1 from a psi4 file
    void read_t1_file(BlockedTensor& T1);

    /// List of large amplitudes
    std::vector<std::pair<std::vector<size_t>, double>> lt1_;
    std::vector<std::pair<std::vector<size_t>, double>> lt2_;

    /// Compute DSRG-transformed Hamiltonian
    void compute_hbar();
    /// Compute DSRG-transformed Hamiltonian Hbar sequentially
    void compute_hbar_sequential();
    /// Compute DSRG-transformed Hamiltonian truncated to 2-nested commutator
    void compute_hbar_qc();

    /// Temporary one-body Hamiltonian
    ambit::BlockedTensor O1_;
    ambit::BlockedTensor C1_;
    /// Temporary two-body Hamiltonian
    ambit::BlockedTensor O2_;
    ambit::BlockedTensor C2_;

    /// Norm of off-diagonal Hbar1 or Hbar2
    double Hbar_od_norm(const int& n, const std::vector<std::string>& blocks);

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

    /// Compute MR-LDSRG(2) truncated to 2-nested commutator
    double compute_energy_ldsrg2_qc();
    /// Compute MR-LDSRG(2)
    double compute_energy_ldsrg2();

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
};
} // namespace forte

#endif // _sa_mrdsrg_h_
