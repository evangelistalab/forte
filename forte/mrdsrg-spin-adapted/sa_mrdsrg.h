/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libdiis/diismanager.h"
#include "psi4/libpsi4util/PsiOutStream.h"

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
    SA_MRDSRG(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
              std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
              std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Compute the corr_level energy with fixed reference
    double compute_energy() override;

  protected:
    /// Start-up function called in the constructor
    void startup();

    /// Read options
    void read_options();

    /// Print a summary of the options
    void print_options();

    /// Fill up integrals
    void build_ints();

    /// Check memory
    void check_memory();

    /// Maximum number of iterations
    int maxiter_;
    /// Energy convergence criteria
    double e_conv_;
    /// Residual convergence criteria
    double r_conv_;

    /// Is the Hbar evaluated sequentially?
    bool sequential_Hbar_;

    /// Omitting blocks with >= 3 virtual indices?
    bool nivo_;

    /// Read amplitudes from previous reference relaxation step
    bool restart_amps_;

    /// Prefix for file name for restart
    std::string restart_file_prefix_;

    /// Dump the converged amplitudes to disk
    void dump_amps_to_disk() override;

    /// Correlation level option
    std::string corrlv_string_;

    /// Correlation level
    enum class CORR_LV { LDSRG2, LDSRG2_QC };
    std::map<std::string, CORR_LV> corrlevelmap{{"LDSRG2", CORR_LV::LDSRG2},
                                                {"LDSRG2_QC", CORR_LV::LDSRG2_QC}};

    /// Max number of commutators considered in recursive single commutator algorithm
    int rsc_ncomm_;
    /// Convergenve threshold for Hbar in recursive single commutator algorithm
    double rsc_conv_;

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

    /// Initial guess of T amplitudes
    void guess_t(BlockedTensor& V, BlockedTensor& T2, BlockedTensor& F, BlockedTensor& T1,
                 BlockedTensor& B);
    /// Update T amplitude in every iteration
    void update_t();

    /// RMS of T2
    double T2rms_;
    /// Norm of T2
    double T2norm_;
    /// Signed max of T2
    double T2max_;
    /// Initial guess of T2
    void guess_t2(BlockedTensor& V, BlockedTensor& T2, BlockedTensor& B);
    /// Initial guess of T2 where T2 has been initialized with H2
    void guess_t2_impl(BlockedTensor& T2);
    /// Update T2 in every iteration
    void update_t2();

    /// RMS of T1
    double T1rms_;
    /// Norm of T1
    double T1norm_;
    /// Signed max of T1
    double T1max_;
    /// Initial guess of T1
    std::string t1_guess_;
    /// Initial guess of T1
    void guess_t1(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1);
    /// Update T1 in every iteration
    void update_t1();

    /// Compute DSRG-transformed Hamiltonian
    void compute_hbar();
    /// Compute DSRG-transformed Hamiltonian Hbar sequentially
    void compute_hbar_sequential();
    /// Compute DSRG-transformed Hamiltonian truncated to 2-nested commutator
    void compute_hbar_qc();

    /// Add H2's Hermitian conjugate to itself, H2 need to contain gGgG block
    void add_hermitian_conjugate(BlockedTensor& H2);

    /// Setup tensors for iterations
    void setup_ldsrg2_tensors();

    /// Temporary one-body Hamiltonian
    ambit::BlockedTensor O1_;
    ambit::BlockedTensor C1_;
    /// Temporary two-body Hamiltonian
    ambit::BlockedTensor O2_;
    ambit::BlockedTensor C2_;

    /// Norm of off-diagonal Hbar1 or Hbar2
    double Hbar_od_norm(const int& n, const std::vector<std::string>& blocks);

    /// Shared pointer of DIISManager object from Psi4
    std::shared_ptr<psi::DIISManager> diis_manager_;
    /// Initialize DIISManager
    void diis_manager_init();
    /// Add entry for DIISManager
    void diis_manager_add_entry();
    /// Extrapolate for DIISManager
    void diis_manager_extrapolate();
    /// Clean up for pointers used for DIIS
    void diis_manager_cleanup();

    /// Compute MR-LDSRG(2)
    double compute_energy_ldsrg2();
};
} // namespace forte

#endif // _sa_mrdsrg_h_
