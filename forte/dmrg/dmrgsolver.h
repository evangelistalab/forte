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

#ifndef _dmrgsolver_h_
#define _dmrgsolver_h_

#include <filesystem>

#include "chemps2/ConvergenceScheme.h"
#include "chemps2/Hamiltonian.h"
#include "chemps2/DMRG.h"

#include "base_classes/active_space_method.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/state_info.h"
#include "base_classes/scf_info.h"
#include "base_classes/forte_options.h"

namespace forte {

class DMRGSolver : public ActiveSpaceMethod {
  public:
    /**
     * @brief DMRGSolver Constructor
     * @param state The state info (symmetry, multiplicity, na, nb, etc.)
     * @param nroot Number of roots of interests
     * @param scf_info SCF information
     * @param options Forte options
     * @param mo_space_info MOSpaceInfo
     * @param as_ints Active space integrals
     */
    DMRGSolver(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
               std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
               std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    /// DMRGSolver Destructor
    ~DMRGSolver();

    /// Compute the energy
    double compute_energy() override;

    /// RDMs override
    std::vector<std::shared_ptr<RDMs>> rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                            int max_rdm_level, RDMsType rdm_type) override;

    /// Transition RDMs override
    std::vector<std::shared_ptr<RDMs>>
    transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                    std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level,
                    RDMsType rdm_type) override;

    /// Set options override
    void set_options(std::shared_ptr<ForteOptions> options) override;

    /// Return the CI wave functions for current state symmetry
    //    std::shared_ptr<psi::Matrix> ci_wave_functions() override { return evecs_; }

    /// Dump wave function to disk (already dumped when computing energies)
    void dump_wave_function(const std::string&) override{};

    /// Read wave function from disk
    /// Return the number of active orbitals, set of determinants, CI coefficients
    //    std::tuple<size_t, std::vector<Determinant>, std::shared_ptr<psi::Matrix>>
    //    read_wave_function(const std::string& filename) override;

  private:
    /// SCFInfo object
    std::shared_ptr<SCFInfo> scf_info_;
    /// ForteOptions
    std::shared_ptr<ForteOptions> options_;

    /// Point group number in CheMPS2
    int pg_number_;
    /// Number of irreps
    int nirrep_;
    /// Wave function symmetry
    int wfn_irrep_;
    /// Multiplicity
    int multiplicity_;
    /// Number of active electrons
    int nelecs_actv_;
    /// Number of active orbitals
    int nactv_;
    /// State label
    std::string state_label_;

    /// Number of reduced renormalized basis states kept during successive DMRG instructions
    std::vector<int> dmrg_sweep_states_;
    /// Energy convergence to stop an instruction during successive DMRG instructions
    std::vector<double> dmrg_sweep_e_convergence_;
    /// Max number of sweeps to stop an instruction during successive DMRG instructions
    std::vector<int> dmrg_sweep_max_sweeps_;
    /// The noise prefactors for successive DMRG instructions
    std::vector<double> dmrg_noise_prefactors_;
    /// The residual tolerances for the Davidson diagonalization during DMRG instructions
    std::vector<double> dmrg_davidson_rtol_;
    /// Whether or not to print the correlation functions after the DMRG calculation
    bool dmrg_print_corr_;

    /// The convergence scheme of CheMPS2
    std::unique_ptr<CheMPS2::ConvergenceScheme> conv_scheme_;
    /// The active-space Hamiltonian of CheMPS2
    std::unique_ptr<CheMPS2::Hamiltonian> hamiltonian_;

    /// Checkpoint files path
    std::filesystem::path tmp_path_;
    /// Directory to save MPS files
    std::filesystem::path mps_files_path_;
    /// Vector of file names
    std::vector<std::string> mps_files_;
    /// Move MPS files around
    void move_mps_files(bool from_cwd_to_folder);

    /// Setup some internal variable
    void startup();

    /// Return the RDMs for the current state
    std::shared_ptr<RDMs> fill_current_rdms(std::shared_ptr<CheMPS2::DMRG> solver,
                                            const int max_rdm_level, RDMsType rdm_type);
};
} // namespace forte
#endif // _dmrgsolver_h_
