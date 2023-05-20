/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#ifndef _dynamic_correlation_solver_h_
#define _dynamic_correlation_solver_h_

#include <map>
#include <memory>
#include <vector>

namespace ambit {
class Tensor;
}

namespace forte {

class ActiveSpaceIntegrals;
class ActiveMultipoleIntegrals;
class ActiveSpaceSolver;
class RDMs;
class SCFInfo;
class StateInfo;
class ForteIntegrals;
class ForteOptions;
class MOSpaceInfo;

class DynamicCorrelationSolver {
  public:
    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    DynamicCorrelationSolver(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
                             std::shared_ptr<ForteOptions> options,
                             std::shared_ptr<ForteIntegrals> ints,
                             std::shared_ptr<MOSpaceInfo> mo_space_info);
    /// Compute energy
    virtual double compute_energy() = 0;

    /// Compute dressed Hamiltonian
    virtual std::shared_ptr<ActiveSpaceIntegrals> compute_Heff_actv() = 0;

    /// Compute dressed multipole integrals
    virtual std::shared_ptr<ActiveMultipoleIntegrals> compute_mp_eff_actv() {
        throw std::runtime_error("Need to overload this function");
    }

    /// Destructor
    virtual ~DynamicCorrelationSolver() = default;

    /// Set whether to read amplitudes or not manually
    void set_read_amps_cwd(bool read) { read_amps_cwd_ = read; }

    /// Clean up amplitudes checkpoint files
    void clean_checkpoints();

    /// Set CI coefficients
    /// TODO: remove this when implemented more efficient way of computing CI response
    virtual void set_ci_vectors(const std::vector<ambit::Tensor>& ci_vectors) {
        ci_vectors_ = ci_vectors;
    }
    /// Set the active space solver
    void set_active_space_solver(std::shared_ptr<ActiveSpaceSolver> as_solver) {
        as_solver_ = as_solver;
    }

    /// Set state to weights
    void set_state_weights_map(const std::map<StateInfo, std::vector<double>>& state_to_weights);

  protected:
    /// The molecular integrals
    std::shared_ptr<ForteIntegrals> ints_;

    /// The MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The RDMs and cumulants of the reference wave function
    std::shared_ptr<RDMs> rdms_;

    /// The SCFInfo
    std::shared_ptr<SCFInfo> scf_info_;

    /// The ForteOptions
    std::shared_ptr<ForteOptions> foptions_;

    /// The CI coefficients
    /// TODO: remove this when implemented more efficient way of computing CI response
    std::vector<ambit::Tensor> ci_vectors_;
    /// Active space solver
    std::shared_ptr<ActiveSpaceSolver> as_solver_ = nullptr;
    /// State to weights map
    std::map<StateInfo, std::vector<double>> state_to_weights_;

    /// Common settings
    void startup();

    /// Nuclear repulsion energy
    double Enuc_;

    /// Frozen core energy
    double Efrzc_;

    /// Compute the reference energy
    double compute_reference_energy();

    /// Printing level
    int print_;

    /// The integral type
    std::string ints_type_;
    /// If ERI density fitted or Cholesky decomposed
    bool eri_df_;

    // ==> DIIS control <==

    /// Cycle number to start DIIS
    int diis_start_;
    /// Minimum number of DIIS vectors
    int diis_min_vec_;
    /// Maximum number of DIIS vectors
    int diis_max_vec_;
    /// Frequency of extrapolating the current DIIS vectors
    int diis_freq_;

    // ==> amplitudes file names <==

    /// Checkpoint file for T1 amplitudes
    std::string t1_file_chk_;
    /// Checkpoint file for T2 amplitudes
    std::string t2_file_chk_;

    /// File name for T1 amplitudes to be saved in current directory
    std::string t1_file_cwd_;
    /// File name for T2 amplitudes to be saved in current directory
    std::string t2_file_cwd_;

    /// Dump amplitudes to current directory
    bool dump_amps_cwd_ = false;
    /// Read amplitudes from current directory
    bool read_amps_cwd_ = false;

    /// Dump the converged amplitudes to disk
    /// Iterative methods should override this function
    virtual void dump_amps_to_disk() {}
};

std::shared_ptr<DynamicCorrelationSolver>
make_dynamic_correlation_solver(const std::string& type, std::shared_ptr<ForteOptions> options,
                                std::shared_ptr<ForteIntegrals> ints,
                                std::shared_ptr<MOSpaceInfo> mo_space_info);

} // namespace forte

#endif // _dynamic_correlation_solver_h_
