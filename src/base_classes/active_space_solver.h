/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _active_space_solver_h_
#define _active_space_solver_h_

#include <vector>
#include <string>

#include "base_classes/state_info.h"

namespace forte {

class ActiveSpaceMethod;
class ActiveSpaceIntegrals;
class ForteIntegrals;
class ForteOptions;
class MOSpaceInfo;
class Reference;
class SCFInfo;

/**
 * @class ActiveSpaceSolver
 *
 * @brief General class for a multi-state active space solver
 *
 * This class can run state-specific, multi-state, and state-averaged computations
 * on small subset of the full orbital space (<30-40 orbitals).
 */
class ActiveSpaceSolver {
  public:
    // ==> Class Constructor and Destructor <==
    /**
     * @brief ActiveSpaceMethod Constructor for a multi-state computation
     * @param states_weights A list of electronic states and their weights stored as vector of
     *        pairs [(state_1, [w_11, w_12, ..., w_1m]), (state_2, [w_21, w_22, ..., w_n]), ...]
     *        where:
     *            state_i specifies the symmetry of a state
     *            w_ij is the weight of the j-th state of symmetry state_i
     * @param state information about the electronic state
     * @param mo_space_info a MOSpaceInfo object
     * @param as_ints integrals for active space
     */
    ActiveSpaceSolver(const std::string& method,
                      std::vector<std::pair<StateInfo, std::vector<double>>>& state_weights_list,
                      std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<MOSpaceInfo> mo_space_info,
                      std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                      std::shared_ptr<ForteOptions> options);

    // ==> Class Interface <==

    /// Compute the energy and return it
    const std::vector<std::pair<StateInfo, std::vector<double>>>& compute_energy();

    /// Compute reference and return it
    Reference reference();

    /// Sets the maximum order RDM/cumulant
    void set_max_rdm_level(size_t value);

    /// Print a summary of the computation information
    void print_options();

    const std::vector<std::pair<StateInfo, std::vector<double>>>& get_state_weights_list() const {
        return state_weights_list_;
    }

    const std::vector<std::shared_ptr<ActiveSpaceMethod>>& get_method_vec() const {
        return method_vec_;
    }

    const std::vector<std::pair<StateInfo, std::vector<double>>>& state_energies_list() const {
        return state_energies_list_;
    }

  protected:
    // a string that specifies the method used (e.g. "FCI", "ACI", ...)
    std::string method_;

    /// A list of electronic states and their weights stored as vector of pairs
    ///   [(state_1, [w_11, w_12, ..., w_1m]), (state_2, [w_21, w_22, ..., w_n]), ...]
    /// where state_i specifies the symmetry of a state and w_ij are are list of weights
    /// for the states of the same symmetry
    std::vector<std::pair<StateInfo, std::vector<double>>> state_weights_list_;

    /// The information about a previous SCF computation
    std::shared_ptr<SCFInfo> scf_info_;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The molecular integrals for the active space
    /// This object holds only the integrals for the orbital contained in the
    /// active_mo_vector.
    /// The one-electron integrals and scalar energy contains contributions from the
    /// doubly occupied orbitals specified by the core_mo_ vector.
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;

    /// User-provided options
    std::shared_ptr<ForteOptions> options_;

    /// A vector of pointers to the ActiveSpaceMethod instantiated for each
    /// of the state symmetries contained in state_weights_list_
    std::vector<std::shared_ptr<ActiveSpaceMethod>> method_vec_;

    /// The maximum order RDM/cumulant to use for all ActiveSpaceMethod objects initialized
    size_t max_rdm_level_ = 1;

    /// Controls which defaulr rdm level to use
    bool set_rdm_ = false; // TODO: remove this hack

    /// Prints a summary of the energies with State info
    void print_energies(std::vector<std::pair<StateInfo, std::vector<double>>>& energies);

    //     * @param states_weights A list of electronic states and their weights stored as vector of
    //     *        pairs [(state_1, [w_11, w_12, ..., w_1m]), (state_2, [w_21, w_22, ..., w_n]),
    //     ...]
    //     *        where:
    //     *            state_i specifies the symmetry of a state
    //     *            w_ij is the weight of the j-th state of symmetry state_i
    /**
     * @brief state_energies_list
     */
    std::vector<std::pair<StateInfo, std::vector<double>>> state_energies_list_;
};

/**
 * @brief Make an active space solver object.
 * @param type a string that specifies the type (e.g. "FCI", "ACI", ...)
 * @param state information about the elecronic state
 * @param scf_info information about a previous SCF computation
 * @param mo_space_info orbital space information
 * @param ints an integral object
 * @param options user-provided options
 * @return a unique pointer for the base class ActiveSpaceMethod
 */
std::unique_ptr<ActiveSpaceSolver> make_active_space_solver(
    const std::string& method,
    std::vector<std::pair<StateInfo, std::vector<double>>>& state_weights_list,
    std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<MOSpaceInfo> mo_space_info,
    std::shared_ptr<ActiveSpaceIntegrals> as_ints, std::shared_ptr<ForteOptions> options);

/**
 * @brief Make a list of states and weights to pass to create an ActiveSpaceSolver object.
 * @param options user-provided options
 * @param wfn a psi wave function
 * @return a unique pointer to an ActiveSpaceSolver object
 */
std::vector<std::pair<StateInfo, std::vector<double>>>
make_state_weights_list(std::shared_ptr<ForteOptions> options,
                        std::shared_ptr<psi::Wavefunction> wfn);

double compute_average_state_energy(
    std::vector<std::pair<StateInfo, std::vector<double>>> state_energies_list,
    std::vector<std::pair<StateInfo, std::vector<double>>> state_weight_list);

} // namespace forte

#endif // _active_space_solver_h_
