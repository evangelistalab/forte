/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "base_classes/state_info.h"
#include "base_classes/scf_info.h"
//#include "base_classes/reference.h"

namespace forte {

class ActiveSpaceIntegrals;
class ForteIntegrals;
class MOSpaceInfo;
class Reference;

/**
 * @class ActiveSpaceSolver
 *
 * @brief Base class for active space solvers
 *
 * This class is the base class for methods that solve for the wavefunction in a
 * small subset of the full orbital space (<30-40 orbitals).
 * This class is responsible for creating and storing the integrals used by
 * active space solvers, which are held by an ActiveSpaceIntegrals object.
 *
 * @note By default, this class assumes that the active orbitals are stored in the MOSpaceInfo
 * object in the space labeled "ACTIVE". Orbitals in the space "RESTRICTED_DOCC"
 * are not correlated and are trated via effective scalar and one-body interactions.
 */
class ActiveSpaceSolver {
  public:
    // Constructor for a single state computation
    /**
     * @brief ActiveSpaceSolver
     * @param state information about the electronic state
     * @param mo_space_info a MOSpaceInfo object
     * @param ints
     */
    ActiveSpaceSolver(StateInfo state, std::shared_ptr<MOSpaceInfo> mo_space_info,
                      std::shared_ptr<ForteIntegrals> ints);
    // Constructor for a multi-state computation
    ActiveSpaceSolver(const std::vector<std::pair<StateInfo, double>>& states_weights,
                      std::shared_ptr<MOSpaceInfo> mo_space_info,
                      std::shared_ptr<ForteIntegrals> ints);

    // Default constructor
    ActiveSpaceSolver() = default;

    /// Virtual destructor to enable deletion of a Derived* through a Base*
    virtual ~ActiveSpaceSolver() = default;

    /// Compute the energy and return it
    virtual double compute_energy() = 0;

    /// Returns the reference
    virtual Reference get_reference() = 0;

    /// Set options from an option object
    /// @param options the options passed in
    virtual void set_options(std::shared_ptr<ForteOptions> options) = 0;

    /// Pass a set of ActiveSpaceIntegrals to the solver (e.g. an effective Hamiltonian)
    /// @param as_ints the integrals passed in
    void set_active_space_integrals(std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
        as_ints_ = as_ints;
    }

  protected:
    /// The orbital space held in MOSpaceInfo that defines the active orbitals
    std::string active_mo_space_ = "ACTIVE";

    /// The orbital space held in MOSpaceInfo that defines the core orbitals (doubly occupied)
    std::string core_mo_space_ = "RESTRICTED_DOCC";

    /// The list of active orbitals (absolute ordering)
    std::vector<size_t> active_mo_;

    /// The list of doubly occupied orbitals (absolute ordering)
    std::vector<size_t> core_mo_;

    /// A list of electronic states and their weights
    std::vector<std::pair<StateInfo, double>> states_weights_;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The molecular integrals object
    std::shared_ptr<ForteIntegrals> ints_;

    /// The molecular integrals for the active space
    /// This object holds only the integrals for the orbital contained in the active_mo_ vector.
    /// The one-electron integrals and scalar energy contains contributions from the
    /// doubly occupied orbitals specified by the core_mo_ vector.
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;

    /// Allocates an ActiveSpaceIntegrals object and fills it with integrals stored in ints_
    void make_active_space_ints();
};

/**
 * @brief make_active_space_solver Make an active space solver object
 * @param type a string that specifies the type (e.g. "FCI", "ACI", ...)
 * @param state information about the elecronic state
 * @param scf_info information about a previous SCF computation
 * @param options user-provided options
 * @param ints an integral object
 * @param mo_space_info
 * @return a shared pointer for the base class ActiveSpaceSolver
 */
std::shared_ptr<ActiveSpaceSolver> make_active_space_solver(
    const std::string& type, StateInfo state, std::shared_ptr<SCFInfo> scf_info,
    std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ForteIntegrals> ints,
    std::shared_ptr<ForteOptions> options);

} // namespace forte

#endif // _active_space_solver_h_
