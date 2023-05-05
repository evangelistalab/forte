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

#ifndef _external_active_space_method_h_
#define _external_active_space_method_h_

#include "base_classes/active_space_method.h"

namespace forte {

/**
 * @class ExternalActiveSpaceMethod
 *
 * @brief Interface for external codes
 *
 */
class ExternalActiveSpaceMethod : public ActiveSpaceMethod {
  public:
    // ==> Class Constructor and Destructor <==
    /**
     * @brief ActiveSpaceMethod Constructor for a single state computation
     * @param state the electronic state to compute
     * @param nroot the number of roots
     * @param mo_space_info a MOSpaceInfo object that defines the orbital spaces
     * @param as_ints molecular integrals defined only for the active space orbitals
     */
    ExternalActiveSpaceMethod(StateInfo state, size_t nroot,
                              std::shared_ptr<MOSpaceInfo> mo_space_info,
                              std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    ~ExternalActiveSpaceMethod() = default;

    // ==> Class Interface <==

    /// Compute the FCI energy
    double compute_energy() override;

    /// Returns the reduced density matrices up to a given rank (max_rdm_level)
    std::vector<std::shared_ptr<RDMs>> rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                            int max_rdm_level, RDMsType type) override;

    /// Returns the transition reduced density matrices between roots of different symmetry up to a
    /// given level (max_rdm_level)
    std::vector<std::shared_ptr<RDMs>>
    transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                    std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level,
                    RDMsType type) override;

    /// Set the options
    void set_options(std::shared_ptr<ForteOptions> options) override;

    /// The number of alpha electrons
    size_t na_;
    /// The number of beta electrons
    size_t nb_;
    /// The number of active orbitals
    size_t nactv_;

  private:
    ///  Whether gamma2 are stored
    bool twopdc_;
    ///  Whether gamma3 are stored
    bool threepdc_;
    /// The alpha 1-RDM
    ambit::Tensor g1a_;
    /// The beta 1-RDM
    ambit::Tensor g1b_;
    /// The alpha-alpha 2-RDM
    ambit::Tensor g2aa_;
    /// The alpha-beta 2-RDM
    ambit::Tensor g2ab_;
    /// The beta-beta 2-RDM
    ambit::Tensor g2bb_;
    /// The alpha-alpha-alpha 3-RDM
    ambit::Tensor g3aaa_;
    /// The alpha-alpha-beta 3-RDM
    ambit::Tensor g3aab_;
    /// The alpha-beta-beta 3-RDM
    ambit::Tensor g3abb_;
    /// The beta-beta-beta 3-RDM
    ambit::Tensor g3bbb_;
};

} // namespace forte

#endif // _external_active_space_method_h_
