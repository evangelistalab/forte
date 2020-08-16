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

#ifndef _mcscf_2step_h_
#define _mcscf_2step_h_

#include <vector>
#include <string>

#include "base_classes/active_space_solver.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/scf_info.h"
#include "base_classes/state_info.h"
#include "integrals/integrals.h"

namespace forte {

class MCSCF_2STEP {
  public:
    /**
     * @brief Constructor of the AO-based CASSCF class
     * @param state_weights_map: The state to weights map of Forte
     * @param options: The ForteOptions pointer
     * @param mo_space_info: The MOSpaceInfo pointer of Forte
     * @param scf_info: The SCF_INFO pointer of Forte
     * @param ints: The ForteIntegral pointer
     *
     * Implementation notes:
     *   See J. Chem. Phys. 142, 224103 (2015) and Theor. Chem. Acc. 97, 88-95 (1997)
     */
    MCSCF_2STEP(const std::map<StateInfo, std::vector<double>>& state_weights_map,
                std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
                std::shared_ptr<forte::SCFInfo> scf_info, std::shared_ptr<ForteIntegrals> ints);

    /// Compute the CASSCF_NEW energy
    double compute_energy();

  private:
    /// The list of states to computed. Passed to the ActiveSpaceSolver
    std::map<StateInfo, std::vector<double>> state_weights_map_;

    /// The Forte options
    std::shared_ptr<ForteOptions> options_;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// SCF information
    std::shared_ptr<SCFInfo> scf_info_;

    /// The Forte integral
    std::shared_ptr<ForteIntegrals> ints_;

    /// Common setup for the class
    void startup();

    /// Read options
    void read_options();

    /// Print options
    void print_options();

    // => Options <=

    /// Integral type
    std::string int_type_;

    /// The printing level
    int print_;
    /// Enable debug printing or not
    bool debug_print_;

    /// Max number of macro iterations
    int maxiter_;
    /// Max number of micro iterations
    int micro_maxiter_;

    /// Energy convergence criteria
    double e_conv_;
    /// Orbital gradient convergence criteria
    double g_conv_;

    /// The name of CI solver
    std::string ci_type_;

    /// Max allowed value for orbital rotation
    double max_rot_;

    /// Keep internal (active-active) rotations
    bool internal_rot_;

    /// Orbital type for redundant pairs
    std::string orb_type_redundant_;

    /// Final total energy
    double energy_;

    /// Solve CI coefficients for the current orbitals
    std::unique_ptr<ActiveSpaceSolver>
    diagonalize_hamiltonian(std::shared_ptr<ActiveSpaceIntegrals> fci_ints, const int print,
                            double& e_c);
};

std::unique_ptr<MCSCF_2STEP>
make_mcscf_two_step(const std::map<StateInfo, std::vector<double>>& state_weight_map,
                    std::shared_ptr<SCFInfo> ref_wfn, std::shared_ptr<ForteOptions> options,
                    std::shared_ptr<MOSpaceInfo> mo_space_info,
                    std::shared_ptr<ForteIntegrals> ints);

} // namespace forte

#endif // _mcscf_2step_h_
