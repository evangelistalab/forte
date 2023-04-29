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
#include "casscf/casscf_orb_grad.h"

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

    /// Compute the MCSCF energy
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

    /// Derivative type for nuclear gradient
    std::string der_type_;

    /// The printing level
    int print_;
    /// Enable debug printing or not
    bool debug_print_;

    /// Max number of macro iterations
    int maxiter_;
    /// Max number of micro iterations
    int micro_maxiter_;
    /// Min number of micro iterations
    int micro_miniter_;

    /// Optimize orbitals or not
    bool opt_orbs_;

    /// Max allowed value for orbital rotation
    double max_rot_;

    /// Keep internal (active-active) rotations
    bool internal_rot_;

    /// Orbital type for redundant pairs
    std::string orb_type_redundant_;

    /// Do DIIS extrapolation for orbitals and CI coefficients
    bool do_diis_;
    /// Iteration number to start adding error vectors
    int diis_start_;
    /// Min number of vectors in DIIS
    int diis_min_vec_;
    /// Max number of vectors in DIIS
    int diis_max_vec_;
    /// DIIS extrapolation frequency
    int diis_freq_;

    /// Energy convergence criteria
    double e_conv_;
    /// Orbital gradient convergence criteria
    double g_conv_;

    /// The name of CI solver
    std::string ci_type_;

    /// Final total energy
    double energy_;

    /// Solve CI coefficients for the current orbitals
    /// @param as_solver the pointer of ActiveSpaceSolver
    /// @param fci_ints the pointer of ActiveSpaceIntegrals
    /// @param params the parameters <print level, e_conv, r_conv, dump_wfn>
    /// @return averaged energy
    double diagonalize_hamiltonian(std::shared_ptr<ActiveSpaceSolver>& as_solver,
                                   std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                   const std::tuple<int, double, double, bool>& params);

    /// Test if we are doing a single-reference orbital optimization
    bool is_single_reference();

    /// Class to store iteration data
    struct CASSCF_HISTORY {
        CASSCF_HISTORY(double ec, double eo, double g, int n)
            : e_c(ec), e_o(eo), g_rms(g), n_micro(n) {}
        double e_c;   // energy from CI
        double e_o;   // energy after orbital optimization
        double g_rms; // RMS of gradient vector
        int n_micro;  // number of micro iteration
    };

    /// Test energy history and return if the energies are converging or not
    bool test_history(const std::vector<CASSCF_HISTORY>& history, const int& n_samples);

    /// Print iteration information
    void print_macro_iteration(const std::vector<CASSCF_HISTORY>& history);
};

std::unique_ptr<MCSCF_2STEP>
make_mcscf_two_step(const std::map<StateInfo, std::vector<double>>& state_weight_map,
                    std::shared_ptr<SCFInfo> ref_wfn, std::shared_ptr<ForteOptions> options,
                    std::shared_ptr<MOSpaceInfo> mo_space_info,
                    std::shared_ptr<ForteIntegrals> ints);

} // namespace forte

#endif // _mcscf_2step_h_
