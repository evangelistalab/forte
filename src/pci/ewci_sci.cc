/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER,
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

#include "ewci_sci.h"

namespace forte {

EWCI_SCI::EWCI_SCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                   std::shared_ptr<MOSpaceInfo> mo_space_info,
                   std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : SelectedCIMethod(state, nroot, scf_info, mo_space_info, as_ints) {}

void EWCI_SCI::set_options(std::shared_ptr<ForteOptions> options) {}

void EWCI_SCI::print_info() {
    psi::outfile->Printf("\n\n\t  ---------------------------------------------------------");
    psi::outfile->Printf("\n\t              Element wise Configuration Interaction"
                    "implementation");
    psi::outfile->Printf("\n\t         by Francesco A. Evangelista and Tianyuan Zhang");
    psi::outfile->Printf("\n\t                      version Aug. 3 2017");
    psi::outfile->Printf("\n\t                    %4d thread(s) %s", num_threads_,
                    have_omp_ ? "(OMP)" : "");
    psi::outfile->Printf("\n\t  ---------------------------------------------------------");

    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{
        {"Symmetry", wavefunction_symmetry_},
        {"Multiplicity", wavefunction_multiplicity_},
        {"Number of roots", nroot_},
        {"Root used for properties", options_->get_int("ROOT")},
        {"Maximum number of iterations", maxiter_},
        {"Energy estimation frequency", energy_estimate_freq_},
        {"Number of threads", num_threads_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Time step (beta)", time_step_},
        {"Spawning threshold", spawning_threshold_},
        {"Initial guess spawning threshold", initial_guess_spawning_threshold_},
        {"Convergence threshold", e_convergence_},
        {"Energy estimate tollerance", energy_estimate_threshold_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Generator type", generator_description_},
        {"Importance functional", functional_description_},
        {"Shift the energy", do_shift_ ? "YES" : "NO"},
        {"Use intermediate normalization", use_inter_norm_ ? "YES" : "NO"},
        {"Fast variational estimate", fast_variational_estimate_ ? "YES" : "NO"},
        {"Result perturbation analysis", do_perturb_analysis_ ? "YES" : "NO"},
        {"Using OpenMP", have_omp_ ? "YES" : "NO"},
    };

    // Print some information
    psi::outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info) {
        psi::outfile->Printf("\n    %-39s %10d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        psi::outfile->Printf("\n    %-39s %10.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        psi::outfile->Printf("\n    %-39s %10s", str_dim.first.c_str(), str_dim.second.c_str());
    }
}

void EWCI_SCI::pre_iter_preparation() {}

void EWCI_SCI::diagonalize_P_space() {}

void EWCI_SCI::find_q_space() {}

void EWCI_SCI::diagonalize_PQ_space() {}

bool EWCI_SCI::check_convergence() {}

void EWCI_SCI::prune_PQ_to_P() {}

void EWCI_SCI::post_iter_process() {}

void EWCI_SCI::set_method_variables(
    std::string ex_alg, size_t nroot_method, size_t root,
    std::vector<std::vector<std::pair<Determinant, double>>> old_roots) {}

DeterminantHashVec EWCI_SCI::get_PQ_space() {}
psi::SharedMatrix EWCI_SCI::get_PQ_evecs() {}
psi::SharedVector EWCI_SCI::get_PQ_evals() {}
WFNOperator EWCI_SCI::get_op() {}
size_t EWCI_SCI::get_ref_root() {}
std::vector<double> EWCI_SCI::get_multistate_pt2_energy_correction() {}
}
