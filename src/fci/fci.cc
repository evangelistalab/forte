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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"

#include "forte_options.h"
#include "helpers/printing.h"
#include "fci_solver.h"

#include "fci.h"

using namespace psi;

namespace forte {

void set_FCI_options(ForteOptions& foptions) {
    foptions.add_int("FCI_NROOT", 1, "The number of roots computed");
    foptions.add_int("FCI_ROOT", 0, "The root selected for state-specific computations");
    foptions.add_int("FCI_MAXITER", 30, "Maximum number of iterations for FCI code");
    foptions.add_int("FCI_MAX_RDM", 1, "The number of trial guess vectors to generate per root");
    foptions.add_bool("FCI_TEST_RDMS", false, "Test the FCI reduced density matrices?");
    foptions.add_bool("FCI_PRINT_NO", false, "Print the NO from the rdm of FCI");
    foptions.add_int("FCI_NTRIAL_PER_ROOT", 10,
                     "The number of trial guess vectors to generate per root");
}

FCI::FCI(StateInfo state, std::shared_ptr<ForteIntegrals> ints,
         std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ActiveSpaceSolver(state, ints, mo_space_info) {
    startup();
}

FCI::FCI(StateInfo state, std::shared_ptr<ForteIntegrals> ints,
         std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<FCIIntegrals> fci_ints)
    : ActiveSpaceSolver(state, ints, mo_space_info) {
    startup();
    fci_ints_ = fci_ints;
}

FCI::~FCI() {}

void FCI::set_max_rdm_level(int value) { max_rdm_level_ = value; }

void FCI::set_fci_iterations(int value) { fci_iterations_ = value; }

void FCI::print_no(bool value) { print_no_ = value; }

void FCI::set_ms(int ms) {
    set_ms_ = true;
    twice_ms_ = ms;
}

// Where should this live?
// charge
// nel
// multiplicity
// ms
// na, nb

void FCI::startup() {
    print_ = ref_wfn_->options().get_int("PRINT");
    max_rdm_level_ = ref_wfn_->options().get_int("FCI_MAX_RDM");
    fci_iterations_ = ref_wfn_->options().get_int("FCI_MAXITER");
    print_no_ = ref_wfn_->options().get_bool("FCI_PRINT_NO");

    if (print_)
        print_method_banner(
            {"String-based Full Configuration Interaction", "by Francesco A. Evangelista"});
}

double FCI::solver_compute_energy() {
    timer method_timer("FCI:energy");
    psi::Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
    size_t nfdocc = mo_space_info_->size("FROZEN_DOCC");
    std::vector<size_t> rdocc = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> active = mo_space_info_->get_corr_abs_mo("ACTIVE");

    if (print_) {
        outfile->Printf("\n  Number of electrons: %d", nel);
        outfile->Printf("\n  Charge: %d", charge);
        outfile->Printf("\n  Multiplicity: %d", multiplicity);
        outfile->Printf("\n  Davidson subspace max dim: %d",
                        wfn->options().get_int("DL_SUBSPACE_PER_ROOT"));
        outfile->Printf("\n  Davidson subspace min dim: %d",
                        wfn->options().get_int("DL_COLLAPSE_PER_ROOT"));
        if (twice_ms_ % 2 == 0) {
            outfile->Printf("\n  M_s: %d", twice_ms_ / 2);
        } else {
            outfile->Printf("\n  M_s: %d/2", twice_ms_);
        }
    }

    fcisolver_ = std::unique_ptr<FCISolver>(new FCISolver(
        active_dim, rdocc, active, na_, nb_, multiplicity_, ref_wfn_->options().get_int("ROOT_SYM"),
        ints_, mo_space_info_, ref_wfn_->options().get_int("FCI_NTRIAL_PER_ROOT"), print_,
        ref_wfn_->options()));

    //    outfile->Printf("\n  B");
    // tweak some options
    fcisolver_->set_max_rdm_level(max_rdm_level_);
    fcisolver_->set_nroot(ref_wfn_->options().get_int("FCI_NROOT"));
    fcisolver_->set_root(ref_wfn_->options().get_int("FCI_ROOT"));
    fcisolver_->set_test_rdms(ref_wfn_->options().get_bool("FCI_TEST_RDMS"));
    fcisolver_->set_fci_iterations(ref_wfn_->options().get_int("FCI_MAXITER"));
    fcisolver_->set_collapse_per_root(ref_wfn_->options().get_int("DL_COLLAPSE_PER_ROOT"));
    fcisolver_->set_subspace_per_root(ref_wfn_->options().get_int("DL_SUBSPACE_PER_ROOT"));
    fcisolver_->set_print_no(print_no_);
    if (fci_ints_ != nullptr) {
        fcisolver_->use_user_integrals_and_restricted_docc(true);
        fcisolver_->set_integral_pointer(fci_ints_);
    }

    double fci_energy = fcisolver_->compute_energy();

    psi::Process::environment.globals["CURRENT ENERGY"] = fci_energy;
    psi::Process::environment.globals["FCI ENERGY"] = fci_energy;

    return fci_energy;
}

Reference FCI::reference() { return fcisolver_->reference(); }
} // namespace forte
