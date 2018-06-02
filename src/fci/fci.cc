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

#include "../forte_options.h"
#include "../helpers.h"
#include "fci_solver.h"

#include "fci.h"

namespace psi {
namespace forte {

void set_FCI_options(ForteOptions& foptions) {
    foptions.add_int("FCI_NROOT", 1, "The number of roots computed");
    foptions.add_int("FCI_ROOT", 0, "The root selected for state-specific computations");
    foptions.add_int("FCI_MAXITER", 30, "Maximum number of iterations for FCI code");
    foptions.add_int("FCI_MAX_RDM", 1, "The number of trial guess vectors to generate per root");
    foptions.add_bool("FCI_TEST_RDMS", false, "Test the FCI reduced density matrices?");
    foptions.add_bool("FCI_PRINT_NO", false, "Print the NO from the rdm of FCI");

    foptions.add_bool("CHUNK_SPACE_ENERGY", false, "Use only a small block of C for variational optemization");
    foptions.add_bool("SOLVE_IN_SUBSPACE", false, "Solve for a subspace after selection via block chop or string trim");

    foptions.add_bool("FCI_TILE_CHOPPER", false, "Chop out unimportant tiles based on some cutoff");
    foptions.add_int("FCI_TC_DIM", 0, "Chop tiles into this size");
    foptions.add_double("FCI_TC_CUT", 0.0, "Tile Chopper threshold");

    foptions.add_bool("FCI_STRING_TRIMMER", false, "Trim out unimportant strings based on some cutoff");
    foptions.add_double("FCI_ST_CUT", 0.0, "String trimmer threshold");

    foptions.add_bool("FCI_SVD", false, "Decompose and reconstruct the FCI wave function");
    foptions.add_str("FCI_SVD_TYPE", "THRESHOLD", {"THRESHOLD","CUMULATIVE"}, "Type of FCI decomposition");
    foptions.add_double("FCI_SVD_TAU", 0.0, "SVD threshold");


    // Says SVD N tiles but means dimension of SVD tile!
    foptions.add_bool("FCI_SVD_TILE", false, "Decompose and reconstruct the FCI wave function with tiles");
    foptions.add_str("FCI_SVD_TILE_TYPE", "THRESHOLD", {"THRESHOLD","CUMULATIVE"}, "Type of FCI-tile decomposition");
    foptions.add_double("FCI_SVD_OMEGA", 0.0, "SVD-tile threshold");
    foptions.add_int("FCI_SVD_N_TILES", 0, "SVD-tile tile dimension");

    foptions.add_int("FCI_NTRIAL_PER_ROOT", 10,
                     "The number of trial guess vectors to generate per root");
}

FCI::FCI(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
         std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ActiveSpaceSolver(ref_wfn, options, ints, mo_space_info) {
    // Copy the wavefunction information
    reference_wavefunction_ = ref_wfn;
    startup();
}

FCI::FCI(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
         std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<FCIIntegrals> fci_ints)
    : ActiveSpaceSolver(ref_wfn, options, ints, mo_space_info) {
    // Copy the wavefunction information
    reference_wavefunction_ = ref_wfn;
    startup();
    fci_ints_ = fci_ints;
}

// FCI::FCI(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
//         std::shared_ptr<MOSpaceInfo> mo_space_info)
//    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info) {
//    // Copy the wavefunction information
//    shallow_copy(ref_wfn);
//    reference_wavefunction_ = ref_wfn;

//    print_ = options_.get_int("PRINT");

//    startup();
//}

FCI::~FCI() {}

void FCI::set_max_rdm_level(int value) { max_rdm_level_ = value; }

void FCI::set_fci_iterations(int value) { fci_iterations_ = value; }

void FCI::print_no(bool value) { print_no_ = value; }

void FCI::set_ms(int ms) {
    set_ms_ = true;
    twice_ms_ = ms;
}

void FCI::startup() {
    print_ = options_.get_int("PRINT");

    if (print_)
        print_method_banner(
            {"String-based Full Configuration Interaction", "by Francesco A. Evangelista"});

    max_rdm_level_ = options_.get_int("FCI_MAX_RDM");
    fci_iterations_ = options_.get_int("FCI_MAXITER");
    print_no_ = options_.get_bool("FCI_PRINT_NO");
}

double FCI::solver_compute_energy() {
    timer method_timer("FCI:energy");
    Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
    size_t nfdocc = mo_space_info_->size("FROZEN_DOCC");
    std::vector<size_t> rdocc = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> active = mo_space_info_->get_corr_abs_mo("ACTIVE");

    int charge = Process::environment.molecule()->molecular_charge();
    if (options_["CHARGE"].has_changed()) {
        charge = options_.get_int("CHARGE");
    }

    int nel = 0;
    int natom = Process::environment.molecule()->natom();
    for (int i = 0; i < natom; i++) {
        nel += static_cast<int>(Process::environment.molecule()->Z(i));
    }
    // If the charge has changed, recompute the number of electrons
    // Or if you cannot find the number of electrons
    nel -= charge;

    int multiplicity = Process::environment.molecule()->multiplicity();
    if (options_["MULTIPLICITY"].has_changed()) {
        multiplicity = options_.get_int("MULTIPLICITY");
    }

    // If the user did not specify ms determine the value from the input or
    // take the lowest value consistent with the value of "MULTIPLICITY"
    if (not set_ms_) {
        if (options_["MS"].has_changed()) {
            twice_ms_ = std::round(2.0 * options_.get_double("MS"));
        } else {
            // Default: lowest spin solution
            twice_ms_ = (multiplicity + 1) % 2;
        }
    }

    //    if(ms < 0){
    //        outfile->Printf("\n  Ms must be no less than 0.");
    //        outfile->Printf("\n  Ms = %2d, MULTIPLICITY = %2d", ms,
    //        multiplicity);
    //        outfile->Printf("\n  Check (specify) Ms value (component of
    //        multiplicity)! \n");
    //        throw PSIEXCEPTION("Ms must be no less than 0. Check output for
    //        details.");
    //    }

    if (print_) {
        outfile->Printf("\n  Number of electrons: %d", nel);
        outfile->Printf("\n  Charge: %d", charge);
        outfile->Printf("\n  Multiplicity: %d", multiplicity);
        outfile->Printf("\n  Davidson subspace max dim: %d",
                        options_.get_int("DL_SUBSPACE_PER_ROOT"));
        outfile->Printf("\n  Davidson subspace min dim: %d",
                        options_.get_int("DL_COLLAPSE_PER_ROOT"));
        if (twice_ms_ % 2 == 0) {
            outfile->Printf("\n  M_s: %d", twice_ms_ / 2);
        } else {
            outfile->Printf("\n  M_s: %d/2", twice_ms_);
        }
    }

    if (((nel - twice_ms_) % 2) != 0)
        throw PSIEXCEPTION("\n\n  FCI: Wrong value of M_s.\n\n");

    // Adjust the number of for frozen and restricted doubly occupied
    size_t nactel = nel - 2 * nfdocc - 2 * rdocc.size();

    size_t na = (nactel + twice_ms_) / 2;
    size_t nb = nactel - na;

    //    outfile->Printf("\n  A");

    fcisolver_ = std::unique_ptr<FCISolver>(new FCISolver(
        active_dim, rdocc, active, na, nb, multiplicity, options_.get_int("ROOT_SYM"), ints_,
        mo_space_info_, options_.get_int("FCI_NTRIAL_PER_ROOT"), print_, options_));

    //    outfile->Printf("\n  B");
    // tweak some options
    fcisolver_->set_max_rdm_level(max_rdm_level_);
    fcisolver_->set_nroot(options_.get_int("FCI_NROOT"));
    fcisolver_->set_root(options_.get_int("FCI_ROOT"));
    fcisolver_->set_test_rdms(options_.get_bool("FCI_TEST_RDMS"));
    fcisolver_->set_fci_iterations(options_.get_int("FCI_MAXITER"));
    fcisolver_->set_collapse_per_root(options_.get_int("DL_COLLAPSE_PER_ROOT"));
    fcisolver_->set_subspace_per_root(options_.get_int("DL_SUBSPACE_PER_ROOT"));
    fcisolver_->set_print_no(print_no_);
    if (fci_ints_ != nullptr) {
        fcisolver_->use_user_integrals_and_restricted_docc(true);
        fcisolver_->set_integral_pointer(fci_ints_);
    }

    double fci_energy = fcisolver_->compute_energy();

    Process::environment.globals["CURRENT ENERGY"] = fci_energy;
    Process::environment.globals["FCI ENERGY"] = fci_energy;

    return fci_energy;
}

Reference FCI::reference() { return fcisolver_->reference(); }
}
}
