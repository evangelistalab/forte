/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "psi4/psi4-dec.h"
//#include "psi4/libmints/molecule.h"
//#include "psi4/libmints/pointgrp.h"
//#include "psi4/libpsio/psio.hpp"

#include "ci-no.h"
//#include "../ci_rdms.h"
//#include "../fci/fci_integrals.h"
#include "../sparse_ci_solver.h"
#include "../stl_bitset_determinant.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

void set_CINO_options(ForteOptions& foptions) {
    foptions.add_bool("CINO", false, "Do a CINO computation?");
    foptions.add_str("CINO_TYPE", "CIS", {"CIS", "CISD"},
                     "The type of wave function.");
    foptions.add_int("CINO_NROOT", 1, "The number of roots computed");

}

CINO::CINO(SharedWavefunction ref_wfn, Options& options,
           std::shared_ptr<ForteIntegrals> ints,
           std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info) {
    // Copy the wavefunction information
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    auto fci_ints_ = std::make_shared<FCIIntegrals>(
        ints, mo_space_info_->get_corr_abs_mo("ACTIVE"),
        mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));

    auto active_mo = mo_space_info_->get_corr_abs_mo("ACTIVE");
    ambit::Tensor tei_active_aa =
        ints->aptei_aa_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_ab =
        ints->aptei_ab_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_bb =
        ints->aptei_bb_block(active_mo, active_mo, active_mo, active_mo);

    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, 
                                    tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();

    STLBitsetDeterminant::set_ints(fci_ints_);
    startup();
}

CINO::~CINO() {}

double CINO::compute_energy() {
    outfile->Printf("\n\n  Computing CIS natural orbitals\n");

    // 1. Build the space of determinants
    std::vector<Determinant> dets = build_dets();

    // 2. Diagonalize the Hamiltonian in this basis
    std::pair<SharedVector, SharedMatrix> evals_evecs =
        diagonalize_hamiltonian(dets);

    // 3. Build the density matrix
    SharedMatrix gamma = build_density_matrix(dets, evals_evecs.second);

    // 4. Diagonalize the density matrix
    std::pair<SharedVector, SharedMatrix> no_U =
        diagonalize_density_matrix(gamma);

    // 5. Find optimal active space and transform the orbitals
    find_active_space_and_transform(no_U);

    return 0.0;
}

void CINO::startup(){
    wavefunction_multiplicity_ = 1;
    if (options_["MULTIPLICITY"].has_changed()) {
        wavefunction_multiplicity_ = options_.get_int("MULTIPLICITY");
    }
    diag_method_ = DLSolver;
    if (options_["DIAG_ALGORITHM"].has_changed()) {
        if (options_.get_str("DIAG_ALGORITHM") == "FULL") {
            diag_method_ = Full;
        } else if (options_.get_str("DIAG_ALGORITHM") == "DLSTRING") {
            diag_method_ = DLString;
        } else if (options_.get_str("DIAG_ALGORITHM") == "DLDISK") {
            diag_method_ = DLDisk;
        }
    }
    nroot_ = options_.get_int("CINO_NROOT");
}

std::vector<Determinant> CINO::build_dets() {
    std::vector<Determinant> dets;

    // build the reference determinant
    size_t nactv = mo_space_info_->size("ACTIVE");
    size_t nrdocc = mo_space_info_->size("RESTRICTED_DOCC");
    int naocc = nalpha_ - nrdocc;
    int nbocc = nbeta_ - nrdocc;
    int navir = nactv - naocc;
    int nbvir = nactv - nbocc;
    std::vector<bool> occupation_a(nactv);
    std::vector<bool> occupation_b(nactv);
    for (int i = 0; i < naocc; i++) {
        occupation_a[i] = true;
    }
    for (int i = 0; i < nbocc; i++) {
        occupation_b[i] = true;
    }
    Determinant ref(occupation_a, occupation_b);
    ref.print();

    // add the reference determinant
    dets.push_back(ref);

    // alpha-alpha single excitation
    for (int i = 0; i < naocc; ++i){
        for (int a = naocc; a < nactv; ++a){
            Determinant single_ia(ref);
            single_ia.set_alfa_bit(i,false);
            single_ia.set_alfa_bit(a,true);
            single_ia.print();
            dets.push_back(single_ia);
        }
    }
    // beta-beta single excitation
    for(int i = 0; i < nbocc; ++i){
        for(int b = nbocc; b < nactv; ++b){
            Determinant single_ib(ref);
            single_ib.set_beta_bit(i,false);
            single_ib.set_beta_bit(b,true);
            single_ib.print();
            dets.push_back(single_ib);
        }
    }

    // CiCi: add beta/beta singles and put determinants in the vector
    return dets;
}

std::pair<SharedVector, SharedMatrix>
CINO::diagonalize_hamiltonian(const std::vector<Determinant>& dets) {
    std::pair<SharedVector, SharedMatrix> evals_evecs;
    // CiCi: talk to Jeff about connecting his code to diagonalize the Hamiltonian

    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);
    sparse_solver.set_e_convergence(options_.get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_.get_int("DL_MAXITER"));
    sparse_solver.set_spin_project(project_out_spin_contaminants_);
    sparse_solver.set_guess_dimension(options_.get_int("DL_GUESS_SIZE"));
    sparse_solver.set_spin_project_full(false);
    sparse_solver.set_print_details(true);

    SharedMatrix evecs;
    SharedVector evals;

    sparse_solver.diagonalize_hamiltonian(dets, evals, evecs, nroot_,
                                          wavefunction_multiplicity_, DLSolver);

    // CiCi: print first 10 excited state energies and check with CIS results (York?)
    //for(int i = 0; i < 10; ++i){
     // outfile->Printf("\n%12f",evals_evecs.first->get(i));
//     }
//    for(auto d:dets){
//    d.print();
//    }
    evals->print();

    return evals_evecs;
}

SharedMatrix CINO::build_density_matrix(const std::vector<Determinant>& dets,
                                        SharedMatrix evecs) {
    SharedMatrix gamma;
    return gamma;
}

/// Diagonalize the density matrix
std::pair<SharedVector, SharedMatrix>
CINO::diagonalize_density_matrix(SharedMatrix gamma) {
    std::pair<SharedVector, SharedMatrix> no_U;
    return no_U;
}

/// Find optimal active space and transform the orbitals
void CINO::find_active_space_and_transform(
    std::pair<SharedVector, SharedMatrix> no_U) {}
}
} // EndNamespaces
