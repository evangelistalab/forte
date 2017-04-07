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
//#include "../sparse_ci_solver.h"
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
}

CINO::CINO(SharedWavefunction ref_wfn, Options& options,
           std::shared_ptr<ForteIntegrals> ints,
           std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info) {
    // Copy the wavefunction information
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;
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

std::vector<Determinant> CINO::build_dets() {
    std::vector<Determinant> dets;
    return dets;
}

std::pair<SharedVector, SharedMatrix>
CINO::diagonalize_hamiltonian(const std::vector<Determinant>& dets) {
    std::pair<SharedVector, SharedMatrix> evals_evecs;
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
