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

#include <fstream>

#include "psi4/libpsi4util/process.h"

#include "orbital-helpers/aosubspace.h"
#include "orbital-helpers/orbital_embedding.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"
#include "helpers/timer.h"

#include "sparse_ci/determinant.h"
#include "version.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/psi4-dec.h"

#ifdef HAVE_CHEMPS2
#include "dmrg/dmrgscf.h"
#include "dmrg/dmrgsolver.h"
#endif

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "forte.h"

using namespace psi;

namespace forte {

#ifdef HAVE_GA
/// These functions replace the Memory Allocator in GA with C/C++ allocator.
void* replace_malloc(size_t bytes, int, char*) { return malloc(bytes); }
void replace_free(void* ptr) { free(ptr); }
#endif

/**
 * @brief Initialize ambit, MPI, and GA. All functions that need to be called
 * once before running forte should go here.
 * @return The pair (my_proc,n_nodes)
 */
std::pair<int, int> startup() {
    ambit::initialize();

#ifdef HAVE_MPI
    MPI_Init(NULL, NULL);
#endif

    int my_proc = 0;
    int n_nodes = 1;
#ifdef HAVE_GA
    GA_Initialize();
    /// Use C/C++ memory allocators
    GA_Register_stack_memory(replace_malloc, replace_free);
    n_nodes = GA_Nnodes();
    my_proc = GA_Nodeid();
    size_t memory = psi::Process::environment.get_memory() / n_nodes;
#endif

#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_proc);
    MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
#endif
    return std::make_pair(my_proc, n_nodes);
}

/**
 * @brief Finalize ambit, MPI, and GA. All functions that need to be called
 * once after running forte should go here.
 */
void cleanup() {

#ifdef HAVE_GA
    GA_Terminate();
#endif

    ambit::finalize();

#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}

void banner() {
    outfile->Printf(
        "\n"
        "  Forte\n"
        "  ----------------------------------------------------------------------------\n"
        "  A suite of quantum chemistry methods for strongly correlated electrons\n\n"
        "    git branch: %s - git commit: %s\n\n"
        "  Developed by:\n"
        "    Francesco A. Evangelista, Chenyang Li, Kevin P. Hannon,\n"
        "    Jeffrey B. Schriber, Tianyuan Zhang, Chenxi Cai,\n"
        "    Nan He, Nicholas Stair, Shuhe Wang, Renke Huang\n"
        "  ----------------------------------------------------------------------------\n",
        GIT_BRANCH, GIT_COMMIT_HASH);
    outfile->Printf("\n  Size of Determinant class: %d bits", sizeof(Determinant) * 8);
}

} // namespace forte
