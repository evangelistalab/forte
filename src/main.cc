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

#include "aosubspace/aosubspace.h"
#include "avas.h"
#include "forte_options.h"
#include "helpers.h"
#include "integrals/integrals.h"
#include "stl_bitset_determinant.h"
#include "version.h"

#ifdef HAVE_CHEMPS2
#include "dmrgscf.h"
#include "dmrgsolver.h"
#endif

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "forte.h"

namespace psi {
namespace forte {

/**
 * @brief Read options from the input file. Called by psi4 before everything
 * else.
 */
extern "C" int read_options(std::string name, Options& options) {

    ForteOptions foptions; // <<

    forte_options(name, foptions);

    if (name == "FORTE" || options.read_globals()) {
        // Old way (deprecated) to pass options to Psi4
        forte_old_options(options);
        // New way to pass options to Psi4
        foptions.add_psi4_options(options);
    }

    return true;
}

/**
 * @brief The main forte function.
 */
extern "C" SharedWavefunction forte(SharedWavefunction ref_wfn, Options& options) {
    // Start a timer
    timer total_time("Forte");

    forte_banner();

    auto my_proc_n_nodes = forte_startup();
    int my_proc = my_proc_n_nodes.first;

    // Make a MOSpaceInfo object
    auto mo_space_info = make_mo_space_info(ref_wfn, options);

    // Make a subspace object
    SharedMatrix Ps = make_aosubspace_projector(ref_wfn, options);

    // Transform the orbitals
    make_avas(ref_wfn, options, Ps);

    // Transform integrals and run forte only if necessary
    if (options.get_str("JOB_TYPE") != "NONE") {
        // Make an integral object
        auto ints = make_forte_integrals(ref_wfn, options, mo_space_info);

        // Compute energy
        forte_old_methods(ref_wfn, options, ints, mo_space_info, my_proc);

//        outfile->Printf("\n\n  Your calculation took %.8f seconds\n", total_time.get());
    }

    forte_cleanup();

    return ref_wfn;
}

/**
 * @brief Initialize ambit, MPI, and GA. All functions that need to be called
 * once before running forte should go here.
 * @return The pair (my_proc,n_nodes)
 */
std::pair<int, int> forte_startup() {
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
    size_t memory = Process::environment.get_memory() / n_nodes;
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
void forte_cleanup() {

#ifdef HAVE_GA
    GA_Terminate();
#endif

    ambit::finalize();

#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}

std::shared_ptr<MOSpaceInfo> make_mo_space_info(SharedWavefunction ref_wfn, Options& options) {
    Dimension nmopi = ref_wfn->nmopi();
    auto mo_space_info = std::make_shared<MOSpaceInfo>(nmopi);
    mo_space_info->read_options(options);
    return mo_space_info;
}

SharedMatrix make_aosubspace_projector(SharedWavefunction ref_wfn, Options& options) {
    // Ps is a SharedMatrix Ps = S^{BA} X X^+ S^{AB}
    auto Ps = create_aosubspace_projector(ref_wfn, options);
    if (Ps) {

        SharedMatrix CPsC = Ps->clone();
        CPsC->transform(ref_wfn->Ca());
        outfile->Printf("\n  Orbital overlap with ao subspace:\n");
        outfile->Printf("    ========================\n");
        outfile->Printf("    Irrep   MO   <phi|P|phi>\n");
        outfile->Printf("    ------------------------\n");
        for (int h = 0; h < CPsC->nirrep(); h++) {
            for (int i = 0; i < CPsC->rowspi(h); i++) {
                outfile->Printf("      %1d   %4d    %.6f\n", h, i + 1, CPsC->get(h, i, i));
            }
        }
        outfile->Printf("    ========================\n");
    }
    return Ps;
}

std::shared_ptr<ForteIntegrals> make_forte_integrals(SharedWavefunction ref_wfn, Options& options,
                                                     std::shared_ptr<MOSpaceInfo> mo_space_info) {
    timer int_timer("Integrals");
    std::shared_ptr<ForteIntegrals> ints;
    if (options.get_str("INT_TYPE") == "CHOLESKY") {
        ints = std::make_shared<CholeskyIntegrals>(options, ref_wfn, UnrestrictedMOs,
                                                   RemoveFrozenMOs, mo_space_info);
    } else if (options.get_str("INT_TYPE") == "DF") {
        ints = std::make_shared<DFIntegrals>(options, ref_wfn, UnrestrictedMOs, RemoveFrozenMOs,
                                             mo_space_info);
    } else if (options.get_str("INT_TYPE") == "DISKDF") {
        ints = std::make_shared<DISKDFIntegrals>(options, ref_wfn, UnrestrictedMOs, RemoveFrozenMOs,
                                                 mo_space_info);
    } else if (options.get_str("INT_TYPE") == "CONVENTIONAL") {
        ints = std::make_shared<ConventionalIntegrals>(options, ref_wfn, UnrestrictedMOs,
                                                       RemoveFrozenMOs, mo_space_info);
    } else if (options.get_str("INT_TYPE") == "DISTDF") {
#ifdef HAVE_GA
        ints = std::make_shared<DistDFIntegrals>(options, ref_wfn, UnrestrictedMOs, RemoveFrozenMOs,
                                                 mo_space_info);
#endif
    } else if (options.get_str("INT_TYPE") == "OWNINTEGRALS") {
        ints = std::make_shared<OwnIntegrals>(options, ref_wfn, UnrestrictedMOs, RemoveFrozenMOs,
                                              mo_space_info);
    } else {
        outfile->Printf("\n Please check your int_type. Choices are CHOLESKY, DF, DISKDF , "
                        "DISTRIBUTEDDF Effective, CONVENTIONAL or OwnIntegrals");
        throw PSIEXCEPTION("INT_TYPE is not correct.  Check options");
    }

    if (options.get_bool("PRINT_INTS")) {
        ints->print_ints();
    }

    return ints;
}

void forte_banner() {
    outfile->Printf(
        "\n"
        "  Forte\n"
        "  ----------------------------------------------------------------------------\n"
        "  A suite of quantum chemistry methods for strongly correlated electrons\n\n"
        "    git branch: %s - git commit: %s\n\n"
        "  Developed by:\n"
        "    Francesco A. Evangelista, Chenyang Li, Kevin P. Hannon,\n"
        "    Jeffrey B. Schriber, Tianyuan Zhang, Chenxi Cai\n"
        "  ----------------------------------------------------------------------------\n",
        GIT_BRANCH, GIT_COMMIT_HASH);
}
}
} // End Namespaces
