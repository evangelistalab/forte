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

#include "forte_options.h"

namespace psi {
namespace forte {

void forte_options(std::string name, ForteOptions& options);

void forte_banner();
}
}

/// These functions replace the Memory Allocator in GA with C/C++ allocator.
void* replace_malloc(size_t bytes, int align, char* name) { return malloc(bytes); }
void replace_free(void* ptr) { free(ptr); }

namespace psi {
namespace forte {

std::pair<int, int> forte_startup();

void forte_cleanup();

std::shared_ptr<MOSpaceInfo> make_mo_space_info(SharedWavefunction ref_wfn, Options& options);

SharedMatrix make_aosubspace_projector(SharedWavefunction ref_wfn, Options& options);

std::shared_ptr<ForteIntegrals> make_forte_integrals(SharedWavefunction ref_wfn, Options& options,
                                                     std::shared_ptr<MOSpaceInfo> mo_space_info);

void make_ci_nos(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
                 std::shared_ptr<MOSpaceInfo> mo_space_info);

void forte_old_methods(SharedWavefunction ref_wfn, Options& options,
                       std::shared_ptr<ForteIntegrals> ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info, int my_proc);

void forte_old_options(Options& options);
}
} // End Namespaces
