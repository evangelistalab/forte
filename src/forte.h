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

//#include <cmath>
//#include <memory>

//#include "mini-boost/boost/format.hpp"
//#include <ambit/tensor.h>

//#include "psi4/psi4-dec.h"
//#include "psi4/psifiles.h"
//#include "psi4/libdpd/dpd.h"
//#include "psi4/libpsio/psio.hpp"
//#include "psi4/libtrans/integraltransform.h"
//#include "psi4/libmints/wavefunction.h"
//#include "psi4/libmints/molecule.h"

//#include "helpers.h"
//#include "aosubspace.h"
//#include "multidimensional_arrays.h"
//#include "mp2_nos.h"
//#include "aci.h"
//#include "pci.h"
//#include "fcimc.h"
//#include "fci_mo.h"
//#include "mrdsrg.h"
//#include "mrdsrg_so.h"
//#include "dsrg_mrpt2.h"
//#include "dsrg_mrpt3.h"
//#include "three_dsrg_mrpt2.h"
//#include "tensorsrg.h"
//#include "mcsrgpt2_mo.h"
//#include "fci_solver.h"
//#include "blockedtensorfactory.h"
//#include "sq.h"
//#include "so-mrdsrg.h"
//#include "dsrg_wick.h"
//#include "casscf.h"
//#include "finite_temperature.h"
//#include "active_dsrgpt2.h"
//#include "dsrg_mrpt.h"
//#include "v2rdm.h"
//#include "localize.h"
//#include "cc.h"

//#ifdef HAVE_CHEMPS2
//#include "dmrgscf.h"
//#include "dmrgsolver.h"
//#endif

//#ifdef HAVE_GA
//#include <ga.h>
//#include <macdecls.h>
//#endif

//#ifdef HAVE_MPI
//#include <mpi.h>
//#endif

void forte_options(std::string name, psi::Options& options);

/// These functions replace the Memory Allocator in GA with C/C++ allocator.
void* replace_malloc(size_t bytes, int align, char* name) {
    return malloc(bytes);
}
void replace_free(void* ptr) { free(ptr); }

namespace psi {
namespace forte {

std::pair<int, int> forte_startup();

void forte_cleanup();

std::shared_ptr<MOSpaceInfo> make_mo_space_info(SharedWavefunction ref_wfn,
                                                Options& options);

SharedMatrix make_aosubspace_projector(SharedWavefunction ref_wfn,
                                       Options& options);

std::shared_ptr<ForteIntegrals>
make_forte_integrals(SharedWavefunction ref_wfn, Options& options,
                     std::shared_ptr<MOSpaceInfo> mo_space_info);

void forte_old_methods(SharedWavefunction ref_wfn, Options& options,
                       std::shared_ptr<ForteIntegrals> ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info, int my_proc);

void forte_old_options(Options& options);
}
} // End Namespaces
