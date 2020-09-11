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

#include "psi4/libpsi4util/process.h"
//#include "psi4/libmints/molecule.h"

//#include "boost/format.hpp"

#include "base_classes/rdms.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"

#include "integrals/active_space_integrals.h"
//#include "sparse_ci/determinant.h"
//#include "sparse_ci/determinant_functions.hpp"
//#include "helpers/iterative_solvers.h"

//#include "helpers/helpers.h"
#include "helpers/printing.h"

#include "external_active_space_method.h"

//#ifdef HAVE_GA
//#include <ga.h>
//#include <macdecls.h>
//#endif

//#include "psi4/psi4-dec.h"

// using namespace psi;

// int fci_debug_level = 4;

namespace forte {

class MOSpaceInfo;

ExternalActiveSpaceMethod::ExternalActiveSpaceMethod(StateInfo state, size_t nroot,
                                                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                                                     std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints) {
    size_t ninact_docc = mo_space_info->size("INACTIVE_DOCC");
    na_ = state.na() - ninact_docc;
    nb_ = state.nb() - ninact_docc;
}

void ExternalActiveSpaceMethod::set_options(std::shared_ptr<ForteOptions> options) {
    //    set_e_convergence(options->get_double("E_CONVERGENCE"));
    //    set_r_convergence(options->get_double("R_CONVERGENCE"));
    //    set_print(options->get_int("PRINT"));
    //    set_root(options->get_int("ROOT"));
}

double ExternalActiveSpaceMethod::compute_energy() {
    print_method_banner({"External Active Space Solver"});

    // call python

    double energy = 0.0;
    energies_.push_back(0.0);

    psi::Process::environment.globals["CURRENT ENERGY"] = energy;
    psi::Process::environment.globals["FCI ENERGY"] = energy;

    return energy;
}

std::vector<RDMs>
ExternalActiveSpaceMethod::rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                int max_rdm_level) {
    throw std::runtime_error("ExternalActiveSpaceMethod::rdms is not implemented!");
    std::vector<RDMs> refs;
    if (max_rdm_level <= 0)
        return refs;
    return refs;
}

std::vector<RDMs> ExternalActiveSpaceMethod::transition_rdms(
    const std::vector<std::pair<size_t, size_t>>& /*root_list*/,
    std::shared_ptr<ActiveSpaceMethod> /*method2*/, int /*max_rdm_level*/) {
    std::vector<RDMs> refs;
    throw std::runtime_error("ExternalActiveSpaceMethod::transition_rdms is not implemented!");
    return refs;
}

} // namespace forte
