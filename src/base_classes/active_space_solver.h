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

#ifndef _active_space_solver_h_
#define _active_space_solver_h_

#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"

#include "helpers/mo_space_info.h"
#include "integrals/integrals.h"

namespace psi {
namespace forte {

class ActiveSpaceSolver : public Wavefunction {
  public:
    // non-virtual interface
    ActiveSpaceSolver(SharedWavefunction ref_wfn, Options& options,
                      std::shared_ptr<ForteIntegrals> ints,
                      std::shared_ptr<MOSpaceInfo> mo_space_info);

    // equivalent to "this->solver_compute_energy()"
    double compute_energy() { return solver_compute_energy(); }

    // enable deletion of a Derived* through a Base*
    virtual ~ActiveSpaceSolver() = default;
    //    virtual ~ActiveSpaceSolver() {};

  protected:
    // pure virtual implementation
    virtual double solver_compute_energy() = 0;

    /// The molecular integrals object
    std::shared_ptr<ForteIntegrals> ints_;
    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
};
}
}

#endif // _active_space_solver_h_
