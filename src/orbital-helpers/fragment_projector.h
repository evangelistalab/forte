/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _fragment_projector_h_
#define _fragment_projector_h_

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/basisset.h"
#include "psi4/liboptions/liboptions.h"

#define _DEBUG_FRAGMENT_PROJECTOR_ 0

namespace forte {

/**
 * @brief The fragment_projector class
 *
 * A class to store information about a (system) fragment
 */

class FragmentProjector {
  public:
    // ==> Constructors <==

    // Simple constructor
    FragmentProjector(std::shared_ptr<psi::Molecule> molecule,
                      std::shared_ptr<psi::BasisSet> basis);
    // Constructor with minAO
    // FragmentProjector::FragmentProjector(std::shared_ptr<Molecule> molecule,
    // std::shared_ptr<BasisSet> minao_basis, 	std::shared_ptr<BasisSet> prime_basis);

    // Build projector and return AO basis matrix Pf_AO
    psi::SharedMatrix build_f_projector(std::shared_ptr<psi::Molecule> molecule,
                                        std::shared_ptr<psi::BasisSet> basis);

  private:
    /// The molecule
    std::shared_ptr<psi::Molecule> molecule_;
    /// The AO basis set
    std::shared_ptr<psi::BasisSet> basis_;

    /// Total number of basis functions (sys + env)
    int nbf_;
    /// Number of basis functions in the system
    int nbf_A_;
    /// Number of atoms in the system
    int natom_A_;

    /// The startup function
    void startup();
};

// Helper function
psi::SharedMatrix make_fragment_projector(psi::SharedWavefunction ref_wfn, psi::Options& options);

} // namespace forte
#endif // _fragment_projector_h_
