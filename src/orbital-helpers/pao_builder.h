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

#ifndef _pao_builder_h_
#define _pao_builder_h_

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/basisset.h"

#include "psi4/libmints/dimension.h"

#define _DEBUG_PAO_BUILDER_ 0

namespace forte {

/**
 * @brief The PAObuilder class
 *
 * A class to build projected atomic orbitals for virtual space.
 *
 * The PAO builder will build the PAOs by:
 *
 * 1. Build density (D_) and AO overlap (S_)
 * 2. Build PAOs using C_pao = I - D_ * S_(truncated according to nbf_A)
 * 3. Orthogonalize PAOs, truncate the number of PAOs to n_pao, according to PAO_THRESHOLD
 * 4. Return a rectangle matrix C with size nmo*n_pao
 *
 * Typical usage:
 *
 *    // Build projected atomic orbitals for fragment virtual space
 *    double tau = options.get_double("PAO_THRESHOLD"); // set a PAO threshold
 *
 *    PAObuilder pao(ref_wfn->Ca(), dimension_of_occupied_orbital, ref_wfn->basisset());
 *    SharedMatrix C_pao = pao.build_A_virtual(nbf_A, tau); // nbf_A: number of fragment AO basis
 *
 *    // Replace the original virtual orbitals with PAOs
 *    ref_wfn->Ca()->set_block(nmopi, dimension_of_fragment_virtual, C_pao);
 *
 */

class PAObuilder {
  public:
    // ==> Constructor <==
    PAObuilder(psi::SharedMatrix C, psi::Dimension noccpi, std::shared_ptr<psi::BasisSet> basis);

    // Build C_PAO_A, return virtual orbitals within a fragment (rectangle shape: nmo*npao)
    psi::SharedMatrix build_A_virtual(int nbf_A, double pao_threshold);

    // Build C_PAO_B, return other orthogonalized virtual orbitals (rectangle)
    psi::SharedMatrix build_B_virtual();

  private:
    /// The AO basis set
    std::shared_ptr<psi::BasisSet> basis_;

    /// The original coefficients
    psi::SharedMatrix C_;

    /// Symmetry info
    int nirrep_;

    /// Basis size info
    int nbf_;

    /// MO info
    psi::Dimension nmopi_;

    /// Docc info
    psi::Dimension noccpi_;

    /// Density
    psi::SharedMatrix D_;

    /// AO Overlap
    psi::SharedMatrix S_;

    /// The startup function
    void startup();
};

} // namespace forte
#endif // _pao_builder_h_
