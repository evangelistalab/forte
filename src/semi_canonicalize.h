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

#ifndef _semi_canonicalize_h_
#define _semi_canonicalize_h_

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"

#include "helpers.h"
#include "integrals/integrals.h"
#include "reference.h"
#include "../blockedtensorfactory.h"

namespace psi {

namespace forte {

/**
 * @brief The SemiCanonical class
 * Computes semi-canonical orbitals
 */
class SemiCanonical {
  public:
    // => Constructor <= //
    SemiCanonical(std::shared_ptr<Wavefunction> wfn, Options& options,
                  std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info,
                  Reference& reference, const bool& quiet = false);

    // Transforms integrals and reference
    void semicanonicalize(Reference& reference);

  private:
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    std::shared_ptr<ForteIntegrals> ints_;

    std::shared_ptr<Wavefunction> wfn_;

    // quiet printing
    bool quiet_;

    // All orbitals
    Dimension nmopi_;
    // Correlated MOs
    Dimension ncmopi_;
    // Frozen core
    Dimension fdocc_;
    // Restricted DOCC
    Dimension rdocc_;
    // Active MOs
    Dimension actv_;
    // Restricted virtuals
    Dimension ruocc_;

    // Total active MOs
    size_t nact_;
    // Total correlated MOs
    size_t ncmo_;
    // Number of irreps
    size_t nirrep_;

    // Builds the generalized fock matrix
    void build_fock_matrix(Reference& reference);

    /**
     * Builds unitary matrices used to diagonalize diagonal blocks of F
     * Ua, Ub span all MOs
     * Ua_t, Ub_t span active MOs
     */
    void build_transformation_matrices(SharedMatrix& Ua, SharedMatrix& Ub, ambit::Tensor& Ua_t,
                                       ambit::Tensor& Ub_t);

    // Transforms integrals
    void transform_ints(SharedMatrix& Ua, SharedMatrix& Ub);

    // Transforms all RDMS/cumulants
    void transform_reference(ambit::Tensor& Ua, ambit::Tensor& Ub, Reference& reference);
};
}
} // End Namespaces

#endif // _mp2_nos_h_
