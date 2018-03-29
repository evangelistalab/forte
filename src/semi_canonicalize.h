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

#include <tuple>

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"

#include "helpers.h"
#include "integrals/integrals.h"
#include "reference.h"
#include "blockedtensorfactory.h"

namespace psi {

namespace forte {

/**
 * @brief The SemiCanonical class
 * Computes semi-canonical orbitals
 */
class SemiCanonical {
  public:
    // => Constructor <= //
    SemiCanonical(std::shared_ptr<Wavefunction> wfn, std::shared_ptr<ForteIntegrals> ints,
                  std::shared_ptr<MOSpaceInfo> mo_space_info, const bool& quiet = false);

    /// Transforms integrals and reference
    void semicanonicalize(Reference& reference, const int& max_rdm_level = 3,
                          const bool& build_fock = true, const bool& transform = true);

    /// Transform integrals
    void transform_ints(SharedMatrix& Ua, SharedMatrix& Ub);

    /// Transform all cumulants, rebuild 2-RDMs using 2-cumulants
    void transform_reference(ambit::Tensor& Ua, ambit::Tensor& Ub, Reference& reference,
                             const int& max_rdm_level);

    /// Back transform integrals
    /// Ua and Ub rotate non-semicanonical to semicanonical
    void back_transform_ints(SharedMatrix& Ua, SharedMatrix& Ub);

    /// Back transform integrals
    void back_transform_ints() {
        back_transform_ints(Ua_, Ub_);
    }

    /// Set active hole and particle dimensions
    void set_actv_dims(const Dimension& actv_docc, const Dimension& actv_virt);

    /// Return the alpha rotation matrix
    SharedMatrix Ua() { return Ua_; }

    /// Return the beta rotation matrix
    SharedMatrix Ub() { return Ub_; }

    /// Return the alpha rotation matrix in the active space
    ambit::Tensor Ua_t() { return Ua_t_; }

    /// Return the beta rotation matrix in the active space
    ambit::Tensor Ub_t() { return Ub_t_; }

  private:
    void startup();

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
    // Active holes
    Dimension actv_docc_;
    // Active particles
    Dimension actv_virt_;

    // Blocks map
    std::map<std::string, Dimension> mo_dims_;

    // Indices (no frozen) map
    std::map<std::string, std::vector<std::vector<size_t>>> cmo_idx_;

    // Figure out indices [[(A1)...], [(A2)...], [(B1)...], [(B2)...]]
    // npi: this mo space; bpi: mo space before npi; tpi: total mo space
    std::vector<std::vector<size_t>> idx_space(const Dimension& npi, const Dimension& bpi,
                                               const Dimension& tpi);

    // Offset of active orbitals
    std::map<std::string, std::vector<int>> actv_offsets_;

    // Offsets
    std::map<std::string, Dimension> offsets_;

    // Total active MOs
    size_t nact_;
    // Total correlated MOs
    size_t ncmo_;
    // Number of irreps
    size_t nirrep_;

    /// Unitary matrix for alpha orbital rotation
    SharedMatrix Ua_;
    /// Unitary matrix for beta orbital rotation
    SharedMatrix Ub_;
    /// Unitary matrix for alpha orbital rotation in the active space
    ambit::Tensor Ua_t_;
    /// Unitary matrix for beta orbital rotation in the active space
    ambit::Tensor Ub_t_;

    /// Set Ua_, Ub_, Ua_t_, and Ub_t_ to identity
    void set_U_to_identity();

    /// Build the generalized fock matrix
    void build_fock_matrix(Reference& reference);

    /// Check Fock matrix, return true if semicanonicalized
    bool check_fock_matrix();

    /// If certain Fock blocks need to be diagonalized
    std::map<std::string, bool> checked_results_;

    /**
     * Builds unitary matrices used to diagonalize diagonal blocks of F
     * Ua, Ub span all MOs
     * Ua_t, Ub_t span active MOs
     */
    void build_transformation_matrices(SharedMatrix& Ua, SharedMatrix& Ub, ambit::Tensor& Ua_t,
                                       ambit::Tensor& Ub_t);
};
}
} // End Namespaces

#endif // _mp2_nos_h_
