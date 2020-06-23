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

#ifndef _semi_canonicalize_h_
#define _semi_canonicalize_h_

#include <tuple>

#include "psi4/libmints/wavefunction.h"

#include "base_classes/mo_space_info.h"
#include "helpers/blockedtensorfactory.h"
#include "integrals/integrals.h"
#include "base_classes/rdms.h"

namespace forte {

class ForteOptions;

/**
 * @brief The SemiCanonical class
 * Computes semi-canonical orbitals
 */
class SemiCanonical {
  public:
    /**
     * @brief SemiCanonical Constructor
     * @param options ForteOptions
     * @param ints ForteInegrals
     * @param mo_space_info MOSpaceInfo
     * @param quiet_banner Method banner is not printed if set to true
     */
    SemiCanonical(std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ForteIntegrals> ints,
                  std::shared_ptr<ForteOptions> options, bool quiet_banner = false);

    /// Transforms integrals and RDMs
    RDMs semicanonicalize(RDMs& rdms, const int& max_rdm_level = 3, const bool& build_fock = true,
                          const bool& transform = true);

    /// Transform all cumulants, rebuild 2-RDMs using 2-cumulants
    RDMs transform_rdms(ambit::Tensor& Ua, ambit::Tensor& Ub, RDMs& rdms, const int& max_rdm_level);

    /// Set active hole and particle dimensions
    void set_actv_dims(const psi::Dimension& actv_docc, const psi::Dimension& actv_virt);

    /// Return the alpha rotation matrix
    psi::SharedMatrix Ua() { return Ua_; }

    /// Return the beta rotation matrix
    psi::SharedMatrix Ub() { return Ub_; }

    /// Return the alpha rotation matrix in the active space
    ambit::Tensor Ua_t() { return Ua_t_; }

    /// Return the beta rotation matrix in the active space
    ambit::Tensor Ub_t() { return Ub_t_; }

  private:
    void startup();

    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    std::shared_ptr<ForteIntegrals> ints_;

    // All orbitals
    psi::Dimension nmopi_;
    // Correlated MOs
    psi::Dimension ncmopi_;
    // Frozen core
    psi::Dimension fdocc_;
    // Restricted DOCC
    psi::Dimension rdocc_;
    // Active MOs
    psi::Dimension actv_;
    // Restricted virtuals
    psi::Dimension ruocc_;
    // Active holes
    psi::Dimension actv_docc_;
    // Active particles
    psi::Dimension actv_virt_;

    // Blocks map
    std::map<std::string, psi::Dimension> mo_dims_;

    // Indices (no frozen) map
    std::map<std::string, std::vector<std::vector<size_t>>> cmo_idx_;

    // Figure out indices [[(A1)...], [(A2)...], [(B1)...], [(B2)...]]
    // npi: this mo space; bpi: mo space before npi; tpi: total mo space
    std::vector<std::vector<size_t>> idx_space(const psi::Dimension& npi, const psi::Dimension& bpi,
                                               const psi::Dimension& tpi);

    // Offset of active orbitals
    std::map<std::string, std::vector<int>> actv_offsets_;

    // Offsets
    std::map<std::string, psi::Dimension> offsets_;

    // Total active MOs
    size_t nact_;
    // Total correlated MOs
    size_t ncmo_;
    // Number of irreps
    size_t nirrep_;

    /// Unitary matrix for alpha orbital rotation
    psi::SharedMatrix Ua_;
    /// Unitary matrix for beta orbital rotation
    psi::SharedMatrix Ub_;
    /// Unitary matrix for alpha orbital rotation in the active space
    ambit::Tensor Ua_t_;
    /// Unitary matrix for beta orbital rotation in the active space
    ambit::Tensor Ub_t_;

    /// Set Ua_, Ub_, Ua_t_, and Ub_t_ to identity
    void set_U_to_identity();

    /// Build the generalized fock matrix
    void build_fock_matrix(RDMs& rdms);

    /// Check Fock matrix, return true if semicanonicalized
    bool check_fock_matrix();
    /// Thresholds for Fock matrix testing
    double threshold_tight_;
    double threshold_loose_;

    /// If certain Fock blocks need to be diagonalized
    std::map<std::string, bool> checked_results_;

    /**
     * Builds unitary matrices used to diagonalize diagonal blocks of F
     * Ua, Ub span all MOs
     * Ua_t, Ub_t span active MOs
     */
    void build_transformation_matrices(psi::SharedMatrix& Ua, psi::SharedMatrix& Ub,
                                       ambit::Tensor& Ua_t, ambit::Tensor& Ub_t);
};
} // namespace forte

#endif // _mp2_nos_h_
