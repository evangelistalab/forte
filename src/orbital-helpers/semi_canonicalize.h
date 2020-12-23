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
    RDMs semicanonicalize(RDMs& rdms, const bool& build_fock = true, const bool& transform = true);

    /// Return the alpha rotation matrix
    psi::SharedMatrix Ua() { return Ua_; }

    /// Return the beta rotation matrix
    psi::SharedMatrix Ub() { return Ub_; }

    /// Return the alpha rotation matrix in the active space
    ambit::Tensor Ua_t() { return Ua_t_; }

    /// Return the beta rotation matrix in the active space
    ambit::Tensor Ub_t() { return Ub_t_; }

    /// Return if the orbital ordering and phases are fixed successfully
    bool fix_orbital_success() { return fix_orbital_success_; }

    /// Set if rotating orbitals to natural orbital basis
    void set_natural_orbital(bool nat_orb) { natural_orb_ = nat_orb; }

  private:
    void read_options(std::shared_ptr<ForteOptions> foptions);

    void startup();

    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    std::shared_ptr<ForteIntegrals> ints_;

    /// Do natural orbitals for active orbitals
    bool natural_orb_ = false;

    /// Mix the frozen and restricted orbitals together
    bool inactive_mix_;
    /// Mix all GAS orbitals together
    bool active_mix_;

    /// Dimension for all orbitals (number of MOs per irrep)
    psi::Dimension nmopi_;

    /// Blocks map
    std::map<std::string, psi::Dimension> mo_dims_;

    /// Offset of GAS orbitals within ACTIVE
    std::map<std::string, psi::Dimension> actv_offsets_;

    /// Number of active MOs
    size_t nact_;

    /// Number of irreps
    size_t nirrep_;

    /// A copy of the alpha Fock matrix
    psi::SharedMatrix Fa_;
    /// A copy of the beta Fock matrix
    psi::SharedMatrix Fb_;
    /// Prepare the Fock matrices for diagonalization
    void prepare_fock(RDMs& rdms);

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

    /// Thresholds for Fock matrix testing
    double threshold_tight_;
    double threshold_loose_;
    /// Thresholds for 1-RDM spin differences
    double threshold_1rdm_;

    /// If certain Fock blocks need to be diagonalized
    std::map<std::string, bool> checked_results_;

    /// Check Fock matrix, return true if semicanonicalized
    bool check_fock_matrix();

    /// Builds unitary matrices used to diagonalize diagonal blocks of Fock
    void build_transformation_matrices();

    /// Fill ambit::Tensor Ua_t_ (Ub_t_) using psi::SharedMatrix Ua_ (Ub_)
    void fill_Uactv(psi::SharedMatrix U, ambit::Tensor Ut);

    /// Successfully fix the orbital ordering and phases
    bool fix_orbital_success_;
};
} // namespace forte

#endif // _semi_canonicalize_h_
