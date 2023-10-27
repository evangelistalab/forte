/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
 * Computes semi-canonical orbitals for given 1RDMs
 */
class SemiCanonical {
  public:
    /**
     * @brief SemiCanonical Constructor
     * @param options ForteOptions
     * @param ints ForteIntegrals
     * @param mo_space_info MOSpaceInfo
     * @param quiet_banner Method banner is not printed if set to true
     */
    SemiCanonical(std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ForteIntegrals> ints,
                  std::shared_ptr<ForteOptions> options, bool quiet = false);

    /// Transforms integrals and RDMs
    void semicanonicalize(std::shared_ptr<RDMs> rdms, const bool& build_fock = true,
                          const bool& nat_orb = false, const bool& transform = true);

    /// Return the alpha rotation matrix
    std::shared_ptr<psi::Matrix> Ua() { return Ua_; }

    /// Return the beta rotation matrix
    std::shared_ptr<psi::Matrix> Ub() { return Ub_; }

    /// Return the alpha rotation matrix in the active space
    ambit::Tensor Ua_t() const { return Ua_t_.clone(); }

    /// Return the beta rotation matrix in the active space
    ambit::Tensor Ub_t() const { return Ub_t_.clone(); }

    /// Return if the orbital ordering and phases are fixed successfully
    bool fix_orbital_success() const { return fix_orbital_success_; }

  private:
    /// startup function to find dimensions and variables
    void startup();

    /// read ForteOptions
    void read_options(const std::shared_ptr<ForteOptions>& foptions);

    /// Forte MOSpaceInfo
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Forte integral
    std::shared_ptr<ForteIntegrals> ints_;

    /// Print level
    int print_;

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

    /// Unitary matrix for alpha orbital rotation
    std::shared_ptr<psi::Matrix> Ua_;
    /// Unitary matrix for beta orbital rotation
    std::shared_ptr<psi::Matrix> Ub_;
    /// Unitary matrix for alpha orbital rotation in the active space
    ambit::Tensor Ua_t_;
    /// Unitary matrix for beta orbital rotation in the active space
    ambit::Tensor Ub_t_;

    /// Set Ua_, Ub_, Ua_t_, and Ub_t_ to identity
    void set_U_to_identity();

    /// Check if orbitals are semicanonicalized
    bool check_orbitals(std::shared_ptr<RDMs> rdms, const bool& nat_orb);

    /// Thresholds for Fock matrix testing
    double threshold_tight_;
    double threshold_loose_;

    /// Blocks of Fock or 1RDM to be checked and diagonalized
    std::map<std::string, std::shared_ptr<psi::Matrix>> mats_;
    /// Prepare blocks of Fock or 1RDM to be checked
    void prepare_matrix_blocks(std::shared_ptr<RDMs> rdms, const bool& nat_orb);

    /// If certain Fock blocks need to be diagonalized
    std::map<std::string, bool> checked_results_;

    /// Builds unitary matrices used to diagonalize diagonal blocks of Fock
    void build_transformation_matrices(const bool& semi);

    /// Fill ambit::Tensor Ua_t_ (Ub_t_) using std::shared_ptr<psi::Matrix> Ua_ (Ub_)
    void fill_Uactv(const std::shared_ptr<psi::Matrix>& U, ambit::Tensor& Ut);

    /// Successfully fix the orbital ordering and phases
    bool fix_orbital_success_;
};
} // namespace forte

#endif // _semi_canonicalize_h_
