/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#pragma once

#include <tuple>

#include "psi4/libmints/wavefunction.h"

#include "base_classes/mo_space_info.h"
#include "helpers/blockedtensorfactory.h"
#include "integrals/integrals.h"
#include "base_classes/rdms.h"

namespace forte {

struct ActiveOrbitalType {
    enum Value { canonical, natural, unspecified };
    Value value_;
    ActiveOrbitalType(Value value) : value_(value) {};
    ActiveOrbitalType(std::string name) {
        for (auto& c : name)
            c = toupper(c);
        if (name == "UNSPECIFIED") {
            value_ = unspecified;
        } else if (name == "NATURAL") {
            value_ = natural;
        } else {
            value_ = canonical;
        }
    }
    // operator Value() const { return value_; }
    // operator bool() const = delete;
    bool operator==(ActiveOrbitalType rhs) const { return value_ == rhs.value_; }
    bool operator!=(ActiveOrbitalType rhs) const { return value_ != rhs.value_; }
    std::string toString() const {
        switch (value_) {
        case unspecified:
            return "UNSPECIFIED";
        case natural:
            return "NATURAL";
        default:
            return "CANONICAL";
        }
    }
};

/// @brief The SemiCanonical class
/// This class computes semi-canonical orbitals from the 1RDM and optionally transforms the
/// integrals and RDMs Semi-canonical orbitals are obtained by diagonalizing the Fock matrix in each
/// orbital space separately The class can also produce natural orbitals. These differ by the
/// semi-canonical orbital only in the active space where they are defined to be eigenvectors of the
/// 1RDM
///
/// The final orbitals are ordered by increasing energy within each irrep and space. Natural
/// orbitals are ordered by decreasing occupation number
class SemiCanonical {
  public:
    /// @brief SemiCanonical Constructor
    /// @param mo_space_info The MOSpaceInfo object
    /// @param ints The ForteIntegrals object
    /// @param scf_info The SCFInfo object
    /// @param threshold The threshold for testing orbitals
    /// @param inactive_mix Mix the frozen and restricted orbitals together?
    /// @param active_mix Mix all GAS orbitals together?
    /// @param quiet_banner Method banner is not printed if set to true
    SemiCanonical(std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ForteIntegrals> ints,
                  std::shared_ptr<SCFInfo> scf_info, bool inactive_mix,
                  bool active_mix, double threshold = 1.0e-8, bool quiet_banner = false);

    /// Transforms integrals and RDMs
    /// @brief Semicanonicalize the orbitals and transform the integrals and RDMs
    /// @param rdms The RDMs of the state to be semicanonicalized
    /// @param build_fock If true, the Fock matrix is built and diagonalized
    /// @param actv_orb_type Orbital type for active orbitals
    /// @param transform If true, the orbitals are transformed
    void semicanonicalize(std::shared_ptr<RDMs> rdms, bool build_fock = true,
                          ActiveOrbitalType orb_type = ActiveOrbitalType::canonical,
                          bool transform = true);

    /// @return the alpha rotation matrix
    std::shared_ptr<psi::Matrix> Ua() { return Ua_; }

    /// @return the beta rotation matrix
    std::shared_ptr<psi::Matrix> Ub() { return Ub_; }

    /// @return the alpha rotation matrix in the active space
    ambit::Tensor Ua_t() const { return Ua_t_.clone(); }

    /// @return the beta rotation matrix in the active space
    ambit::Tensor Ub_t() const { return Ub_t_.clone(); }

    /// @return if the orbital ordering and phases are fixed successfully
    bool fix_orbital_success() const { return fix_orbital_success_; }

  private:
    /// startup function to find dimensions and variables
    void startup();

    /// Forte MOSpaceInfo
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Forte integral
    std::shared_ptr<ForteIntegrals> ints_;

    /// SCF information
    std::shared_ptr<SCFInfo> scf_info_;

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
    bool check_orbitals(std::shared_ptr<RDMs> rdms, ActiveOrbitalType orb_type);

    /// Thresholds for Fock matrix testing
    double threshold_tight_;
    double threshold_loose_;

    /// Blocks of Fock or 1RDM to be checked and diagonalized
    std::map<std::string, std::shared_ptr<psi::Matrix>> mats_;
    /// Prepare blocks of Fock or 1RDM to be checked
    void prepare_matrix_blocks(std::shared_ptr<RDMs> rdms, ActiveOrbitalType orb_type);

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