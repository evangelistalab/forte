/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#ifndef _custom_integrals_h_
#define _custom_integrals_h_

#include "integrals.h"

namespace forte {

/**
 * @brief The CustomIntegrals class stores user-provided integrals
 */
class CustomIntegrals : public ForteIntegrals {
  public:
    /// Contructor of CustomIntegrals
    /// @param options a pointer to ForteOptions
    /// @param mo_space_info a pointer to Forte MOSpaceInfo
    /// @param restricted the type of integral transformation
    /// @param scalar the nuclear repulsion energy
    /// @param oei_a the alpha one-electron integrals in MO basis
    /// @param oei_b the beta one-electron integrals in MO basis
    /// @param tei_aa the alpha-alpha two-electron integrals in MO basis
    /// @param tei_ab the alpha-beta two-electron integrals in MO basis
    /// @param tei_bb the beta-beta two-electron integrals in MO basis
    CustomIntegrals(std::shared_ptr<ForteOptions> options,
                    std::shared_ptr<MOSpaceInfo> mo_space_info, IntegralSpinRestriction restricted,
                    double scalar, const std::vector<double>& oei_a,
                    const std::vector<double>& oei_b, const std::vector<double>& tei_aa,
                    const std::vector<double>& tei_ab, const std::vector<double>& tei_bb);

    void initialize() override;

    /// Grabs the antisymmetriced TEI - assumes storage in aphy_tei_*
    double aptei_aa(size_t p, size_t q, size_t r, size_t s) override;
    double aptei_ab(size_t p, size_t q, size_t r, size_t s) override;
    double aptei_bb(size_t p, size_t q, size_t r, size_t s) override;

    /// Grabs the antisymmetrized TEI - assumes storage of ambit tensor
    ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                 const std::vector<size_t>& r,
                                 const std::vector<size_t>& s) override;
    ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                 const std::vector<size_t>& r,
                                 const std::vector<size_t>& s) override;
    ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                 const std::vector<size_t>& r,
                                 const std::vector<size_t>& s) override;

    /// Make the generalized Fock matrix
    void make_fock_matrix(ambit::Tensor Da, ambit::Tensor Db) override;

    /// Make the closed-shell Fock matrix
    std::tuple<psi::SharedMatrix, psi::SharedMatrix, double>
    make_fock_inactive(psi::Dimension dim_start, psi::Dimension dim_end) override;

    /// Make the active Fock matrix
    std::tuple<psi::SharedMatrix, psi::SharedMatrix> make_fock_active(ambit::Tensor Da,
                                                                      ambit::Tensor Db) override;

    /// Make the active Fock matrix using restricted equation
    psi::SharedMatrix make_fock_active_restricted(psi::SharedMatrix D) override;

    /// Make the active Fock matrix using unrestricted equation
    std::tuple<psi::SharedMatrix, psi::SharedMatrix>
    make_fock_active_unrestricted(psi::SharedMatrix Da, psi::SharedMatrix Db) override;

    size_t nthree() const override { throw std::runtime_error("Wrong Integral type"); }

    void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                 bool alpha2) override;

  private:
    // ==> Class private data <==

    /// Full two-electron integrals stored as a vector with redundant elements (no permutational
    /// symmetry) (includes frozen orbitals)
    std::vector<double> full_aphys_tei_aa_;
    std::vector<double> full_aphys_tei_ab_;
    std::vector<double> full_aphys_tei_bb_;

    std::vector<double> original_full_one_electron_integrals_a_;
    std::vector<double> original_full_one_electron_integrals_b_;

    bool save_original_tei_ = false;
    ambit::Tensor original_V_aa_;
    ambit::Tensor original_V_ab_;
    ambit::Tensor original_V_bb_;
    std::vector<double> original_full_aphys_tei_aa_;
    std::vector<double> original_full_aphys_tei_ab_;
    std::vector<double> original_full_aphys_tei_bb_;

    // ==> Class private functions <==

    void resort_four(std::vector<double>& tei, std::vector<size_t>& map);
    /// An addressing function to for two-electron integrals
    /// @return the address of the integral <pq|rs> or <pq||rs>
    size_t aptei_index(size_t p, size_t q, size_t r, size_t s) {
        return aptei_idx_ * aptei_idx_ * aptei_idx_ * p + aptei_idx_ * aptei_idx_ * q +
               aptei_idx_ * r + s;
    }

    void update_orbitals(std::shared_ptr<psi::Matrix> Ca, std::shared_ptr<psi::Matrix> Cb) override;

    // ==> Class private virtual functions <==
    void gather_integrals() override;
    void resort_integrals_after_freezing() override;
    void compute_frozen_one_body_operator() override;

    void transform_one_electron_integrals();
    void transform_two_electron_integrals();
};

} // namespace forte

#endif // _integrals_h_
