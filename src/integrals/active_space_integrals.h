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

#ifndef _active_space_integrals_
#define _active_space_integrals_

#include "integrals/integrals.h"
#include "sparse_ci/determinant.h"

class Dimension;

namespace forte {

/**
 * @brief The ActiveSpaceIntegrals class stores integrals necessary for FCI calculations
 */
class ActiveSpaceIntegrals {
  public:
    // ==> Class Constructors <==

    /**
     * @brief Contructor to create integrals for an active space
     * @param ints forte integral object
     * @param active_mo the list of active orbitals
     * @param active_mo_symmetry the symmetry of the active orbitals
     * @param rdocc_mo the list of orbitals that are doubly occupied. This information is used
     *                 to form the effective one-electron operator
     */
    ActiveSpaceIntegrals(std::shared_ptr<ForteIntegrals> ints, const std::vector<size_t>& active_mo,
                         const std::vector<int>& active_mo_symmetry,
                         const std::vector<size_t>& rdocc_mo);

    // ==> Class Interface <==

    /// Return the
    std::shared_ptr<ForteIntegrals> ints() { return ints_; }

    /// Return the number of MOs
    size_t nmo() const { return nmo_; }

    std::vector<int> active_mo_symmetry() const;

    std::vector<size_t> active_mo() const;

    std::vector<size_t> restricted_docc_mo() const;

    /// Return nuclear repulsion energy
    double nuclear_repulsion_energy() const { return ints_->nuclear_repulsion_energy(); }

    /// Return the frozen core energy (contribution from FROZEN_DOCC)
    double frozen_core_energy() const { return frozen_core_energy_; }

    /// Return the scalar_energy energy (contribution from RESTRICTED_DOCC)
    double scalar_energy() const { return scalar_energy_; }

    /// Set scalar_energy();
    void set_scalar_energy(double scalar_energy) { scalar_energy_ = scalar_energy; }

    /// Compute a determinant's energy
    double energy(const Determinant& det) const;

    /// Compute the matrix element of the Hamiltonian between this determinant
    /// and a given one
    double slater_rules(const Determinant& lhs, const Determinant& rhs) const;
    /// Compute the matrix element of the Hamiltonian between this determinant
    /// and a given one
    double slater_rules_single_alpha(const Determinant& det, int i, int a) const;
    /// Compute the matrix element of the Hamiltonian between this determinant
    /// and a given one
    double slater_rules_single_beta(const Determinant& det, int i, int a) const;
    /// Compute the matrix element of the Hamiltonian between this determinant
    /// and a given one
    double slater_rules_single_alpha_abs(const Determinant& det, int i, int a) const;
    /// Compute the matrix element of the Hamiltonian between this determinant
    /// and a given one
    double slater_rules_single_beta_abs(const Determinant& det, int i, int a) const;

    /// Return the alpha effective one-electron integral
    double oei_a(size_t p, size_t q) const { return oei_a_[p * nmo_ + q]; }
    /// Return the beta effective one-electron integral
    double oei_b(size_t p, size_t q) const { return oei_b_[p * nmo_ + q]; }
    std::vector<double> oei_a_vector() { return oei_a_; }
    std::vector<double> oei_b_vector() { return oei_b_; }

    /// Return the alpha-alpha antisymmetrized two-electron integral <pq||rs>
    double tei_aa(size_t p, size_t q, size_t r, size_t s) const {
        return tei_aa_[nmo3_ * p + nmo2_ * q + nmo_ * r + s];
    }
    /// Return the alpha-beta two-electron integral <pq|rs>
    double tei_ab(size_t p, size_t q, size_t r, size_t s) const {
        return tei_ab_[nmo3_ * p + nmo2_ * q + nmo_ * r + s];
    }
    /// Return the beta-beta antisymmetrized two-electron integral <pq||rs>
    double tei_bb(size_t p, size_t q, size_t r, size_t s) const {
        return tei_bb_[nmo3_ * p + nmo2_ * q + nmo_ * r + s];
    }

    /// Return a vector of alpha-alpha antisymmetrized two-electron integrals
    const std::vector<double>& tei_aa_vector() const { return tei_aa_; }
    /// Return a vector of alpha-beta antisymmetrized two-electron integrals
    const std::vector<double>& tei_ab_vector() const { return tei_ab_; }
    /// Return a vector of beta-beta antisymmetrized two-electron integrals
    const std::vector<double>& tei_bb_vector() const { return tei_bb_; }

    /// Return the alpha-alpha antisymmetrized two-electron integral <pq||pq>
    double diag_tei_aa(size_t p, size_t q) const {
        return tei_aa_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
    }
    /// Return the alpha-beta two-electron integral <pq|rs>
    double diag_tei_ab(size_t p, size_t q) const {
        return tei_ab_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
    }
    /// Return the beta-beta antisymmetrized two-electron integral <pq||rs>
    double diag_tei_bb(size_t p, size_t q) const {
        return tei_bb_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
    }
    IntegralType get_integral_type() { return integral_type_; }
    /// Set the active integrals
    void set_active_integrals(const ambit::Tensor& tei_aa, const ambit::Tensor& tei_ab,
                              const ambit::Tensor& tei_bb);
    /// Compute the restricted_docc operator
    void compute_restricted_one_body_operator();
    /// Set the restricted_one_body_operator
    void set_restricted_one_body_operator(const std::vector<double>& oei_a,
                                          const std::vector<double>& oei_b) {
        oei_a_ = oei_a;
        oei_b_ = oei_b;
    }

    /// Streamline the process of setting up active integrals and
    /// restricted_docc
    /// Sets active integrals based on active space and restricted_docc
    /// If you want more control, don't use this function.
    void set_active_integrals_and_restricted_docc();

    /// Print the alpha-alpha integrals
    void print();

  private:
    // ==> Class Private Data <==

    /// The number of MOs
    size_t nmo_;
    /// The number of MOs squared
    size_t nmo2_;
    /// The number of MOs cubed
    size_t nmo3_;
    /// The number of MOs to the fourth power
    size_t nmo4_;
    /// The integral type
    IntegralType integral_type_;
    /// The integrals object
    std::shared_ptr<ForteIntegrals> ints_;
    /// The frozen core energy
    double frozen_core_energy_;
    /// The scalar contribution to the energy
    double scalar_energy_;
    /// The alpha one-electron integrals
    std::vector<double> oei_a_;
    /// The beta one-electron integrals
    std::vector<double> oei_b_;
    /// The alpha-alpha antisymmetrized two-electron integrals in physicist
    /// notation
    std::vector<double> tei_aa_;
    /// The alpha-beta antisymmetrized two-electron integrals in physicist
    /// notation
    std::vector<double> tei_ab_;
    /// The beta-beta antisymmetrized two-electron integrals in physicist
    /// notation
    std::vector<double> tei_bb_;
    /// The diagonal alpha-alpha antisymmetrized two-electron integrals in
    /// physicist notation
    std::vector<double> diag_tei_aa_;
    /// The diagonal alpha-beta antisymmetrized two-electron integrals in
    /// physicist notation
    std::vector<double> diag_tei_ab_;
    /// The diagonal beta-beta antisymmetrized two-electron integrals in
    /// physicist notation
    std::vector<double> diag_tei_bb_;
    /// A vector of indices for the active molecular orbitals
    std::vector<size_t> active_mo_;
    /// A vector of the symmetry ofthe active molecular orbitals
    std::vector<int> active_mo_symmetry_;
    /// A Vector of indices for the restricted_docc molecular orbitals
    std::vector<size_t> restricted_docc_mo_;

    // ==> Class Private Functions <==

    inline size_t tei_index(size_t p, size_t q, size_t r, size_t s) const {
        return nmo3_ * p + nmo2_ * q + nmo_ * r + s;
    }
    /// F^{Restricted}_{uv} = h_{uv} + \sum_{i = frozen_core}^{restricted_core}
    /// 2(uv | ii) - (ui|vi)
    void RestrictedOneBodyOperator(std::vector<double>& oei_a, std::vector<double>& oei_b);
    void startup();
};

std::shared_ptr<ActiveSpaceIntegrals>
make_active_space_ints(std::shared_ptr<forte::MOSpaceInfo> mo_space_info,
                       std::shared_ptr<ForteIntegrals> ints, const std::string& active_space,
                       const std::vector<std::string>& core_spaces);

} // namespace forte

#endif // _active_space_integrals_
