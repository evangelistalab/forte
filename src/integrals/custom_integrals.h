/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
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
    /// Contructor of the class.  Calls std::shared_ptr<ForteIntegrals> ints
    /// constructor
    CustomIntegrals(std::shared_ptr<ForteOptions> options, std::shared_ptr<psi::Wavefunction> ref_wfn,
                    std::shared_ptr<MOSpaceInfo> mo_space_info, IntegralSpinRestriction restricted);

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

    ambit::Tensor three_integral_block(const std::vector<size_t>&, const std::vector<size_t>&,
                                       const std::vector<size_t>&) override;
    ambit::Tensor three_integral_block_two_index(const std::vector<size_t>&, size_t,
                                                 const std::vector<size_t>&) override;
    double** three_integral_pointer() override;

    void make_fock_matrix(std::shared_ptr<psi::Matrix> gamma_a,
                          std::shared_ptr<psi::Matrix> gamma_b) override;

    size_t nthree() const override { throw std::runtime_error("Wrong Integral type"); }

    void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                 bool alpha2) override;

  private:
    // ==> Class data <==

    /// Used to store the two-electron integrals (pq|rs) in chemist notation with 8-fold symmetry
    /// and addressed with the function four(p,q,r,s)
    std::vector<double> aphys_tei_aa;
    std::vector<double> aphys_tei_ab;
    std::vector<double> aphys_tei_bb;

    // ==> Class private functions <==

    void resort_four(std::vector<double>& tei, std::vector<size_t>& map);
    /// An addressing function to for two-electron integrals
    /// @return the address of the integral <pq|rs> or <pq||rs>
    size_t aptei_index(size_t p, size_t q, size_t r, size_t s) {
        return aptei_idx_ * aptei_idx_ * aptei_idx_ * p + aptei_idx_ * aptei_idx_ * q +
               aptei_idx_ * r + s;
    }
    /// Set the number of orbitals and allocate the memory
    void custom_integrals_allocate(int norb, const std::vector<int>& orbsym);

    // ==> Class private virtual functions <==

    void gather_integrals() override;
    void resort_integrals_after_freezing() override;
};

} // namespace forte

#endif // _integrals_h_
