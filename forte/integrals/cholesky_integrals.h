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

#ifndef _cholesky_integrals_h_
#define _cholesky_integrals_h_

#include "integrals.h"

namespace forte {

/// Class written by Kevin Hannon
/**
 * @brief The CholeskyIntegrals class approximates two-electron integrals via Cholesky decomposition
 *
 * This class assumes the Cholesky tensors can be stored in memory.
 */
class CholeskyIntegrals : public Psi4Integrals {
  public:
    /// Contructor of CholeskyIntegrals
    CholeskyIntegrals(std::shared_ptr<ForteOptions> options,
                      std::shared_ptr<psi::Wavefunction> ref_wfn,
                      std::shared_ptr<MOSpaceInfo> mo_space_info,
                      IntegralSpinRestriction restricted);

    void initialize() override;

    /// aptei_x will grab antisymmetriced integrals and creates DF/CD integrals
    /// on the fly
    double aptei_aa(size_t p, size_t q, size_t r, size_t s) override;
    double aptei_ab(size_t p, size_t q, size_t r, size_t s) override;
    double aptei_bb(size_t p, size_t q, size_t r, size_t s) override;

    /// Return the antisymmetrized alpha-alpha chunck as an ambit::Tensor
    ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                 const std::vector<size_t>& r,
                                 const std::vector<size_t>& s) override;
    /// Return the antisymmetrized alpha-beta chunck as an ambit::Tensor
    ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                 const std::vector<size_t>& r,
                                 const std::vector<size_t>& s) override;
    /// Return the antisymmetrized beta-beta chunck as an ambit::Tensor
    ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                 const std::vector<size_t>& r,
                                 const std::vector<size_t>& s) override;

    double three_integral(size_t A, size_t p, size_t q) const;

    double** three_integral_pointer() override;
    ambit::Tensor three_integral_block(const std::vector<size_t>& A, const std::vector<size_t>& p,
                                       const std::vector<size_t>& q) override;
    ambit::Tensor three_integral_block_two_index(const std::vector<size_t>&, size_t,
                                                 const std::vector<size_t>&) override;
    /// Do not use this if you are using CD/DF integrals
    void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                 bool alpha2) override;

    size_t nthree() const override;
    std::shared_ptr<psi::Matrix> L_ao_;

  private:
    // ==> Class data <==

    std::shared_ptr<psi::Matrix> ThreeIntegral_;
    size_t nthree_ = 0;

    // ==> Class private functions <==

    void resort_three(std::shared_ptr<psi::Matrix>& threeint, std::vector<size_t>& map);
    void transform_integrals();

    // ==> Class private virtual functions <==

    void gather_integrals() override;
    void resort_integrals_after_freezing() override;
};

} // namespace forte

#endif // _cholesky_integrals_h_
