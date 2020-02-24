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

#ifndef _df_integrals_h_
#define _df_integrals_h_

#include "integrals.h"

namespace forte {

/**
 * @brief The DFIntegrals class approximates two-electron integrals via density fitting
 *
 * This class assumes the density-fitting tensors can be stored in memory.
 */
class DFIntegrals : public ForteIntegrals {
  public:
    DFIntegrals(std::shared_ptr<ForteOptions> options, std::shared_ptr<psi::Wavefunction> ref_wfn,
                std::shared_ptr<MOSpaceInfo> mo_space_info, IntegralSpinRestriction restricted);

    double aptei_aa(size_t p, size_t q, size_t r, size_t s) override;
    double aptei_ab(size_t p, size_t q, size_t r, size_t s) override;
    double aptei_bb(size_t p, size_t q, size_t r, size_t s) override;

    /// Reads the antisymmetrized alpha-alpha chunck and returns an
    /// ambit::Tensor
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

    double three_integral(size_t A, size_t p, size_t q);

    ambit::Tensor three_integral_block(const std::vector<size_t>& A, const std::vector<size_t>& p,
                                       const std::vector<size_t>& q) override;
    ambit::Tensor three_integral_block_two_index(const std::vector<size_t>&, size_t,
                                                 const std::vector<size_t>&) override;
    double** three_integral_pointer() override;
    void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                 bool alpha2) override;

    void make_fock_matrix(std::shared_ptr<psi::Matrix> gamma_a,
                          std::shared_ptr<psi::Matrix> gamma_b) override;

    size_t nthree() const override;

  private:
    // ==> Class data <==

    std::shared_ptr<psi::Matrix> ThreeIntegral_;
    size_t nthree_ = 0;

    // ==> Class private functions <==

    void resort_three(std::shared_ptr<psi::Matrix>& threeint, std::vector<size_t>& map);

    // ==> Class private virtual functions <==

    void gather_integrals() override;
    void resort_integrals_after_freezing() override;
};

} // namespace forte

#endif // _df_integrals_h_
