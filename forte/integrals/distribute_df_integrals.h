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

#ifndef _conventional_integrals_h_
#define _conventional_integrals_h_

#include "integrals.h"

namespace forte {

#ifdef HAVE_GA
class DistDFIntegrals : public Psi4Integrals {
  public:
    DistDFIntegrals(std::shared_ptr<ForteOptions> options,
                    std::shared_ptr<psi::Wavefunction> ref_wfn, IntegralSpinRestriction restricted,
                    std::shared_ptr<MOSpaceInfo> mo_space_info);

    void initialize() override;
    virtual void retransform_integrals();
    /// aptei_xy functions are slow.  try to use three_integral_block

    virtual double aptei_aa(size_t /*p*/, size_t /*q*/, size_t /*r*/, size_t /*s*/) {}
    virtual double aptei_ab(size_t /*p*/, size_t /*q*/, size_t /*r*/, size_t /*s*/) {}
    virtual double aptei_bb(size_t /*p*/, size_t /*q*/, size_t /*r*/, size_t /*s*/) {}

    /// Reads the antisymmetrized alpha-alpha chunck and returns an
    /// ambit::Tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s) {}
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s) {}
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s) {}

    virtual double diag_aptei_aa(size_t, size_t) {}
    virtual double diag_aptei_ab(size_t, size_t) {}
    virtual double diag_aptei_bb(size_t, size_t) {}
    virtual double three_integral(size_t, size_t, size_t) {}
    virtual double** three_integral_pointer() {
        throw psi::PSIEXCEPTION("Integrals are distributed.  Pointer does not exist");
    }
    /// Read a block of the DFIntegrals and return an Ambit tensor of size A by
    /// p by q
    virtual ambit::Tensor three_integral_block(const std::vector<size_t>& /*A*/,
                                               const std::vector<size_t>& /*p*/,
                                               const std::vector<size_t>& /*q*/);
    /// return ambit tensor of size A by q
    virtual ambit::Tensor three_integral_block_two_index(const std::vector<size_t>& /*A*/,
                                                         size_t /*p*/,
                                                         const std::vector<size_t>& /*q*/) {}

    virtual void set_tei(size_t, size_t, size_t, size_t, double, bool, bool) {
        outfile->Printf("DistributedDF will not work with set_tei");
        throw psi::PSIEXCEPTION("DistDF can not use set_tei");
    }

    /// Make a Fock matrix computed with respect to a given determinant
    virtual size_t nthree() const { return nthree_; }
    virtual int ga_handle() { return DistDF_ga_; }

  private:
    virtual void gather_integrals();
    virtual void resort_integrals_after_freezing() {}

    /// This is the handle for GA
    int DistDF_ga_;

    std::shared_ptr<psi::Matrix> ThreeIntegral_;
    size_t nthree_ = 0;
    /// Assuming integrals are stored on disk
    /// Reads the block of integrals present for each process
    ambit::Tensor read_integral_chunk(std::shared_ptr<Tensor>& B, std::vector<int>& lo,
                                      std::vector<int>& hi);
    /// Distributes tensor according to naux dimension
    void create_dist_df();
    void test_distributed_integrals();
};
#endif

} // namespace forte

#endif // _conventional_integrals_h_
