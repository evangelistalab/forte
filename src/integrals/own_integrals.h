/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
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

#ifndef _own_integrals_h_
#define _own_integrals_h_

#include "integrals.h"



class Tensor;

namespace forte {

class ForteOptions;
class MOSpaceInfo;

/// This class is used if the user wants to generate their own integrals for
/// their method.
/// This would be very useful for CI based methods (the integrals class is
/// wasteful and dumb for this area)
/// Also, I am putting this here if I(Kevin) ever get around to implementing
/// AO-DSRG-MRPT2
class OwnIntegrals : public ForteIntegrals {
  public:
    OwnIntegrals(psi::Options& options, psi::SharedWavefunction ref_wfn,
                 IntegralSpinRestriction restricted, std::shared_ptr<MOSpaceInfo> mo_space_info);

    virtual void retransform_integrals() {}
    /// aptei_xy functions are slow.  try to use three_integral_block

    virtual double aptei_aa(size_t /*p*/, size_t /*q*/, size_t /*r*/, size_t /*s*/) { return 0.0; }
    virtual double aptei_ab(size_t /*p*/, size_t /*q*/, size_t /*r*/, size_t /*s*/) { return 0.0; }
    virtual double aptei_bb(size_t /*p*/, size_t /*q*/, size_t /*r*/, size_t /*s*/) { return 0.0; }

    /// Reads the antisymmetrized alpha-alpha chunck and returns an
    /// ambit::Tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& /*p*/,
                                         const std::vector<size_t>& /*q*/,
                                         const std::vector<size_t>& /*r*/,
                                         const std::vector<size_t>& /*s*/) {
        return blank_tensor_;
    }
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& /*p*/,
                                         const std::vector<size_t>& /*q*/,
                                         const std::vector<size_t>& /*r*/,
                                         const std::vector<size_t>& /*s*/) {
        return blank_tensor_;
    }
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& /*p*/,
                                         const std::vector<size_t>& /*q*/,
                                         const std::vector<size_t>& /*r*/,
                                         const std::vector<size_t>& /*s*/) {
        return blank_tensor_;
    }

    virtual double diag_aptei_aa(size_t, size_t) { return 0.0; }
    virtual double diag_aptei_ab(size_t, size_t) { return 0.0; }
    virtual double diag_aptei_bb(size_t, size_t) { return 0.0; }
    virtual double three_integral(size_t, size_t, size_t) { return 0.0; }
    virtual double** three_integral_pointer() {
        throw PSIEXCEPTION("Integrals are distributed.  Pointer does not exist");
    }
    /// Read a block of the DFIntegrals and return an Ambit tensor of size A by
    /// p by q
    virtual ambit::Tensor three_integral_block(const std::vector<size_t>& /*A*/,
                                               const std::vector<size_t>& /*p*/,
                                               const std::vector<size_t>& /*q*/) {
        return blank_tensor_;
    }
    /// return ambit tensor of size A by q
    virtual ambit::Tensor three_integral_block_two_index(const std::vector<size_t>& /*A*/,
                                                         size_t /*p*/,
                                                         const std::vector<size_t>& /*q*/) {
        return blank_tensor_;
    }

    virtual void set_tei(size_t, size_t, size_t, size_t, double, bool, bool) {}
    virtual ~OwnIntegrals();

    virtual void make_fock_matrix(SharedMatrix /*gamma_a*/, SharedMatrix /*gamma_b*/) {}
    virtual size_t nthree() const { return 1; }

  private:
    // ==> Class data <==

    ambit::Tensor blank_tensor_;

    // ==> Class private virtual functions <==

    virtual void gather_integrals() {}
    virtual void resort_integrals_after_freezing() {}
};

} // namespace forte
} // namespace psi

#endif // _own_integrals_h_
