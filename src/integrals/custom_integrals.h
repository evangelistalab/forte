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

#ifndef _custom_integrals_h_
#define _custom_integrals_h_

#include "integrals.h"

namespace psi {

class Tensor;

namespace forte {

class ForteOptions;
class MOSpaceInfo;

/**
 * @brief The CustomIntegrals class stores user-provided integrals
 */
class CustomIntegrals : public ForteIntegrals {
  public:
    /// Contructor of the class.  Calls std::shared_ptr<ForteIntegrals> ints
    /// constructor
    CustomIntegrals(psi::Options& options, SharedWavefunction ref_wfn,
                    IntegralSpinRestriction restricted, std::shared_ptr<MOSpaceInfo> mo_space_info);
    virtual ~CustomIntegrals();

    /// Grabs the antisymmetriced TEI - assumes storage in aphy_tei_*
    virtual double aptei_aa(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_ab(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_bb(size_t p, size_t q, size_t r, size_t s);

    /// Grabs the antisymmetrized TEI - assumes storage of ambit tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);

    virtual double three_integral(size_t, size_t, size_t) {
        outfile->Printf("\n Oh no!, you tried to grab a ThreeIntegral but this "
                        "is not there!!");
        throw PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral");
    }
    virtual ambit::Tensor three_integral_block(const std::vector<size_t>&,
                                               const std::vector<size_t>&,
                                               const std::vector<size_t>&) {
        outfile->Printf("\n Oh no!, you tried to grab a ThreeIntegral but this "
                        "is not there!!");
        throw PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral");
    }
    virtual ambit::Tensor three_integral_block_two_index(const std::vector<size_t>&, size_t,
                                                         const std::vector<size_t>&) {
        outfile->Printf("\n Oh no! this isn't here");
        throw PSIEXCEPTION("INT_TYPE=DISKDF");
    }

    virtual double** three_integral_pointer() {
        outfile->Printf("\n Doh! There is no Three_integral here.  Use DF/CD");
        throw PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral!");
    }

    virtual void make_fock_matrix(SharedMatrix gamma_a, SharedMatrix gamma_b);

    virtual size_t nthree() const { throw PSIEXCEPTION("Wrong Integral type"); }

    virtual void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                         bool alpha2);
  private:
    // ==> Class data <==

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

    // ==> Class private virtual functions <==

    virtual void gather_integrals();
    virtual void resort_integrals_after_freezing();
};

} // namespace forte
} // namespace psi

#endif // _integrals_h_
