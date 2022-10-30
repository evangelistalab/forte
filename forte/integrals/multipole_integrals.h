/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _multipole_integrals_
#define _multipole_integrals_

#include <vector>

#include "psi4/libmints/matrix.h"

#include <ambit/tensor.h>

#include "base_classes/mo_space_info.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"

namespace forte {
class MultipoleIntegrals {
  public:
    /// @brief Contructor to create integrals for an active space
    /// @param ints forte integral object
    /// @param mo_space_info The MOSpaceInfo object
    /// @param order the order of multipole (1=dipole, 2=quadrupole)
    MultipoleIntegrals(std::shared_ptr<ForteIntegrals> ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info, int order);

    /// @brief Electronic multipole moment matrix element
    /// @param direction The direction of multipole moment
    /// @param p The bra index
    /// @param q The ket index
    /// @param corr Whether indices p, q start counting from correlated orbitals
    /// @return electronic multipole matrix element M_{pq}
    double mp_ints(int direction, size_t p, size_t q, bool corr = true) const;

    /// Nuclear contributions to dipole moments in X, Y, Z order
    std::vector<double> nuclear_dipole() const;

    /// Frozen-orbital contributions to dipole moments
    std::vector<double> mp_frozen_core() const;

    /// Return the MO space info object
    std::shared_ptr<MOSpaceInfo> mo_space_info() const;

    /// Return the order of multipole
    int order() const;
    /// Return the number of directions
    int ndirs() const;

  private:
    /// The MO space info object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Mutipole order
    int order_;
    /// Number of directions of multipole
    int ndirs_;

    /// The mapping from correlated MO to full MO (frozen + correlated)
    std::vector<size_t> cmotomo_;

    /// Nuclear dipole moment
    std::vector<double> nuc_dipole_;
    /// Frozen-core contributions
    std::vector<double> mp_frzc_;

    /// MO multipole integrals (frozen orbitals included)
    /// each element is a nmo x nmo psi::SharedMatrix in Pitzer order
    /// dipole order: X, Y, Z
    /// quadrupole order: XX, XY, XZ, YY, YZ, ZZ
    std::vector<psi::SharedMatrix> mp_ints_;
};

class ActiveMultipoleIntegrals {
  public:
    // ==> Class Constructors <==

    /// @brief Contructor to create integrals for an active space
    /// @param mpints MultipoleIntegrals object
    ActiveMultipoleIntegrals(std::shared_ptr<MultipoleIntegrals> mpints);

    // ==> Class Interface <==

    /// Nuclear contributions to dipole moment
    std::vector<double> nuclear_dipole() const;

    /// Compute electronic contributions
    /// Dipole in X, Y, Z order
    /// Quadrupole in XX, XY, XZ, YY, YZ, ZZ order
    std::vector<double> compute_electronic_multipole(std::shared_ptr<RDMs> rdms);

    /// Frozen-core contributions
    std::vector<double> scalars_fdocc() const;
    /// Core (restricted docc) contribution
    std::vector<double> scalars_rdocc() const;
    /// Inactive (frozen docc + restricted docc) contribution
    std::vector<double> scalars() const;

    /// Set scalar term
    void set_scalar_rdocc(int direction, double value);
    /// Set 1-body integrals
    void set_1body(int direction, ambit::Tensor M1);
    /// Set spin-free 2-body similarity transformed integrals
    void set_2body(int direction, ambit::Tensor M2);
    /// Set spin-dependent 2-body similarity transformed integrals
    void set_2body(int direction, ambit::Tensor M2aa, ambit::Tensor M2ab, ambit::Tensor M2bb);

  private:
    // ==> Class Private Data <==

    /// The integrals object
    std::shared_ptr<MultipoleIntegrals> mpints_;
    /// Order of multipole
    int order_;

    /// The number of MOs
    size_t nmo_;
    /// The number of MOs squared
    size_t nmo2_;
    /// The number of MOs cubed
    size_t nmo3_;
    /// The number of MOs to the fourth power
    size_t nmo4_;

    /// Contributions of inactive orbitals
    std::vector<double> scalars_rdocc_;
    /// One-body integrals
    std::vector<ambit::Tensor> one_body_ints_;
    /// Two-body integrals, spin free
    std::vector<ambit::Tensor> two_body_ints_;
    /// Two-body integrals, alpha-alpha spin
    std::vector<ambit::Tensor> two_body_ints_aa_;
    /// Two-body integrals, alpha-beta spin
    std::vector<ambit::Tensor> two_body_ints_ab_;
    /// Two-body integrals, beta-beta spin
    std::vector<ambit::Tensor> two_body_ints_bb_;

    /// Test the dimension of a given tensor
    void _test_tensor_dims(ambit::Tensor T);
};
} // namespace forte

#endif // _multipole_integrals_