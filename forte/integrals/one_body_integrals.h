/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _one_body_integrals_
#define _one_body_integrals_

#include <vector>

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/vector3.h"

#include <ambit/tensor.h>

#include "base_classes/mo_space_info.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"

namespace forte {
class MultipoleIntegrals {
  public:
    /**
     * @brief Construct a new Multipole Integrals object
     *
     * @param ints forte integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    MultipoleIntegrals(std::shared_ptr<ForteIntegrals> ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info);

    /**
     * @brief Electronic dipole moment integrals matrix element
     *
     * @param direction The direction of dipole moment [X, Y, Z]
     * @param p The bra index (frozen orbitals included)
     * @param q The ket index (frozen orbitals included)
     * @return electronic dipole matrix element M_{pq}
     */
    double dp_ints(int direction, size_t p, size_t q) const;

    /**
     * @brief Electronic quadrupole moment integrals matrix element
     *
     * @param direction The direction of quadrupole moment [XX, XY, XZ, YY, YZ, ZZ]
     * @param p The bra index (frozen orbitals included)
     * @param q The ket index (frozen orbitals included)
     * @return electronic quadrupole matrix element M_{pq}
     */
    double qp_ints(int direction, size_t p, size_t q) const;

    /**
     * @brief Electronic dipole moment integrals matrix element
     *
     * @param direction The direction of dipole moment [X, Y, Z]
     * @param p The bra index (frozen orbitals excluded)
     * @param q The ket index (frozen orbitals excluded)
     * @return electronic dipole matrix element M_{pq}
     */
    double dp_ints_corr(int direction, size_t p, size_t q) const;

    /**
     * @brief Electronic quadrupole moment integrals matrix element
     *
     * @param direction The direction of quadrupole moment [XX, XY, XZ, YY, YZ, ZZ]
     * @param p The bra index (frozen orbitals excluded)
     * @param q The ket index (frozen orbitals excluded)
     * @return electronic quadrupole matrix element M_{pq}
     */
    double qp_ints_corr(int direction, size_t p, size_t q) const;

    /// Nuclear contributions to dipole moments in X, Y, Z order
    std::shared_ptr<psi::Vector> nuclear_dipole(const psi::Vector3& origin = {0.0, 0.0, 0.0}) const;

    /// Nuclear contributions to quadrupole moments in XX, XY, XZ, YY, YZ, ZZ order
    std::shared_ptr<psi::Vector> nuclear_quadrupole(const psi::Vector3& origin = {0.0, 0.0,
                                                                                  0.0}) const;

    /// Frozen-orbital contributions to dipole moments
    std::shared_ptr<psi::Vector> dp_frozen_core() const;

    /// Frozen-orbital contributions to quadrupole moments
    std::shared_ptr<psi::Vector> qp_frozen_core() const;

    /// Return the MO space info object
    std::shared_ptr<MOSpaceInfo> mo_space_info() const;

  private:
    /// The MO space info object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The mapping from correlated MO to full MO (frozen + correlated)
    std::vector<size_t> cmotomo_;
    /// The molecule object used to compute nuclear contribution
    std::shared_ptr<psi::Molecule> molecule_;

    /// MO dipole integrals (frozen orbitals included)
    /// each element is a nmo x nmo std::shared_ptr<psi::Matrix> in Pitzer order
    /// X, Y, Z
    std::vector<std::shared_ptr<psi::Matrix>> dp_ints_;

    /// MO quadrupole integrals (frozen orbitals included)
    /// each element is a nmo x nmo std::shared_ptr<psi::Matrix> in Pitzer order
    /// XX, XY, XZ, YY, YZ, ZZ
    std::vector<std::shared_ptr<psi::Matrix>> qp_ints_;
};

class ActiveMultipoleIntegrals {
  public:
    // ==> Class Constructors <==

    /**
     * @brief Construct a new Active Multipole Integrals object
     *
     * @param mpints MultipoleIntegrals object
     */
    ActiveMultipoleIntegrals(std::shared_ptr<MultipoleIntegrals> mpints);

    // ==> Class Interface <==

    /// Nuclear contributions to dipole moment
    std::shared_ptr<psi::Vector> nuclear_dipole(const psi::Vector3& origin = {0.0, 0.0, 0.0}) const;
    /// Nuclear contributions to quadrupole moment
    std::shared_ptr<psi::Vector> nuclear_quadrupole(const psi::Vector3& origin = {0.0, 0.0,
                                                                                  0.0}) const;

    /// Compute electronic contributions to dipole moment [X, Y, Z]
    std::shared_ptr<psi::Vector> compute_electronic_dipole(std::shared_ptr<RDMs> rdms,
                                                           bool transition = false);
    /// Compute electronic contributions to dipole moment [XX, XY, XZ, YY, YZ, ZZ]
    std::shared_ptr<psi::Vector> compute_electronic_quadrupole(std::shared_ptr<RDMs> rdms,
                                                               bool transition = false);

    /// Dipole from frozen orbitals
    std::shared_ptr<psi::Vector> dp_scalars_fdocc() const;
    /// Dipole from restricted docc orbitals
    std::shared_ptr<psi::Vector> dp_scalars_rdocc() const;
    /// Dipole from doubly occupied orbitals
    std::shared_ptr<psi::Vector> dp_scalars() const;

    /// Set dipole scalar term from restricted docc orbitals
    void set_dp_scalar_rdocc(int direction, double value);
    /// Set 1-body dipole integrals
    void set_dp1_ints(int direction, ambit::Tensor M1);
    /// Set spin-free 2-body similarity transformed dipole integrals
    void set_dp2_ints(int direction, ambit::Tensor M2);
    /// Set spin-dependent 2-body similarity transformed dipole integrals
    void set_dp2_ints(int direction, ambit::Tensor M2aa, ambit::Tensor M2ab, ambit::Tensor M2bb);

    /// Quadrupole from frozen orbitals
    std::shared_ptr<psi::Vector> qp_scalars_fdocc() const;
    /// Quadrupole from restricted docc orbitals
    std::shared_ptr<psi::Vector> qp_scalars_rdocc() const;
    /// Quadrupole from doubly occupied orbitals
    std::shared_ptr<psi::Vector> qp_scalars() const;

    /// Set quadrupole scalar term from restricted docc orbitals
    void set_qp_scalar_rdocc(int direction, double value);
    /// Set 1-body quadrupole integrals
    void set_qp1_ints(int direction, ambit::Tensor M1);
    /// Set spin-free 2-body similarity transformed quadrupole integrals
    void set_qp2_ints(int direction, ambit::Tensor M2);
    /// Set spin-dependent 2-body similarity transformed quadrupole integrals
    void set_qp2_ints(int direction, ambit::Tensor M2aa, ambit::Tensor M2ab, ambit::Tensor M2bb);

    /// Return the dipole many-body level
    int dp_many_body_level() const;
    /// Return the quadrupole many-body level
    int qp_many_body_level() const;

    /// Set dipole integrals name
    void set_dp_name(const std::string& name) { dp_name_ = name; }
    /// Set quadrupole integrals name
    void set_qp_name(const std::string& name) { qp_name_ = name; }
    /// Return dipole integrals name
    std::string dp_name() const { return dp_name_; }
    /// Return quadrupole integrals name
    std::string qp_name() const { return qp_name_; }

  private:
    // ==> Class Private Data <==

    /// The integrals object
    std::shared_ptr<MultipoleIntegrals> mpints_;
    /// Many-body level of dipole integrals
    int dp_many_body_level_;
    /// Many-body level of quadrupole integrals
    int qp_many_body_level_;

    /// Dipole integrals name
    std::string dp_name_ = "";
    /// Quadrupole integrals name
    std::string qp_name_ = "";

    /// The number of MOs
    size_t nmo_;
    /// The number of MOs squared
    size_t nmo2_;
    /// The number of MOs cubed
    size_t nmo3_;
    /// The number of MOs to the fourth power
    size_t nmo4_;

    /// Dipole from inactive orbitals
    std::shared_ptr<psi::Vector> dp0_rdocc_;
    /// One-body dipole integrals
    std::vector<ambit::Tensor> dp1_ints_;
    /// Two-body dipole integrals, spin free
    std::vector<ambit::Tensor> dp2_ints_;
    /// Two-body dipole integrals, alpha-alpha spin
    std::vector<ambit::Tensor> dp2_ints_aa_;
    /// Two-body dipole integrals, alpha-beta spin
    std::vector<ambit::Tensor> dp2_ints_ab_;
    /// Two-body dipole integrals, beta-beta spin
    std::vector<ambit::Tensor> dp2_ints_bb_;

    /// Quadrupole from inactive orbitals
    std::shared_ptr<psi::Vector> qp0_rdocc_;
    /// One-body quadrupole integrals
    std::vector<ambit::Tensor> qp1_ints_;
    /// Two-body quadrupole integrals, spin free
    std::vector<ambit::Tensor> qp2_ints_;
    /// Two-body quadrupole integrals, alpha-alpha spin
    std::vector<ambit::Tensor> qp2_ints_aa_;
    /// Two-body quadrupole integrals, alpha-beta spin
    std::vector<ambit::Tensor> qp2_ints_ab_;
    /// Two-body quadrupole integrals, beta-beta spin
    std::vector<ambit::Tensor> qp2_ints_bb_;

    /// Test the dimension of a given tensor
    void _test_tensor_dims(ambit::Tensor T);
};
} // namespace forte

#endif // _one_body_integrals_