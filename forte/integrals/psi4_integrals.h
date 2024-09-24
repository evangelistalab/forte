/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER,
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

#pragma once

#include <vector>

#include "psi4/libfock/jk.h"
#include "psi4/libmints/dimension.h"
#include "ambit/blocked_tensor.h"

#include "integrals/integrals.h"

class Tensor;

namespace psi {
class Options;
class Matrix;
class Vector;
class Wavefunction;
class Dimension;
class BasisSet;
} // namespace psi

namespace forte {

class ForteOptions;
class MOSpaceInfo;
class SCFInfo;

/**
 * @brief Interface to integrals read from psi4
 */
class Psi4Integrals : public ForteIntegrals {
  public:
    Psi4Integrals(std::shared_ptr<ForteOptions> options, std::shared_ptr<SCFInfo> scf_info,
                  std::shared_ptr<psi::Wavefunction> ref_wfn,
                  std::shared_ptr<MOSpaceInfo> mo_space_info, IntegralType integral_type,
                  IntegralSpinRestriction restricted);

    std::shared_ptr<psi::Wavefunction> wfn() override;
    std::shared_ptr<psi::JK> jk() override;
    void jk_finalize() override;

    std::shared_ptr<psi::Matrix> Ca_SO2AO(std::shared_ptr<const psi::Matrix> Ca_SO) const override;

    /// Make the generalized Fock matrix using Psi4 JK object
    void make_fock_matrix(ambit::Tensor Da, ambit::Tensor Db) override;

    /// Make the closed-shell Fock matrix using Psi4 JK object
    std::tuple<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>, double>
    make_fock_inactive(psi::Dimension dim_start, psi::Dimension dim_end) override;

    /// Make the active Fock matrix using Psi4 JK object
    std::tuple<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>>
    make_fock_active(ambit::Tensor Da, ambit::Tensor Db) override;

    /// Make the active Fock matrix using restricted equation
    std::shared_ptr<psi::Matrix>
    make_fock_active_restricted(std::shared_ptr<psi::Matrix> D) override;

    /// Make the active Fock matrix using unrestricted equation
    std::tuple<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>>
    make_fock_active_unrestricted(std::shared_ptr<psi::Matrix> Da,
                                  std::shared_ptr<psi::Matrix> Db) override;

    /// Orbital coefficients in AO x MO basis, where MO is in Pitzer order
    std::shared_ptr<psi::Matrix> Ca_AO() const override;

    /// Build and return MO dipole integrals (X, Y, Z) in Pitzer order
    std::vector<std::shared_ptr<psi::Matrix>> mo_dipole_ints() const override;

    /// Build and return MO quadrupole integrals (XX, XY, XZ, YY, YZ, ZZ) in Pitzer order
    std::vector<std::shared_ptr<psi::Matrix>> mo_quadrupole_ints() const override;

  private:
    void base_initialize_psi4();
    void setup_psi4_ints();
    void transform_one_electron_integrals();
    void compute_frozen_one_body_operator() override;
    void update_orbitals(std::shared_ptr<psi::Matrix> Ca, std::shared_ptr<psi::Matrix> Cb,
                         bool re_transform = true) override;
    void rotate_mos() override;

    /// Build AO dipole and quadrupole integrals
    void build_multipole_ints_ao() override;

    /// Make a shared pointer to a Psi4 JK object
    void make_psi4_JK();
    /// Call JK intialize
    void jk_initialize(double mem_percentage = 0.8, int print_level = 1);

    /// AO Fock control
    enum class FockAOStatus { none, inactive, generalized };
    FockAOStatus fock_ao_level_ = FockAOStatus::none;

  protected:
    void freeze_core_orbitals() override;

    /// The Wavefunction object
    std::shared_ptr<psi::Wavefunction> wfn_;

    /// JK object from Psi4
    std::shared_ptr<psi::JK> JK_;

    // threshold for DF fitting condition (Psi4)
    double df_fitting_cutoff_;
    // threshold for Schwarz cutoff (Psi4)
    double schwarz_cutoff_;
};

} // namespace forte
