/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _mp2_nos_h_
#define _mp2_nos_h_

#include "psi4/libmints/wavefunction.h"

#include "base_classes/mo_space_info.h"
#include "integrals/integrals.h"
#include "base_classes/rdms.h"
#include "base_classes/orbital_transform.h"

namespace forte {

/**
 * @brief The MP2_NOS class
 * Computes MP2 natural orbitals
 */
class MP2_NOS : public OrbitalTransform {
  public:
    // => Constructor <= //
    MP2_NOS(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
            std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    void compute_transformation();
    psi::SharedMatrix get_Ua();
    psi::SharedMatrix get_Ub();

  private:
    std::shared_ptr<SCFInfo> scf_info_;
    std::shared_ptr<ForteOptions> options_;
    psi::SharedMatrix Ua_;
    psi::SharedMatrix Ub_;

    /// List of alpha occupied MOs
    std::vector<size_t> a_occ_mos_;
    /// List of beta occupied MOs
    std::vector<size_t> b_occ_mos_;
    /// List of alpha virtual MOs
    std::vector<size_t> a_vir_mos_;
    /// List of beta virtual MOs
    std::vector<size_t> b_vir_mos_;
    /// List of auxiliary MOs
    std::vector<size_t> aux_mos_;

    /// Fock matrix alpha
    std::vector<double> Fa_;
    /// Fock matrix beta
    std::vector<double> Fb_;

    /// Build MP2 1-RDM using conventional integrals
    ambit::BlockedTensor build_1rdm_conv();

    /// Build MP2 1-RDM using DF integrals
    ambit::BlockedTensor build_1rdm_df();
    /// Build OO part of DF-MP2 1-RDM
    //    void compute_df_1rdm_oo(ambit::Tensor& Da, ambit::Tensor& Db);
    /// Build VV part of DF-MP2 1-RDM
    //    void compute_df_1rdm_vv(ambit::Tensor& Da, ambit::Tensor& Db);
};
} // namespace forte

#endif // _mp2_nos_h_
