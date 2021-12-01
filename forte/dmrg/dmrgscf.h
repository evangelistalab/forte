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

#ifndef DMRG_H
#define DMRG_H

#include "psi4/libfock/jk.h"
#include "psi4/libtrans/mospace.h"
#include "base_classes/rdms.h"
#include "base_classes/active_space_method.h"
#include "integrals/integrals.h"
#include "base_classes/mo_space_info.h"

#include "chemps2/Irreps.h"
#include "chemps2/Problem.h"
#include "chemps2/CASSCF.h"
#include "chemps2/Initialize.h"
#include "chemps2/EdmistonRuedenberg.h"

namespace forte {

class DMRGSCF : public ActiveSpaceMethod {
  public:
    DMRGSCF(StateInfo state, std::shared_ptr<SCFInfo> scf_info,
            std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
            std::shared_ptr<MOSpaceInfo> mo_space_info);

    double compute_energy();

    RDMs rmds() { return dmrg_rdms_; }
    void set_iterations(int dmrg_iterations) { dmrg_iterations_ = dmrg_iterations; }

  private:
    RDMs dmrg_rdms_;
    int dmrg_iterations_ = 1;

    StateInfo state_;
    std::shared_ptr<SCFInfo> scf_info_;
    std::shared_ptr<ForteOptions> options_;
    std::shared_ptr<ForteIntegrals> ints_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    void set_up_ints();
    void compute_reference(double* one_rdm, double* two_rdm, double* three_rdm,
                           CheMPS2::DMRGSCFindices* iHandler);
    /// Ported over codes from DMRGSCF plugin
    void startup();

    /// Form the active fock matrix
    void buildJK(psi::SharedMatrix MO_RDM, psi::SharedMatrix MO_JK, psi::SharedMatrix Cmat,
                 std::shared_ptr<psi::JK> myJK);
    /// Form Inactive fock matrix
    void buildQmatOCC(CheMPS2::DMRGSCFmatrix* theQmatOCC, CheMPS2::DMRGSCFindices* iHandler,
                      psi::SharedMatrix MO_RDM, psi::SharedMatrix MO_JK, psi::SharedMatrix Cmat,
                      std::shared_ptr<psi::JK> myJK);
    void buildTmatrix(CheMPS2::DMRGSCFmatrix* theTmatrix, CheMPS2::DMRGSCFindices* iHandler,
                      std::shared_ptr<psi::PSIO> psio, psi::SharedMatrix Cmat);

    /// Form active fock matrix
    void buildQmatACT(CheMPS2::DMRGSCFmatrix* theQmatACT, CheMPS2::DMRGSCFindices* iHandler,
                      double* DMRG1DM, psi::SharedMatrix MO_RDM, psi::SharedMatrix MO_JK,
                      psi::SharedMatrix Cmat, std::shared_ptr<psi::JK> myJK);

    void buildHamDMRG(std::shared_ptr<psi::IntegralTransform> ints,
                      std::shared_ptr<psi::MOSpace> Aorbs_ptr, CheMPS2::DMRGSCFmatrix* theTmatrix,
                      CheMPS2::DMRGSCFmatrix* theQmatOCC, CheMPS2::DMRGSCFindices* iHandler,
                      CheMPS2::Hamiltonian* HamDMRG, std::shared_ptr<psi::PSIO> psio);
    void buildHamDMRGForte(CheMPS2::DMRGSCFmatrix* theQmatOCC, CheMPS2::DMRGSCFindices* iHandler,
                           CheMPS2::Hamiltonian* HamDMRG, std::shared_ptr<ForteIntegrals> ints);

    void fillRotatedTEI_coulomb(std::shared_ptr<psi::IntegralTransform> ints,
                                std::shared_ptr<psi::MOSpace> OAorbs_ptr,
                                CheMPS2::DMRGSCFmatrix* theTmatrix,
                                CheMPS2::DMRGSCFintegrals* theRotatedTEI,
                                CheMPS2::DMRGSCFindices* iHandler, std::shared_ptr<psi::PSIO> psio);
    void fillRotatedTEI_exchange(std::shared_ptr<psi::IntegralTransform> ints,
                                 std::shared_ptr<psi::MOSpace> OAorbs_ptr,
                                 std::shared_ptr<psi::MOSpace> Vorbs_ptr,
                                 CheMPS2::DMRGSCFintegrals* theRotatedTEI,
                                 CheMPS2::DMRGSCFindices* iHandler,
                                 std::shared_ptr<psi::PSIO> psio);
    void copyUNITARYtoPSIMX(CheMPS2::DMRGSCFunitary* unitary, CheMPS2::DMRGSCFindices* iHandler,
                            psi::SharedMatrix target);
    void update_WFNco(psi::SharedMatrix orig_coeff, CheMPS2::DMRGSCFindices* iHandler,
                      CheMPS2::DMRGSCFunitary* unitary, psi::SharedMatrix work1,
                      psi::SharedMatrix work2);

    /// Makes sure that CHEMPS2 and PSI4 have same symmetry
    int chemps2_groupnumber(const string SymmLabel);
    /// Copies PSI4Matrices to CHEMPS2 matrices and vice versa
    void copyPSIMXtoCHEMPS2MX(psi::SharedMatrix source, CheMPS2::DMRGSCFindices* iHandler,
                              CheMPS2::DMRGSCFmatrix* target);
    void copyCHEMPS2MXtoPSIMX(CheMPS2::DMRGSCFmatrix* source, CheMPS2::DMRGSCFindices* iHandler,
                              psi::SharedMatrix target);
};
} // namespace forte

#endif // DMRG_H
