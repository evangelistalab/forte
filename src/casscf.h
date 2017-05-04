/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef CASSCF_H
#define CASSCF_H

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libfock/jk.h"

#include "integrals/integrals.h"
#include "ambit/blocked_tensor.h"
#include "reference.h"
#include "helpers.h"
#include "blockedtensorfactory.h"
#include "fci/fci_vector.h"
#include "fci/fci_integrals.h"
#include "semi_canonicalize.h"

namespace psi {
namespace forte {

class CASSCF : public Wavefunction {
  public:
    /**
     * @brief CASSCF::CASSCF
     * @param options -> Options object
     * @param ints    -> The integral object.  I may not use this as I need the
     * AO based integrals
     * @param mo_space_info -> The MOSpaceInfo object for getting active space
     * information
     * This class will implement the AO based CASSCF by Hohenstein and Martinez.
     * Ref is .  Hohenstein J.Chem.Phys, 142, 224103.
     * This reference has a nice algorithmic flowchart.  Look it up
     *
     */
    CASSCF(SharedWavefunction ref_wfn, Options& options,
           std::shared_ptr<ForteIntegrals> ints,
           std::shared_ptr<MOSpaceInfo> mo_space_info);
    /// Compute CASSCF given a 1RDM and 2RDM
    void compute_casscf();
    /// Use daniels code to compute Orbital optimization
    // void compute_casscf_soscf();
    /// Return the final gamma1
    ambit::Tensor gamma1() { return gamma1_; }
    /// Return the final gamma2;
    ambit::Tensor gamma2() { return gamma2_; }
    double compute_energy() { return E_casscf_; }

  private:
    /// The active one RDM in the MO basis
    ambit::Tensor gamma1_;

    /// The active two RDM (may need to be symmetrized)
    ambit::Tensor gamma2_;
    /// The reference object generated from Francesco's Full CI
    Reference cas_ref_;
    /// The energy computed in FCI with updates from CASSCF and CI
    double E_casscf_;
    /// The OPtions object
    Options options_;
    std::shared_ptr<ForteIntegrals> ints_;
    /// The mo_space_info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The dimension for number of molecular orbitals (CORRELATED or ALL)
    Dimension nmopi_;
    /// The number of correlated molecular orbitals (Restricted Core + Active +
    /// Restricted_UOCC + Frozen_Virt
    size_t nmo_;
    /// The number of active orbitals
    size_t na_;
    /// The number of irreps
    size_t nirrep_;
    /// The number of SO (AO for C matrices)
    Dimension nsopi_;
    /// the number of restricted_docc
    size_t nrdocc_;
    /// The number of frozen_docc
    size_t nfrozen_;
    /// The number of virtual orbitals
    size_t nvir_;
    /// The number of NMO including frozen core
    size_t all_nmo_;

    /// These member variables are all summarized in Algorithm 1
    /// Equation 9

    /// The Fock matrix due to Frozen core orbitals
    SharedMatrix F_froze_;
    /// The One Electron integrals (H = T + V)  (in AO basis)
    SharedMatrix Hcore_;
    /// The JK object.  Built in constructor
    std::shared_ptr<JK> JK_;
    /// Perform a CAS-CI with the updated MO coefficients
    void cas_ci();
    /// Sets up the FCISolver
    void set_up_fci();
    /// Set up a SA-FCI
    void set_up_sa_fci();
    /// check the cas_ci energy with spin-free RDM
    double cas_check(Reference cas);
    /// Read all the mospace info and assign correct dimensions
    void startup();
    /// Compute overlap between old_c and new_c
    void overlap_orbitals(const SharedMatrix& C_old, const SharedMatrix& C_new);
    void overlap_coefficients();
    void write_orbitals_molden();
    /// Diagonalize F_I + F_A
    std::pair<SharedMatrix, SharedVector> casscf_canonicalize();

    /// DEBUG PRINTING
    bool casscf_debug_print_;
    /// Freeze the core and leave them unchanged
    /// set frozen_core_orbitals
    std::shared_ptr<Matrix> set_frozen_core_orbitals();
    /// Compute the restricted_one_body operator for FCI(done also in
    /// OrbitalOptimizer)

    std::vector<std::vector<double>> compute_restricted_docc_operator();

    double scalar_energy_ = 0.0;
    /// The Dimensions for the major orbitals spaces involved in CASSCF
    /// Trying to get these all in the startup, so I can use them repeatly
    /// rather than create them in different places
    Dimension frozen_docc_dim_;
    Dimension restricted_docc_dim_;
    Dimension active_dim_;
    Dimension restricted_uocc_dim_;
    Dimension inactive_docc_dim_;

    std::vector<size_t> frozen_docc_abs_;
    std::vector<size_t> restricted_docc_abs_;
    std::vector<size_t> active_abs_;
    std::vector<size_t> restricted_uocc_abs_;
    std::vector<size_t> inactive_docc_abs_;
    std::vector<size_t> nmo_abs_;
    /// Transform the active integrals
    ambit::Tensor transform_integrals();
    std::pair<ambit::Tensor, std::vector<double>> CI_Integrals();
    /// The transform integrals computed from transform_integrals
    ambit::Tensor tei_paaa_;
    int print_;
    /// The CISolutions per iteration
    std::vector<std::vector<std::shared_ptr<FCIWfn>>> CISolutions_;
};
}
}

#endif // CASSCF_H
