/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef CASSCF_H
#define CASSCF_H


#include "psi4/libmints/wavefunction.h"
#include "psi4/libfock/jk.h"

#include "integrals/integrals.h"
#include "ambit/blocked_tensor.h"
#include "base_classes/rdms.h"
#include "base_classes/mo_space_info.h"
#include "helpers/blockedtensorfactory.h"
#include "fci/fci_vector.h"
#include "integrals/active_space_integrals.h"
#include "orbital-helpers/semi_canonicalize.h"
#include "base_classes/active_space_method.h"

namespace forte {

// class ActiveSpaceIntegrals;
class SCFInfo;

class CASSCF : public ActiveSpaceMethod {
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
    CASSCF(StateInfo state, size_t nroot, std::shared_ptr<forte::SCFInfo> scf_info,
           std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
           std::shared_ptr<ActiveSpaceIntegrals> as_ints);
    /// Use daniels code to compute Orbital optimization
    // void compute_casscf_soscf();
    /// Return the final gamma1
    ambit::Tensor gamma1() { return gamma1_; }
    /// Return the final gamma2;
    ambit::Tensor gamma2() { return gamma2_; }
    double compute_energy() override;

    /// Compute CASSCF gradient
    psi::SharedMatrix compute_gradient();

    void set_options(std::shared_ptr<ForteOptions>) override{};

    /// Returns the reduced density matrices up to a given level (max_rdm_level)
    std::vector<RDMs> rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                           int max_rdm_level) override;

    /// Returns the transition reduced density matrices between roots of different symmetry up to a
    /// given level (max_rdm_level)
    std::vector<RDMs> transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                      std::shared_ptr<ActiveSpaceMethod> method2,
                                      int max_rdm_level) override;

    /// check the cas_ci energy with spin-free RDM
    double cas_check(RDMs cas);

  private:
    /// SCF information
    std::shared_ptr<SCFInfo> scf_info_;
    /// The options
    std::shared_ptr<ForteOptions> options_;

    /// The active one RDM in the MO basis
    ambit::Tensor gamma1_;

    /// The active two RDM (may need to be symmetrized)
    ambit::Tensor gamma2_;
    /// The reference object generated from Francesco's Full CI
    RDMs cas_ref_;
    /// The energy computed in FCI with updates from CASSCF and CI
    double E_casscf_;
    std::shared_ptr<ForteIntegrals> ints_;

    /// The dimension for number of molecular orbitals (CORRELATED or ALL)
    psi::Dimension nmopi_;
    /// The number of correlated molecular orbitals (Restricted Core + Active +
    /// Restricted_UOCC + Frozen_Virt
    size_t nmo_;
    /// The number of active orbitals
    size_t na_;
    /// The number of irreps
    size_t nirrep_;
    /// The number of SO (AO for C matrices)
    psi::Dimension nsopi_;
    /// the number of restricted_docc
    size_t nrdocc_;
    /// The number of frozen_docc
    size_t nfrozen_;
    /// The number of virtual orbitals
    size_t nvir_;
    /// The number of NMO including frozen core
    size_t all_nmo_;

    // These are essential variables and functions for computing CASSCF gradient.
    /// Set Ambit tensor labels
    void set_ambit_space();
    /// Set density
    void set_density();
    /// Set Fock matrix
    void set_fock();
    /// Set Hamiltonian
    void set_h();
    /// Set two-electron integrals
    void set_v();
    /// Set the Lagrangian
    void set_lagrangian();
    /// Write the Lagrangian
    void write_lagrangian();
    /// Set MO space environment and global variables
    void set_all_variables();
    /// Set H, V, F and densities using ambit tensors
    void set_tensor();
    /// Write spin_dependent one-RDMs coefficients
    void write_1rdm_spin_dependent();
    /// Write spin_dependent two-RDMs coefficients using IWL
    void write_2rdm_spin_dependent();
    /// TPDM backtransform
    void tpdm_backtransform();
    /// List of core MOs (Correlated)
    std::vector<size_t> core_mos_;
    /// List of active MOs (Correlated)
    std::vector<size_t> actv_mos_;
    /// List of virtual MOs (Correlated)
    std::vector<size_t> virt_mos_;
    /// List of core MOs (Absolute)
    std::vector<size_t> core_all_;
    /// List of active MOs (Absolute)
    std::vector<size_t> actv_all_;
    /// List of relative core MOs
    std::vector<std::pair<unsigned long, unsigned long>,
                std::allocator<std::pair<unsigned long, unsigned long>>>
        core_mos_relative;
    /// List of relative active MOs
    std::vector<std::pair<unsigned long, unsigned long>,
                std::allocator<std::pair<unsigned long, unsigned long>>>
        actv_mos_relative;
    /// Dimension of different irreps
    psi::Dimension irrep_vec;
    /// One-particle density matrix
    ambit::BlockedTensor Gamma1_;
    /// Two-body denisty tensor
    ambit::BlockedTensor Gamma2_;
    // Lagrangian tensor
    ambit::BlockedTensor W_;
    // core Hamiltonian
    ambit::BlockedTensor H_;
    // two-electron integrals
    ambit::BlockedTensor V_;
    // Fock matrix
    ambit::BlockedTensor F_;
    /// Kevin's Tensor Wrapper
    std::shared_ptr<BlockedTensorFactory> BTF_;

    /// These member variables are all summarized in Algorithm 1
    /// Equation 9

    /// The Fock matrix due to Frozen core orbitals
    psi::SharedMatrix F_froze_;
    /// The One Electron integrals (H = T + V)  (in AO basis)
    psi::SharedMatrix Hcore_;
    /// The JK object.  Built in constructor
    std::shared_ptr<psi::JK> JK_;
    /// Perform a CAS-CI with the updated MO coefficients
    void cas_ci();
    /// Sets up the FCI
    void set_up_fci();
    /// Set up a SA-FCI
    //  void set_up_sa_fci();
    /// Read all the mospace info and assign correct dimensions
    void startup();
    /// Compute overlap between old_c and new_c
    void overlap_orbitals(const psi::SharedMatrix& C_old, const psi::SharedMatrix& C_new);
    void overlap_coefficients();
    void write_orbitals_molden();
    /// Diagonalize F_I + F_A
    std::pair<psi::SharedMatrix, psi::SharedVector> casscf_canonicalize();

    /// DEBUG PRINTING
    bool casscf_debug_print_;
    /// Freeze the core and leave them unchanged
    /// set frozen_core_orbitals
    std::shared_ptr<psi::Matrix> set_frozen_core_orbitals();
    /// Compute the restricted_one_body operator for FCI(done also in
    /// OrbitalOptimizer)

    // Recompute reference
    void cas_ci_final();

    std::vector<std::vector<double>> compute_restricted_docc_operator();

    double scalar_energy_ = 0.0;
    /// The psi::Dimensions for the major orbitals spaces involved in CASSCF
    /// Trying to get these all in the startup, so I can use them repeatly
    /// rather than create them in different places
    psi::Dimension frozen_docc_dim_;
    psi::Dimension restricted_docc_dim_;
    psi::Dimension active_dim_;
    psi::Dimension restricted_uocc_dim_;
    psi::Dimension inactive_docc_dim_;

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
    std::vector<std::vector<std::shared_ptr<FCIVector>>> CISolutions_;
    std::shared_ptr<ActiveSpaceIntegrals> get_ci_integrals();
};
} // namespace forte

#endif // CASSCF_H
