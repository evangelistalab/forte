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

#ifndef _casscf_hess_h_
#define _casscf_hess_h_

#include <map>

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

class SCFInfo;

class CASSCF_HESS {
  public:
    CASSCF_HESS(const std::map<StateInfo, std::vector<double>>& state_weights_map,
                std::shared_ptr<forte::SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ForteIntegrals> ints);

    /// Compute the CASSCF energy
    double compute_energy();

  private:
    /// Read options
    void read_options();
    /// Set up MO spaces
    void setup_mo_spaces();
    /// Set up orbital spaces for ambit
    void setup_ambit();

    /// Nuclear repulsion energy
    double Enuc_;
    /// The one-electron integrals
    ambit::BlockedTensor oei_;
    /// The two-electron integrals in chemists' notation
    ambit::BlockedTensor tei_;
    /// Fill in integrals
    void build_ints();

    /// Spin-summed 1RDM
    ambit::BlockedTensor d1_;
    /// Spin-summed 2RDM symmetrized
    ambit::BlockedTensor d2_;
    /// Grab densities from RDMs
    void build_density();

    /// Orthogonal transformation matrix for orbitals
    ambit::BlockedTensor U_;
    psi::SharedMatrix Um_;
    /// Transformation matrix for orbitals
    ambit::BlockedTensor T_;
    psi::SharedMatrix Tm_;
    /// Antisymmetric matrix for orbital rotation
    psi::SharedMatrix R_;
    /// Increamental orbital rotation
    psi::SharedMatrix dR_;

    /// Identity matrix
    ambit::BlockedTensor I_;
    /// Closed-shell energy
    double Ec_;
    /// Closed-shell Fock matrix
    ambit::BlockedTensor Fc_;
    /// Compute closed-shell contributions
    void compute_closed_shell();

    /// Initial CASCI energy in a macroiteration
    double E0_;
    /// Compute energy using initial orbitals and CI in a macroiteration
    void compute_init_macro_energy();

    /// Averaged Fock matrix
    ambit::BlockedTensor F_;
    /// L intermediate tensor
    ambit::BlockedTensor L4_;
    /// Build Fock matrix
    void build_fock();
    /// Build intermediate L
    void build_L4();

    /// Gradient matrix A
    ambit::BlockedTensor A_;
    /// Gradient matrix B
    ambit::BlockedTensor B_;
    /// Hessian tensor G
    ambit::BlockedTensor G4_;
    /// Compute the second-order energy for fixed CI coefficients
    double compute_2nd_o();

    /// Second-order expansion of the closed-shell energy
    double Ec2_;
    /// Second-order expansion of one-electron integrals
    ambit::BlockedTensor Fc2_;
    /// Second-order expansion of two-electron integrals
    ambit::BlockedTensor Vab2_;
    /// Build the second-order expansion of integrals for active space solver
    void build_2nd_actv_ints();
    /// Compute the second-order energy for a given orthogonal matrix U
    double compute_2nd_ci();

    /// Orbital gradient matrix A tilde
    ambit::BlockedTensor Atilde_;
    /// Orbital hessian tensor G tilde
    ambit::BlockedTensor Gtilde_;
    /// Solve the orbital optimization equations
    void orbital_step();

    /// List of correlated MOs
    std::vector<size_t> corr_mos_;
    /// List of core MOs (Correlated)
    std::vector<size_t> rdocc_mos_;
    /// List of active MOs (Correlated)
    std::vector<size_t> actv_mos_;
    /// List of virtual MOs (Correlated)
    std::vector<size_t> ruocc_mos_;
    ///
    std::map<std::string, std::vector<size_t>> label_to_mos_;

    /// The list of states to computed. Passed to the ActiveSpaceSolver
    std::map<StateInfo, std::vector<double>> state_weights_map_;
    /// SCF information
    std::shared_ptr<SCFInfo> scf_info_;
    /// The options
    std::shared_ptr<ForteOptions> options_;
    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// The active one RDM in the MO basis
    ambit::Tensor gamma1_;
    /// The active two RDM (may need to be symmetrized)
    ambit::Tensor gamma2_;
    /// The reference object generated from Francesco's Full CI
    RDMs rdms_;
    /// The energy computed in FCI with updates from CASSCF and CI
    double E_casscf_;
    std::shared_ptr<ForteIntegrals> ints_;

    /// The number of irreps
    size_t nirrep_;
    /// The number of SO (AO for C matrices)
    psi::Dimension nsopi_;

    /// The number of active orbitals
    size_t nactv_;
    /// the number of restricted_docc
    size_t nrdocc_;
    /// The number of frozen_docc
    size_t nfdocc_;
    /// The number of virtual orbitals
    size_t nruocc_;
    /// The number of correlated molecular orbitals (not frozen)
    size_t ncmo_;
    /// The number of NMO including frozen core
    size_t nmo_;

    /// List of core MOs (Absolute)
    std::vector<size_t> core_mos_abs_;
    /// List of active MOs (Absolute)
    std::vector<size_t> actv_mos_abs_;

    /// The psi::Dimensions for frozen docc
    psi::Dimension frozen_docc_dim_;
    /// The psi::Dimensions for restricted docc
    psi::Dimension restricted_docc_dim_;
    /// The psi::Dimensions for active
    psi::Dimension active_dim_;
    /// The psi::Dimensions for restricted uocc
    psi::Dimension restricted_uocc_dim_;
    /// The psi::Dimensions for frozen uocc
    psi::Dimension inactive_docc_dim_;
    /// The psi::Dimensions for all correlated orbitals
    psi::Dimension corr_dim_;
    /// The psi::Dimensions for all orbitals
    psi::Dimension nmo_dim_;

    /// List of relative MOs for restricted docc
    std::vector<std::pair<size_t, size_t>> core_mos_rel_;
    /// List of relative MOs for active
    std::vector<std::pair<size_t, size_t>> actv_mos_rel_;
    /// List of relative MOs for restricted docc
    std::vector<std::pair<size_t, size_t>> virt_mos_rel_;

    std::vector<size_t> frozen_docc_abs_;
    std::vector<size_t> restricted_docc_abs_;
    std::vector<size_t> active_abs_;
    std::vector<size_t> restricted_uocc_abs_;
    std::vector<size_t> inactive_docc_abs_;
    std::vector<size_t> nmo_abs_;

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

//    /// One-particle density matrix
//    ambit::BlockedTensor Gamma1_;
//    /// Two-body denisty tensor
//    ambit::BlockedTensor Gamma2_;
//    // Lagrangian tensor
//    ambit::BlockedTensor W_;
//    // core Hamiltonian
//    ambit::BlockedTensor H_;
//    // two-electron integrals
//    ambit::BlockedTensor V_;
//    // Fock matrix
//    ambit::BlockedTensor F_;
//    /// Kevin's Tensor Wrapper
//    std::shared_ptr<BlockedTensorFactory> BTF_;

    /// These member variables are all summarized in Algorithm 1
    /// Equation 9

    /// The Fock matrix due to frozen core orbitals
    psi::SharedMatrix F_frozen_core_;
    /// The one-electron integral matrix in the AO basis (H = T + V)
    psi::SharedMatrix Hcore_;
    /// The JK object.  Built in constructor
    std::shared_ptr<psi::JK> JK_;
    /// Diagonalize the Hamiltonian using the updated MO coefficients (does FCI, sCI, DMRG, etc.)
    void diagonalize_hamiltonian();
    /// Read all the mospace info and assign correct dimensions
    void startup();
    /// Compute overlap between old_c and new_c
    void overlap_orbitals(const psi::SharedMatrix& C_old, const psi::SharedMatrix& C_new);
    void overlap_coefficients();
    void write_orbitals_molden();

    /// DEBUG PRINTING
    bool debug_print_;
    /// Freeze the core and leave them unchanged
    /// set frozen_core_orbitals
    std::shared_ptr<psi::Matrix> set_frozen_core_orbitals();
    /// Compute the restricted_one_body operator for FCI(done also in
    /// OrbitalOptimizer)

    std::vector<double> compute_restricted_docc_operator();

    double scalar_energy_ = 0.0;

    /// Transform the active integrals
    ambit::Tensor transform_integrals(std::shared_ptr<psi::Matrix> Ca);

    /// The transform integrals computed from transform_integrals
    ambit::Tensor tei_gaaa_;

    /// The print level
    int print_ = 0;

    /// The CISolutions per iteration
    std::vector<std::vector<std::shared_ptr<FCIVector>>> CISolutions_;
    std::shared_ptr<ActiveSpaceIntegrals> get_ci_integrals();

    /// Semi-canonicalize orbital and return the rotation matrix
    std::shared_ptr<psi::Matrix> semicanonicalize(std::shared_ptr<psi::Matrix> Ca);
    /// Build Fock matrix
    std::shared_ptr<psi::Matrix> build_fock(std::shared_ptr<psi::Matrix> Ca);
    /// Build the inactive Fock (part that does not depend on 1RDM), includes frozen docc
    std::shared_ptr<psi::Matrix> build_fock_inactive(std::shared_ptr<psi::Matrix> Ca);
    /// Build the active Fock (part that does depend on 1RDM)
    std::shared_ptr<psi::Matrix> build_fock_active(std::shared_ptr<psi::Matrix> Ca);

    /// Maximum macroiteration
    int maxiter_macro_;

    /// CI solver name
    std::string ci_type_;
};
} // namespace forte

#endif // _casscf_h_
