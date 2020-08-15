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

#ifndef _casscf_orb_h_
#define _casscf_orb_h_

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

#include "psi4/libdiis/diismanager.h"
#include "psi4/libfock/jk.h"
#include "psi4/libmints/matrix.h"

#include "ambit/blocked_tensor.h"

#include "base_classes/active_space_method.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"

namespace forte {

class SCFInfo;

class CASSCF_GRAD {
  public:
    /**
     * @brief Constructor of the AO-based CASSCF class
     * @param state_weights_map: The state to weights map of Forte
     * @param options: The ForteOptions pointer
     * @param mo_space_info: The MOSpaceInfo pointer of Forte
     * @param scf_info: The SCF_INFO pointer of Forte
     * @param ints: The ForteIntegral pointer
     *
     * Implementation notes:
     *   See J. Chem. Phys. 142, 224103 (2015) and Theor. Chem. Acc. 97, 88-95 (1997)
     */
    CASSCF_GRAD(std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
                std::shared_ptr<ForteIntegrals> ints);

    /// Evaluate the energy and orbital gradient
    double evaluate(psi::SharedVector x, psi::SharedVector g);

    /// Evaluate the diagonal orbital Hessian
    void hess_diag(psi::SharedVector x, psi::SharedVector h0);

    /// Set RDMs used for orbital optimization
    void set_rdms(RDMs& rdms);

    /// Prepare active space integrals
    std::shared_ptr<ActiveSpaceIntegrals> active_space_ints();

    /// MO coefficients
    psi::SharedMatrix Ca() { return C_; }

    /// Orbital gradients
    double grad_norm() { return grad_->norm(); }

  private:
    /// The list of states to computed. Passed to the ActiveSpaceSolver
    std::map<StateInfo, std::vector<double>> state_weights_map_;

    /// The Forte options
    std::shared_ptr<ForteOptions> options_;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// SCF information
    std::shared_ptr<SCFInfo> scf_info_;

    /// The Forte integral
    std::shared_ptr<ForteIntegrals> ints_;

    /// Common setup for the class
    void startup();

    /// Read options
    void read_options();

    /// Print options
    void print_options();

    /// Prepare MO spaces
    void setup_mos();

    /// Number of non-redundant pairs for orbital optimization
    void nonredundant_pairs();

    /// Prepare JK object
    void setup_JK();

    /// Set up Ambit blocks
    void setup_ambit();

    /// Initialize/Allocate tensors and matrices
    void init_tensors();

    /// The JK object of Psi4
    std::shared_ptr<psi::JK> JK_;

    // => MO spaces related <=

    /// The number of irreps
    int nirrep_;

    /// The number of SO per irrep (AO for C matrices)
    psi::Dimension nsopi_;
    /// The number of MO per irrep
    psi::Dimension nmopi_;
    /// The number of non-frozen MO per irrep
    psi::Dimension ncmopi_;
    /// The number of DOCC (including frozen core) per irrep
    psi::Dimension ndoccpi_;
    /// The number of frozen DOCC per irrep
    psi::Dimension nfrzcpi_;
    /// The number of active per irrep
    psi::Dimension nactvpi_;

    /// The number of SOs
    size_t nso_;
    /// The number of MOs
    size_t nmo_;
    /// The number of non-frozen MOs
    size_t ncmo_;
    /// The number of active orbitals
    size_t nactv_;

    /// List of core MOs (Correlated)
    std::vector<size_t> core_mos_;
    /// List of active MOs (Correlated)
    std::vector<size_t> actv_mos_;
    /// List of virtual MOs (Correlated)
    std::vector<size_t> virt_mos_;
    /// Map from MO space label to the MO indices
    std::map<std::string, std::vector<size_t>> label_to_mos_;

    /// Relative indices within an irrep for correlated MOs
    std::vector<std::pair<size_t, size_t>> corr_mos_rel_;

    /// Relative indices within an MO space for correlated MOs
    std::vector<std::pair<std::string, size_t>> corr_mos_rel_space_;

    /// Number of orbital rotations considered
    size_t nrot_;
    /// List of rotation pairs in <irrep, index1, index2> format
    std::vector<std::tuple<int, size_t, size_t>> rot_mos_irrep_;
    /// List of rotation pairs in <block, index1, index2> format
    std::vector<std::tuple<std::string, size_t, size_t>> rot_mos_block_;

    /// The symmetry of every active orbitals
    std::vector<int> actv_sym_;

    // => Options <=

    /// Integral type
    std::string int_type_;

    /// The printing level
    int print_;
    /// Enable debug printing or not
    bool debug_print_;

    /// Orbital gradient convergence criteria
    double g_conv_;

    /// Keep internal (active-active) rotations
    bool internal_rot_;

    /// User specified zero rotations
    /// vector of irrep, map from index i to other indices uncoupled with index i
    std::vector<std::unordered_map<size_t, std::unordered_set<size_t>>> zero_rots_;

    /// Orbital type for redundant pairs
    std::string orb_type_redundant_;

    // => Tensors and matrices <=

    /// Initial orbital coefficients
    std::shared_ptr<psi::Matrix> C0_;
    /// Current orbital coefficients
    std::shared_ptr<psi::Matrix> C_;

    /// The bare AO one-electron integrals
    psi::SharedMatrix H_ao_;
    /// The bare MO one-electron intergals (nmo x nmo)
    psi::SharedMatrix H_mo_;

    /// The inactive Fock matrix in MO basis
    psi::SharedMatrix F_closed_; // nmo x nmo
    ambit::BlockedTensor Fc_;    // ncmo x ncmo
    /// The active Fock matrix in MO basis
    psi::SharedMatrix F_active_; // nmo x nmo
    /// The generalized Fock matrix in MO basis
    psi::SharedMatrix Fock_; // nmo x nmo
    ambit::BlockedTensor F_; // ncmo x ncmo
    /// Diagonal elements of the generalized Fock matrix (Pitzer ordering)
    std::vector<double> Fd_;

    /// Two-electron integrals in chemists' notation (pq|rs)
    ambit::BlockedTensor V_;

    /// Spin-summed 1-RDM
    ambit::BlockedTensor D1_;
    psi::SharedMatrix rdm1_;
    /// Spin-summed averaged 2-RDM in 1^+ 1 2^+ 2 ordering
    ambit::BlockedTensor D2_;

    /// The orbital Lagrangian matrix
    ambit::BlockedTensor A_;

    /// The orthogonal transformation matrix U = exp(R)
    psi::SharedMatrix U_;

    /// The orbital gradients
    ambit::BlockedTensor g_;
    psi::SharedVector grad_;
    /// The orbital diagonal Hessian
    ambit::BlockedTensor h_diag_;
    psi::SharedVector hess_diag_;

    /// G intermediates when forming internal diagonal Hessian
    ambit::BlockedTensor Guu_;
    ambit::BlockedTensor Guv_;
    /// Intermediate (TEI) when forming internal diagonal Hessian
    ambit::BlockedTensor jk_internal_;
    /// Intermediate (2RDM) when forming internal diagonal Hessian
    ambit::BlockedTensor d2_internal_;

    // => functions used in every iteration <=

    /// Build integrals for gradients and Hessian
    void build_mo_integrals();

    /// Build MO one-electron integrals
    void build_oei_from_ao();

    /// Build two-electron integrals
    void build_tei_from_ao();

    /// Build Fock matrix
    void build_fock(bool rebuild_inactive = false);
    /// Build the inactive Fock (does not depend on 1RDM), includes frozen docc
    void build_fock_inactive();
    /// Build the active Fock (does depend on 1RDM)
    void build_fock_active();

    /// Compute closed-shell energy (frozen_docc + restricted_docc)
    void compute_energy_closed();
    /// The closed-shell energy
    double e_closed_;

    /// Compute the energy for given sets of orbitals and density
    void compute_reference_energy();
    /// The energy computed using the current orbitals and CI coefficients
    double energy_;

    /// Compute the orbital gradients
    void compute_orbital_grad();
    /// Compute the diagonal Hessian for orbital rotations
    void compute_orbital_hess_diag();

    // => Some helper functions <=

    /// Format the Fock matrix from SharedMatrix to BlockedTensor
    void format_fock(psi::SharedMatrix Fock, ambit::BlockedTensor F);

    /// Format the 1RDM from BlockedTensor to SharedMatrix
    void format_1rdm();

    /// Reshape the orbital rotation related BlockedTensor to SharedVector
    void reshape_rot_ambit(ambit::BlockedTensor bt, psi::SharedVector sv);

    /// Fix redundant orbitals and return the rotation matrix
    std::shared_ptr<psi::Matrix> canonicalize();
};
} // namespace forte

#endif // _casscf_orb_
