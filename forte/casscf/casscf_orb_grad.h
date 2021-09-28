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

#ifndef _casscf_orb_grad_h_
#define _casscf_orb_grad_h_

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

class CASSCF_ORB_GRAD {
  public:
    /**
     * @brief Constructor of the AO-based CASSCF class
     * @param options: The ForteOptions pointer
     * @param mo_space_info: The MOSpaceInfo pointer of Forte
     * @param ints: The ForteIntegral pointer
     *
     * Implementation notes:
     *   See J. Chem. Phys. 142, 224103 (2015) and Theor. Chem. Acc. 97, 88-95 (1997)
     */
    CASSCF_ORB_GRAD(std::shared_ptr<ForteOptions> options,
                    std::shared_ptr<MOSpaceInfo> mo_space_info,
                    std::shared_ptr<ForteIntegrals> ints);

    /// Evaluate the energy and orbital gradient
    double evaluate(psi::SharedVector x, psi::SharedVector g, bool do_g = true);

    /// Evaluate the diagonal orbital Hessian
    void hess_diag(psi::SharedVector x, psi::SharedVector h0);

    /// Set RDMs used for orbital optimization
    void set_rdms(RDMs& rdms);

    /// Return active space integrals for CI
    std::shared_ptr<ActiveSpaceIntegrals> active_space_ints();

    /// Return the number of nonredundant orbital rotations
    size_t nrot() { return nrot_; }

    /// Return the initial (not optimized) MO coefficients
    psi::SharedMatrix Ca_initial() { return C0_; }

    /// Return the optimized MO coefficients
    psi::SharedMatrix Ca() { return C_; }

    /// Return the generalized Fock matrix
    psi::SharedMatrix fock() { return Fock_; }

    /// Canonicalize the final orbitals
    void canonicalize_final(psi::SharedMatrix U);

    /// Compute nuclear gradient
    void compute_nuclear_gradient();

  private:
    /// The Forte options
    std::shared_ptr<ForteOptions> options_;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The Forte integral
    std::shared_ptr<ForteIntegrals> ints_;

    /// Common setup for the class
    void startup();

    /// Read options
    void read_options();

    /// Prepare MO spaces
    void setup_mos();

    /// Number of non-redundant pairs for orbital optimization
    void nonredundant_pairs();

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
    /// The number of frozen UOCC per irrep
    psi::Dimension nfrzvpi_;
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
    /// The number of frozen-core orbitals
    size_t nfrzc_;

    /// List of core MOs (Absolute)
    std::vector<size_t> core_mos_;
    /// List of active MOs (Absolute)
    std::vector<size_t> actv_mos_;
    /// Map from MO space label to the absolute MO indices
    std::map<std::string, std::vector<size_t>> label_to_mos_;
    /// Map from MO space label to the correlated MO indices
    std::map<std::string, std::vector<size_t>> label_to_cmos_;

    /// Relative indices within an irrep <irrep, relative indices>
    std::vector<std::pair<int, size_t>> mos_rel_;

    /// Relative indices within an MO space <space, relative indices>
    std::vector<std::pair<std::string, size_t>> mos_rel_space_;

    /// Number of orbital rotations considered
    size_t nrot_;
    /// List of rotation pairs in <irrep, index1, index2> format
    std::vector<std::tuple<int, size_t, size_t>> rot_mos_irrep_;
    /// List of rotation pairs in <block, index1, index2> format
    std::vector<std::tuple<std::string, size_t, size_t>> rot_mos_block_;

    // => Options <=

    /// The printing level
    int print_;
    /// Enable debug printing or not
    bool debug_print_;

    /// Orbital gradient convergence criteria
    double g_conv_;

    /// Keep internal (GASn-GASn) rotations
    bool internal_rot_;
    /// If the active space is from GAS
    bool gas_ref_;

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

    /// The inactive Fock matrix in MO basis
    psi::SharedMatrix F_closed_; // nmo x nmo
    ambit::BlockedTensor Fc_;    // ncmo x ncmo
    /// The generalized Fock matrix in MO basis
    psi::SharedMatrix Fock_; // nmo x nmo
    ambit::BlockedTensor F_; // ncmo x ncmo
    /// Diagonal elements of the generalized Fock matrix (Pitzer ordering)
    std::vector<double> Fd_;

    /// Two-electron integrals in chemists' notation (pu|xy)
    ambit::BlockedTensor V_;

    /// Spin-summed 1-RDM
    ambit::BlockedTensor D1_;
    psi::SharedMatrix rdm1_;
    /// Spin-summed averaged 2-RDM in 1^+ 1 2^+ 2 ordering
    ambit::BlockedTensor D2_;

    /// The orbital response of MCSCF energy
    ambit::BlockedTensor A_;
    psi::SharedMatrix Am_;

    /// The orbital rotation matrix
    psi::SharedMatrix R_;
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

    // => functions used in micro iteration <=

    /// Build integrals for gradients and Hessian
    void build_mo_integrals();

    /// Build two-electron integrals
    void build_tei_from_ao();

    /// Fill two-electron integrals for custom integrals
    void fill_tei_custom(ambit::BlockedTensor V);

    /// JK build for Fock-like terms
    void JK_build(psi::SharedMatrix Cl, psi::SharedMatrix Cr);

    /// Build Fock matrix
    void build_fock(bool rebuild_inactive = false);
    /// Build the inactive Fock (does not depend on 1RDM), includes frozen docc
    void build_fock_inactive();
    /// Build the active Fock (does depend on 1RDM)
    void build_fock_active();

    /// Compute the energy for given sets of orbitals and density
    void compute_reference_energy();
    /// The energy computed using the current orbitals and CI coefficients
    double energy_;
    /// The closed-shell energy
    double e_closed_;

    /// Compute the orbital gradients
    void compute_orbital_grad();
    /// Compute the diagonal Hessian for orbital rotations
    void compute_orbital_hess_diag();

    /// Update orbitals using the given rotation matrix in vector form
    bool update_orbitals(psi::SharedVector x);

    // => Nuclear gradient related functions <=

    /// compute AO Lagrangian matrix and push to Psi4
    void compute_Lagrangian();

    /// compute AO 1-RDM and push to Psi4
    void compute_opdm_ao();

    /// Dump the MCSCF MO 2-RDM to file using IWL
    void dump_tpdm_iwl();
    /// Dump the Hartree-Fock MO 2-RDM to file using IWL
    void dump_tpdm_iwl_hf();

    /// Are there any frozen orbitals?
    bool is_frozen_orbs_;

    /// Start up for doing gradient with frozen orbitals
    void setup_grad_frozen();

    /// Doubly occupied MOs from Hartree-Fock
    psi::Dimension hf_ndoccpi_;
    /// Unoccupied MOs from Hartree-Fock
    psi::Dimension hf_nuoccpi_;
    /// List of occupied MOs from Hartree-Fock
    std::vector<size_t> hf_docc_mos_;
    /// List of unoccupied MOs from Hartree-Fock
    std::vector<size_t> hf_uocc_mos_;

    /// Hartree-Fock orbital energies
    psi::SharedVector epsilon_;

    /// Compute the frozen part of the A matrix
    void build_Am_frozen();

    /// Z vector for CPSCF equations
    psi::SharedMatrix Z_;
    /// Solve Z vector equation if there are frozen orbitals
    void solve_cpscf();

    /**
     * Contract Roothaan-Bagus supermatrix with Z: sum_{pq} Z_{pq} L_{pq,rs}
     * Roothaan-Bagus supermatrix L_{pq,rs} = 4 * (pq|rs) - (pr|sq) - (ps|rq)
     *
     * Express contraction in AO basis:
     * sum_{pq} sum_{PQRS} CZrow_{Pp} Z_{pq} CZcol_{Qq} L_{PQ,RS} Crow_{Rr} Ccol_{Ss}
     */
    psi::SharedMatrix contract_RB_Z(psi::SharedMatrix Z, psi::SharedMatrix C_Zrow,
                                    psi::SharedMatrix C_Zcol, psi::SharedMatrix C_row,
                                    psi::SharedMatrix C_col);

    // => Some helper functions <=

    /// Format the Fock matrix from SharedMatrix to BlockedTensor
    void format_fock(psi::SharedMatrix Fock, ambit::BlockedTensor F);

    /// Format the 1RDM from BlockedTensor to SharedMatrix
    void format_1rdm();

    /// Fill Am_ matrix from BlockedTensor A
    void fill_A_matrix_data(ambit::BlockedTensor A);

    /// Reshape the orbital rotation related BlockedTensor to SharedVector
    void reshape_rot_ambit(ambit::BlockedTensor bt, psi::SharedVector sv);

    /// Fix redundant orbitals and return the rotation matrix
    std::shared_ptr<psi::Matrix> canonicalize();

    /// Compute the exponential of a skew-symmetric matrix
    psi::SharedMatrix matrix_exponential(psi::SharedMatrix A, int n);

    /// Grab part of the orbital coefficients
    psi::SharedMatrix C_subset(const std::string& name, psi::SharedMatrix C,
                               psi::Dimension dim_start, psi::Dimension dim_end);
};
} // namespace forte

#endif // _casscf_orb_grad_h_
