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

#ifndef _orbital_optimizer_h_
#define _orbital_optimizer_h_

#include "ambit/blocked_tensor.h"
#include "base_classes/rdms.h"
//#include "base_classes/mo_space_info.h"
#include "helpers/blockedtensorfactory.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libfock/jk.h"

using namespace psi;

namespace forte {
class MOSpaceInfo;
class ForteOptions;
class ForteIntegrals;

/**
* @brief OrbitalOptimizer does an orbital optimization given an 1RDM, 2RDM, and the integrals
* Forms an orbital gradient:  g_{pq} = [ h_{pq} gamma_{pq} + Gamma_{pq}^{rs} <pq||rs> - A(p->q)]
* Here only form a diagonal hessian of orbitals: Look at Hohenstein J. Chem. Phys. 142, 224103.
* Diagonal Hessian only requires (pu|xy) integrals (many are built in JK library)
* Daniel Smith's CASSCF code in PSI4 was integral in debugging.
*
* Usage of this class:  Contructor will just set_up the basic values
* I learn best from examples:  Here is how I use it in the CASSCF code.
* Note:  If you are not freezing core, you do not need F_froze
*         OrbitalOptimizer orbital_optimizer(gamma1_,
                                           gamma2_,
                                           ints_->aptei_ab_block(nmo_abs_,
active_abs_, active_abs_, active_abs_) ,
                                           options_,
                                           mo_space_info_);
        orbital_optimizer.set_one_body(OneBody)
        orbital_optimizer.set_frozen_one_body(F_froze_);
        orbital_optimizer.set_no_symmetry_mo(Call_);
        orbital_optimizer.set_symmmetry_mo(Ca);

        orbital_optimizer.update()
        S = orbital_optimizer.approx_solve()
        C_new = orbital_optimizer.rotate(Ca, S) -> Right now, if this is an
iterative procedure, you should
        use the Ca that was previously updated, ie (C_new  =
Cold(exp(S_previous)) * exp(S))
*/
class OrbitalOptimizer {
  public:
    OrbitalOptimizer();

    /**
     * @brief Given 1RDM, 2RDM, (pu|xy) integrals, and space information, do an
     * orbital optimization
     * OrbitalOptimizer returns a orbital rotation parameter that allows you to
     * update your orbitals
     * @param Gamma1 The SYMMETRIZED 1-RDM:  gamma1_a + gamma1_b
     * @param Gamma2 The SYMMETRIZED 2-RDM:  Look at code ( Gamma = rdm_2aa +
     * rdm_2ab) with prefactors
     * @param two_body_ab (pu|xy) integrals (NOTE:  This is only valid if you are
     * doing an orbital optimization at the level of CASSCF
     * @param options The options object
     * @param mo_space_info MOSpace object for handling active/rdocc/ruocc
     */
    OrbitalOptimizer(ambit::Tensor Gamma1, ambit::Tensor Gamma2, ambit::Tensor two_body_ab,
                     std::shared_ptr<ForteOptions> options,
                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                     std::shared_ptr<ForteIntegrals> ints);

    /// You have to set these at the start of the computation
    /// The MO Coefficient you get from wfn_->Ca()
    void set_symmmetry_mo(psi::SharedMatrix C) { Ca_sym_ = C; }
    /// The MO Coefficient in pitzer ordering (symmetry-aware)
    /// The workhouse of the program:  Computes gradient, hessian.
    void update();
    /// Solution of g + Hx = 0 (with diagonal H), so x = - g / H
    psi::SharedMatrix approx_solve();
    /// Exponentiate the orbital rotation parameter and use this to update your
    /// MO coefficients
    psi::SharedMatrix rotate_orbitals(psi::SharedMatrix C, psi::SharedMatrix S);
    /// The norm of the orbital gradient
    double orbital_gradient_norm() { return (g_->rms()); }
    /// Must compute the frozen_one_body fock matrix
    void set_frozen_one_body(psi::SharedMatrix F_froze) { F_froze_ = F_froze; }
    /// Give the AO one electron integrals (H = T + V)
    void one_body(psi::SharedMatrix H) { H_ = H; }
    /// Print a summary of timings
    void set_print_timings(bool timing) { timings_ = timing; }
    /// Set the JK object
    void set_jk(std::shared_ptr<JK>& JK) { JK_ = JK; }

  protected:
    /// SCF information
    std::shared_ptr<SCFInfo> scf_info_;

    /// The 1-RDM (usually of size na_^2)
    ambit::Tensor gamma1_;
    /// The 1-RDM psi::SharedMatrix
    psi::SharedMatrix gamma1M_;
    /// The 2-RDM (usually of size na^4)
    ambit::Tensor gamma2_;
    /// The 2-RDM psi::SharedMatrix
    psi::SharedMatrix gamma2M_;
    ambit::Tensor integral_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    std::shared_ptr<JK> JK_;

    std::shared_ptr<ForteOptions> options_;
    /// The mo_space_info
    /// The MO Coefficient matrix in Pfitzer ordering in whatever symmetry
    /// this matrix is ao by nmo
    psi::SharedMatrix Call_;
    /// C matrix in the SO basis
    psi::SharedMatrix Ca_sym_;

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

    /// These member variables are all summarized in Algorithm 1
    /// Equation 9

    /// H = T + V
    psi::SharedMatrix H_;
    /// The Fock matrix due to Frozen core orbitals
    psi::SharedMatrix F_froze_;
    /// The core Fock Matrix
    psi::SharedMatrix F_core_;
    /// The F_act_ -> ie the fock matrix of nmo by nmo generated using the all
    /// active portion of the OPM
    /// Equation 10
    psi::SharedMatrix F_act_;
    /// The JK object
    psi::SharedMatrix JK_fock_;
    /// Intermediate in forming orbital gradient matrix
    psi::SharedMatrix Y_;
    /// Z intermediate
    psi::SharedMatrix Z_;
    /// The Orbital Gradient
    psi::SharedMatrix g_;
    /// The Orbital Hessian
    psi::SharedMatrix d_;
    /// Solution of g + HS = 0
    psi::SharedMatrix S_;

    /// private functions

    /// This function will implement steps 4 and 9 of algorithm
    virtual void form_fock_intermediates() = 0;
    /// Assemble the orbital gradient (10-15)
    void orbital_gradient();
    /// Assemble the diagonal Hessian (20-22)
    void diagonal_hessian();
    /// check the cas_ci energy with spin-free RDM
    void orbital_rotation_parameter();
    /// Perform the exponential of x
    psi::SharedMatrix matrix_exp(const psi::SharedMatrix&);
    /// form SharedMatrices of Gamma1 and Gamma2 (Tensor library not great for
    /// non contractions)
    void fill_shared_density_matrices();
    /// Diagonalize an augmented Hessian and take lowest eigenvector as solution
    psi::SharedMatrix AugmentedHessianSolve();

    //    psi::SharedMatrix make_c_sym_aware(psi::SharedMatrix aotoso);

    void startup();

    /// DEBUG PRINTING
    bool casscf_debug_print_;
    /// Freeze the core and leave them unchanged
    /// Uses this to override MOSPACEINFO
    bool casscf_freeze_core_;
    /// Print timings
    bool timings_ = false;

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

    /// Allows for easy acess of unitary parameter
    std::map<size_t, size_t> nhole_map_;
    std::map<size_t, size_t> npart_map_;
    bool cas_;
    bool gas_;
    /// Number of GAS
    size_t gas_num_;
    enum MATRIX_EXP { PSI4, TAYLOR };
    void zero_redunant(psi::SharedMatrix& matrix);

    /// Information about GAS
    std::pair<int, std::map<std::string, SpaceInfo>> gas_info_;

    std::vector<std::vector<size_t>> relative_gas_mo_;
};

/// A Class for use in CASSCFOrbitalOptimizer (computes FockCore and FockActive
/// using JK builders)
class CASSCFOrbitalOptimizer : public OrbitalOptimizer {
  public:
    CASSCFOrbitalOptimizer(ambit::Tensor Gamma1, ambit::Tensor Gamma2, ambit::Tensor two_body_ab,
                           std::shared_ptr<ForteOptions> options,
                           std::shared_ptr<MOSpaceInfo> mo_space_info,
                           std::shared_ptr<ForteIntegrals> ints);
    virtual ~CASSCFOrbitalOptimizer();

  private:
    virtual void form_fock_intermediates();
};

/// A Class for OrbitalOptimization in PostCASSCF methods through use of
/// renormalized integrals
/// Pass (pq | ij) and (pj | iq) type integrals
class PostCASSCFOrbitalOptimizer : public OrbitalOptimizer {
  public:
    PostCASSCFOrbitalOptimizer(ambit::Tensor Gamma1, ambit::Tensor Gamma2,
                               ambit::Tensor two_body_ab, std::shared_ptr<ForteOptions> options,
                               std::shared_ptr<MOSpaceInfo> mo_space_info,
                               std::shared_ptr<ForteIntegrals> ints);
    virtual ~PostCASSCFOrbitalOptimizer();

    void set_fock_integrals_pq_mm(const ambit::Tensor& pq_mm) { pq_mm_ = pq_mm; }
    void set_fock_integrals_pm_qm(const ambit::Tensor& pm_qm) { pm_qm_ = pm_qm; }
    void set_fock_integrals_pq_uv(const ambit::Tensor& pq_uv) { pq_uv_ = pq_uv; }
    void set_fock_integrals_pu_qv(const ambit::Tensor& pu_qv) { pu_qv_ = pu_qv; }

  private:
    ambit::Tensor pq_mm_;
    ambit::Tensor pm_qm_;
    ambit::Tensor pq_uv_;
    ambit::Tensor pu_qv_;
    virtual void form_fock_intermediates();
};
} // namespace forte

#endif // _orbital_optimizer_h_
