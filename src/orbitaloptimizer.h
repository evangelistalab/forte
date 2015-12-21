#ifndef ORBITALOPTIMIZER_H
#define ORBITALOPTIMIZER_H
#include "ambit/blocked_tensor.h"
#include "reference.h"
#include "helpers.h"
#include "blockedtensorfactory.h"
#include <libmints/matrix.h>

using namespace psi;

namespace psi{ namespace forte{

/**
* @brief OrbitalOptimizer does an orbital optimization given an RDM-1, RDM-2, and the integrals
* Forms an orbital gradient:  g_{pq} =[ h_{pq} gamma_{pq} + Gamma_{pq}^{rs} <pq || rs> - A(p->q)]
* Right now, Only forms a diagonal hessian of orbitals:  Look at  Hohenstein J.Chem.Phys, 142, 224103.
* Diagonal Hessian only requires (p u | x y) integrals (many are built in JK library)
* Daniel Smith's CASSCF code in PSI4 was integral in debugging.
*
* Usage of this class:  Contructor will just set_up the basic values
* I learn best from examples:  Here is how I use it in the CASSCF code.
* Note:  If you are not freezing core, you do not need F_froze
*         OrbitalOptimizer orbital_optimizer(gamma1_,
                                           gamma2_,
                                           ints_->aptei_ab_block(nmo_abs_, active_abs_, active_abs_, active_abs_) ,
                                           options_,
                                           mo_space_info_);
        orbital_optimizer.set_one_body(OneBody)
        orbital_optimizer.set_frozen_one_body(F_froze_);
        orbital_optimizer.set_no_symmetry_mo(Call_);
        orbital_optimizer.set_symmmetry_mo(Ca);
        orbital_optimizer.update()
        S = orbital_optimizer.approx_solve()
        C_new = orbital_optimizer.rotate(Ca, S)

*/
class OrbitalOptimizer
{
public:
    OrbitalOptimizer();

    /**
     * @brief Given 1RDM, 2RDM, (pu|xy) integrals, and space information, do an orbital optimization
     * OrbitalOptimizer returns a orbital rotation parameter that allows you to update your orbitals
     * @param Gamma1 The SYMMETRIZED 1-RDM:  gamma1_a + gamma2_b
     * @param Gamma2 The SYMMETRIZTED 2-RDM: 1/4 * (gamma2_aa + gamma2_ab + gamma_2_ba + gamma2_bb)
     * @param two_body_ab (pu|xy) integrals(NOTE:  This is only valid if you are doing an orbital optimization at the level of CASSCF
     * @param options The options object
     * @param mo_space_info MOSpace object for handling active/rdocc/ruocc
     */

    OrbitalOptimizer( ambit::Tensor Gamma1,
                      ambit::Tensor Gamma2,
                      ambit::Tensor two_body_ab,
                      Options& options,
                      std::shared_ptr<MOSpaceInfo> mo_space_info);

    ///You have to set these at the start of the computation
    /// The MO Coefficient you get from wfn_->Ca()
    void set_symmmetry_mo(SharedMatrix C)  {Ca_sym_ = C;}
    /// The MO Coefficient in pitzer ordering (symmetry-aware)
    void set_no_symmetry_mo(SharedMatrix C){Call_ = C;}
    /// The workhouse of the program:  Computes gradient, hessian.
    void update();
    /// Solution of g + Hx = 0 (with diagonal H), so x = - g / H
    SharedMatrix approx_solve();
    /// Exponentiate the orbital rotation parameter and use this to update your MOCoefficient
    SharedMatrix rotate_orbitals(SharedMatrix C, SharedMatrix S);
    /// The norm of the orbital gradient
    double orbital_gradient_norm(){return (g_->rms());}
    /// Must compute the frozen_one_body fock matrix
    void set_frozen_one_body(SharedMatrix F_froze){F_froze_ = F_froze;}
    /// Give the AO one electron integrals (H = T + V)
    void one_body(SharedMatrix H){H_ = H;}
protected:
    ///The 1-RDM (usually of size na_^2)
    ambit::Tensor gamma1_;
    ///The 1-RDM SharedMatrix
    SharedMatrix gamma1M_;
    ///The 2-RDM (usually of size na^4)
    ambit::Tensor gamma2_;
    ///The 2-RDM SharedMatrix
    SharedMatrix gamma2M_;
    ambit::Tensor integral_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;


    Options options_;
    /// The ForteIntegrals pointer
    boost::shared_ptr<Wavefunction> wfn_;
    /// The mo_space_info
    /// The MO Coefficient matrix in Pfitzer ordering in whatever symmetry
    /// this matrix is ao by nmo
    SharedMatrix Call_;
    /// C matrix in the SO basis
    SharedMatrix Ca_sym_;

    /// The dimension for number of molecular orbitals (CORRELATED or ALL)
    Dimension nmopi_;
    /// The number of correlated molecular orbitals (Restricted Core + Active + Restricted_UOCC + Frozen_Virt
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

    /// H = T + V
    SharedMatrix H_;
    /// The Fock matrix due to Frozen core orbitals
    SharedMatrix F_froze_;
    /// The core Fock Matrix
    SharedMatrix F_core_;
    /// The F_act_ -> ie the fock matrix of nmo by nmo generated using the all active portion of the OPM
    /// Equation 10
    SharedMatrix F_act_;
    /// Intermediate in forming orbital gradient matrix
    SharedMatrix Y_;
    /// Z intermediate
    SharedMatrix Z_;
    ///The Orbital Gradient
    SharedMatrix g_;
    ///The Orbital Hessian
    SharedMatrix d_;
    ///Solution of g + HS = 0
    SharedMatrix S_;

    /// private functions

    /// This function will implement steps 4 and 9 of algorithm
    void form_fock_core();
    ///Implement step 9 of algoritm
    void form_fock_active();
    /// Assemble the orbital gradient (10-15)
    void orbital_gradient();
    /// Assemble the diagonal Hessian (20-22)
    void diagonal_hessian();
    /// check the cas_ci energy with spin-free RDM
    void orbital_rotation_parameter();

    ///form SharedMatrices of Gamma1 and Gamma2 (Tensor library not great for non contractions)
    void fill_shared_density_matrices();

    SharedMatrix make_c_sym_aware();

    void startup();

    /// DEBUG PRINTING
    bool casscf_debug_print_;
    /// Freeze the core and leave them unchanged
    /// Uses this to override MOSPACEINFO
    bool casscf_freeze_core_;

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


};

}}
#endif // ORBITALOPTIMIZER_H
