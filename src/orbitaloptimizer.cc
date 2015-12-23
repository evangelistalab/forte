#include "orbitaloptimizer.h"
#include "helpers.h"
#include "ambit/blocked_tensor.h"
#include <libfock/jk.h>
#include "reference.h"
#include "integrals.h"
#include <libqt/qt.h>
#include <libmints/matrix.h>
#include "helpers.h"
#include "fci_solver.h"
#include <psifiles.h>
#include <lib3index/cholesky.h>
using namespace psi;

namespace psi{ namespace forte{

OrbitalOptimizer::OrbitalOptimizer()
{
}

OrbitalOptimizer::OrbitalOptimizer(ambit::Tensor Gamma1,
                                   ambit::Tensor Gamma2,
                                   ambit::Tensor two_body_ab,
                                   Options& options,
                                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : gamma1_(Gamma1), gamma2_(Gamma2), integral_(two_body_ab), mo_space_info_(mo_space_info), options_(options)
{
    startup();

}
void OrbitalOptimizer::update()
{
    /// F^{I}_{pq} = h_{pq} + 2 (pq | kk) - (pk |qk)
    /// This is done using JK builder: F^{I}_{pq} = h_{pq} + C^{T}[2J - K]C
    /// Built in the AO basis for efficiency
    form_fock_core();

    /// F^{A}_{pq} = \gamma_{uv} (pq | uv) - 1/2 (pu |qv)
    /// First a Cholesky Decomposition is performed on Gamma (maybe isn't necessary)
    /// Use JK builder: J(\gamma_{uv} - 1/2 K(\gamma_{uv})
    form_fock_active();

    orbital_gradient();

    diagonal_hessian();

}

void OrbitalOptimizer::startup()
{
    frozen_docc_dim_     = mo_space_info_->get_dimension("FROZEN_DOCC");
    restricted_docc_dim_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    active_dim_          = mo_space_info_->get_dimension("ACTIVE");
    restricted_uocc_dim_ = mo_space_info_->get_dimension("RESTRICTED_UOCC");
    inactive_docc_dim_   = mo_space_info_->get_dimension("INACTIVE_DOCC");
    nmopi_ = mo_space_info_->get_dimension("CORRELATED");

    frozen_docc_abs_     = mo_space_info_->get_corr_abs_mo("FROZEN_DOCC");
    restricted_docc_abs_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    active_abs_          = mo_space_info_->get_corr_abs_mo("ACTIVE");
    restricted_uocc_abs_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    inactive_docc_abs_   = mo_space_info_->get_corr_abs_mo("INACTIVE_DOCC");
    nmo_abs_             = mo_space_info_->get_corr_abs_mo("CORRELATED");

    casscf_freeze_core_  = options_.get_bool("CASSCF_FREEZE_CORE");

    nmo_ = mo_space_info_->size("CORRELATED");
    all_nmo_ = mo_space_info_->size("ALL");

    nrdocc_ = restricted_docc_abs_.size();
    na_     = active_abs_.size();
    nvir_  = restricted_uocc_abs_.size();
    wfn_ = Process::environment.wavefunction();
    casscf_debug_print_ = options_.get_bool("CASSCF_DEBUG_PRINTING");
    nirrep_ = wfn_->nirrep();
    nsopi_  = wfn_->nsopi();

}

void OrbitalOptimizer::form_fock_core()
{
    /// Get the CoreHamiltonian in AO basis

    if(Ca_sym_ == nullptr)
    {
        outfile->Printf("\n\n Please give your OrbitalOptimize an Orbital");
        throw PSIEXCEPTION("Please set CMatrix before you call orbital rotation casscf");
    }
    if(H_ == nullptr)
    {
        outfile->Printf("\n\n Please set the OneBody operator");
        throw PSIEXCEPTION("Please set H before you call orbital rotation casscf");
    }
    H_->transform(Ca_sym_);
    if(casscf_debug_print_){
        H_->set_name("CORR_HAMIL");
        H_->print();
    }

    ///Step 2: From Hamiltonian elements
    ///This will use JK builds (Equation 18 - 22)
    /// F_{pq}^{core} = C_{mu p}C_{nu q} [h_{uv} + 2J^{(D_c) - K^{(D_c)}]
    ///Have to go from the full C matrix to the C_core in the SO basis
    /// tricky...tricky
    //SharedMatrix C_core(new Matrix("C_core", nmo_, inactive_dim_abs.size()));

    // Need to get the inactive block of the C matrix
    SharedMatrix C_core(new Matrix("C_core", nirrep_, nsopi_, restricted_docc_dim_));
    SharedMatrix F_core_c1(new Matrix("F_core_no_sym", nmo_, nmo_));
    F_core_c1->zero();

    ///If there is no restricted_docc, there is no C_core
    if(restricted_docc_dim_.sum() > 0)
    {
        for(size_t h = 0; h < nirrep_; h++){
            for(int mu = 0; mu < nsopi_[h]; mu++){
                for(int i = 0; i <  restricted_docc_dim_[h]; i++){
                    C_core->set(h, mu, i, Ca_sym_->get(h, mu, i + frozen_docc_dim_[h]));
                }
            }
        }
        if(casscf_debug_print_){
            C_core->print();
        }

        boost::shared_ptr<JK> JK_core = JK::build_JK();

        JK_core->set_memory(Process::environment.get_memory() * 0.8);
        /// Already transform everything to C1 so make sure JK does not do this.

        /////TODO: Make this an option in my code
        //JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
        JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
        JK_core->initialize();

        std::vector<boost::shared_ptr<Matrix> >&Cl = JK_core->C_left();

        Cl.clear();
        Cl.push_back(C_core);

        JK_core->compute();

        SharedMatrix J_core = JK_core->J()[0];
        SharedMatrix K_core = JK_core->K()[0];

        J_core->scale(2.0);
        SharedMatrix F_core = J_core->clone();
        F_core->subtract(K_core);

        /// If there are frozen orbitals, need to add
        /// FrozenCore Fock matrix to inactive block
        if(casscf_freeze_core_)
        {
            F_core->add(F_froze_);
        }
        F_core->transform(Ca_sym_);
        F_core->add(H_);

        int offset = 0;
        for(size_t h = 0; h < nirrep_; h++){
            for(int p = 0; p < nmopi_[h]; p++){
                for(int q = 0; q < nmopi_[h]; q++){
                    F_core_c1->set(p + offset, q + offset, F_core->get(h, p + frozen_docc_dim_[h], q + frozen_docc_dim_[h]));
                }
            }
            offset += nmopi_[h];
        }
    }
    F_core_   = F_core_c1;
    if(casscf_debug_print_)
    {
        F_core_->set_name("INACTIVE_FOCK");
        F_core_->print();
    }

}
void OrbitalOptimizer::form_fock_active()
{
    Call_ = make_c_sym_aware();
    ///Step 3:
    ///Compute equation 10:
    /// The active OPM is defined by gamma = gamma_{alpha} + gamma_{beta}

    size_t nso = wfn_->nso();
    SharedMatrix C_active(new Matrix("C_active", nso,na_));
    auto active_abs_corr = mo_space_info_->get_absolute_mo("ACTIVE");

    for(size_t mu = 0; mu < nso; mu++){
        for(size_t u = 0; u <  na_; u++){
            C_active->set(mu, u, Call_->get(mu, active_abs_corr[u]));
        }
    }

    ambit::Tensor Cact = ambit::Tensor::build(ambit::CoreTensor, "Cact", {nso, na_});
    ambit::Tensor OPDM_aoT = ambit::Tensor::build(ambit::CoreTensor, "OPDM_aoT", {nso, nso});

    Cact.iterate([&](const std::vector<size_t>& i,double& value){
        value = C_active->get(i[0], i[1]);});

    ///Transfrom the all active OPDM to the AO basis
    OPDM_aoT("mu,nu") = gamma1_("u,v")*Cact("mu, u") * Cact("nu, v");
    SharedMatrix OPDM_ao(new Matrix("OPDM_AO", nso, nso));

    OPDM_aoT.iterate([&](const std::vector<size_t>& i,double& value){
        OPDM_ao->set(i[0], i[1], value);});

    ///In order to use JK builder for active part, need a Cmatrix like matrix
    /// AO OPDM looks to be semi positive definite so perform CholeskyDecomp and feed this to JK builder
    boost::shared_ptr<CholeskyMatrix> Ch (new CholeskyMatrix(OPDM_ao, 1e-12, Process::environment.get_memory()));
    Ch->choleskify();
    SharedMatrix L_C = Ch->L();
    SharedMatrix L_C_correct(new Matrix("L_C_order", nso, Ch->Q()));

    for(size_t mu = 0; mu < nso; mu++){
        for(size_t Q = 0; Q < Ch->Q(); Q++){
            L_C_correct->set(mu, Q, L_C->get(Q, mu));
        }
    }

    boost::shared_ptr<JK> JK_act = JK::build_JK();

    JK_act->set_memory(Process::environment.get_memory() * 0.8);

    /////TODO: Make this an option in my code
    JK_act->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
    JK_act->initialize();

    std::vector<boost::shared_ptr<Matrix> >&Cl = JK_act->C_left();

    Cl.clear();
    Cl.push_back(L_C_correct);

    JK_act->set_allow_desymmetrization(false);
    JK_act->compute();

    SharedMatrix J_core = JK_act->J()[0];
    SharedMatrix K_core = JK_act->K()[0];

    SharedMatrix F_act = J_core->clone();
    K_core->scale(0.5);
    F_act->subtract(K_core);
    F_act->transform(Call_);
    F_act->set_name("FOCK_ACTIVE");

    SharedMatrix F_act_no_frozen(new Matrix("F_act", nmo_, nmo_));
    int offset_nofroze = 0;
    int offset_froze   = 0;
    Dimension no_frozen_dim = mo_space_info_->get_dimension("ALL");

    for(size_t h = 0; h < nirrep_; h++){
        int froze = frozen_docc_dim_[h];
        for(int p = froze; p < no_frozen_dim[h]; p++){
            for(int q = froze; q < no_frozen_dim[h]; q++){
                F_act_no_frozen->set(p  - froze + offset_froze, q - froze + offset_froze, F_act->get(p + offset_nofroze, q + offset_nofroze));
            }
        }
        offset_froze   += nmopi_[h];
        offset_nofroze += no_frozen_dim[h];
    }
    if(casscf_debug_print_){
        F_act_no_frozen->print();
    }

    if(nfrozen_ == 0)
    {
        F_act_ = F_act;
    }
    else{
        F_act_ = F_act_no_frozen;
    }

}

void OrbitalOptimizer::orbital_gradient()
{
    //std::vector<size_t> nmo_array = mo_space_info_->get_corr_abs_mo("CORRELATED");
    ///From Y_{pt} = F_{pu}^{core} * Gamma_{tu}
    ambit::Tensor Y = ambit::Tensor::build(ambit::CoreTensor,"Y",{nmo_, na_});
    ambit::Tensor F_pu = ambit::Tensor::build(ambit::CoreTensor, "F_pu", {nmo_, na_});
    if(nrdocc_ > 0)
    {
        F_pu.iterate([&](const std::vector<size_t>& i,double& value){
            value = F_core_->get(nmo_abs_[i[0]],active_abs_[i[1]]);});
    }
    Y("p,t") = F_pu("p,u") * gamma1_("t, u");

    SharedMatrix Y_m(new Matrix("Y_m", nmo_, na_));

    Y.iterate([&](const std::vector<size_t>& i,double& value){
        Y_m->set(nmo_abs_[i[0]],i[1], value);});
    Y_ = Y_m;
    Y_->set_name("F * gamma");
    if(casscf_debug_print_)
    {
        Y_->print();
    }

    //Form Z (pu | v w) * Gamma2(tuvw)
    //One thing I am not sure about for Gamma2->how to get spin free RDM from spin based RDM
    //gamma1 = gamma1a + gamma1b;
    //gamma2 = gamma2aa + gamma2ab + gamma2ba + gamma2bb
    /// lambda2 = gamma1*gamma1

    //std::vector<size_t> na_array = mo_space_info_->get_corr_abs_mo("ACTIVE");
    ///SInce the integrals class assumes that the indices are relative,
    /// pass the relative indices to the integrals code.
    ambit::Tensor Z = ambit::Tensor::build(ambit::CoreTensor, "Z", {nmo_, na_});

    //(pu | x y) -> <px | uy> * gamma2_{"t, u, x, y"
    Z("p, t") = integral_("p,u,x,y") * gamma2_("t, u, x, y");

    SharedMatrix Zm(new Matrix("Zm", nmo_, na_));
    Z.iterate([&](const std::vector<size_t>& i,double& value){
        Zm->set(nmo_abs_[i[0]],i[1], value);});

    Z_ = Zm;
    Z_->set_name("g * rdm2");
    if(casscf_debug_print_)
    {
        Z_->print();
    }
    // g_ia = 4F_core + 2F_act
    // g_ta = 2Y + 4Z
    // g_it = 4F_core + 2 F_act - 2Y - 4Z;

    //GOTCHA:  Z and T are of size nmo by na
    //The absolute MO should not be used to access elements of Z, Y, or Gamma since these are of 0....na_ arrays
    //auto occ_array = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    //auto virt_array = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    //auto active_array = mo_space_info_->get_corr_abs_mo("ACTIVE");


    SharedMatrix Orb_grad(new Matrix("G_pq", nmo_, nmo_));
    Orb_grad->set_name("CASSCF Gradient");

    for(size_t ii = 0; ii < restricted_docc_abs_.size(); ii++)
        for(size_t ti = 0; ti < active_abs_.size(); ti++){
            {
                size_t i = restricted_docc_abs_[ii];
                size_t t = active_abs_[ti];
                //double value_it = 4 * F_core_->get(i, t) + 2 * F_act_->get(i,t) - 2 * Y_->get(i,ti) - 4 * Z_->get(i, ti);
                double value_it = 4 * F_core_->get(i, t) + 4 * F_act_->get(i,t) - 2 * Y_->get(i,ti) - 2 * Z_->get(i, ti);
                Orb_grad->set(i,t, value_it) ;
            }
    }
    if(casscf_debug_print_){outfile->Printf("\n i, t %8.8f", Orb_grad->rms());}
    for(auto i : restricted_docc_abs_){
        for(auto a : restricted_uocc_abs_){
            double value_ia = F_core_->get(i, a) * 4.0 + F_act_->get(i, a) * 4.0;
            Orb_grad->set(i, a, value_ia);
        }
    }
    if(casscf_debug_print_){outfile->Printf("\n i, a %8.8f", Orb_grad->rms());}

    for(size_t ai = 0; ai < restricted_uocc_abs_.size(); ai++){
        for(size_t ti = 0; ti < active_abs_.size(); ti++){
            size_t t = active_abs_[ti];
            size_t a = restricted_uocc_abs_[ai];
            double value_ta = 2.0 * Y_->get(a, ti) + 2.0 * Z_->get(a, ti);
            Orb_grad->set(t,a, value_ta);
        }
    }
    if(casscf_debug_print_){outfile->Printf("\n t, a %8.8f", Orb_grad->rms());}

    std::vector<size_t> active_rel = mo_space_info_->get_corr_abs_mo("ACTIVE");

    for(size_t u = 0; u < na_; u++){
        for(size_t v = 0; v < na_; v++){
            Orb_grad->set(active_rel[u], active_rel[v], 0.0);
        }
    }
    if(casscf_debug_print_)
    {
        Orb_grad->print();
    }

    g_ = Orb_grad;
    g_->set_name("CASSCF_GRADIENT");


}
void OrbitalOptimizer::diagonal_hessian()
{
    fill_shared_density_matrices();
    SharedMatrix D(new Matrix("DH", nmo_, nmo_));

    for(size_t ii = 0; ii < restricted_docc_abs_.size(); ii++){
        for(size_t ai = 0; ai < restricted_uocc_abs_.size(); ai++){
            size_t a = restricted_uocc_abs_[ai];
            size_t i = restricted_docc_abs_[ii];
            //double value_ia = F_core_->get(a,a) * 4.0 + 2 * F_act_->get(a,a);
            //value_ia -= 4.0 * F_core_->get(i,i)  - 2 * F_act_->get(i,i);
            double value_ia = (F_core_->get(a,a) * 4.0 + 4.0 * F_act_->get(a,a));
            value_ia -= (4.0 * F_core_->get(i,i)  + 4.0 * F_act_->get(i,i));
            D->set(i,a,value_ia);
            D->set(a, i, value_ia);
        }
    }
    for(size_t ai = 0; ai < restricted_uocc_abs_.size(); ai++){
        for(size_t ti = 0; ti < active_abs_.size(); ti++){
            size_t a = restricted_uocc_abs_[ai];
            size_t t = active_abs_[ti];
            //double value_ta = 2.0 * gamma1M_->get(ti,ti) * F_core_->get(a,a);
            //value_ta += gamma1M_->get(ti,ti) * F_act_->get(a,a);
            //value_ta -= 2*Y_->get(t,ti) + 4.0 *Z_->get(t,ti);
            double value_ta = 2.0 * gamma1M_->get(ti,ti) * F_core_->get(a,a);
            value_ta += 2.0 * gamma1M_->get(ti,ti) * F_act_->get(a,a);
            value_ta -= (2*Y_->get(t,ti) + 2.0 *Z_->get(t,ti));
            D->set(t,a, value_ta);
            D->set(a,t, value_ta);
        }
    }
    for(size_t ii = 0; ii < restricted_docc_abs_.size(); ii++){
        for(size_t ti = 0; ti < active_abs_.size(); ti++){
            size_t i = restricted_docc_abs_[ii];
            size_t t = active_abs_[ti];

            double value_it = 4.0 * F_core_->get(t,t)
                    + 4.0 * F_act_->get(t,t)
                    + 2.0 * gamma1M_->get(ti,ti) * F_core_->get(i,i);
            value_it+=2.0 * gamma1M_->get(ti,ti) * F_act_->get(i,i);
            value_it-=(4.0 * F_core_->get(i,i) + 4.0 * F_act_->get(i,i));
            value_it-=(2.0*Y_->get(t,ti) + 2.0 * Z_->get(t,ti));
            D->set(i,t, value_it);
            D->set(t,i, value_it);
        }
    }

    for(size_t u = 0; u < na_; u++){
        for(size_t v = 0; v < na_; v++){
            D->set(active_abs_[u], active_abs_[v], 1.0);
        }
    }
    d_ = D;


}
SharedMatrix OrbitalOptimizer::approx_solve()
{
    ///Create an orbital rotation matrix of size (NMO - frozen)
    SharedMatrix S(new Matrix("S", nmo_, nmo_));
    Dimension true_nmopi = wfn_->nmopi();
    SharedMatrix S_sym(new Matrix("S_sym", nirrep_, true_nmopi, true_nmopi));
    SharedMatrix S_sym_AH(new Matrix("S_sym", nirrep_, true_nmopi, true_nmopi));

    int offset = 0;
    for(size_t h = 0; h < nirrep_; h++){
        for(int p = 0; p < nmopi_[h]; p++){
            int poff = p + offset;
            for(int q = 0; q < nmopi_[h]; q++){
                int qoff = q + offset;
                if(poff < qoff)
                {
                    if(d_->get(poff,qoff) > 1e-8)
                        S->set(poff,qoff, g_->get(poff,qoff) / d_->get(poff,qoff));
                }
                else if(poff > qoff)
                {
                    if(d_->get(qoff,poff) > 1e-8)
                        S->set(poff,qoff,-1.0 * g_->get(qoff,poff) / d_->get(qoff,poff));
                }

            }
        }
    offset += nmopi_[h];
    }
    ///Since this is CASSCF, the active rotations are reduntant.  Zero those!
    auto na_vec = mo_space_info_->get_corr_abs_mo("ACTIVE");
    for(size_t u = 0; u < na_; u++)
        for(size_t v = 0; v < na_; v++)
            S->set(na_vec[u], na_vec[v], 0.0);

    ///Convert to a symmetry matrix
    offset = 0;
    int frozen = 0;
    //SharedMatrix S_diag = AugmentedHessianSolve();
    for(size_t h = 0; h < nirrep_; h++){
        frozen = frozen_docc_dim_[h];
        for(int p = 0; p < nmopi_[h]; p++){
            for(int q = 0; q < nmopi_[h]; q++){
                S_sym->set(h, p + frozen, q + frozen, S->get(p + offset, q + offset));
            }
        }
        if(casscf_freeze_core_)
        {
            for(int fr = 0; fr < frozen_docc_dim_[h]; fr++)
            {
                S_sym->set(h, fr, fr, 1.0);
            }
        }
        offset += nmopi_[h];
    }
    S->print();
    S_diag->print();

    return S_sym;
}
SharedMatrix OrbitalOptimizer::AugmentedHessianSolve()
{
    SharedMatrix AugmentedHessian(new Matrix("Augmented Hessian", 2 * nmo_ + 1, 2 * nmo_ + 1));

    for(size_t p = 0; p < nmo_; p++){

        for(size_t q = 0; q < nmo_; q++){
            AugmentedHessian->set(p, q, d_->get(p, q));
        }
        for(size_t q = 0; q < nmo_; q++){
            AugmentedHessian->set(p + nmo_, q, g_->get(p, q));
            AugmentedHessian->set(q, p + nmo_, g_->get(q, p));
        }
    }
    AugmentedHessian->set(2*nmo_, 2*nmo_, 0.0);
    g_->print();
    d_->print();
    AugmentedHessian->print();
    SharedMatrix HessianEvec(new Matrix("HessianEvec",  2*nmo_ + 1, 2*nmo_ + 1));
    SharedVector HessianEval(new Vector("HessianEval", 2*nmo_ + 1));
    AugmentedHessian->diagonalize(HessianEvec, HessianEval);
    HessianEval->print();
    return HessianEvec;

}

SharedMatrix OrbitalOptimizer::rotate_orbitals(SharedMatrix C, SharedMatrix S)
{
    ///Clone the C matrix
    SharedMatrix C_rot(C->clone());
    SharedMatrix S_mat(S->clone());

    S_mat->expm();
    for(size_t h = 0; h < nirrep_; h++)
    {
        for(int fr = 0; fr < frozen_docc_dim_[h]; fr++){
            S_mat->set(h, fr, fr, 1.0);
        }
    }

    C_rot = Matrix::doublet(C, S_mat);
    return C_rot;


}

void OrbitalOptimizer::fill_shared_density_matrices()
{
    SharedMatrix gamma_spin_free(new Matrix("Gamma", na_, na_));
    gamma1_.iterate([&](const std::vector<size_t>& i,double& value){
        gamma_spin_free->set(i[0], i[1], value);});
    gamma1M_ = gamma_spin_free;

    SharedMatrix gamma2_matrix(new Matrix("Gamma2", na_ * na_, na_ * na_));
    gamma2_.iterate([&](const std::vector<size_t>& i,double& value){
        gamma2_matrix->set(i[0] * i[1] + i[1], i[2] * i[3] + i[3], value);});
    gamma2M_ = gamma2_matrix;
}
boost::shared_ptr<Matrix> OrbitalOptimizer::make_c_sym_aware()
{
    ///Step 1: Obtain guess MO coefficients C_{mup}
    /// Since I want to use these in a symmetry aware basis,
    /// I will move the C matrix into a Pfitzer ordering

    Dimension nmopi = mo_space_info_->get_dimension("ALL");

    SharedMatrix aotoso = wfn_->aotoso();

    /// I want a C matrix in the C1 basis but symmetry aware
    size_t nso = wfn_->nso();
    nirrep_ = wfn_->nirrep();
    SharedMatrix Call(new Matrix(nso, nmopi.sum()));

    // Transform from the SO to the AO basis for the C matrix.
    // just transfroms the C_{mu_ao i} -> C_{mu_so i}
    for (size_t h = 0, index = 0; h < nirrep_; ++h){
        for (int i = 0; i < nmopi[h]; ++i){
            size_t nao = nso;
            size_t nso = nsopi_[h];

            if (!nso) continue;

            C_DGEMV('N',nao,nso,1.0,aotoso->pointer(h)[0],nso,&Ca_sym_->pointer(h)[0][i],nmopi[h],0.0,&Call->pointer()[0][index],nmopi.sum());

            index += 1;
        }

    }

    return Call;
}

}}
