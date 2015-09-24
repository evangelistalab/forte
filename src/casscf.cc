#include "casscf.h"
#include "reference.h"
#include "integrals.h"
#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libmints/molecule.h>
#include <libqt/qt.h>
#include <libmints/matrix.h>
#include "helpers.h"
#include <libfock/jk.h>
#include <libmints/mints.h>
#include "fci_solver.h"
#include <psifiles.h>
#include <libmints/factory.h>
#include <libmints/mintshelper.h>
#include <lib3index/cholesky.h>

namespace psi{ namespace forte{

CASSCF::CASSCF(Options &options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
         : options_(options),
         ints_(ints),
         mo_space_info_(mo_space_info)
{
    if(options_.get_str("SCF_TYPE") == "PK")
    {
        outfile->Printf("\n Kevin has yet to figure out PKJK so only works for DF/CD ");
        throw PSIEXCEPTION("Need to set SCF_TYPE==DF/CD to use CASSCF code");
    }
    startup();
}
void CASSCF::compute_casscf()
{
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    outfile->Printf("\n nmo:%lu  na_:%lu", nmo_, na_);
    if(na_ == 0)
    {
        outfile->Printf("\n\n\n Please set the active space");
        throw PSIEXCEPTION(" The active space is zero.  Set the active space");
    }
    else if(na_ == nmo_)
    {
        outfile->Printf("\n Your about to do an all active CASSCF");
        throw PSIEXCEPTION("The active space is all the MOs.  Orbitals don't matter at this point");
    }
    SharedMatrix Cold(Call_->clone());
    int maxiter = options_.get_int("CASSCF_ITERATIONS");
    std::vector<int> iter_con;
    std::vector<double> g_norm_con;
    std::vector<double> E_casscf_con;
    for(int iter = 0; iter < maxiter; iter++)
    {
       iter_con.push_back(iter);
        if(iter==0)
        {
            print_h2("CASSCF Iteration");
        }

        cas_ci();
        E_casscf_con.push_back(E_casscf_);

        form_fock_core();

        form_fock_active();

        orbital_gradient();
        double g_norm = g_->sum_of_squares();
        g_norm_con.push_back(g_norm);

        if(g_norm < options_.get_double("CASSCF_CONVERGENCE"))
        {
            break;
            outfile->Printf("\n\n CASSCF CONVERGED \n\n");
            outfile->Printf("\n %8.8f", g_norm);
        }


        diagonal_hessian();

        //Update MO coefficients
        SharedMatrix S(new Matrix("S", nmo_, nmo_));
        auto mo_array = mo_space_info_->get_absolute_mo("ALL");
        for(auto p : mo_array){
            for(auto q : mo_array){

                if(p < q)
                {
                    if(d_->get(p, q) > 1e-8)
                        S->set(p,q, g_->get(p,q) / d_->get(p,q));
                }
                else if(p > q)
                {
                    if(d_->get(q, p) > 1e-8)
                        S->add(p,q,-1.0 * g_->get(q,p) / d_->get(q,p));
                }
            }
        }

        ///Last Step: Update MO Coefficients
        //C' = C * Exp(S)

        S->power(exp(1.0));
        Call_->gemm('n', 't', 1.0, Cold, S, 0.0);
        SharedMatrix Ca = wfn->Ca();
        Ca->copy(Call_);
        // With updated C coefficients, need to retransform integrals so I can run FCI with transformed integrals
        //ints_->retransform_integrals();

        if(options_.get_int("PRINT") > 1)
        {
            S->print();
            g_->print();
            d_->print();
            Ca->print();
        }


    }
    outfile->Printf("\n iter    g_norm      E_CASSCF");
    for(size_t i = 0; i < iter_con.size(); i++)
    {
        outfile->Printf("\n %d  %8.8f   %8.8f", i, g_norm_con[i], E_casscf_con[i]);
    }


}

void CASSCF::startup()
{
    print_method_banner({"Complete Active Space Self Consistent Field","Kevin Hannon"});
    ///Step 1: Obtain guess MO coefficients C_{mup}
    /// Since I want to use these in a symmetry aware basis,
    /// I will move the C matrix into a Pfitzer ordering
    nmopi_ = mo_space_info_->get_dimension("ALL");

    nmo_ = mo_space_info_->size("ALL");
    na_  = mo_space_info_->size("ACTIVE");

    auto wfn = Process::environment.wavefunction();
    SharedMatrix Call_sym = wfn->Ca();
    // This is commented out because I may need it at some point, but I don't not think it is necessary as of now.
    //Dimension nsopi_ = wfn->nsopi();
    SharedMatrix aotoso = wfn->aotoso();
    SharedMatrix Call(new Matrix("Call_", nmo_, nmo_));
    Dimension nsopi_ = wfn->nsopi();
    size_t nso = wfn->nso();
    size_t nirrep_ = wfn->nirrep();

    // Transform from the SO to the AO basis
    for (size_t h = 0, index = 0; h < nirrep_; ++h){
        for (int i = 0; i < nmopi_[h]; ++i){
            size_t nao = nso;
            size_t nso = nsopi_[h];

            if (!nso) continue;

            C_DGEMV('N',nao,nso,1.0,aotoso->pointer(h)[0],nso,&Call_sym->pointer(h)[0][i],nmopi_[h],0.0,&Call->pointer()[0][index],nmopi_.sum());

            index += 1;
        }

    }
    Call_ = Call;
}
void CASSCF::cas_ci()
{
    boost::shared_ptr<Wavefunction> wfn_casscf = Process::environment.wavefunction();
    boost::shared_ptr<FCI> fci_casscf(new FCI(wfn_casscf,options_,ints_,mo_space_info_));
    fci_casscf->compute_energy();
    cas_ref_ = fci_casscf->reference();
    E_casscf_ = cas_ref_.get_Eref();
}

void CASSCF::form_fock_core()
{

    /// Get the CoreHamiltonian in AO basis

    //boost::shared_ptr<PSIO> psio_ = PSIO::shared_object();
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    if(true)
    {
        boost::shared_ptr<MintsHelper> mints(new MintsHelper());
        SharedMatrix T = mints->ao_kinetic();
        SharedMatrix V = mints->ao_potential();
        SharedMatrix H = T->clone();
        H->add(V);

        H->transform(Call_);


        ///Step 2: From Hamiltonian elements
        ///This will use JK builds (Equation 18 - 22)
        /// F_{pq}^{core} = C_{mu p}C_{nu q} [h_{uv} + 2J^{(D_c) - K^{(D_c)}]


        ///Have to go from the full C matrix to the C_core in the SO basis
        /// tricky...tricky
        std::vector<size_t> inactive_dim_abs = mo_space_info_->get_absolute_mo("INACTIVE_DOCC");
        SharedMatrix C_core(new Matrix("C_core",nmo_, inactive_dim_abs.size()));
        // Need to get the inactive block of the C matrix
        for(size_t mu = 0; mu < nmo_; mu++){
            for(size_t i = 0; i <  inactive_dim_abs.size(); i++){
                C_core->set(mu, i, Call_->get(mu, inactive_dim_abs[i]));
            }
        }

        boost::shared_ptr<JK> JK_core = JK::build_JK();

        JK_core->set_memory(Process::environment.get_memory() * 0.8);
        /// Already transform everything to C1 so make sure JK does not do this.
        JK_core->set_allow_desymmetrization(false);

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
        F_core->add(H);
        F_core->transform(Call_);
        F_core_ = F_core;

   }
    // Commented out because this is for PK algorithm and I am not sure why it is required to be in the SO basis
//   else
//   {
//        outfile->Printf("\n You are about to run a DF-CASSCF.");
//        boost::shared_ptr<PSIO> psio_ = PSIO::shared_object();
//        boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
//
//        SharedMatrix T = SharedMatrix(wfn->matrix_factory()->create_matrix(PSIF_SO_T));
//        SharedMatrix V = SharedMatrix(wfn->matrix_factory()->create_matrix(PSIF_SO_V));
//
//        SharedMatrix Ca = wfn->Ca();
//        T->load(psio_, PSIF_OEI);
//        V->load(psio_, PSIF_OEI);
//        SharedMatrix H = T->clone();
//        H->add(V);
//        H->transform(Ca);
//
//        Dimension inactive_dim = mo_space_info_->get_dimension("INACTIVE_DOCC");
//        Dimension nmopi_       = mo_space_info_->get_dimension("ALL");
//        SharedMatrix C_core(new Matrix("C_core",nirrep_, nmopi_, inactive_dim));
//        // Need to get the inactive block of the C matrix
//        for(size_t h = 0; h < nirrep_; h++){
//            for(size_t mu = 0; mu < nmopi_[h]; mu++){
//                for(size_t i = 0; i <  inactive_dim[h]; i++){
//                    C_core->set(h,mu, i, Ca->get(h,mu, i));
//                }
//            }
//        }
//        boost::shared_ptr<JK> JK_core = JK::build_JK();
//
//        JK_core->set_memory(Process::environment.get_memory() * 0.8);
//        /// Already transform everything to C1 so make sure JK does not do this.
//        JK_core->set_allow_desymmetrization(false);
//
//        /////TODO: Make this an option in my code
//        //JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
//        JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
//        JK_core->initialize();
//
//
//
//        std::vector<boost::shared_ptr<Matrix> >&Cl = JK_core->C_left();
//
//        Cl.clear();
//        Cl.push_back(C_core);
//
//        JK_core->compute();
//
//        SharedMatrix J_core = JK_core->J()[0];
//        SharedMatrix K_core = JK_core->K()[0];
//
//        J_core->scale(2.0);
//        F_core_ = J_core->clone();
//        F_core_->subtract(K_core);
//        F_core_->add(H);
//        F_core_->print();
//
//   }

}
void CASSCF::form_fock_active()
{
    ///Step 3:
    ///Compute equation 10:
    /// The active OPM is defined by gamma = gamma_{alpha} + gamma_{beta}

    ambit::Tensor gamma_no_spin = ambit::Tensor::build(ambit::kCore,"Return",{na_, na_});
    gamma1_ = ambit::Tensor::build(ambit::kCore,"Return",{na_, na_});
    ambit::Tensor gamma1a = cas_ref_.L1a();
    ambit::Tensor gamma1b = cas_ref_.L1b();

    gamma_no_spin("i,j") = gamma1a("i,j") + gamma1b("i,j");

    SharedMatrix gamma_spin_free(new Matrix("Gamma", na_, na_));
    gamma_no_spin.iterate([&](const std::vector<size_t>& i,double& value){
        gamma_spin_free->set(i[0], i[1], value);});
    gamma1M_ = gamma_spin_free;
    gamma1_ = gamma_no_spin;

    std::vector<size_t> active_abs_mo = mo_space_info_->get_absolute_mo("ACTIVE");
    SharedMatrix C_active(new Matrix("C_active", nmo_,na_));

    for(size_t mu = 0; mu < nmo_; mu++){
        for(size_t u = 0; u <  na_; u++){
            C_active->set(mu, u, Call_->get(mu, active_abs_mo[u]));
        }
    }

    ambit::Tensor Cact = ambit::Tensor::build(ambit::kCore, "Cact", {nmo_, na_});
    ambit::Tensor OPDM_aoT = ambit::Tensor::build(ambit::kCore, "OPDM_aoT", {nmo_, nmo_});

    Cact.iterate([&](const std::vector<size_t>& i,double& value){
        value = C_active->get(i[0], i[1]);});

    ///Transfrom the all active OPDM to the AO basis
    OPDM_aoT("mu,nu") = gamma_no_spin("u,v")*Cact("mu, u") * Cact("nu, v");
    SharedMatrix OPDM_ao(new Matrix("OPDM_AO", nmo_, nmo_));

    OPDM_aoT.iterate([&](const std::vector<size_t>& i,double& value){
        OPDM_ao->set(i[0], i[1], value);});

    ///In order to use JK builder for active part, need a Cmatrix like matrix
    /// AO OPDM looks to be semi positive definite so perform CholeskyDecomp and feed this to JK builder
    boost::shared_ptr<CholeskyMatrix> Ch (new CholeskyMatrix(OPDM_ao, 1e-12, Process::environment.get_memory()));
    Ch->choleskify();
    SharedMatrix L_C = Ch->L();
    SharedMatrix L_C_correct(new Matrix("L_C_order", nmo_, Ch->Q()));

    for(size_t mu = 0; mu < nmo_; mu++){
        for(size_t Q = 0; Q < Ch->Q(); Q++){
            L_C_correct->set(mu, Q, L_C->get(Q, mu));
        }
    }

    boost::shared_ptr<JK> JK_act = JK::build_JK();

    JK_act->set_memory(Process::environment.get_memory() * 0.8);
    JK_act->set_allow_desymmetrization(false);

    /////TODO: Make this an option in my code
    JK_act->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
    JK_act->initialize();

    std::vector<boost::shared_ptr<Matrix> >&Cl = JK_act->C_left();

    Cl.clear();
    Cl.push_back(L_C_correct);

    JK_act->compute();

    SharedMatrix J_core = JK_act->J()[0];
    SharedMatrix K_core = JK_act->K()[0];

    SharedMatrix F_act = J_core->clone();
    F_act->scale(2.0);
    F_act->subtract(K_core);
    F_act->transform(Call_);
    F_act_ = F_act;


}
void CASSCF::orbital_gradient()
{
    ///From Y_{pt} = F_{pu}^{core} * Gamma_{tu}
    ambit::Tensor Y = ambit::Tensor::build(ambit::kCore,"Y",{nmo_, na_});
    ambit::Tensor F_pu = ambit::Tensor::build(ambit::kCore, "F_pu", {nmo_, na_});
    auto active_mo = mo_space_info_->get_absolute_mo("ACTIVE");
    F_pu.iterate([&](const std::vector<size_t>& i,double& value){
        value = F_core_->get(i[0],active_mo[i[1]]);});
    Y("p,t") = F_pu("p,u") * gamma1_("t, u");
    SharedMatrix Y_m(new Matrix("Y_m", nmo_, na_));

    Y.iterate([&](const std::vector<size_t>& i,double& value){
        Y_m->set(i[0],i[1], value);});
    Y_ = Y_m;

    //Form Z (pu | v w) * Gamma2(tuvw)
    //One thing I am not sure about for Gamma2->how to get spin free RDM from spin based RDM
    //gamma1 = gamma1a + gamma1b;
    //gamma2 = gamma2aa + gamma2ab + gamma2ba + gamma2bb
    /// lambda2 = gamma1*gamma1
    ambit::Tensor L1a  = cas_ref_.L1a();
    ambit::Tensor L1b  = cas_ref_.L1b();
    ambit::Tensor L2aa = cas_ref_.L2aa();
    ambit::Tensor L2ab = cas_ref_.L2ab();
    ambit::Tensor L2bb = cas_ref_.L2bb();

    //lambda^{pq}_{rs} = gamma^{pq}_{rs} - gamma^p_r gamma^q_s + gamma^p_s gamma^q_r

    L2aa("p,q,r,s") += L1a("p,r") * L1a("q,s");
    L2aa("p,q,r,s") -= L1a("p,s") * L1a("q,r");

    L2ab("pqrs") += L1a("pr") * L1a("qs");

    L2bb("pqrs") += L1b("pr") * L1b("qs");
    L2bb("pqrs") -= L1b("ps") * L1b("qr");


    gamma2_ = ambit::Tensor::build(ambit::kCore, "gamma2", {na_, na_, na_, na_});

    // This may or may not be correct.  Really need to find a way to check this code
    gamma2_("u,v,x,y") = L2aa("uvxy") + L2ab("uvxy") + L2bb("uvxy");
    //gamma2_("u,v,x,y") = gamma2_("x,y,u,v");
    //gamma2_("u,v,x,y") = gamma2_("")

    ambit::Tensor tei_puvy = ambit::Tensor::build(ambit::kCore, "puvy", {nmo_, na_, na_, na_});
    std::vector<size_t> nmo_array = mo_space_info_->get_absolute_mo("ALL");
    std::vector<size_t> na_array = mo_space_info_->get_absolute_mo("ACTIVE");
    tei_puvy = ints_->aptei_ab_block(nmo_array, na_array, na_array, na_array);
    ambit::Tensor Z = ambit::Tensor::build(ambit::kCore, "Z", {nmo_, na_});

    Z("p, t") = tei_puvy("p,u,x,y") * gamma2_("t, u, x, y");
    SharedMatrix Zm(new Matrix("Zm", nmo_, na_));
    Z.iterate([&](const std::vector<size_t>& i,double& value){
        Zm->set(i[0],i[1], value);});

    Z_ = Zm;
    //Forming Orbital gradient
    // g_ia = 4F_core + 2F_act
    // g_ta = 2Y + 4Z
    // g_it = 4F_core + 2 F_act - 2Y - 4Z;

    //GOTCHA:  Z and T are of size nmo by na
    //The absolute MO should not be used to access elements of Z, Y, or Gamma since these are of 0....na_ arrays
    auto occ_array = mo_space_info_->get_absolute_mo("INACTIVE_DOCC");
    auto virt_array = mo_space_info_->get_absolute_mo("RESTRICTED_UOCC");
    auto active_array = mo_space_info_->get_absolute_mo("ACTIVE");
    SharedMatrix Orb_grad(new Matrix("G_pq", nmo_, nmo_));
    for(size_t ii = 0; ii < occ_array.size(); ii++)
        for(size_t ti = 0; ti < active_array.size(); ti++){
            {
                size_t i = occ_array[ii];
                size_t t = active_array[ti];
                double value_it = 4 * F_core_->get(i, t) + 2 * F_act_->get(i,t) - 2 * Y_->get(i,ti) - 4 * Z_->get(i, ti);
                Orb_grad->set(i,t, value_it) ;
            }
    }
    for(auto i : occ_array){
        for(auto a : virt_array){
            double value_ia = F_core_->get(i, a) * 4 + F_act_->get(i, a);
            Orb_grad->set(i, a, value_ia);

        }
    }
    for(size_t ai = 0; ai < virt_array.size(); ai++){
        for(size_t ti = 0; ti < active_array.size(); ti++){
            size_t t = active_array[ti];
            size_t a = virt_array[ai];
            double value_ta = 2 * Y_->get(a, ti) + 4 * Z_->get(a,ti);
            Orb_grad->set(t,a, value_ta);
        }
    }

    g_ = Orb_grad;


}
void CASSCF::diagonal_hessian()
{
    SharedMatrix D(new Matrix("DH", nmo_, nmo_));
    auto i_array = mo_space_info_->get_absolute_mo("INACTIVE_DOCC");
    auto a_array = mo_space_info_->get_absolute_mo("RESTRICTED_UOCC");
    auto t_array = mo_space_info_->get_absolute_mo("ACTIVE");

    for(size_t ii = 0; ii < i_array.size(); ii++){
        for(size_t ai = 0; ai < a_array.size(); ai++){
            size_t a = a_array[ai];
            size_t i = i_array[ii];
            double value_ia = F_core_->get(a,a) * 4.0 + 2 * F_act_->get(a,a);
            value_ia -= 4.0 * F_core_->get(i,i)  - 2 * F_act_->get(i,i);
            D->set(i,a,value_ia);
        }
    }
    for(size_t ai = 0; ai < a_array.size(); ai++){
        for(size_t ti = 0; ti < t_array.size(); ti++){
            size_t a = a_array[ai];
            size_t t = t_array[ti];
            double value_ta = 2.0 * gamma1M_->get(ti,ti) * F_core_->get(a,a);
            value_ta += gamma1M_->get(ti,ti) * F_act_->get(a,a);
            value_ta -= 2*Y_->get(t,ti) + 4.0 *Z_->get(t,ti);
            D->set(t,a, value_ta);
        }
    }
    for(size_t ii = 0; ii < i_array.size(); ii++){
        for(size_t ti = 0; ti < t_array.size(); ti++){
            size_t i = i_array[ii];
            size_t t = t_array[ti];
            double value_it = 4.0 * F_core_->get(t,t)
                    + 2.0 * F_act_->get(t,t)
                    + 2 * gamma1M_->get(ti,ti) * F_core_->get(i,i);
            value_it+=gamma1M_->get(ti,ti) * F_act_->get(i,i);
            value_it-=4.0 * F_core_->get(i,i) + 2.0 * F_act_->get(i,i);
            value_it-=2.0*Y_->get(t,ti) + 4.0 * Z_->get(t,ti);
            D->set(i,t, value_it);
        }
    }
    d_ = D;


}

}}
