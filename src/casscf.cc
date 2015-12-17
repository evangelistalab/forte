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
#include "fci_mo.h"
#include <libthce/lreri.h>
#include "orbitaloptimizer.h"
namespace psi{ namespace forte{

CASSCF::CASSCF(Options &options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
         : options_(options),
         wfn_(Process::environment.wavefunction()),
         ints_(ints),
         mo_space_info_(mo_space_info)
{
    startup();
}
void CASSCF::compute_casscf()
{
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

    int maxiter = options_.get_int("CASSCF_ITERATIONS");

    /// Provide a nice summary at the end for iterations
    std::vector<int> iter_con;
    std::vector<double> g_norm_con;
    std::vector<double> E_casscf_con;
    /// FrozenCore C Matrix is never rotated
    /// Can bring this out of loop
    if(nfrozen_ > 0)
    {
        F_froze_ = set_frozen_core_orbitals();
    }


    ///Start the iteration
    ///
    outfile->Printf("\n iter    g_norm      E_CASSCF");
    for(int iter = 0; iter < maxiter; iter++)
    {
       iter_con.push_back(iter);
        if(iter==0)
        {
            print_h2("CASSCF Iteration");
        }

        /// Perform a CAS-CI using either York's code or Francesco's
        /// If CASSCF_DEBUG_PRINTING is on, will compare CAS-CI with SPIN-FREE RDM
        E_casscf_ = 0.0;
        cas_ci();
        SharedMatrix Ca = wfn_->Ca();
        SharedMatrix Cb = wfn_->Cb();

        OrbitalOptimizer orbital_optimizer(gamma1_,
                                           gamma2_,
                                           ints_->aptei_ab_block(nmo_abs_, active_abs_, active_abs_, active_abs_) ,
                                           options_,
                                           mo_space_info_);
        orbital_optimizer.set_frozen_one_body(F_froze_);
        orbital_optimizer.set_no_symmetry_mo(Call_);
        orbital_optimizer.set_symmmetry_mo(Ca);
        SharedMatrix S_sym = orbital_optimizer.orbital_rotation_casscf();
        double g_norm = orbital_optimizer.orbital_gradient_norm();


        if(g_norm < options_.get_double("CASSCF_CONVERGENCE"))
        {
            break;
            outfile->Printf("\n\n CASSCF CONVERGED: @ %8.8f \n\n",E_casscf_ );
            outfile->Printf("\n %8.8f", g_norm);
        }
        Call_->set_name("symmetry aware C");


        SharedMatrix Cp = Matrix::doublet(Ca, S_sym);
        Cp->set_name("Updated C");
        if(casscf_debug_print_)
        {
            Cp->print();
            S_sym->set_name("OrbitalRotationMatrix");
            S_sym->print();
        }

        Ca->copy(Cp);
        Cb->copy(Cp);
        ///Right now, this retransforms all the integrals
        ///Think of ways of avoiding this
        ///This is used as way to retransform our integrals so Francesco's FCI code can use the updated CI
        ///Can redefi
        ints_->retransform_integrals();


        /// Use the newly transformed MO to create a CMatrix that is aware of symmetry
        Call_->zero();
        Call_ = make_c_sym_aware();


        outfile->Printf("\n %d %10.12f   %10.12f", iter, g_norm, E_casscf_);
    }

    if(iter_con.size() == size_t(maxiter))
    {
        outfile->Printf("\n CASSCF did not converged");
        throw PSIEXCEPTION("CASSCF did not converged.");
    }
    Process::environment.globals["CURRENT ENERGY"] = E_casscf_;
    Process::environment.globals["CASSCF_ENERGY"] = E_casscf_;



}
void CASSCF::startup()
{
    print_method_banner({"Complete Active Space Self Consistent Field","Kevin Hannon"});
    na_  = mo_space_info_->size("ACTIVE");
    nsopi_ = wfn_->nsopi();
    Call_ = make_c_sym_aware();

    casscf_debug_print_ = options_.get_bool("CASSCF_DEBUG_PRINTING");

    casscf_freeze_core_ = options_.get_bool("CASSCF_FREEZE_CORE");

    frozen_docc_dim_     = mo_space_info_->get_dimension("FROZEN_DOCC");
    restricted_docc_dim_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    active_dim_          = mo_space_info_->get_dimension("ACTIVE");
    restricted_uocc_dim_ = mo_space_info_->get_dimension("RESTRICTED_UOCC");
    inactive_docc_dim_   = mo_space_info_->get_dimension("INACTIVE_DOCC");

    frozen_docc_abs_     = mo_space_info_->get_corr_abs_mo("FROZEN_DOCC");
    restricted_docc_abs_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    active_abs_          = mo_space_info_->get_corr_abs_mo("ACTIVE");
    restricted_uocc_abs_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    inactive_docc_abs_   = mo_space_info_->get_corr_abs_mo("INACTIVE_DOCC");
    nmo_abs_             = mo_space_info_->get_corr_abs_mo("CORRELATED");
    nmo_ = mo_space_info_->size("CORRELATED");
    all_nmo_ = mo_space_info_->size("ALL");
    nmopi_ = mo_space_info_->get_dimension("CORRELATED");
    nrdocc_ = restricted_docc_abs_.size();
    nvir_  = restricted_uocc_abs_.size();

    nfrozen_ = frozen_docc_abs_.size();
    ///If the user wants to freeze core after casscf, this section of code sets frozen_docc to zero
    if(!casscf_freeze_core_)
    {
        restricted_docc_abs_ = mo_space_info_->get_absolute_mo("INACTIVE_DOCC");
        restricted_docc_dim_ = mo_space_info_->get_dimension("INACTIVE_DOCC");
        active_abs_          = mo_space_info_->get_absolute_mo("ACTIVE");
        restricted_uocc_abs_ = mo_space_info_->get_absolute_mo("RESTRICTED_UOCC");

        for(size_t h = 0; h < nirrep_; h++){frozen_docc_dim_[h] = 0;}
        for(size_t i = 0; i < frozen_docc_abs_.size(); i++){frozen_docc_abs_[i] = 0;}
        nfrozen_ = 0;
        nmo_ = mo_space_info_->size("ALL");
        nmopi_ = mo_space_info_->get_dimension("ALL");
        nmo_abs_ = mo_space_info_->get_absolute_mo("ALL");
    }
    if(casscf_debug_print_)
    {
        outfile->Printf("\n Total Number of NMO: %d", nmo_);
        outfile->Printf("\n ACTIVE: ");
        for(auto active : active_abs_){outfile->Printf(" %d", active);}
        outfile->Printf("\n RESTRICTED: ");
        for(auto restricted : restricted_docc_abs_){outfile->Printf(" %d", restricted);}
        outfile->Printf("\n VIRTUAL: ");
        for(auto virtual_index : restricted_uocc_abs_){outfile->Printf(" %d", virtual_index);}

    }

}
boost::shared_ptr<Matrix> CASSCF::make_c_sym_aware()
{
    ///Step 1: Obtain guess MO coefficients C_{mup}
    /// Since I want to use these in a symmetry aware basis,
    /// I will move the C matrix into a Pfitzer ordering

    SharedMatrix Call_sym = wfn_->Ca();
    Ca_sym_ = Call_sym;
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

            C_DGEMV('N',nao,nso,1.0,aotoso->pointer(h)[0],nso,&Call_sym->pointer(h)[0][i],nmopi[h],0.0,&Call->pointer()[0][index],nmopi.sum());

            index += 1;
        }

    }

    return Call;
}
void CASSCF::cas_ci()
{
    ///Calls francisco's FCI code and does a CAS-CI with the active given in the input
    SharedMatrix gamma2_matrix(new Matrix("gamma2", na_*na_, na_*na_));
    if(options_.get_str("CAS_TYPE") == "FCI")
    {
        boost::shared_ptr<FCI> fci_casscf(new FCI(wfn_,options_,ints_,mo_space_info_));
        fci_casscf->set_max_rdm_level(3);
        fci_casscf->compute_energy();

        //Used to grab the computed energy and RDMs.
        cas_ref_ = fci_casscf->reference();
        E_casscf_ = cas_ref_.get_Eref();
        if(options_.get_bool("CASSCF_DEBUG_PRINTING"))
        {
            double E_casscf_check = cas_check(cas_ref_);
            outfile->Printf("\n E_casscf_check - E_casscf_ = difference\n");
            outfile->Printf("\n %8.8f - %8.8f = %8.8f", E_casscf_check, E_casscf_, E_casscf_check - E_casscf_);
        }
        ambit::Tensor L2aa = cas_ref_.g2aa();
        ambit::Tensor L2ab = cas_ref_.g2ab();
        ambit::Tensor L2bb = cas_ref_.g2bb();

        ambit::Tensor gamma2 = ambit::Tensor::build(ambit::kCore, "gamma2", {na_, na_, na_, na_});

        //// This may or may not be correct.  Really need to find a way to check this code
        gamma2("u, v, x, y") = L2aa("u, v, x, y") + L2ab("u, v, x, y") + L2ab("v, u, y, x") + L2bb("u, v, x, y");
        gamma2_ = ambit::Tensor::build(ambit::kCore, "gamma2", {na_, na_, na_, na_});
        gamma2_("u, v, x, y") = gamma2("u, v, x, y") + gamma2("v, u, x, y") + gamma2("u, v, y, x") + gamma2("v, u, y, x");
        gamma2_.scale(0.25);
        gamma2_.iterate([&](const std::vector<size_t>& i,double& value){
            gamma2_matrix->set(i[0] * i[1] + i[1], i[2] * i[3] + i[3], value);});


    }
    else if(options_.get_str("CAS_TYPE") == "CAS")
    {
        FCI_MO cas(wfn_, options_, ints_, mo_space_info_);
        cas.compute_energy();
        cas_ref_ = cas.reference();
        E_casscf_ = cas_ref_.get_Eref();

        ambit::Tensor L2aa = cas_ref_.L2aa();
        ambit::Tensor L2ab = cas_ref_.L2ab();
        ambit::Tensor L2bb = cas_ref_.L2bb();
        ambit::Tensor L1a  = cas_ref_.L1a();
        ambit::Tensor L1b  = cas_ref_.L1b();

        L2aa("p,q,r,s") += L1a("p,r") * L1a("q,s");
        L2aa("p,q,r,s") -= L1a("p,s") * L1a("q,r");

        L2ab("pqrs") +=  L1a("pr") * L1b("qs");
        //L2ab("pqrs") += L1b("pr") * L1a("qs");

        L2bb("pqrs") += L1b("pr") * L1b("qs");
        L2bb("pqrs") -= L1b("ps") * L1b("qr");

        ambit::Tensor gamma2 = ambit::Tensor::build(ambit::kCore, "gamma2", {na_, na_, na_, na_});

        // This may or may not be correct.  Really need to find a way to check this code
        gamma2("u,v,x,y") +=  L2aa("u,v,x, y");
        gamma2("u,v,x,y") +=  L2ab("u,v,x,y");
        //gamma2("u,v,x,y") +=  L2ab("v, u, y, x");
        //gamma2("u,v,x,y") +=  L2bb("u,v,x,y");

        //gamma2_("u,v,x,y") = gamma2_("x,y,u,v");
        //gamma2_("u,v,x,y") = gamma2_("")
        gamma2_ = ambit::Tensor::build(ambit::kCore, "gamma2", {na_, na_, na_, na_});
        gamma2_.copy(gamma2);
        gamma2_.scale(2.0);
        gamma2_.iterate([&](const std::vector<size_t>& i,double& value){
            gamma2_matrix->set(i[0] * i[1] + i[1], i[2] * i[3] + i[3], value);});
        cas_ref_ = cas.reference();
        E_casscf_ = cas_ref_.get_Eref();
        if(options_.get_bool("CASSCF_DEBUG_PRINTING"))
        {
            double E_casscf_check = cas_check(cas_ref_);
            outfile->Printf("\n E_casscf_check - E_casscf_ = difference\n");
            outfile->Printf("\n %8.8f - %8.8f = %8.8f", E_casscf_check, E_casscf_, E_casscf_check - E_casscf_);
        }

    }
    /// Compute the 1RDM
    ambit::Tensor gamma_no_spin = ambit::Tensor::build(ambit::kCore,"Return",{na_, na_});
    gamma1_ = ambit::Tensor::build(ambit::kCore,"Return",{na_, na_});
    ambit::Tensor gamma1a = cas_ref_.L1a();
    ambit::Tensor gamma1b = cas_ref_.L1b();

    gamma_no_spin("i,j") = (gamma1a("i,j") + gamma1b("i,j"));

    gamma1_ = gamma_no_spin;

}

double CASSCF::cas_check(Reference cas_ref)
{
    ambit::Tensor gamma1 = ambit::Tensor::build(ambit::kCore, "Gamma1", {na_, na_});
    ambit::Tensor gamma2 = ambit::Tensor::build(ambit::kCore, "Gamma2", {na_, na_, na_, na_});
    std::shared_ptr<FCIIntegrals> fci_ints = std::make_shared<FCIIntegrals>(ints_, mo_space_info_->get_corr_abs_mo("ACTIVE"), mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));
    fci_ints->set_active_integrals_and_restricted_docc();

    /// Spin-free ORDM = gamma1_a + gamma1_b
    ambit::Tensor L1b = cas_ref.L1b();
    ambit::Tensor L1a = cas_ref.L1a();
    gamma1("u, v") = (L1a("u, v") + L1b("u, v"));
    std::string cas_type = options_.get_str("CAS_TYPE");
    if(cas_type=="FCI")
    {
        ambit::Tensor L2aa = cas_ref.g2aa();
        ambit::Tensor L2ab = cas_ref.g2ab();
        ambit::Tensor L2bb = cas_ref.g2bb();

        gamma2("u, v, x, y") = L2aa("u, v, x, y") + L2bb("u, v, x, y") + L2ab("u, v, x, y") + L2ab("v, u, y, x");
    }
    else if(cas_type=="CAS")
    {
        ambit::Tensor L2aa = cas_ref.L2aa();
        ambit::Tensor L2ab = cas_ref.L2ab();
        ambit::Tensor L2bb = cas_ref.L2bb();
        ambit::Tensor L1a  = cas_ref.L1a();
        ambit::Tensor L1b  = cas_ref.L1b();

        L2aa("p,q,r,s") += L1a("p,r") * L1a("q,s");
        L2aa("p,q,r,s") -= L1a("p,s") * L1a("q,r");

        L2ab("pqrs") +=  L1a("pr") * L1b("qs");
        //L2ab("pqrs") += L1b("pr") * L1a("qs");

        L2bb("pqrs") += L1b("pr") * L1b("qs");
        L2bb("pqrs") -= L1b("ps") * L1b("qr");

        // This may or may not be correct.  Really need to find a way to check this code
        gamma2.copy(L2aa);
        gamma2("u,v,x,y") +=  L2ab("u,v,x,y");
        //gamma2("u,v,x,y") +=  L2ab("v, u, y, x");
        //gamma2("u,v,x,y") +=  L2bb("u,v,x,y");

        //gamma2_("u,v,x,y") = gamma2_("x,y,u,v");
        //gamma2_("u,v,x,y") = gamma2_("")
        gamma2.scale(2.0);

    }

    double E_casscf = 0.0;

    std::vector<size_t> na_array = mo_space_info_->get_corr_abs_mo("ACTIVE");

    ambit::Tensor tei_ab = ints_->aptei_ab_block(na_array, na_array, na_array, na_array);

    for (size_t p = 0; p < na_array.size(); ++p){
        for (size_t q = 0; q < na_array.size(); ++q){
            E_casscf += gamma1.data()[na_ * p + q] * fci_ints->oei_a(p,q);
        }
    }

    E_casscf += 0.5 * gamma2("u, v, x, y") * tei_ab("u, v, x, y");
    E_casscf += ints_->frozen_core_energy();
    E_casscf += fci_ints->scalar_energy();
    E_casscf += Process::environment.molecule()->nuclear_repulsion_energy();
    return E_casscf;

}
boost::shared_ptr<Matrix> CASSCF::set_frozen_core_orbitals()
{
    Dimension nmopi = mo_space_info_->get_dimension("ALL");
    Dimension frozen_dim = mo_space_info_->get_dimension("FROZEN_DOCC");
    SharedMatrix C_core(new Matrix("C_core",nirrep_, nmopi, frozen_dim));
    // Need to get the frozen block of the C matrix
    for(size_t h = 0; h < nirrep_; h++){
        for(int mu = 0; mu < nmopi[h]; mu++){
            for(int i = 0; i < frozen_dim[h]; i++){
                C_core->set(h, mu, i, Ca_sym_->get(h, mu, i));
            }
        }
    }

    boost::shared_ptr<JK> JK_core = JK::build_JK();

    JK_core->set_memory(Process::environment.get_memory() * 0.8);
    /// Already transform everything to C1 so make sure JK does not do this.
    //JK_core->set_allow_desymmetrization(false);

    /////TODO: Make this an option in my code
    //JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
    JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
    JK_core->initialize();

    std::vector<boost::shared_ptr<Matrix> >&Cl = JK_core->C_left();

    Cl.clear();
    Cl.push_back(C_core);

    JK_core->compute();

    SharedMatrix F_core = JK_core->J()[0];
    SharedMatrix K_core = JK_core->K()[0];

    F_core->scale(2.0);
    F_core->subtract(K_core);

    return F_core;

}

}}
