#include "casscf.h"
#include "reference.h"
#include "integrals.h"
#include <libqt/qt.h>
#include <libmints/matrix.h>
#include "helpers.h"
#include <libfock/jk.h>
#include "fci_solver.h"
#include <psifiles.h>
#include "fci_mo.h"
#include "orbitaloptimizer.h"
#include <libdiis/diismanager.h>
#include <libdiis/diisentry.h>
#include <libmints/factory.h>
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
    /// FrozenCore C Matrix is never rotated
    /// Can bring this out of loop
    if(nfrozen_ > 0)
    {
        F_froze_ = set_frozen_core_orbitals();
    }

    ///Setup the DIIS manager
    int diis_freq = options_.get_int("CASSCF_DIIS_FREQ");
    int diis_start = options_.get_int("CASSCF_DIIS_START");
    int diis_max_vec = options_.get_int("CASSCF_DIIS_MAX_VEC");
    double hessian_scale_value = options_.get_double("CASSCF_MAX_HESSIAN");
    bool   scale_hessian       = options_.get_bool("CASSCF_SCALE_HESSIAN");
    bool do_diis = options_.get_bool("CASSCF_DO_DIIS");

    Dimension all_nmopi = wfn_->nmopi();
    SharedMatrix S(new Matrix("Orbital Rotation", nirrep_, all_nmopi, all_nmopi));
    SharedMatrix Sstep;

    std::shared_ptr<DIISManager> diis_manager(new DIISManager(diis_max_vec, "MCSCF DIIS", DIISManager::OldestAdded, DIISManager::InCore));
    diis_manager->set_error_vector_size(1, DIISEntry::Matrix, S.get());
    diis_manager->set_vector_size(1, DIISEntry::Matrix, S.get());

    int diis_count = 0;

    print_h2("CASSCF Iteration");
    outfile->Printf("\n iter    ||g||           Delta_E            E_CASSCF       CONV_TYPE");

    E_casscf_ =  Process::environment.globals["SCF ENERGY"];
    outfile->Printf("\n E_casscf: %8.8f", E_casscf_);
    double E_casscf_old = 0.0;
    //tei_paaa_ = transform_integrals();
    for(int iter = 0; iter < maxiter; iter++)
    {
       iter_con.push_back(iter);

        /// Perform a CAS-CI using either York's code or Francesco's
        /// If CASSCF_DEBUG_PRINTING is on, will compare CAS-CI with SPIN-FREE RDM
        E_casscf_old = E_casscf_;
        cas_ci();
        SharedMatrix Ca = wfn_->Ca();
        SharedMatrix Cb = wfn_->Cb();

        OrbitalOptimizer orbital_optimizer(gamma1_,
                                           gamma2_,
                                           ints_->aptei_ab_block(nmo_abs_, active_abs_, active_abs_, active_abs_),
                                           options_,
                                           mo_space_info_);

        orbital_optimizer.set_frozen_one_body(F_froze_);
        orbital_optimizer.set_symmmetry_mo(Ca);
        orbital_optimizer.one_body(ints_->OneBodyAO());

        orbital_optimizer.update();
        double g_norm = orbital_optimizer.orbital_gradient_norm();

        if(g_norm < options_.get_double("CASSCF_G_CONVERGENCE") && fabs(E_casscf_old - E_casscf_)  < options_.get_double("CASSCF_E_CONVERGENCE")&& iter > 1)
        {
            outfile->Printf("\n\n @E_CASSCF: = %12.12f \n\n",E_casscf_ );
            outfile->Printf("\n Norm of orbital_gradient is %8.8f", g_norm);
            outfile->Printf("\n\n Energy difference: %12.12f", fabs(E_casscf_old - E_casscf_));
            break;
        }

        Sstep = orbital_optimizer.approx_solve();
        ///"Borrowed"(Stolen) from Daniel Smith's code.
        double maxS = 0.0;
        for (int h = 0; h < Sstep->nirrep(); h++){
            for(int i = 0; i < Sstep->rowspi()[h]; i++){
                for(int j = 0; j < Sstep->colspi()[h]; j++){
                    if( fabs(Sstep->get(h, i, j)) > maxS) maxS = fabs(Sstep->get(h, i, j));
                }
            }
        }
        if(maxS > hessian_scale_value && scale_hessian){
            Sstep->scale(hessian_scale_value / maxS);
        }
        //Sstep->print();
        // Add step to overall rotation
        S->copy(Sstep);
        SharedMatrix Cp = orbital_optimizer.rotate_orbitals(Ca, Sstep);

        // TODO:  Add options controlled.  Iteration and g_norm
        if(do_diis && (iter > diis_start && g_norm < options_.get_double("CASSCF_G_CONVERGENCE")))
        {
            diis_manager->add_entry(2, Sstep.get(), S.get());
            diis_count++;
        }

        if(do_diis && (!(diis_count % diis_freq) && iter > diis_start))
        {
            diis_manager->extrapolate(1, S.get());
        }


        Cp->set_name("Updated C");
        if(casscf_debug_print_)
        {
            Cp->print();
            Sstep->set_name("OrbitalRotationMatrix");
            Sstep->print();
        }

        ///ENFORCE Ca = Cb
        Ca->copy(Cp);
        Cb->copy(Cp);
        ///Right now, this retransforms all the integrals
        ///Think of ways of avoiding this
        ///This is used as way to retransform our integrals so Francesco's FCI code can use the updated CI
        ///Can redefi

        ints_->retransform_integrals();
        //tei_paaa_ = transform_integrals();
        //ambit::Tensor active_trans_int = ints_->aptei_ab_block(active_abs_, active_abs_, active_abs_, active_abs_);
        //active_trans_int.print(stdout);



        std::string diis_start_label = "";
        if(iter >= diis_start && do_diis==true && g_norm < 1e-4){diis_start_label = "DIIS";}
        outfile->Printf("\n %4d   %10.12f   %10.12f   %10.12f   %4s", iter, g_norm, fabs(E_casscf_ - E_casscf_old), E_casscf_, diis_start_label.c_str());
    }
    diis_manager->delete_diis_file();
    diis_manager.reset();

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
    nirrep_ = wfn_->nirrep();

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
void CASSCF::cas_ci()
{
    ///Calls francisco's FCI code and does a CAS-CI with the active given in the input
    SharedMatrix gamma2_matrix(new Matrix("gamma2", na_*na_, na_*na_));
    if(options_.get_str("CAS_TYPE") == "FCI")
    {

        //Used to grab the computed energy and RDMs.
        set_up_fci();
        if(options_.get_bool("CASSCF_DEBUG_PRINTING"))
        {
            double E_casscf_check = cas_check(cas_ref_);
            outfile->Printf("\n E_casscf_check - E_casscf_ = difference\n");
            outfile->Printf("\n %8.8f - %8.8f = %8.8f", E_casscf_check, E_casscf_, E_casscf_check - E_casscf_);
        }
        ambit::Tensor L2aa = cas_ref_.g2aa();
        ambit::Tensor L2ab = cas_ref_.g2ab();
        ambit::Tensor L2bb = cas_ref_.g2bb();

        ambit::Tensor gamma2 = ambit::Tensor::build(ambit::CoreTensor, "gamma2", {na_, na_, na_, na_});

        //// This may or may not be correct.  Really need to find a way to check this code
        gamma2("u, v, x, y") = L2aa("u, v, x, y") + L2ab("u, v, x, y") + L2ab("v, u, y, x") + L2bb("u, v, x, y");
        gamma2_ = ambit::Tensor::build(ambit::CoreTensor, "gamma2", {na_, na_, na_, na_});
        gamma2_("u, v, x, y") = gamma2("u, v, x, y") + gamma2("v, u, x, y") + gamma2("u, v, y, x") + gamma2("v, u, y, x");
        gamma2_.scale(0.25);
        gamma2_.iterate([&](const std::vector<size_t>& i,double& value){
            gamma2_matrix->set(i[0] * i[1] + i[1], i[2] * i[3] + i[3], value);});


    }
    else if(options_.get_str("CAS_TYPE") == "CAS")
    {
        FCI_MO cas(wfn_, options_, ints_, mo_space_info_);
        cas.use_default_orbitals(true);
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

        ambit::Tensor gamma2 = ambit::Tensor::build(ambit::CoreTensor, "gamma2", {na_, na_, na_, na_});

        // This may or may not be correct.  Really need to find a way to check this code
        gamma2("u,v,x,y") +=  L2aa("u,v,x, y");
        gamma2("u,v,x,y") +=  L2ab("u,v,x,y");
        //gamma2("u,v,x,y") +=  L2ab("v, u, y, x");
        //gamma2("u,v,x,y") +=  L2bb("u,v,x,y");

        //gamma2_("u,v,x,y") = gamma2_("x,y,u,v");
        //gamma2_("u,v,x,y") = gamma2_("")
        gamma2_ = ambit::Tensor::build(ambit::CoreTensor, "gamma2", {na_, na_, na_, na_});
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
    ambit::Tensor gamma_no_spin = ambit::Tensor::build(ambit::CoreTensor,"Return",{na_, na_});
    gamma1_ = ambit::Tensor::build(ambit::CoreTensor,"Return",{na_, na_});
    ambit::Tensor gamma1a = cas_ref_.L1a();
    ambit::Tensor gamma1b = cas_ref_.L1b();

    gamma_no_spin("i,j") = (gamma1a("i,j") + gamma1b("i,j"));

    gamma1_ = gamma_no_spin;

}

double CASSCF::cas_check(Reference cas_ref)
{
    ambit::Tensor gamma1 = ambit::Tensor::build(ambit::CoreTensor, "Gamma1", {na_, na_});
    ambit::Tensor gamma2 = ambit::Tensor::build(ambit::CoreTensor, "Gamma2", {na_, na_, na_, na_});
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
    SharedMatrix Ca = wfn_->Ca();
    Dimension nmopi = mo_space_info_->get_dimension("ALL");
    Dimension frozen_dim = mo_space_info_->get_dimension("FROZEN_DOCC");
    SharedMatrix C_core(new Matrix("C_core",nirrep_, nmopi, frozen_dim));
    // Need to get the frozen block of the C matrix
    for(size_t h = 0; h < nirrep_; h++){
        for(int mu = 0; mu < nmopi[h]; mu++){
            for(int i = 0; i < frozen_dim[h]; i++){
                C_core->set(h, mu, i, Ca->get(h, mu, i));
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
ambit::Tensor CASSCF::transform_integrals()
{
    ///This function will do an integral transformation using the JK builder
    /// This was borrowed from Kevin Hannon's IntegralTransform Plugin
    SharedMatrix Identity(new Matrix("I",nmo_ ,nmo_));
    Identity->identity();
    SharedMatrix CAct(new Matrix("CAct", nsopi_.sum(), na_));
    auto active_abs = mo_space_info_->get_absolute_mo("ACTIVE");

    ///Step 1: Obtain guess MO coefficients C_{mup}
    /// Since I want to use these in a symmetry aware basis,
    /// I will move the C matrix into a Pfitzer ordering

    Dimension nmopi = mo_space_info_->get_dimension("ALL");

    SharedMatrix aotoso = wfn_->aotoso();

    /// I want a C matrix in the C1 basis but symmetry aware
    size_t nso = wfn_->nso();
    nirrep_ = wfn_->nirrep();
    SharedMatrix Call(new Matrix(nso, nmopi.sum()));
    SharedMatrix Ca_sym = wfn_->Ca();

    // Transform from the SO to the AO basis for the C matrix.
    // just transfroms the C_{mu_ao i} -> C_{mu_so i}
    for (size_t h = 0, index = 0; h < nirrep_; ++h){
        for (int i = 0; i < nmopi[h]; ++i){
            size_t nao = nso;
            size_t nso = nsopi_[h];

            if (!nso) continue;

            C_DGEMV('N',nao,nso,1.0,aotoso->pointer(h)[0],nso,&Ca_sym->pointer(h)[0][i],nmopi[h],0.0,&Call->pointer()[0][index],nmopi.sum());

            index += 1;
        }

    }

    for(int mu = 0; mu < nsopi_.sum(); mu++){
        for(size_t v = 0; v < na_;v++){
            CAct->set(mu, v, Call->get(mu, active_abs[v]));
        }
    }

    ambit::Tensor active_int = ambit::Tensor::build(ambit::CoreTensor, "Gamma2", {nmo_, na_, na_, na_});
    std::vector<double>& active_int_data = active_int.data();
    for(size_t i = 0; i < na_; i++){
        SharedVector C_i = CAct->get_column(0, i);
        for(size_t j = 0; j < na_; j++){
            SharedMatrix D(new Matrix("D", nmo_, nmo_));
            SharedVector C_j = CAct->get_column(0, j);
            C_DGER(nmo_, nmo_, 1.0, &(C_i->pointer()[0]), 1, &(C_j->pointer()[0]), 1, D->pointer()[0], nmo_);
            ///Form D = C_rC_s'
            ///Use this to compute J matrix
            /// (pq | rs) = C J C'

            boost::shared_ptr<JK> JK_trans = JK::build_JK();
            JK_trans->set_memory(Process::environment.get_memory() * 0.8);
            JK_trans->initialize();

            std::vector<boost::shared_ptr<Matrix> > &Cl = JK_trans->C_left();
            std::vector<boost::shared_ptr<Matrix> > &Cr = JK_trans->C_right();
            Cl.clear();
            Cr.clear();
            Cl.push_back(D);
            Cr.push_back(Identity);
            JK_trans->set_allow_desymmetrization(false);
            JK_trans->set_do_K(false);
            JK_trans->compute();

            SharedMatrix J = JK_trans->J()[0];
            SharedMatrix half_trans(new Matrix("Trans", nmo_, na_));
            half_trans = Matrix::triplet(Call, J, CAct, true, false, false);
            for(size_t p = 0; p < nmo_; p++){
                for(size_t q = 0; q < na_; q++){
                    active_int_data[p * na_ * na_ * na_ + i * na_ * na_ + q * na_ + j] = half_trans->get(p, q);
                }
            }
        }
    }
    return active_int;

}
void CASSCF::set_up_fci()
{
    Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
    size_t nfdocc = mo_space_info_->size("FROZEN_DOCC");
    std::vector<size_t> rdocc = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> active = mo_space_info_->get_corr_abs_mo("ACTIVE");

    int charge = Process::environment.molecule()->molecular_charge();
    if(options_["CHARGE"].has_changed()){
        charge = options_.get_int("CHARGE");
    }

    int nel = 0;
    int natom = Process::environment.molecule()->natom();
    for(int i=0; i < natom;i++){
        nel += static_cast<int>(Process::environment.molecule()->Z(i));
    }
    // If the charge has changed, recompute the number of electrons
    // Or if you cannot find the number of electrons
    nel -= charge;

    int multiplicity = Process::environment.molecule()->multiplicity();
    if(options_["MULTIPLICITY"].has_changed()){
        multiplicity = options_.get_int("MULTIPLICITY");
    }

    // Default: lowest spin solution
    int ms = (multiplicity + 1) % 2;

    if(options_["MS"].has_changed()){
        ms = options_.get_int("MS");
    }

    if(ms < 0){
        outfile->Printf("\n  Ms must be no less than 0.");
        outfile->Printf("\n  Ms = %2d, MULTIPLICITY = %2d", ms, multiplicity);
        outfile->Printf("\n  Check (specify) Ms value (component of multiplicity)! \n");
        throw PSIEXCEPTION("Ms must be no less than 0. Check output for details.");
    }

    if (options_.get_int("PRINT")){
        outfile->Printf("\n  Number of electrons: %d",nel);
        outfile->Printf("\n  Charge: %d",charge);
        outfile->Printf("\n  Multiplicity: %d",multiplicity);
        outfile->Printf("\n  Davidson subspace max dim: %d",options_.get_int("DAVIDSON_SUBSPACE_PER_ROOT"));
        outfile->Printf("\n  Davidson subspace min dim: %d",options_.get_int("DAVIDSON_COLLAPSE_PER_ROOT"));
        if (ms % 2 == 0){
            outfile->Printf("\n  M_s: %d",ms / 2);
        }else{
            outfile->Printf("\n  M_s: %d/2",ms);
        }
    }

    if( ((nel - ms) % 2) != 0)
        throw PSIEXCEPTION("\n\n  FCI: Wrong value of M_s.\n\n");

    // Adjust the number of for frozen and restricted doubly occupied
    size_t nactel = nel - 2 * nfdocc - 2 * rdocc.size();

    size_t na = (nactel + ms) / 2;
    size_t nb =  nactel - na;

    FCISolver fcisolver(active_dim,rdocc,active,na,nb,multiplicity,options_.get_int("ROOT_SYM"),ints_, mo_space_info_,
                               options_.get_int("NTRIAL_PER_ROOT"),options_.get_int("PRINT"), options_);
    // tweak some options
    fcisolver.set_max_rdm_level(2);
    fcisolver.set_nroot(options_.get_int("NROOT"));
    fcisolver.set_root(options_.get_int("ROOT"));
    fcisolver.test_rdms(options_.get_bool("TEST_RDMS"));
    fcisolver.set_fci_iterations(options_.get_int("FCI_ITERATIONS"));
    fcisolver.set_collapse_per_root(options_.get_int("DAVIDSON_COLLAPSE_PER_ROOT"));
    fcisolver.set_subspace_per_root(options_.get_int("DAVIDSON_SUBSPACE_PER_ROOT"));
    fcisolver.print_no(false);
    fcisolver.use_user_integrals_and_restricted_docc(true);

    std::shared_ptr<FCIIntegrals> fci_ints = std::make_shared<FCIIntegrals>(ints_, active, rdocc);

    ambit::Tensor active_aa = ints_->aptei_aa_block(active, active, active, active);
    ambit::Tensor active_ab = ints_->aptei_ab_block(active, active, active, active);
    ambit::Tensor active_bb = ints_->aptei_bb_block(active, active, active, active);
    fci_ints->set_active_integrals(active_aa, active_ab, active_bb);
    std::vector<std::vector<double> > oei_vector = compute_restricted_docc_operator();
    fci_ints->set_restricted_one_body_operator(oei_vector[0], oei_vector[1]);
    fci_ints->set_scalar_energy(scalar_energy_);
    fcisolver.set_integral_pointer(fci_ints);


    E_casscf_ = fcisolver.compute_energy();
    cas_ref_ = fcisolver.reference();
}
std::vector<std::vector<double> > CASSCF::compute_restricted_docc_operator()
{
    Dimension restricted_docc_dim = mo_space_info_->get_dimension("INACTIVE_DOCC");
    Dimension nsopi           = Process::environment.wavefunction()->nsopi();
    int nirrep               = Process::environment.wavefunction()->nirrep();
    Dimension nmopi = mo_space_info_->get_dimension("ALL");

    SharedMatrix Cdocc(new Matrix("C_RESTRICTED", nirrep, nsopi, restricted_docc_dim));
    SharedMatrix Ca = Process::environment.wavefunction()->Ca();
    for(int h = 0; h < nirrep; h++)
    {
        for(int mu = 0; mu < nsopi[h]; mu++)
        {
            for(int i = 0; i < restricted_docc_dim[h]; i++)
            {
                Cdocc->set(h, mu, i, Ca->get(h, mu, i));
            }
        }
    }
    ///F_frozen = D_{uv}^{frozen} * (2<uv|rs> - <ur | vs>)
    ///F_restricted = D_{uv}^{restricted} * (2<uv|rs> - <ur | vs>)
    ///F_inactive = F_frozen + F_restricted + H_{pq}^{core}
    /// D_{uv}^{frozen} = \sum_{i = 0}^{frozen}C_{ui} * C_{vi}
    /// D_{uv}^{inactive} = \sum_{i = 0}^{inactive}C_{ui} * C_{vi}
    /// This section of code computes the fock matrix for the INACTIVE_DOCC("RESTRICTED_DOCC")

    boost::shared_ptr<JK> JK_inactive = JK::build_JK();

    JK_inactive->set_memory(Process::environment.get_memory() * 0.8);
    JK_inactive->initialize();

    std::vector<boost::shared_ptr<Matrix> >&Cl = JK_inactive->C_left();
    Cl.clear();
    Cl.push_back(Cdocc);
    JK_inactive->compute();
    SharedMatrix J_restricted = JK_inactive->J()[0];
    SharedMatrix K_restricted = JK_inactive->K()[0];

    J_restricted->scale(2.0);
    SharedMatrix F_restricted = J_restricted->clone();
    F_restricted->subtract(K_restricted);

    boost::shared_ptr<PSIO> psio_ = PSIO::shared_object();
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    SharedMatrix T = SharedMatrix(wfn->matrix_factory()->create_matrix(PSIF_SO_T));
    SharedMatrix V = SharedMatrix(wfn->matrix_factory()->create_matrix(PSIF_SO_V));
    SharedMatrix OneInt = T;
    OneInt->zero();

    T->load(psio_, PSIF_OEI);
    V->load(psio_, PSIF_OEI);
    SharedMatrix H(T->clone());
    H->add(V);

    F_restricted->add(H);
    F_restricted->transform(Ca);
    H->transform(Ca);
    size_t all_nmo = mo_space_info_->size("ALL");
    SharedMatrix F_restric_c1(new Matrix("F_restricted", all_nmo, all_nmo));
    size_t offset = 0;
    for(int h = 0; h < nirrep; h++){
        for(int p = 0; p < nmopi[h]; p++){
            for(int q = 0; q < nmopi[h]; q++){
                F_restric_c1->set(p + offset, q + offset, F_restricted->get(h, p, q ));
            }
        }
        offset += nmopi[h];
    }
    size_t nmo2 = na_ * na_;
    std::vector<double> oei_a(nmo2);
    std::vector<double> oei_b(nmo2);

    auto absolute_active = mo_space_info_->get_absolute_mo("ACTIVE");
    for(size_t u = 0; u < na_; u++){
        for(size_t v = 0; v < na_; v++){
            double value = F_restric_c1->get(absolute_active[u], absolute_active[v]);
            //double h_value = H->get(absolute_active[u], absolute_active[v]);
            oei_a[u * na_ + v ] = value;
            oei_b[u * na_ + v ] = value;
        }
    }
    Dimension restricted_docc = mo_space_info_->get_dimension("INACTIVE_DOCC");
    double E_restricted = 0.0;
    for(int h = 0; h < nirrep; h++){
        for(int rd = 0; rd < restricted_docc[h]; rd++){
            E_restricted += H->get(h, rd, rd) + F_restricted->get(h, rd, rd);
        }
    }
    /// Since F^{INACTIVE} includes frozen_core in fock build, the energy contribution includes frozen_core_energy
    if(casscf_debug_print_)
    {
        outfile->Printf("\n Inactive Energy = %8.8f", E_restricted - ints_->frozen_core_energy());
    }
    scalar_energy_ = ints_->scalar();
    scalar_energy_ += E_restricted - ints_->frozen_core_energy();
    std::vector<std::vector<double> > oei_container;
    oei_container.push_back(oei_a);
    oei_container.push_back(oei_b);
    return oei_container;

}

}}
