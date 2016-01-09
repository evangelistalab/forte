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
    Timer overall_update;

    Timer fock_core;
    form_fock_core();
    if(timings_){outfile->Printf("\n\n FormFockCore took %8.8f s.", fock_core.get());}

    /// F^{A}_{pq} = \gamma_{uv} (pq | uv) - 1/2 (pu |qv)
    /// First a Cholesky Decomposition is performed on Gamma (maybe isn't necessary)
    /// Use JK builder: J(\gamma_{uv} - 1/2 K(\gamma_{uv})
    Timer fock_active;
    form_fock_active();
    if(timings_){outfile->Printf("\n\n FormFockActive took %8.8f s.", fock_active.get());}

    Timer orbital_grad;
    orbital_gradient();
    if(timings_){outfile->Printf("\n\n FormOrbitalGradient took %8.8f s.", orbital_grad.get());}

    Timer diag_hess;
    diagonal_hessian();
    if(timings_){outfile->Printf("\n\n FormDiagHessian took %8.8f s.", diag_hess.get());}

    if(timings_){outfile->Printf("\n\n Update function takes %8.8f s", overall_update.get());}

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
    SharedMatrix F_core(new Matrix("InactiveTemp", nirrep_, nmopi_, nmopi_));
    F_core_c1->zero();

    ///If there is no restricted_docc, there is no C_core
    if(restricted_docc_dim_.sum() > 0)
    {
        for(size_t h = 0; h < nirrep_; h++){
            for(int i = 0; i <  restricted_docc_dim_[h]; i++){
                C_core->set_column(h,i, Ca_sym_->get_column(h, i + frozen_docc_dim_[h]));
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
        F_core = J_core->clone();
        F_core->subtract(K_core);
    }

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

    for(size_t u = 0; u <  na_; u++){
            C_active->set_column(0, u, Call_->get_column(0, active_abs_corr[u]));
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
            L_C_correct->set_row(0, mu, L_C->get_column(0, mu));
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
    if(nrdocc_ > 0 or nfrozen_ > 0)
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
    if(casscf_debug_print_)
    {
        outfile->Printf("\n\n integral_: %8.8f  gamma2_: %8.8f", integral_.norm(2), gamma2_.norm(2));
    }

    SharedMatrix Zm(new Matrix("Zm", nmo_, na_));
    Z.iterate([&](const std::vector<size_t>& i,double& value){
        Zm->set(i[0],i[1], value);});

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


    size_t nhole = nrdocc_ + na_;
    size_t npart = na_ + nvir_;
    SharedMatrix Orb_grad(new Matrix("G_pq", nhole, npart));
    Orb_grad->set_name("CASSCF Gradient");

    auto generalized_hole_abs = mo_space_info_->get_corr_abs_mo("GENERALIZED HOLE");
    auto generalized_part_abs = mo_space_info_->get_corr_abs_mo("GENERALIZED PARTICLE");
    Dimension general_hole_dim = mo_space_info_->get_dimension("GENERALIZED HOLE");
    Dimension general_part_dim = mo_space_info_->get_dimension("GENERALIZED PARTICLE");
    auto generalized_hole_rel = mo_space_info_->get_relative_mo("GENERALIZED HOLE");
    auto generalized_part_rel = mo_space_info_->get_relative_mo("GENERALIZED PARTICLE");
    if(casscf_debug_print_)
    {
        outfile->Printf("Generalized_hole_abs\n");
        for(auto gha : generalized_hole_abs)
        {
            outfile->Printf(" %d", gha);
        }
        outfile->Printf("\n Generalized_part_abs\n");
        for(auto gpa : generalized_part_abs)
        {
            outfile->Printf(" %d", gpa);
        }
        outfile->Printf("Generalized_hole_rel\n");
        for(auto ghr : generalized_hole_rel)
        {
            outfile->Printf(" (%d, %d) ", ghr.first, ghr.second);
        }
        outfile->Printf("Generalized_part_rel\n");
        for(auto gpr : generalized_part_rel)
        {
            outfile->Printf(" (%d, %d) ", gpr.first, gpr.second);
        }
    }
    int offset_hole = 0;
    int offset_part = 0;
    std::vector<size_t> hole_offset_vector(nirrep_);
    std::vector<size_t> particle_offset_vector(nirrep_);
    for(size_t h = 0; h < nirrep_; h++){
            hole_offset_vector[h] = offset_hole;
            particle_offset_vector[h] = offset_part;
            offset_hole += general_hole_dim[h];
            offset_part += general_part_dim[h];
    }

    /// Create a map that takes absolute index (in nmo indexing) and returns a value that corresponds to either hole or particle
    /// IE: 0, 1, 2, 9 for hole where 2 and 9 are active
    /// This map returns 0, 1, 2, 3
    for(size_t i = 0; i < nhole; i++)
    {
        int sym_irrep = generalized_hole_rel[i].first;
        //nhole_map_[generalized_hole_abs[i]] = (generalized_hole_rel[i].second + restricted_docc_dim_[generalized_hole_rel[i].first]);
        /// generalized_hole_rel[i].second tells where the entry is in with respect to the irrep
        /// hole_offset is an offset that accounts for the previous irreps
        nhole_map_[generalized_hole_abs[i]] = generalized_hole_rel[i].second
                + hole_offset_vector[sym_irrep]
                - frozen_docc_dim_[sym_irrep];
    }
    for(size_t a = 0; a < npart; a++)
    {
        int sym_irrep = generalized_part_rel[a].first;
        /// generalized_part_rel[a].second tells where the entry is in with respect to the irrep
        /// part_offset_vector is an offset that accounts for the previous irreps
        npart_map_[generalized_part_abs[a]] = (generalized_part_rel[a].second
                                               + particle_offset_vector[sym_irrep]
                                               - restricted_docc_dim_[sym_irrep])
                                               - frozen_docc_dim_[sym_irrep];
    }

    if(casscf_debug_print_)
    {
        for(auto hole : nhole_map_)
        {
            outfile->Printf("\n nhole_map[%d] = %d", hole.first, hole.second);
        }
        for(auto part : npart_map_)
        {
            outfile->Printf("\n npart_map[%d] = %d", part.first, part.second);
        }
    }


    ///Some wierdness going on
    /// Since G is nhole by npart
    /// I need to make sure that the matrix is ordered in pitzer ordering
    /// The offset allows me to place the correct values with pitzer ordering

    if(nrdocc_ > 0)
    {
        for(size_t ii = 0; ii < nrdocc_; ii++){
            for(size_t ti = 0; ti < na_; ti++){
                {
                    size_t i = restricted_docc_abs_[ii];
                    size_t t = active_abs_[ti];
                    //size_t ti_offset = generalized_part_rel[ti].second + gen_part[generalized_part_rel[ti].first];
                    //double value_it = 4 * F_core_->get(i, t) + 2 * F_act_->get(i,t) - 2 * Y_->get(i,ti) - 4 * Z_->get(i, ti);
                    double value_it = 4 * F_core_->get(i, t) + 4 * F_act_->get(i,t) - 2 * Y_->get(i,ti) - 2 * Z_->get(i, ti);
                    Orb_grad->set(nhole_map_[i] ,npart_map_[t], value_it) ;
                }
            }
        }
    }
    if(casscf_debug_print_)
    {
        outfile->Printf("\n i, t %8.8f", Orb_grad->rms());
        Orb_grad->print();
    }
    if(nrdocc_ > 0)
    {
        for(size_t ii = 0; ii < nrdocc_; ii++){
            for(size_t aa = 0; aa < nvir_; aa++){
                size_t i = restricted_docc_abs_[ii];
                size_t a = restricted_uocc_abs_[aa];
                //size_t a_offset = generalized_part_rel[aa + na_].second + gen_part[generalized_part_rel[aa + na_].first];
                double value_ia = F_core_->get(i, a) * 4.0 + F_act_->get(i, a) * 4.0;
                Orb_grad->set(nhole_map_[i],npart_map_[a], value_ia);
            }
        }
    }
    if(casscf_debug_print_)
    {
        outfile->Printf("\n i, a %8.8f", Orb_grad->rms());
        Orb_grad->print();
    }

    for(size_t ai = 0; ai < nvir_; ai++){
        for(size_t ti = 0; ti < na_; ti++){
            size_t t = active_abs_[ti];
            size_t a = restricted_uocc_abs_[ai];
            //size_t a_offset = generalized_part_rel[ti + na_].second + gen_part[generalized_part_rel[ti + na_].first];
            double value_ta = 2.0 * Y_->get(a, ti) + 2.0 * Z_->get(a, ti);
            Orb_grad->set(nhole_map_[t],npart_map_[a], value_ta);
        }
    }
    if(casscf_debug_print_)
    {
        outfile->Printf("\n t, a %8.8f", Orb_grad->rms());
        Orb_grad->print();
    }


    for(size_t ui = 0; ui < na_; ui++){
        for(size_t v = 0; v < na_; v++){
            size_t u = active_abs_[ui];
            size_t vo = active_abs_[v];
            Orb_grad->set(nhole_map_[u],npart_map_[vo] , 0.0);
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
    size_t nhole = nrdocc_ + na_;
    size_t npart = na_ + nvir_;
    SharedMatrix D(new Matrix("D_pq", nhole, npart));
    D->set_name("Diagonal Hessian");

    for(size_t ii = 0; ii < nrdocc_; ii++){
        for(size_t ai = 0; ai < nvir_; ai++){
            size_t a = restricted_uocc_abs_[ai];
            size_t i = restricted_docc_abs_[ii];
            //double value_ia = F_core_->get(a,a) * 4.0 + 2 * F_act_->get(a,a);
            //value_ia -= 4.0 * F_core_->get(i,i)  - 2 * F_act_->get(i,i);
            double value_ia = (F_core_->get(a,a) * 4.0 + 4.0 * F_act_->get(a,a));
            value_ia -= (4.0 * F_core_->get(i,i)  + 4.0 * F_act_->get(i,i));
            D->set(nhole_map_[i],npart_map_[a],value_ia);
        }
    }
    for(size_t ai = 0; ai < nvir_; ai++){
        for(size_t ti = 0; ti < na_; ti++){
            size_t a = restricted_uocc_abs_[ai];
            size_t t = active_abs_[ti];
            //double value_ta = 2.0 * gamma1M_->get(ti,ti) * F_core_->get(a,a);
            //value_ta += gamma1M_->get(ti,ti) * F_act_->get(a,a);
            //value_ta -= 2*Y_->get(t,ti) + 4.0 *Z_->get(t,ti);
            double value_ta = 2.0 * gamma1M_->get(ti,ti) * F_core_->get(a,a);
            value_ta += 2.0 * gamma1M_->get(ti,ti) * F_act_->get(a,a);
            value_ta -= (2*Y_->get(t,ti) + 2.0 *Z_->get(t,ti));
            D->set(nhole_map_[t],npart_map_[a], value_ta);
        }
    }
    for(size_t ii = 0; ii < nrdocc_; ii++){
        for(size_t ti = 0; ti < na_; ti++){
            size_t i = restricted_docc_abs_[ii];
            size_t t = active_abs_[ti];
            double value_it = 4.0 * F_core_->get(t,t)
                    + 4.0 * F_act_->get(t,t)
                    + 2.0 * gamma1M_->get(ti,ti) * F_core_->get(i,i);
            value_it+=2.0 * gamma1M_->get(ti,ti) * F_act_->get(i,i);
            value_it-=(4.0 * F_core_->get(i,i) + 4.0 * F_act_->get(i,i));
            value_it-=(2.0*Y_->get(t,ti) + 2.0 * Z_->get(t,ti));
            D->set(nhole_map_[i],npart_map_[t], value_it);
        }
    }

    for(size_t u = 0; u < na_; u++){
        for(size_t v = 0; v < na_; v++){
            size_t uo = active_abs_[u];
            size_t vo = active_abs_[v];
            D->set(nhole_map_[uo],npart_map_[vo], 1.0);
        }
    }
    d_ = D;
    if(casscf_debug_print_)
    {
        d_->print();
    }


}
SharedMatrix OrbitalOptimizer::approx_solve()
{
    Dimension nhole_dim = restricted_docc_dim_ + active_dim_;
    Dimension nvirt_dim = restricted_uocc_dim_ + active_dim_;

    SharedMatrix G_grad(new Matrix("GradientSym", nhole_dim, nvirt_dim));
    SharedMatrix D_grad(new Matrix("HessianSym", nhole_dim, nvirt_dim));

    int offset_hole = 0;
    int offset_part = 0;
    for(size_t h = 0; h < nirrep_; h++){
        for(int i = 0; i < nhole_dim[h]; i++){
            int ioff = i + offset_hole;
            for(int a = 0; a < nvirt_dim[h]; a++){
                int aoff = a + offset_part;
                G_grad->set(h, i, a, g_->get(ioff, aoff));
                D_grad->set(h, i, a, d_->get(ioff, aoff));
            }
        }
        offset_hole += nhole_dim[h];
        offset_part += nvirt_dim[h];
    }
    SharedMatrix S_tmp = G_grad->clone();
    S_tmp->apply_denominator(D_grad);
    for(size_t h = 0; h < nirrep_; h++)
    {
        for(int u = 0; u < active_dim_[h]; u++)
        {
            for(int v = 0; v < active_dim_[h]; v++)
            {
                S_tmp->set(h, restricted_docc_dim_[h] + u, v, 0.0);
            }
        }
    }

    if(casscf_debug_print_)
    {
        G_grad->print();
        D_grad->print();
        S_tmp->set_name("g / d");
        S_tmp->print();
    }
    return S_tmp;
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
    SharedMatrix HessianEvec(new Matrix("HessianEvec",  2*nmo_ + 1, 2*nmo_ + 1));
    SharedVector HessianEval(new Vector("HessianEval", 2*nmo_ + 1));
    AugmentedHessian->diagonalize(HessianEvec, HessianEval);
    if(casscf_debug_print_)
    {
        g_->print();
        d_->print();
        AugmentedHessian->print();
        HessianEval->print();
    }
    return HessianEvec;

}

SharedMatrix OrbitalOptimizer::rotate_orbitals(SharedMatrix C, SharedMatrix S)
{
    Dimension nhole_dim = mo_space_info_->get_dimension("GENERALIZED HOLE");
    Dimension nvirt_dim = mo_space_info_->get_dimension("GENERALIZED PARTICLE");
    ///Clone the C matrix
    SharedMatrix C_rot(C->clone());
    SharedMatrix S_mat(S->clone());
    SharedMatrix S_sym(new Matrix("Exp(K)", wfn_->nirrep(), wfn_->nmopi(), wfn_->nmopi()));
    SharedMatrix U = S_sym->clone();
    int offset_hole = 0;
    int offset_part = 0;
    for(size_t h = 0; h < nirrep_; h++){
        for(int i = 0; i < frozen_docc_dim_[h]; i++)
        {
            S_sym->set(h, i, i, 1.0);
        }
        for(int i = 0; i < nhole_dim[h]; i++){
            for(int a = std::max(restricted_docc_dim_[h], i); a < nmopi_[h]; a++){
                int ioff = i + frozen_docc_dim_[h];
                int aoff = a + frozen_docc_dim_[h];
                S_sym->set(h, ioff, aoff, S_mat->get(h, i, a - restricted_docc_dim_[h]));
                S_sym->set(h, aoff, ioff, -1.0 * S_mat->get(h, i, a - restricted_docc_dim_[h]));
            }
        }
    offset_hole += nhole_dim[h];
    offset_part += nvirt_dim[h];
    }
    S_sym->expm();
    for(size_t h = 0; h < nirrep_; h++)
    {
        for(int f = 0; f < frozen_docc_dim_[h]; f++)
        {
            S_sym->set(h, f, f, 1.0);
        }
    }


    C_rot = Matrix::doublet(C, S_sym);
    C_rot->set_name("ROTATED_ORBITAL");
    S_sym->set_name("Orbital Rotation (S = exp(x))");
    if(casscf_debug_print_)
    {
        C_rot->print();
        S_sym->print();
    }
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
