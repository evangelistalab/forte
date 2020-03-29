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

#include "casscf/casscf.h"
#include "integrals/integrals.h"
#include "base_classes/rdms.h"

#include "psi4/libfock/jk.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libqt/qt.h"
#include "psi4/psifiles.h"
#include "psi4/libmints/basisset.h"

#include "helpers/printing.h"
#include "helpers/helpers.h"
#include "helpers/timer.h"
#include "sci/aci.h"
#include "fci/fci_solver.h"
#include "base_classes/active_space_solver.h"

#include "sci/fci_mo.h"
#include "orbital-helpers/mp2_nos.h"
#include "orbital-helpers/orbitaloptimizer.h"
#include "orbital-helpers/semi_canonicalize.h"

#ifdef HAVE_CHEMPS2
#include "dmrg/dmrgsolver.h"
#endif
#include "psi4/libdiis/diisentry.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libmints/factory.h"

namespace forte {

CASSCF::CASSCF(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
               std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
               std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints), scf_info_(scf_info),
      options_(options), ints_(as_ints->ints()) {
    print_method_banner({"Complete Active Space Self Consistent Field", "Kevin Hannon"});
    startup();
}

void CASSCF::startup() {
    auto scf_type = options_->get_str("SCF_TYPE");
    if (scf_type == "PK" or scf_type == "OUT_OF_CORE") {
        outfile->Printf("\n\n  Please change SCF_TYPE to DIRECT for conventional integrals.");
        throw psi::PSIEXCEPTION("Please change SCF_TYPE to DIRECT.");
    }

    print_ = options_->get_int("PRINT");
    casscf_debug_print_ = options_->get_bool("CASSCF_DEBUG_PRINTING");

    nsopi_ = scf_info_->nsopi();
    nirrep_ = mo_space_info_->nirrep();

    // Set MOs containers.
    core_mos_abs_ = mo_space_info_->absolute_mo("RESTRICTED_DOCC");
    actv_mos_abs_ = mo_space_info_->absolute_mo("ACTIVE");
    core_mos_rel_ = mo_space_info_->get_relative_mo("RESTRICTED_DOCC");
    actv_mos_rel_ = mo_space_info_->get_relative_mo("ACTIVE");

    frzc_dim_ = mo_space_info_->dimension("FROZEN_DOCC");
    core_dim_ = mo_space_info_->dimension("RESTRICTED_DOCC");
    actv_dim_ = mo_space_info_->dimension("ACTIVE");
    virt_dim_ = mo_space_info_->dimension("RESTRICTED_UOCC");
    closed_dim_ = mo_space_info_->dimension("INACTIVE_DOCC");
    corr_dim_ = mo_space_info_->dimension("CORRELATED");
    nmo_dim_ = mo_space_info_->dimension("ALL");

    frzc_mos_ = mo_space_info_->corr_absolute_mo("FROZEN_DOCC");
    core_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->corr_absolute_mo("ACTIVE");
    virt_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");
    closed_mos_ = mo_space_info_->corr_absolute_mo("INACTIVE_DOCC");
    corr_mos_ = mo_space_info_->corr_absolute_mo("CORRELATED");

    ncmo_ = mo_space_info_->size("CORRELATED");
    nmo_ = mo_space_info_->size("ALL");
    ncore_ = core_mos_.size();
    nvirt_ = virt_mos_.size();
    nfrzc_ = frzc_mos_.size();
    nactv_ = mo_space_info_->size("ACTIVE");

    Hcore_ = SharedMatrix(ints_->wfn()->H()->clone());

    local_timer JK_initialize;
    if (options_->get_str("SCF_TYPE") == "GTFOCK") {
#ifdef HAVE_JK_FACTORY
        psi::Process::environment.set_legacy_molecule(ints_->wfn()->molecule());
        JK_ = std::shared_ptr<JK>(new GTFockJK(ints_->basisset()));
#else
        throw psi::PSIEXCEPTION("GTFock was not compiled in this version");
#endif
    } else {
        if (options_->get_str("SCF_TYPE") == "DF") {
            auto df_basis = ints_->get_basisset("DF_BASIS_SCF");
            JK_ = std::make_shared<DiskDFJK>(ints_->basisset(), df_basis);
        } else {
            JK_ = JK::build_JK(ints_->basisset(), psi::BasisSet::zero_ao_basis_set(),
                               psi::Process::environment.options);
        }
    }

    JK_->set_memory(psi::Process::environment.get_memory() * 0.85);
    JK_->initialize();
    JK_->C_left().clear();
    JK_->C_right().clear();

    if (print_ > 0)
        outfile->Printf("\n    JK takes %5.5f s to initialize while using %s", JK_initialize.get(),
                        scf_type.c_str());
}

double CASSCF::compute_energy() {
    if (nactv_ == 0) {
        outfile->Printf("\n\n\n Please set the active space");
        throw psi::PSIEXCEPTION(" The active space is zero.  Set the active space");
    } else if (nactv_ == ncmo_) {
        outfile->Printf("\n Your about to do an all active CASSCF");
        throw psi::PSIEXCEPTION("The active space is all the MOs.  Orbitals don't "
                                "matter at this point");
    }

    int maxiter = options_->get_int("CASSCF_ITERATIONS");

    // Provide a nice summary at the end for iterations
    std::vector<int> iter_con;

    // Frozen-core C Matrix is never rotated
    if (nfrzc_ > 0) {
        F_froze_ = set_frozen_core_orbitals();
    }

    // DIIS options
    bool do_diis = options_->get_bool("CASSCF_DO_DIIS");
    int diis_freq = options_->get_int("CASSCF_DIIS_FREQ");
    int diis_start = options_->get_int("CASSCF_DIIS_START");
    int diis_max_vec = options_->get_int("CASSCF_DIIS_MAX_VEC");
    double diis_gradient_norm = options_->get_double("CASSCF_DIIS_NORM");

    // CI step options
    bool ci_step = options_->get_bool("CASSCF_CI_STEP");
    int casscf_freq = options_->get_int("CASSCF_CI_FREQ");

    double rotation_max_value = options_->get_double("CASSCF_MAX_ROTATION");

    psi::Dimension nhole_dim = mo_space_info_->dimension("GENERALIZED HOLE");
    psi::Dimension npart_dim = mo_space_info_->dimension("GENERALIZED PARTICLE");
    psi::SharedMatrix S(new psi::Matrix("Orbital Rotation", nirrep_, nhole_dim, npart_dim));
    psi::SharedMatrix Sstep;

    // Setup the DIIS manager
    auto diis_manager = std::make_shared<DIISManager>(
        diis_max_vec, "MCSCF DIIS", DIISManager::OldestAdded, DIISManager::InCore);
    diis_manager->set_error_vector_size(1, DIISEntry::Matrix, S.get());
    diis_manager->set_vector_size(1, DIISEntry::Matrix, S.get());

    int diis_count = 0;

    E_casscf_ = 0.0;
    double E_casscf_old = 0.0, Ediff = 0.0;
    psi::SharedMatrix C_start(ints_->Ca()->clone());
    double econv = options_->get_double("CASSCF_E_CONVERGENCE");
    double gconv = options_->get_double("CASSCF_G_CONVERGENCE");

    psi::SharedMatrix Ca = ints_->Ca();
    psi::SharedMatrix Cb = ints_->Cb();
    print_h2("CASSCF Iteration");
    outfile->Printf("\n iter    ||g||           Delta_E            E_CASSCF       CONV_TYPE");

    for (int iter = 0; iter < maxiter; iter++) {
        local_timer casscf_total_iter;

        local_timer transform_integrals_timer;
        tei_paaa_ = transform_integrals();
        if (print_ > 0) {
            outfile->Printf("\n\n Transform Integrals takes %8.8f s.",
                            transform_integrals_timer.get());
        }
        iter_con.push_back(iter);

        // Perform a CASCI
        // If CASSCF_DEBUG_PRINTING is on, will compare CAS-CI with SPIN-FREE RDM
        E_casscf_old = E_casscf_;
        if (print_ > 0) {
            outfile->Printf("\n\n  Performing a CAS with %s",
                            options_->get_str("CASSCF_CI_SOLVER").c_str());
        }
        local_timer cas_timer;
        // Perform a DMRG-CI, ACI, FCI inside an active space
        // If casscf_freq is on, do the CI step every casscf_iteration
        // for example, every 5 iterations, run a CASCI.
        if (ci_step) {
            if ((iter < casscf_freq) || (iter % casscf_freq) == 0) {
                cas_ci();
            }
        } else {
            cas_ci();
        }
        if (print_ > 0) {
            outfile->Printf("\n\n CAS took %8.6f seconds.", cas_timer.get());
        }
        Ca = ints_->Ca();
        Cb = ints_->Cb();

        CASSCFOrbitalOptimizer orbital_optimizer(gamma1_, gamma2_, tei_paaa_, options_,
                                                 mo_space_info_);

        orbital_optimizer.set_scf_info(scf_info_);
        orbital_optimizer.set_frozen_one_body(F_froze_);
        orbital_optimizer.set_symmmetry_mo(Ca);
        // orbital_optimizer.one_body(Hcore_);
        psi::SharedMatrix Hcore(Hcore_->clone());
        orbital_optimizer.one_body(Hcore);
        if (print_ > 0) {
            orbital_optimizer.set_print_timings(true);
        }
        orbital_optimizer.set_jk(JK_);
        orbital_optimizer.update();
        double g_norm = orbital_optimizer.orbital_gradient_norm();

        Ediff = E_casscf_ - E_casscf_old;
        if ((std::fabs(Ediff) < econv) && (g_norm < gconv) && (iter > 1)) {

            outfile->Printf("\n %4d   %10.12f   %10.12f   %10.12f  %10.6f s", iter, g_norm, Ediff,
                            E_casscf_, casscf_total_iter.get());

            outfile->Printf(
                "\n\n A miracle has come to pass. The CASSCF iterations have converged.");
            break;
        }

        Sstep = orbital_optimizer.approx_solve();

        ///"Borrowed"(Stolen) from Daniel Smith's code.
        double maxS = 0.0;
        for (int h = 0; h < Sstep->nirrep(); h++) {
            for (int i = 0; i < Sstep->rowspi()[h]; i++) {
                for (int j = 0; j < Sstep->colspi()[h]; j++) {
                    if (std::fabs(Sstep->get(h, i, j)) > maxS)
                        maxS = std::fabs(Sstep->get(h, i, j));
                }
            }
        }
        if (maxS > rotation_max_value) {
            Sstep->scale(rotation_max_value / maxS);
        }

        // Add step to overall rotation

        S->add(Sstep);

        // TODO:  Add options controlled.  Iteration and g_norm
        if (do_diis && (iter > diis_start && g_norm < diis_gradient_norm)) {
            diis_manager->add_entry(2, Sstep.get(), S.get());
            diis_count++;
        }

        if (do_diis && (!(diis_count % diis_freq) && iter > diis_start)) {
            diis_manager->extrapolate(1, S.get());
        }
        psi::SharedMatrix Cp = orbital_optimizer.rotate_orbitals(C_start, S);

        /// ENFORCE Ca = Cb
        Ca->copy(Cp);
        Cb->copy(Cp);

        ints_->update_orbitals(Ca, Cb); // Perhaps slow down code.

        std::string diis_start_label = "";
        if (iter >= diis_start && do_diis == true && g_norm < diis_gradient_norm) {
            diis_start_label = "DIIS";
        }
        outfile->Printf("\n %4d   %10.12f   %10.12f   %10.12f  %10.6f s  %4s ~", iter, g_norm,
                        Ediff, E_casscf_, casscf_total_iter.get(), diis_start_label.c_str());
    }
    // if(casscf_debug_print_)
    //{
    //    overlap_orbitals(this->Ca(), C_start);
    //}
    //    if (options_->get_bool("MONITOR_SA_SOLUTION")) {
    //        overlap_coefficients();
    //    }
    diis_manager->delete_diis_file();
    diis_manager.reset();

    if (iter_con.size() == size_t(maxiter) && maxiter > 1) {
        outfile->Printf("\n CASSCF did not converge");
        throw psi::PSIEXCEPTION("CASSCF did not converge.");
    }

    // INSERT HERE
    // restransform integrals using DF_BASIS_MP2 for
    // consistent energies in correlation treatment
    //    ints_->update_orbitals(Ca, Cb);

    cas_ci();
    outfile->Printf("\n @E(CASSCF) = %18.12f \n", E_casscf_);
    psi::Process::environment.globals["CURRENT ENERGY"] = E_casscf_;
    psi::Process::environment.globals["CASSCF_ENERGY"] = E_casscf_;

    return E_casscf_;
}

void CASSCF::cas_ci() {
    // perform a CAS-CI with the active given in the input
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints = get_ci_integrals();

    auto state_weights_map = make_state_weights_map(options_, ints_->wfn());
    auto state_map = to_state_nroots_map(state_weights_map);

    std::string casscf_ci_type = options_->get_str("CASSCF_CI_SOLVER");
    auto active_space_solver = make_active_space_solver(casscf_ci_type, state_map, scf_info_,
                                                        mo_space_info_, fci_ints, options_);
    const auto state_energies_map = active_space_solver->compute_energy();
    cas_ref_ = active_space_solver->compute_average_rdms(state_weights_map, 2);
    E_casscf_ = compute_average_state_energy(state_energies_map, state_weights_map);

    /*
        if (options_->get_str("CASSCF_CI_SOLVER") == "FCI") {
            // Used to grab the computed energy and RDMs.
            if (options_->psi_options()["AVG_STATE"].size() == 0) {
                set_up_fci();
           // } else {
           //     set_up_sa_fci();
            }
        } else if (options_->get_str("CASSCF_CI_SOLVER") == "CAS") {
            set_up_fcimo();
        } else if (options_->get_str("CASSCF_CI_SOLVER") == "ACI") {
            as_ints_ = get_ci_integrals();
            AdaptiveCI aci(state_, nroot_, scf_info_, options_, mo_space_info_, as_ints_);
            aci.set_max_rdm(2);
            aci.set_quiet(quiet);
            E_casscf_ = aci.compute_energy();
            std::vector<std::pair<size_t,size_t>> roots;
            roots.push_back(std::make_pair(0,0));
            cas_ref_ = aci.reference(roots)[0];
        } else if (options_->get_str("CASSCF_CI_SOLVER") == "DMRG") {
    #ifdef HAVE_CHEMPS2
            DMRGSolver dmrg(state_, scf_info_, options_, ints_, mo_space_info_);
            dmrg.set_max_rdm(2);
            dmrg.spin_free_rdm(true);
            std::pair<ambit::Tensor, std::vector<double>> integral_pair = CI_Integrals();
            dmrg.set_up_integrals(integral_pair.first, integral_pair.second);
            dmrg.set_scalar(scalar_energy_ + ints_->frozen_core_energy() +
                            ints_->nuclear_repulsion_energy());
            E_casscf_ = dmrg.compute_energy();

            cas_ref_ = dmrg.reference();
    #else
            throw psi::PSIEXCEPTION("Did not compile with CHEMPS2 so DMRG will not work");
    #endif
        }

        if (options_->get_str("CASSCF_CI_SOLVER") == "DMRG") {
            gamma2_ = cas_ref_.SFg2();
        }
        */

    // Compute 1-RDM
    gamma1_ = cas_ref_.g1a().clone();
    gamma1_("ij") += cas_ref_.g1b()("ij");

    // Compute 2-RDM
    gamma2_ = cas_ref_.SFg2();
}

double CASSCF::cas_check(RDMs cas_ref) {
    ambit::Tensor gamma1 = ambit::Tensor::build(ambit::CoreTensor, "Gamma1", {nactv_, nactv_});
    ambit::Tensor gamma2 =
        ambit::Tensor::build(ambit::CoreTensor, "Gamma2", {nactv_, nactv_, nactv_, nactv_});

    std::vector<size_t> rdocc = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    std::vector<size_t> active = mo_space_info_->corr_absolute_mo("ACTIVE");
    std::vector<int> active_sym = mo_space_info_->symmetry("ACTIVE");
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints =
        std::make_shared<ActiveSpaceIntegrals>(ints_, active, active_sym, rdocc);

    fci_ints->set_active_integrals_and_restricted_docc();

    /// Spin-free ORDM = gamma1_a + gamma1_b
    ambit::Tensor L1b = cas_ref.g1b();
    ambit::Tensor L1a = cas_ref.g1a();
    gamma1("u, v") = (L1a("u, v") + L1b("u, v"));
    ambit::Tensor L2aa = cas_ref.L2aa();
    ambit::Tensor L2ab = cas_ref.L2ab();
    ambit::Tensor L2bb = cas_ref.L2bb();

    L2aa("p,q,r,s") += L1a("p,r") * L1a("q,s");
    L2aa("p,q,r,s") -= L1a("p,s") * L1a("q,r");

    L2ab("pqrs") += L1a("pr") * L1b("qs");
    // L2ab("pqrs") += L1b("pr") * L1a("qs");

    L2bb("pqrs") += L1b("pr") * L1b("qs");
    L2bb("pqrs") -= L1b("ps") * L1b("qr");

    // This may or may not be correct.  Really need to find a way to check this
    // code
    gamma2.copy(L2aa);
    gamma2("u,v,x,y") += L2ab("u,v,x,y");
    // gamma2("u,v,x,y") +=  L2ab("v, u, y, x");
    // gamma2("u,v,x,y") +=  L2bb("u,v,x,y");

    // gamma2_("u,v,x,y") = gamma2_("x,y,u,v");
    // gamma2_("u,v,x,y") = gamma2_("")
    gamma2.scale(2.0);

    double E_casscf = 0.0;

    std::vector<size_t> na_array = mo_space_info_->corr_absolute_mo("ACTIVE");

    ambit::Tensor tei_ab = ints_->aptei_ab_block(na_array, na_array, na_array, na_array);

    double OneBody = 0.0;
    double TwoBody = 0.0;
    double Frozen = 0.0;
    double fci_ints_scalar = 0.0;
    for (size_t p = 0; p < na_array.size(); ++p) {
        for (size_t q = 0; q < na_array.size(); ++q) {
            E_casscf += gamma1.data()[nactv_ * p + q] * fci_ints->oei_a(p, q);
        }
    }
    OneBody = E_casscf;
    outfile->Printf("\n OneBodyE_CASSCF: %8.8f", E_casscf);

    E_casscf += 0.5 * gamma2("u, v, x, y") * tei_ab("u, v, x, y");
    TwoBody += 0.5 * gamma2("u, v, x, y") * tei_ab("u, v, x, y");
    E_casscf += ints_->frozen_core_energy();
    Frozen = ints_->frozen_core_energy();
    E_casscf += fci_ints->scalar_energy();
    fci_ints_scalar = fci_ints->scalar_energy();
    E_casscf += ints_->nuclear_repulsion_energy();
    outfile->Printf("\n\n OneBody: %8.8f TwoBody: %8.8f Frozen: %8.8f "
                    "fci_ints_scalar: %8.8f",
                    OneBody, TwoBody, Frozen, fci_ints_scalar);

    return E_casscf;
}
std::shared_ptr<psi::Matrix> CASSCF::set_frozen_core_orbitals() {
    psi::SharedMatrix Ca = ints_->Ca();
    psi::Dimension nsopi = scf_info_->nsopi();
    psi::Dimension frozen_dim = mo_space_info_->dimension("FROZEN_DOCC");
    psi::SharedMatrix C_core(new psi::Matrix("C_core", nirrep_, nsopi, frozen_dim));
    // Need to get the frozen block of the C matrix
    for (size_t h = 0; h < nirrep_; h++) {
        for (int i = 0; i < frozen_dim[h]; i++) {
            C_core->set_column(h, i, Ca->get_column(h, i));
        }
    }

    //    JK_->set_allow_desymmetrization(true);
    JK_->set_do_K(true);
    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();

    Cl.clear();
    Cl.push_back(C_core);
    Cr.clear();
    Cr.push_back(C_core);

    JK_->compute();

    psi::SharedMatrix F_core = JK_->J()[0];
    psi::SharedMatrix K_core = JK_->K()[0];

    F_core->scale(2.0);
    F_core->subtract(K_core);

    return F_core;
}
ambit::Tensor CASSCF::transform_integrals() {
    // This function will do an integral transformation using the JK builder,
    // and return the integrals of type <px|uy> = (pu|xy).
    // This was borrowed from Kevin Hannon's IntegralTransform Plugin.

    // Transform C matrix to C1 symmetry
    size_t nso = scf_info_->nso();
    psi::SharedMatrix Ca_sym = ints_->Ca();
    psi::SharedMatrix aotoso = ints_->aotoso();
    auto Call = std::make_shared<psi::Matrix>(nso, nmo_);

    // Transform from the SO to the AO basis for the C matrix.
    // just transfroms the C_{mu_ao i} -> C_{mu_so i}
    local_timer CSO2AO;
    for (size_t h = 0, index = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nmo_dim_[h]; ++i) {
            size_t nao = nso;
            size_t nso = nsopi_[h];

            if (!nso)
                continue;

            C_DGEMV('N', nao, nso, 1.0, aotoso->pointer(h)[0], nso, &Ca_sym->pointer(h)[0][i],
                    nmo_dim_[h], 0.0, &Call->pointer()[0][index], nmo_dim_.sum());

            index += 1;
        }
    }
    if (print_ > 1) {
        outfile->Printf("\n  CSO2AO takes %8.4f s.", CSO2AO.get());
    }

    // set up the active part of the C matrix
    auto Cact = std::make_shared<psi::Matrix>("Cact", nso, nactv_);
    std::vector<std::shared_ptr<psi::Matrix>> Cact_vec(nactv_);
    for (size_t x = 0; x < nactv_; x++) {
        psi::SharedVector Call_vec = Call->get_column(0, actv_mos_abs_[x]);
        Cact->set_column(0, x, Call_vec);

        std::string name = "Cact slice " + std::to_string(x);
        auto temp = std::make_shared<psi::Matrix>(name, nso, 1);
        temp->set_column(0, 0, Call_vec);
        Cact_vec[x] = temp;
    }

    // The following type of integrals are needed:
    // (pu|xy) = C_{Mp}^T C_{Nu} C_{Rx}^T C_{Sy} (MN|RS)
    //         = C_{Mp}^T C_{Nu} J_{MN}^{xy}
    //         = C_{Mp}^T J_{MN}^{xy} C_{Nu}

    JK_->set_memory(psi::Process::environment.get_memory() * 0.85);
    JK_->set_do_K(false);
    std::vector<psi::SharedMatrix>& Cl = JK_->C_left();
    std::vector<psi::SharedMatrix>& Cr = JK_->C_right();
    Cl.clear();
    Cr.clear();

    for (size_t x = 0; x < nactv_; ++x) {
        for (size_t y = x; y < nactv_; ++y) {
            Cl.push_back(Cact_vec[x]);
            Cr.push_back(Cact_vec[y]);
        }
    }

    JK_->compute();

    auto half_trans = std::make_shared<psi::Matrix>("Trans", nmo_, nactv_);
    auto corr_mos_abs = mo_space_info_->absolute_mo("CORRELATED");

    std::vector<size_t> ints_dim{ncmo_, nactv_, nactv_, nactv_};
    auto ints = ambit::Tensor::build(ambit::CoreTensor, "gaaa ints", ints_dim);
    std::vector<double>& ints_data = ints.data();
    size_t nactv2 = nactv_ * nactv_;
    size_t nactv3 = nactv2 * nactv_;

    for (size_t x = 0, shift = 0; x < nactv_; ++x) {
        shift += x;
        for (size_t y = x; y < nactv_; ++y) {
            psi::SharedMatrix J = JK_->J()[x * nactv_ + y - shift];
            half_trans = psi::linalg::triplet(Call, J, Cact, true, false, false);

            for (size_t p = 0; p < ncmo_; ++p) {
                for (size_t u = 0; u < nactv_; ++u) {
                    double value = half_trans->get(corr_mos_abs[p], u);
                    ints_data[corr_mos_[p] * nactv3 + x * nactv2 + u * nactv_ + y] = value;
                    ints_data[corr_mos_[p] * nactv3 + y * nactv2 + u * nactv_ + x] = value;
                }
            }
        }
    }

    return ints;
}

void CASSCF::set_up_fci() {
    auto as_ints = make_active_space_ints(mo_space_info_, ints_, "ACTIVE", {{"RESTRICTED_DOCC"}});

    std::shared_ptr<ActiveSpaceMethod> fcisolver = make_active_space_method(
        "FCI", state_, nroot_, scf_info_, mo_space_info_, as_ints, options_);

    fcisolver->set_root(options_->get_int("ROOT"));
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints = get_ci_integrals();
    fcisolver->set_active_space_integrals(fci_ints);
    E_casscf_ = fcisolver->compute_energy();

    std::vector<std::pair<size_t, size_t>> roots;
    roots.push_back(std::make_pair(0, 0));
    cas_ref_ = fcisolver->rdms(roots, 2)[0];
}

std::shared_ptr<ActiveSpaceIntegrals> CASSCF::get_ci_integrals() {

    std::vector<size_t> rdocc = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    std::vector<size_t> active = mo_space_info_->corr_absolute_mo("ACTIVE");
    std::vector<int> active_sym = mo_space_info_->symmetry("ACTIVE");
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints =
        std::make_shared<ActiveSpaceIntegrals>(ints_, active, active_sym, rdocc);
    if (!(options_->get_bool("RESTRICTED_DOCC_JK"))) {
        fci_ints->set_active_integrals_and_restricted_docc();
    } else {
        auto na_array = mo_space_info_->corr_absolute_mo("ACTIVE");

        ambit::Tensor active_aa = ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsAA",
                                                       {nactv_, nactv_, nactv_, nactv_});
        ambit::Tensor active_ab = ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsAB",
                                                       {nactv_, nactv_, nactv_, nactv_});
        ambit::Tensor active_bb = ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsBB",
                                                       {nactv_, nactv_, nactv_, nactv_});
        const std::vector<double>& tei_paaa_data = tei_paaa_.data();

        active_ab.iterate([&](const std::vector<size_t>& i, double& value) {
            value = tei_paaa_data[na_array[i[0]] * nactv_ * nactv_ * nactv_ +
                                  i[1] * nactv_ * nactv_ + i[2] * nactv_ + i[3]];
        });

        active_aa.copy(active_ab);
        active_bb.copy(active_ab);
        active_aa("u,v,x,y") -= active_ab("u, v, y, x");
        active_bb.copy(active_aa);

        fci_ints->set_active_integrals(active_aa, active_ab, active_bb);
        if (casscf_debug_print_) {
            outfile->Printf("\n\n tei_active_aa: %8.8f tei_active_ab: %8.8f", active_aa.norm(2),
                            active_ab.norm(2));
        }

        std::vector<std::vector<double>> oei_vector;
        if ((ncore_ + nfrzc_) > 0) {
            oei_vector = compute_restricted_docc_operator();
            fci_ints->set_restricted_one_body_operator(oei_vector[0], oei_vector[1]);
            fci_ints->set_scalar_energy(scalar_energy_);
        } else {
            std::vector<double> oei_a(nactv_ * nactv_);
            std::vector<double> oei_b(nactv_ * nactv_);

            for (size_t p = 0; p < nactv_; ++p) {
                size_t pp = active[p];
                for (size_t q = 0; q < nactv_; ++q) {
                    size_t qq = active[q];
                    size_t idx = nactv_ * p + q;
                    oei_a[idx] = ints_->oei_a(pp, qq);
                    oei_b[idx] = ints_->oei_b(pp, qq);
                }
            }
            oei_vector.push_back(oei_a);
            oei_vector.push_back(oei_b);
            scalar_energy_ = 0.00;
            fci_ints->set_restricted_one_body_operator(oei_vector[0], oei_vector[1]);
            fci_ints->set_scalar_energy(scalar_energy_);
        }
    }
    return fci_ints;
}

std::vector<std::vector<double>> CASSCF::compute_restricted_docc_operator() {
    ///
    psi::Dimension restricted_docc_dim = mo_space_info_->dimension("INACTIVE_DOCC");
    psi::Dimension nsopi = scf_info_->nsopi();
    int nirrep = mo_space_info_->nirrep();
    psi::Dimension nmopi = mo_space_info_->dimension("ALL");

    psi::SharedMatrix Cdocc(new psi::Matrix("C_RESTRICTED", nirrep, nsopi, restricted_docc_dim));
    psi::SharedMatrix Ca = ints_->Ca();
    for (int h = 0; h < nirrep; h++) {
        for (int i = 0; i < restricted_docc_dim[h]; i++) {
            Cdocc->set_column(h, i, Ca->get_column(h, i));
        }
    }
    /// F_frozen = D_{uv}^{frozen} * (2<uv|rs> - <ur | vs>)
    /// F_restricted = D_{uv}^{restricted} * (2<uv|rs> - <ur | vs>)
    /// F_inactive = F_frozen + F_restricted + H_{pq}^{core}
    /// D_{uv}^{frozen} = \sum_{i = 0}^{frozen}C_{ui} * C_{vi}
    /// D_{uv}^{inactive} = \sum_{i = 0}^{inactive}C_{ui} * C_{vi}
    /// This section of code computes the fock matrix for the
    /// INACTIVE_DOCC("RESTRICTED_DOCC")

    //    std::shared_ptr<JK> JK_inactive = JK::build_JK(this->basisset(),
    //    this->options_);
    //
    //    JK_inactive->set_memory(psi::Process::environment.get_memory() * 0.8);
    //    JK_inactive->initialize();
    //
    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();
    //    JK_->set_allow_desymmetrization(true);
    JK_->set_do_K(true);
    Cl.clear();
    Cl.push_back(Cdocc);
    Cr.clear();
    Cr.push_back(Cdocc);
    JK_->compute();
    psi::SharedMatrix J_restricted = JK_->J()[0];
    psi::SharedMatrix K_restricted = JK_->K()[0];

    J_restricted->scale(2.0);
    psi::SharedMatrix F_restricted = J_restricted->clone();
    F_restricted->subtract(K_restricted);

    /// Just create the OneInt integrals from scratch

    psi::SharedMatrix Hcore(Hcore_->clone());
    F_restricted->add(Hcore);
    F_restricted->transform(Ca);
    Hcore->transform(Ca);

    size_t all_nmo = mo_space_info_->size("ALL");
    psi::SharedMatrix F_restric_c1(new psi::Matrix("F_restricted", all_nmo, all_nmo));
    size_t offset = 0;
    for (int h = 0; h < nirrep; h++) {
        for (int p = 0; p < nmopi[h]; p++) {
            for (int q = 0; q < nmopi[h]; q++) {
                F_restric_c1->set(p + offset, q + offset, F_restricted->get(h, p, q));
            }
        }
        offset += nmopi[h];
    }
    size_t nmo2 = nactv_ * nactv_;
    std::vector<double> oei_a(nmo2);
    std::vector<double> oei_b(nmo2);

    auto absolute_active = mo_space_info_->absolute_mo("ACTIVE");
    for (size_t u = 0; u < nactv_; u++) {
        for (size_t v = 0; v < nactv_; v++) {
            double value = F_restric_c1->get(absolute_active[u], absolute_active[v]);
            // double h_value = H->get(absolute_active[u], absolute_active[v]);
            oei_a[u * nactv_ + v] = value;
            oei_b[u * nactv_ + v] = value;
            if (casscf_debug_print_)
                outfile->Printf("\n oei(%d, %d) = %8.8f", u, v, value);
        }
    }
    psi::Dimension restricted_docc = mo_space_info_->dimension("INACTIVE_DOCC");
    double E_restricted = 0.0;
    for (int h = 0; h < nirrep; h++) {
        for (int rd = 0; rd < restricted_docc[h]; rd++) {
            E_restricted += Hcore->get(h, rd, rd) + F_restricted->get(h, rd, rd);
        }
    }
    /// Since F^{INACTIVE} includes frozen_core in fock build, the energy
    /// contribution includes frozen_core_energy
    if (casscf_debug_print_) {
        outfile->Printf("\n Frozen Core Energy = %8.8f", ints_->frozen_core_energy());
        outfile->Printf("\n Restricted Energy = %8.8f", E_restricted - ints_->frozen_core_energy());
        outfile->Printf("\n Scalar Energy = %8.8f",
                        ints_->scalar() + E_restricted - ints_->frozen_core_energy());
    }
    scalar_energy_ = ints_->scalar();
    scalar_energy_ += (E_restricted - ints_->frozen_core_energy());
    std::vector<std::vector<double>> oei_container;
    oei_container.push_back(oei_a);
    oei_container.push_back(oei_b);
    return oei_container;
}
void CASSCF::overlap_orbitals(const psi::SharedMatrix& C_old, const psi::SharedMatrix& C_new) {
    psi::SharedMatrix S_orbitals(
        new psi::Matrix("Overlap", scf_info_->nsopi(), scf_info_->nsopi()));
    psi::SharedMatrix S_basis = ints_->wfn()->S();
    S_orbitals = psi::linalg::triplet(C_old, S_basis, C_new, true, false, false);
    S_orbitals->set_name("C^T S C (Overlap)");
    for (size_t h = 0; h < nirrep_; h++) {
        for (int i = 0; i < S_basis->rowspi(h); i++) {
            if (std::fabs(S_basis->get(h, i, i) - 1.0000000) > 1e-6) {
                //    S_basis->get_row(h, i)->print();
            }
        }
    }
}
/*
void CASSCF::set_up_sa_fci() {
    SA_FCISolver sa_fcisolver(options_->psi_options(), ints_->wfn());
    sa_fcisolver.set_mo_space_info(mo_space_info_);
    sa_fcisolver.set_integrals(ints_);
    std::vector<size_t> rdocc = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    std::vector<size_t> active = mo_space_info_->corr_absolute_mo("ACTIVE");
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints =
        std::make_shared<ActiveSpaceIntegrals>(ints_, active, rdocc);
    auto na_array = mo_space_info_->corr_absolute_mo("ACTIVE");

    ambit::Tensor active_aa =
        ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsAA", {na_, na_, na_, na_});
    ambit::Tensor active_ab =
        ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsAB", {na_, na_, na_, na_});
    ambit::Tensor active_bb =
        ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsBB", {na_, na_, na_, na_});
    const std::vector<double>& tei_paaa_data = tei_paaa_.data();

    active_ab.iterate([&](const std::vector<size_t>& i, double& value) {
        value =
            tei_paaa_data[na_array[i[0]] * na_ * na_ * na_ + i[1] * na_ * na_ + i[2] * na_ + i[3]];
    });
    active_aa.copy(active_ab);
    active_bb.copy(active_ab);
    active_aa("u,v,x,y") -= active_ab("u, v, y, x");
    active_bb.copy(active_aa);

    fci_ints->set_active_integrals(active_aa, active_ab, active_bb);
    if (casscf_debug_print_) {
        outfile->Printf("\n\n tei_active_aa: %8.8f tei_active_ab: %8.8f", active_aa.norm(2),
                        active_ab.norm(2));
    }

    std::vector<std::vector<double>> oei_vector;
    if ((nrdocc_ + nfrozen_) > 0) {
        oei_vector = compute_restricted_docc_operator();
        fci_ints->set_restricted_one_body_operator(oei_vector[0], oei_vector[1]);
        fci_ints->set_scalar_energy(scalar_energy_);
        sa_fcisolver.set_integral_pointer(fci_ints);
    } else {
        std::vector<double> oei_a(na_ * na_);
        std::vector<double> oei_b(na_ * na_);

        for (size_t p = 0; p < na_; ++p) {
            size_t pp = active[p];
            for (size_t q = 0; q < na_; ++q) {
                size_t qq = active[q];
                size_t idx = na_ * p + q;
                oei_a[idx] = ints_->oei_a(pp, qq);
                oei_b[idx] = ints_->oei_b(pp, qq);
            }
        }
        oei_vector.push_back(oei_a);
        oei_vector.push_back(oei_b);
        scalar_energy_ = 0.00;
        fci_ints->set_restricted_one_body_operator(oei_vector[0], oei_vector[1]);
        fci_ints->set_scalar_energy(scalar_energy_);
        sa_fcisolver.set_integral_pointer(fci_ints);
    }

    E_casscf_ = sa_fcisolver.compute_energy();
    cas_ref_ = sa_fcisolver.reference();
    //    if (options_->get_bool("MONITOR_SA_SOLUTION")) {
    //        std::vector<std::shared_ptr<FCIVector>> StateAveragedFCISolver =
    //            sa_fcisolver.StateAveragedCISolution();
    //        CISolutions_.push_back(StateAveragedFCISolver);
    //    }
}
*/

void CASSCF::write_orbitals_molden() {
    psi::SharedVector occ_vector(new psi::Vector(nirrep_, corr_dim_));
    view_modified_orbitals(ints_->wfn(), ints_->Ca(), scf_info_->epsilon_a(), occ_vector);
}
// void CASSCF::overlap_coefficients() {
//    outfile->Printf("\n iter  Overlap_{i-1} Overlap_{i}");
//    for (size_t iter = 1; iter < CISolutions_.size(); ++iter) {
//        for (size_t cisoln = 0; cisoln < CISolutions_[iter].size(); cisoln++) {
//            for (size_t j = 0; j < CISolutions_[iter].size(); j++) {
//                if (std::fabs(CISolutions_[0][cisoln]->dot(CISolutions_[iter][j])) > 0.90) {
//                    outfile->Printf("\n %d:%d %d:%d %8.8f", 0, cisoln, iter, j,
//                                    CISolutions_[0][cisoln]->dot(CISolutions_[iter][j]));
//                }
//            }
//        }
//        outfile->Printf("\n");
//    }
//}
std::pair<ambit::Tensor, std::vector<double>> CASSCF::CI_Integrals() {
    std::vector<std::vector<double>> oei_vector = compute_restricted_docc_operator();

    auto na_array = mo_space_info_->corr_absolute_mo("ACTIVE");
    const std::vector<double>& tei_paaa_data = tei_paaa_.data();
    ambit::Tensor active_ab = ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsBB",
                                                   {nactv_, nactv_, nactv_, nactv_});

    active_ab.iterate([&](const std::vector<size_t>& i, double& value) {
        value = tei_paaa_data[na_array[i[0]] * nactv_ * nactv_ * nactv_ + i[1] * nactv_ * nactv_ +
                              i[2] * nactv_ + i[3]];
    });
    std::pair<ambit::Tensor, std::vector<double>> pair_return =
        std::make_pair(active_ab, oei_vector[0]);
    return pair_return;
}

std::vector<RDMs> CASSCF::rdms(const std::vector<std::pair<size_t, size_t>>& /*root_list*/,
                               int /*max_rdm_level*/) {
    // TODO (York): this does not seem the correct thing to do.
    std::vector<RDMs> refs;
    refs.push_back(cas_ref_);
    return refs;
}

std::vector<RDMs>
CASSCF::transition_rdms(const std::vector<std::pair<size_t, size_t>>& /*root_list*/,
                        std::shared_ptr<ActiveSpaceMethod> /*method2*/, int /*max_rdm_level*/) {
    std::vector<RDMs> refs;
    throw std::runtime_error("FCISolver::transition_rdms is not implemented!");
    return refs;
}

} // namespace forte
