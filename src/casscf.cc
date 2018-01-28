/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "casscf.h"
#include "integrals/integrals.h"
#include "reference.h"

#include "psi4/libfock/jk.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libqt/qt.h"
#include "psi4/psifiles.h"

#include "helpers.h"

#include "aci/aci.h"
#include "fci/fci_solver.h"

#include "fci_mo.h"
#include "mp2_nos.h"
#include "orbitaloptimizer.h"
#include "sa_fcisolver.h"
#include "semi_canonicalize.h"

#ifdef HAVE_CHEMPS2
#include "dmrgsolver.h"
#endif
#include "psi4/libdiis/diisentry.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libmints/factory.h"

namespace psi {
namespace forte {

CASSCF::CASSCF(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
               std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), options_(options), ints_(ints), mo_space_info_(mo_space_info) {
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;
    startup();
}
void CASSCF::compute_casscf() {
    if (na_ == 0) {
        outfile->Printf("\n\n\n Please set the active space");
        throw PSIEXCEPTION(" The active space is zero.  Set the active space");
    } else if (na_ == nmo_) {
        outfile->Printf("\n Your about to do an all active CASSCF");
        throw PSIEXCEPTION("The active space is all the MOs.  Orbitals don't "
                           "matter at this point");
    }

    int maxiter = options_.get_int("CASSCF_ITERATIONS");

    /// Provide a nice summary at the end for iterations
    std::vector<int> iter_con;
    /// FrozenCore C Matrix is never rotated
    /// Can bring this out of loop
    if (nfrozen_ > 0) {
        F_froze_ = set_frozen_core_orbitals();
    }

    /// Setup the DIIS manager
    int diis_freq = options_.get_int("CASSCF_DIIS_FREQ");
    int diis_start = options_.get_int("CASSCF_DIIS_START");
    int diis_max_vec = options_.get_int("CASSCF_DIIS_MAX_VEC");
    int casscf_freq = options_.get_int("CASSCF_CI_FREQ");
    bool ci_step = options_.get_bool("CASSCF_CI_STEP");
    bool do_diis = options_.get_bool("CASSCF_DO_DIIS");
    double diis_gradient_norm = options_.get_double("CASSCF_DIIS_NORM");
    double rotation_max_value = options_.get_double("CASSCF_MAX_ROTATION");

    Dimension nhole_dim = mo_space_info_->get_dimension("GENERALIZED HOLE");
    Dimension npart_dim = mo_space_info_->get_dimension("GENERALIZED PARTICLE");
    SharedMatrix S(new Matrix("Orbital Rotation", nirrep_, nhole_dim, npart_dim));
    SharedMatrix Sstep;

    std::shared_ptr<DIISManager> diis_manager(
        new DIISManager(diis_max_vec, "MCSCF DIIS", DIISManager::OldestAdded, DIISManager::InCore));
    diis_manager->set_error_vector_size(1, DIISEntry::Matrix, S.get());
    diis_manager->set_vector_size(1, DIISEntry::Matrix, S.get());

    int diis_count = 0;

    E_casscf_ = 0.0;
    double E_casscf_old = 0.0, Ediff = 0.0;
    SharedMatrix C_start(this->Ca()->clone());
    double econv = options_.get_double("CASSCF_E_CONVERGENCE");
    double gconv = options_.get_double("CASSCF_G_CONVERGENCE");

    print_h2("CASSCF Iteration");
    outfile->Printf("\n iter    ||g||           Delta_E            E_CASSCF       CONV_TYPE");

    for (int iter = 0; iter < maxiter; iter++) {
        Timer casscf_total_iter;

        Timer transform_integrals_timer;
        tei_paaa_ = transform_integrals();
        if (print_ > 0) {
            outfile->Printf("\n\n Transform Integrals takes %8.8f s.",
                            transform_integrals_timer.get());
        }
        iter_con.push_back(iter);

        /// Perform a CAS-CI using either York's code or Francesco's
        /// If CASSCF_DEBUG_PRINTING is on, will compare CAS-CI with SPIN-FREE RDM
        E_casscf_old = E_casscf_;
        if (print_ > 0) {
            outfile->Printf("\n\n  Performing a CAS with %s", options_.get_str("CAS_TYPE").c_str());
        }
        Timer cas_timer;
        /// Perform a DMRG-CI, ACI, FCI inside an active space
        /// If casscf_freq is on, do the CI step every casscf_iteration
        /// Ie, every 5 iterations, run a CAS_CI.
        /// TODO:  Maybe make this run the orbital optimization
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
        SharedMatrix Ca = this->Ca();
        SharedMatrix Cb = this->Cb();

        CASSCFOrbitalOptimizer orbital_optimizer(gamma1_, gamma2_, tei_paaa_, options_,
                                                 mo_space_info_);

        orbital_optimizer.set_wavefunction(reference_wavefunction_);
        orbital_optimizer.set_frozen_one_body(F_froze_);
        orbital_optimizer.set_symmmetry_mo(Ca);
        // orbital_optimizer.one_body(Hcore_);
        SharedMatrix Hcore(Hcore_->clone());
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
            outfile->Printf("\n @E(CASSCF) = %18.12f \n", E_casscf_);
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
        SharedMatrix Cp = orbital_optimizer.rotate_orbitals(C_start, S);

        /// ENFORCE Ca = Cb
        Ca->copy(Cp);
        Cb->copy(Cp);

        std::string diis_start_label = "";
        if (iter >= diis_start && do_diis == true && g_norm < diis_gradient_norm) {
            diis_start_label = "DIIS";
        }
        outfile->Printf("\n %4d   %10.12f   %10.12f   %10.12f  %10.6f s  %4s", iter, g_norm, Ediff,
                        E_casscf_, casscf_total_iter.get(), diis_start_label.c_str());
    }
    // if(casscf_debug_print_)
    //{
    //    overlap_orbitals(this->Ca(), C_start);
    //}
    //    if (options_.get_bool("MONITOR_SA_SOLUTION")) {
    //        overlap_coefficients();
    //    }
    diis_manager->delete_diis_file();
    diis_manager.reset();

    if (iter_con.size() == size_t(maxiter) && maxiter > 1) {
        outfile->Printf("\n CASSCF did not converged");
        throw PSIEXCEPTION("CASSCF did not converged.");
    }
    Process::environment.globals["CURRENT ENERGY"] = E_casscf_;
    Process::environment.globals["CASSCF_ENERGY"] = E_casscf_;
    Timer retrans_ints;
    if (print_ > 0) {
        outfile->Printf("\n Overall retranformation of integrals takes %6.4f s.\n",
                        retrans_ints.get());
    }

    if (options_.get_bool("SEMI_CANONICAL")) {
        ints_->retransform_integrals();
        SemiCanonical semi(reference_wavefunction_, ints_, mo_space_info_);
        semi.semicanonicalize(cas_ref_, 0);
    }
}
void CASSCF::startup() {
    print_method_banner({"Complete Active Space Self Consistent Field", "Kevin Hannon"});
    na_ = mo_space_info_->size("ACTIVE");
    print_ = options_.get_int("PRINT");
    nsopi_ = this->nsopi();
    nirrep_ = this->nirrep();
    if (options_.get_str("SCF_TYPE") == "PK") {
        outfile->Printf("\n\n CASSCF algorithm can not use PK");
        throw PSIEXCEPTION("PK should not be used for CASSCF");
    }

    casscf_debug_print_ = options_.get_bool("CASSCF_DEBUG_PRINTING");

    frozen_docc_dim_ = mo_space_info_->get_dimension("FROZEN_DOCC");
    restricted_docc_dim_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    active_dim_ = mo_space_info_->get_dimension("ACTIVE");
    restricted_uocc_dim_ = mo_space_info_->get_dimension("RESTRICTED_UOCC");
    inactive_docc_dim_ = mo_space_info_->get_dimension("INACTIVE_DOCC");

    frozen_docc_abs_ = mo_space_info_->get_corr_abs_mo("FROZEN_DOCC");
    restricted_docc_abs_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    active_abs_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    restricted_uocc_abs_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    inactive_docc_abs_ = mo_space_info_->get_corr_abs_mo("INACTIVE_DOCC");
    nmo_abs_ = mo_space_info_->get_corr_abs_mo("CORRELATED");
    nmo_ = mo_space_info_->size("CORRELATED");
    all_nmo_ = mo_space_info_->size("ALL");
    nmopi_ = mo_space_info_->get_dimension("CORRELATED");
    nrdocc_ = restricted_docc_abs_.size();
    nvir_ = restricted_uocc_abs_.size();

    nfrozen_ = frozen_docc_abs_.size();
    /// If the user wants to freeze core after casscf, this section of code sets
    /// frozen_docc to zero
    if (casscf_debug_print_) {
        outfile->Printf("\n Total Number of NMO: %d", nmo_);
        outfile->Printf("\n ACTIVE: ");
        for (auto active : active_abs_) {
            outfile->Printf(" %d", active);
        }
        outfile->Printf("\n RESTRICTED: ");
        for (auto restricted : restricted_docc_abs_) {
            outfile->Printf(" %d", restricted);
        }
        outfile->Printf("\n VIRTUAL: ");
        for (auto virtual_index : restricted_uocc_abs_) {
            outfile->Printf(" %d", virtual_index);
        }
    }
    std::shared_ptr<PSIO> psio_ = PSIO::shared_object();
    SharedMatrix T = SharedMatrix(matrix_factory()->create_matrix(PSIF_SO_T));
    SharedMatrix V = SharedMatrix(matrix_factory()->create_matrix(PSIF_SO_V));
    SharedMatrix OneInt = T;
    OneInt->zero();

    T->load(psio_, PSIF_OEI);
    V->load(psio_, PSIF_OEI);
    Hcore_ = matrix_factory()->create_shared_matrix("Core Hamiltonian");
    Hcore_->add(T);
    Hcore_->add(V);

    Timer JK_initialize;
    if (options_.get_str("SCF_TYPE") == "GTFOCK") {
#ifdef HAVE_JK_FACTORY
        Process::environment.set_legacy_molecule(this->molecule());
        JK_ = std::shared_ptr<JK>(new GTFockJK(this->basisset()));
#else
        throw PSIEXCEPTION("GTFock was not compiled in this version");
#endif
    } else {
        if (options_.get_str("SCF_TYPE") == "DF") {
            JK_ = JK::build_JK(basisset(), get_basisset("DF_BASIS_SCF"), options_);
        } else {
            JK_ = JK::build_JK(basisset(), BasisSet::zero_ao_basis_set(), options_);
        }
    }
    JK_->set_memory(Process::environment.get_memory() * 0.8);
    JK_->initialize();
    JK_->C_left().clear();
    JK_->C_right().clear();
    if (print_ > 0)
        outfile->Printf("\n     JK takes %5.5f s to initialize while using %s", JK_initialize.get(),
                        options_.get_str("SCF_TYPE").c_str());
}
void CASSCF::cas_ci() {
    /// Calls francisco's FCI code and does a CAS-CI with the active given in
    /// the input
    // tei_paaa_ = transform_integrals();
    SharedMatrix gamma2_matrix(new Matrix("gamma2", na_ * na_, na_ * na_));
    bool quiet = true;
    if (print_ > 0) {
        quiet = false;
    }
    if (options_.get_str("CAS_TYPE") == "FCI") {
        // Used to grab the computed energy and RDMs.
        if (options_["AVG_STATE"].size() == 0) {
            set_up_fci();
        } else {
            set_up_sa_fci();
        }
    } else if (options_.get_str("CAS_TYPE") == "CAS") {
        set_up_fcimo();
    } else if (options_.get_str("CAS_TYPE") == "ACI") {
        ints_->retransform_integrals();
        AdaptiveCI aci(reference_wavefunction_, options_, ints_, mo_space_info_);
        aci.set_max_rdm(2);
        aci.set_quiet(quiet);
        aci.compute_energy();
        cas_ref_ = aci.reference();
        E_casscf_ = cas_ref_.get_Eref();
    } else if (options_.get_str("CAS_TYPE") == "DMRG") {
#ifdef HAVE_CHEMPS2
        DMRGSolver dmrg(reference_wavefunction_, options_, mo_space_info_, ints_);
        dmrg.set_max_rdm(2);
        dmrg.spin_free_rdm(true);
        std::pair<ambit::Tensor, std::vector<double>> integral_pair = CI_Integrals();
        dmrg.set_up_integrals(integral_pair.first, integral_pair.second);
        dmrg.set_scalar(scalar_energy_ + ints_->frozen_core_energy() +
                        Process::environment.molecule()->nuclear_repulsion_energy(reference_wavefunction_->get_dipole_field_strength()));
        dmrg.compute_energy();

        cas_ref_ = dmrg.reference();
        E_casscf_ = cas_ref_.get_Eref();
#else
        throw PSIEXCEPTION("Did not compile with CHEMPS2 so DMRG will not work");
#endif
    }

    ambit::Tensor L2aa = cas_ref_.L2aa();
    ambit::Tensor L2ab = cas_ref_.L2ab();
    ambit::Tensor L2bb = cas_ref_.L2bb();
    ambit::Tensor L1a = cas_ref_.L1a();
    ambit::Tensor L1b = cas_ref_.L1b();

    L2aa("p,q,r,s") += L1a("p,r") * L1a("q,s");
    L2aa("p,q,r,s") -= L1a("p,s") * L1a("q,r");

    L2ab("pqrs") += L1a("pr") * L1b("qs");
    // L2ab("pqrs") += L1b("pr") * L1a("qs");

    L2bb("pqrs") += L1b("pr") * L1b("qs");
    L2bb("pqrs") -= L1b("ps") * L1b("qr");
    if (options_.get_str("CAS_TYPE") == "DMRG")
        L2aa.scale(0.5);

    ambit::Tensor gamma2 = ambit::Tensor::build(ambit::CoreTensor, "gamma2", {na_, na_, na_, na_});

    gamma2("u,v,x,y") += L2aa("u,v,x,y");
    gamma2("u,v,x,y") += L2ab("u,v,x,y");
    gamma2_ = ambit::Tensor::build(ambit::CoreTensor, "gamma2", {na_, na_, na_, na_});
    gamma2_.copy(gamma2);
    gamma2_.scale(2.0);
    gamma2_.iterate([&](const std::vector<size_t>& i, double& value) {
        gamma2_matrix->set(i[0] * i[1] + i[1], i[2] * i[3] + i[3], value);
    });

    /// Compute the 1RDM
    ambit::Tensor gamma_no_spin = ambit::Tensor::build(ambit::CoreTensor, "Return", {na_, na_});
    gamma1_ = ambit::Tensor::build(ambit::CoreTensor, "Return", {na_, na_});
    ambit::Tensor gamma1a = cas_ref_.L1a();
    ambit::Tensor gamma1b = cas_ref_.L1b();

    gamma_no_spin("i,j") = (gamma1a("i,j") + gamma1b("i,j"));

    gamma1_ = gamma_no_spin;
    if (options_.get_str("CAS_TYPE") == "DMRG") {
        gamma2_ = cas_ref_.SFg2();
    }
}

double CASSCF::cas_check(Reference cas_ref) {
    ambit::Tensor gamma1 = ambit::Tensor::build(ambit::CoreTensor, "Gamma1", {na_, na_});
    ambit::Tensor gamma2 = ambit::Tensor::build(ambit::CoreTensor, "Gamma2", {na_, na_, na_, na_});
    ints_->retransform_integrals();
    std::shared_ptr<FCIIntegrals> fci_ints =
        std::make_shared<FCIIntegrals>(ints_, mo_space_info_->get_corr_abs_mo("ACTIVE"),
                                       mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));
    fci_ints->set_active_integrals_and_restricted_docc();

    /// Spin-free ORDM = gamma1_a + gamma1_b
    ambit::Tensor L1b = cas_ref.L1b();
    ambit::Tensor L1a = cas_ref.L1a();
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

    std::vector<size_t> na_array = mo_space_info_->get_corr_abs_mo("ACTIVE");

    ambit::Tensor tei_ab = ints_->aptei_ab_block(na_array, na_array, na_array, na_array);

    double OneBody = 0.0;
    double TwoBody = 0.0;
    double Frozen = 0.0;
    double fci_ints_scalar = 0.0;
    for (size_t p = 0; p < na_array.size(); ++p) {
        for (size_t q = 0; q < na_array.size(); ++q) {
            E_casscf += gamma1.data()[na_ * p + q] * fci_ints->oei_a(p, q);
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
    E_casscf += Process::environment.molecule()->nuclear_repulsion_energy(reference_wavefunction_->get_dipole_field_strength());
    outfile->Printf("\n\n OneBody: %8.8f TwoBody: %8.8f Frozen: %8.8f "
                    "fci_ints_scalar: %8.8f",
                    OneBody, TwoBody, Frozen, fci_ints_scalar);

    return E_casscf;
}
std::shared_ptr<Matrix> CASSCF::set_frozen_core_orbitals() {
    SharedMatrix Ca = this->Ca();
    Dimension nsopi = this->nsopi();
    Dimension frozen_dim = mo_space_info_->get_dimension("FROZEN_DOCC");
    SharedMatrix C_core(new Matrix("C_core", nirrep_, nsopi, frozen_dim));
    // Need to get the frozen block of the C matrix
    for (size_t h = 0; h < nirrep_; h++) {
        for (int i = 0; i < frozen_dim[h]; i++) {
            C_core->set_column(h, i, Ca->get_column(h, i));
        }
    }

    //    JK_->set_allow_desymmetrization(true);
    JK_->set_do_K(true);
    std::vector<std::shared_ptr<Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<Matrix>>& Cr = JK_->C_right();

    Cl.clear();
    Cl.push_back(C_core);
    Cr.clear();
    Cr.push_back(C_core);

    JK_->compute();

    SharedMatrix F_core = JK_->J()[0];
    SharedMatrix K_core = JK_->K()[0];

    F_core->scale(2.0);
    F_core->subtract(K_core);

    return F_core;
}
ambit::Tensor CASSCF::transform_integrals() {
    if (options_.get_str("SCF_TYPE") == "OUT_OF_CORE") {
        outfile->Printf("\n To use Out_of_core for scf_type, I need to "
                        "implement integral transform with symmetry");
        throw PSIEXCEPTION("Need to use scf_type direct for CASSCF if you want "
                           "conventional integrals");
    }
    /// This function will do an integral transformation using the JK builder
    /// This was borrowed from Kevin Hannon's IntegralTransform Plugin
    size_t nmo_no_froze = mo_space_info_->size("ALL");
    size_t nmo_with_froze = mo_space_info_->size("CORRELATED");
    SharedMatrix CAct(new Matrix("CAct", nsopi_.sum(), na_));
    auto active_abs = mo_space_info_->get_absolute_mo("ACTIVE");

    /// Step 1: Obtain guess MO coefficients C_{mup}
    /// Since I want to use these in a symmetry aware basis,
    /// I will move the C matrix into a Pfitzer ordering

    Dimension nmopi = mo_space_info_->get_dimension("ALL");

    SharedMatrix aotoso = this->aotoso();

    /// I want a C matrix in the C1 basis but symmetry aware
    size_t nso = this->nso();
    nirrep_ = this->nirrep();
    SharedMatrix Call(new Matrix(nso, nmo_no_froze));
    SharedMatrix Ca_sym = this->Ca();
    SharedMatrix Identity(new Matrix("I", nso, nso));
    Identity->identity();

    // Transform from the SO to the AO basis for the C matrix.
    // just transfroms the C_{mu_ao i} -> C_{mu_so i}
    Timer CSO2AO;
    for (size_t h = 0, index = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nmopi[h]; ++i) {
            size_t nao = nso;
            size_t nso = nsopi_[h];

            if (!nso)
                continue;

            C_DGEMV('N', nao, nso, 1.0, aotoso->pointer(h)[0], nso, &Ca_sym->pointer(h)[0][i],
                    nmopi[h], 0.0, &Call->pointer()[0][index], nmopi.sum());

            index += 1;
        }
    }
    outfile->Printf("\n CSO2SO takes %8.4f s.", CSO2AO.get());

    for (size_t v = 0; v < na_; v++) {
        SharedVector Call_vec = Call->get_column(0, active_abs[v]);
        CAct->set_column(0, v, Call_vec);
    }

    ambit::Tensor active_int =
        ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegrals", {nmo_with_froze, na_, na_, na_});
    std::vector<double>& active_int_data = active_int.data();

    /// (pu|xy) = C_{Rx} C_{Sy}(MN|RS)
    ///         = C_{Mp}C_{Nu}J(D_{MN})^{xy})
    ///         = C_{Mp}^{T} D_{MN}^{xy} C_{Nu}

    std::vector<std::pair<std::shared_ptr<Matrix>, std::vector<int>>> D_vec;
    Timer c_dger;
    for (size_t i = 0; i < na_; i++) {
        SharedVector C_i = CAct->get_column(0, i);
        for (size_t j = i; j < na_; j++) {
            SharedMatrix D(new Matrix("D", nso, nso));
            std::vector<int> ij(2);
            ij[0] = i;
            ij[1] = j;
            SharedVector C_j = CAct->get_column(0, j);
            /// D_{uv}^{ij} = C_i C_j^T
            C_DGER(nso, nso, 1.0, &(C_i->pointer()[0]), 1, &(C_j->pointer()[0]), 1, D->pointer()[0],
                   nso);

            D_vec.push_back(std::make_pair(D, ij));
        }
    }
    if (print_ > 1) {
        outfile->Printf("\n C_DGER takes %8.5f", c_dger.get());
    }
    JK_->set_memory(Process::environment.get_memory() * 0.8);
    //    JK_->set_allow_desymmetrization(false);
    JK_->set_do_K(false);
    // JK_->initialize();
    std::vector<std::shared_ptr<Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<Matrix>>& Cr = JK_->C_right();
    Cl.clear();
    Cr.clear();
    for (size_t d = 0; d < D_vec.size(); d++) {
        Cl.push_back(D_vec[d].first);
        Cr.push_back(Identity);
    }
    Timer jk_build;
    JK_->compute();
    if (print_ > 1) {
        outfile->Printf("\n JK builder takes %8.6f s", jk_build.get());
    }

    SharedMatrix half_trans(new Matrix("Trans", nmo_no_froze, na_));
    int count = 0;
    auto absolute_all = mo_space_info_->get_absolute_mo("CORRELATED");
    auto corr_abs = mo_space_info_->get_corr_abs_mo("CORRELATED");
    int p = 0;
    for (auto d : D_vec) {
        int i = d.second[0];
        int j = d.second[1];
        SharedMatrix J = JK_->J()[count];
        half_trans->zero();
        half_trans = Matrix::triplet(Call, J, CAct, true, false, false);
        count++;
        for (size_t p = 0; p < nmo_with_froze; p++) {
            for (size_t q = 0; q < na_; q++) {
                active_int_data[corr_abs[p] * na_ * na_ * na_ + i * na_ * na_ + q * na_ + j] =
                    half_trans->get(absolute_all[p], q);
                active_int_data[corr_abs[p] * na_ * na_ * na_ + j * na_ * na_ + q * na_ + i] =
                    half_trans->get(absolute_all[p], q);
            }
        }
    }
    return active_int;
}
void CASSCF::set_up_fci() {
    Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
    size_t nfdocc = mo_space_info_->size("FROZEN_DOCC");
    std::vector<size_t> rdocc = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> active = mo_space_info_->get_corr_abs_mo("ACTIVE");

    int charge = Process::environment.molecule()->molecular_charge();
    if (options_["CHARGE"].has_changed()) {
        charge = options_.get_int("CHARGE");
    }

    int nel = 0;
    int natom = Process::environment.molecule()->natom();
    for (int i = 0; i < natom; i++) {
        nel += static_cast<int>(Process::environment.molecule()->Z(i));
    }
    // If the charge has changed, recompute the number of electrons
    // Or if you cannot find the number of electrons
    nel -= charge;

    int multiplicity = Process::environment.molecule()->multiplicity();
    if (options_["MULTIPLICITY"].has_changed()) {
        multiplicity = options_.get_int("MULTIPLICITY");
    }
    if (options_["MULTIPLICITY"].has_changed() && options_["CASSCF_MULTIPLICITY"].has_changed()) {
        multiplicity = options_.get_int("CASSCF_MULTIPLICITY");
    }

    // Default: lowest spin solution
    int twice_ms = (multiplicity + 1) % 2;

    if (options_["MS"].has_changed()) {
        twice_ms = std::round(2.0 * options_.get_double("MS"));
    }

    if (twice_ms < 0) {
        outfile->Printf("\n  Ms must be no less than 0.");
        outfile->Printf("\n  Ms = %2d, MULTIPLICITY = %2d", twice_ms, multiplicity);
        outfile->Printf("\n  Check (specify) Ms value (component of multiplicity)! \n");
        throw PSIEXCEPTION("Ms must be no less than 0. Check output for details.");
    }

    if (options_.get_int("PRINT")) {
        print_h2("FCI Summary");
        outfile->Printf("\n    Number of electrons: %d", nel);
        outfile->Printf("\n    Charge: %d", charge);
        outfile->Printf("\n    Multiplicity: %d", multiplicity);
        outfile->Printf("\n    Davidson subspace max dim: %d",
                        options_.get_int("DL_SUBSPACE_PER_ROOT"));
        outfile->Printf("\n    Davidson subspace min dim: %d",
                        options_.get_int("DL_COLLAPSE_PER_ROOT"));
        if (twice_ms % 2 == 0) {
            outfile->Printf("\n    M_s: %d", twice_ms / 2);
        } else {
            outfile->Printf("\n    M_s: %d/2", twice_ms);
        }
    }

    if (((nel - twice_ms) % 2) != 0)
        throw PSIEXCEPTION("\n\n  FCI: Wrong value of M_s.\n\n");

    // Adjust the number of for frozen and restricted doubly occupied
    size_t nactel = nel - 2 * nfdocc - 2 * rdocc.size();

    size_t na = (nactel + twice_ms) / 2;
    size_t nb = nactel - na;

    FCISolver fcisolver(active_dim, rdocc, active, na, nb, multiplicity,
                        options_.get_int("ROOT_SYM"), ints_, mo_space_info_,
                        options_.get_int("NTRIAL_PER_ROOT"), options_.get_int("PRINT"), options_);
    // tweak some options
    fcisolver.set_max_rdm_level(2);
    fcisolver.set_nroot(options_.get_int("NROOT"));
    fcisolver.set_root(options_.get_int("ROOT"));
    fcisolver.set_test_rdms(options_.get_bool("FCI_TEST_RDMS"));
    fcisolver.set_fci_iterations(options_.get_int("FCI_MAXITER"));
    fcisolver.set_collapse_per_root(options_.get_int("DL_COLLAPSE_PER_ROOT"));
    fcisolver.set_subspace_per_root(options_.get_int("DL_SUBSPACE_PER_ROOT"));
    fcisolver.set_print_no(false);

    std::shared_ptr<FCIIntegrals> fci_ints = std::make_shared<FCIIntegrals>(ints_, active, rdocc);
    if (!(options_.get_bool("RESTRICTED_DOCC_JK"))) {
        ints_->retransform_integrals();
        fci_ints->set_active_integrals_and_restricted_docc();
    } else {
        auto na_array = mo_space_info_->get_corr_abs_mo("ACTIVE");
        fcisolver.use_user_integrals_and_restricted_docc(true);

        ambit::Tensor active_aa =
            ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsAA", {na_, na_, na_, na_});
        ambit::Tensor active_ab =
            ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsAB", {na_, na_, na_, na_});
        ambit::Tensor active_bb =
            ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsBB", {na_, na_, na_, na_});
        const std::vector<double>& tei_paaa_data = tei_paaa_.data();

        active_ab.iterate([&](const std::vector<size_t>& i, double& value) {
            value = tei_paaa_data[na_array[i[0]] * na_ * na_ * na_ + i[1] * na_ * na_ + i[2] * na_ +
                                  i[3]];
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
            fcisolver.set_integral_pointer(fci_ints);
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
            fcisolver.set_integral_pointer(fci_ints);
        }
    }

    E_casscf_ = fcisolver.compute_energy();
    /// Get the CIVector for each iteration
    std::vector<std::shared_ptr<FCIWfn>> FCIWfnSolution(1);
    FCIWfnSolution.push_back(fcisolver.get_FCIWFN());
    CISolutions_.push_back(FCIWfnSolution);

    cas_ref_ = fcisolver.reference();
}
std::vector<std::vector<double>> CASSCF::compute_restricted_docc_operator() {
    ///
    Dimension restricted_docc_dim = mo_space_info_->get_dimension("INACTIVE_DOCC");
    Dimension nsopi = this->nsopi();
    int nirrep = this->nirrep();
    Dimension nmopi = mo_space_info_->get_dimension("ALL");

    SharedMatrix Cdocc(new Matrix("C_RESTRICTED", nirrep, nsopi, restricted_docc_dim));
    SharedMatrix Ca = this->Ca();
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
    //    JK_inactive->set_memory(Process::environment.get_memory() * 0.8);
    //    JK_inactive->initialize();
    //
    std::vector<std::shared_ptr<Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<Matrix>>& Cr = JK_->C_right();
    //    JK_->set_allow_desymmetrization(true);
    JK_->set_do_K(true);
    Cl.clear();
    Cl.push_back(Cdocc);
    Cr.clear();
    Cr.push_back(Cdocc);
    JK_->compute();
    SharedMatrix J_restricted = JK_->J()[0];
    SharedMatrix K_restricted = JK_->K()[0];

    J_restricted->scale(2.0);
    SharedMatrix F_restricted = J_restricted->clone();
    F_restricted->subtract(K_restricted);

    /// Just create the OneInt integrals from scratch

    SharedMatrix Hcore(Hcore_->clone());
    F_restricted->add(Hcore);
    F_restricted->transform(Ca);
    Hcore->transform(Ca);

    size_t all_nmo = mo_space_info_->size("ALL");
    SharedMatrix F_restric_c1(new Matrix("F_restricted", all_nmo, all_nmo));
    size_t offset = 0;
    for (int h = 0; h < nirrep; h++) {
        for (int p = 0; p < nmopi[h]; p++) {
            for (int q = 0; q < nmopi[h]; q++) {
                F_restric_c1->set(p + offset, q + offset, F_restricted->get(h, p, q));
            }
        }
        offset += nmopi[h];
    }
    size_t nmo2 = na_ * na_;
    std::vector<double> oei_a(nmo2);
    std::vector<double> oei_b(nmo2);

    auto absolute_active = mo_space_info_->get_absolute_mo("ACTIVE");
    for (size_t u = 0; u < na_; u++) {
        for (size_t v = 0; v < na_; v++) {
            double value = F_restric_c1->get(absolute_active[u], absolute_active[v]);
            // double h_value = H->get(absolute_active[u], absolute_active[v]);
            oei_a[u * na_ + v] = value;
            oei_b[u * na_ + v] = value;
            if (casscf_debug_print_)
                outfile->Printf("\n oei(%d, %d) = %8.8f", u, v, value);
        }
    }
    Dimension restricted_docc = mo_space_info_->get_dimension("INACTIVE_DOCC");
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
void CASSCF::overlap_orbitals(const SharedMatrix& C_old, const SharedMatrix& C_new) {
    SharedMatrix S_orbitals(new Matrix("Overlap", this->nsopi(), this->nsopi()));
    SharedMatrix S_basis = this->S();
    S_orbitals = Matrix::triplet(C_old, S_basis, C_new, true, false, false);
    S_orbitals->set_name("C^T S C (Overlap)");
    for (size_t h = 0; h < nirrep_; h++) {
        for (int i = 0; i < S_basis->rowspi(h); i++) {
            if (std::fabs(S_basis->get(h, i, i) - 1.0000000) > 1e-6) {
                //    S_basis->get_row(h, i)->print();
            }
        }
    }
}
void CASSCF::set_up_sa_fci() {
    SA_FCISolver sa_fcisolver(options_, reference_wavefunction_);
    sa_fcisolver.set_mo_space_info(mo_space_info_);
    sa_fcisolver.set_integrals(ints_);
    std::vector<size_t> rdocc = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> active = mo_space_info_->get_corr_abs_mo("ACTIVE");
    std::shared_ptr<FCIIntegrals> fci_ints = std::make_shared<FCIIntegrals>(ints_, active, rdocc);
    auto na_array = mo_space_info_->get_corr_abs_mo("ACTIVE");

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
    //    if (options_.get_bool("MONITOR_SA_SOLUTION")) {
    //        std::vector<std::shared_ptr<FCIWfn>> StateAveragedFCISolver =
    //            sa_fcisolver.StateAveragedCISolution();
    //        CISolutions_.push_back(StateAveragedFCISolver);
    //    }
}
void CASSCF::set_up_fcimo() {
    // setup FCIIntegrals for FCI_MO
    std::vector<size_t> rdocc = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> active = mo_space_info_->get_corr_abs_mo("ACTIVE");
    std::shared_ptr<FCIIntegrals> fci_ints = std::make_shared<FCIIntegrals>(ints_, active, rdocc);

    if (!(options_.get_bool("RESTRICTED_DOCC_JK"))) {
        ints_->retransform_integrals();
        fci_ints->set_active_integrals_and_restricted_docc();
    } else {
        auto na_array = mo_space_info_->get_corr_abs_mo("ACTIVE");

        ambit::Tensor active_aa =
            ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsAA", {na_, na_, na_, na_});
        ambit::Tensor active_ab =
            ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsAB", {na_, na_, na_, na_});
        ambit::Tensor active_bb =
            ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsBB", {na_, na_, na_, na_});
        const std::vector<double>& tei_paaa_data = tei_paaa_.data();

        active_ab.iterate([&](const std::vector<size_t>& i, double& value) {
            value = tei_paaa_data[na_array[i[0]] * na_ * na_ * na_ + i[1] * na_ * na_ + i[2] * na_ +
                                  i[3]];
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
        }
    }

    FCI_MO cas(reference_wavefunction_, options_, ints_, mo_space_info_, fci_ints);
    cas.set_quite_mode(print_ > 0 ? false : true);
    cas.compute_energy();
    cas_ref_ = cas.reference(2);
    E_casscf_ = cas_ref_.get_Eref();
}
void CASSCF::write_orbitals_molden() {
    SharedVector occ_vector(new Vector(nirrep_, nmopi_));
    view_modified_orbitals(reference_wavefunction_, this->Ca(), this->epsilon_a(), occ_vector);
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

    auto na_array = mo_space_info_->get_corr_abs_mo("ACTIVE");
    const std::vector<double>& tei_paaa_data = tei_paaa_.data();
    ambit::Tensor active_ab =
        ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsBB", {na_, na_, na_, na_});

    active_ab.iterate([&](const std::vector<size_t>& i, double& value) {
        value =
            tei_paaa_data[na_array[i[0]] * na_ * na_ * na_ + i[1] * na_ * na_ + i[2] * na_ + i[3]];
    });
    std::pair<ambit::Tensor, std::vector<double>> pair_return =
        std::make_pair(active_ab, oei_vector[0]);
    return pair_return;
}
}
}
