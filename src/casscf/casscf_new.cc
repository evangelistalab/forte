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

#include "casscf/casscf_new.h"
#include "integrals/integrals.h"
#include "base_classes/rdms.h"

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
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

using namespace ambit;

namespace forte {

CASSCF_NEW::CASSCF_NEW(const std::map<StateInfo, std::vector<double>>& state_weights_map,
                       std::shared_ptr<ForteOptions> options,
                       std::shared_ptr<MOSpaceInfo> mo_space_info,
                       std::shared_ptr<forte::SCFInfo> scf_info,
                       std::shared_ptr<ForteIntegrals> ints)
    : state_weights_map_(state_weights_map), options_(options), mo_space_info_(mo_space_info),
      scf_info_(scf_info), ints_(ints) {
    startup();
}

void CASSCF_NEW::startup() {
    print_method_banner({"Complete Active Space Self Consistent Field",
                         "written by Kevin P. Hannon and Chenyang Li"});

    // nuclear repulsion energy
    e_nuc_ = ints_->nuclear_repulsion_energy();

    // setup MO spaces
    setup_mos();

    // read and print options
    read_options();
    print_options();

    // nonredundant pairs
    nonredundant_pairs();

    // setup JK
    setup_JK();

    // setup ambit spaces
    setup_ambit();

    // allocate memory for tensors and matrices
    init_tensors();
}

void CASSCF_NEW::setup_mos() {
    nirrep_ = mo_space_info_->nirrep();

    nsopi_ = ints_->nsopi();
    nmopi_ = mo_space_info_->dimension("ALL");
    ncmopi_ = mo_space_info_->dimension("CORRELATED");
    ndoccpi_ = mo_space_info_->dimension("INACTIVE_DOCC");
    nfrzcpi_ = mo_space_info_->dimension("FROZEN_DOCC");
    nactvpi_ = mo_space_info_->dimension("ACTIVE");

    nso_ = nsopi_.sum();
    nmo_ = mo_space_info_->size("ALL");
    ncmo_ = mo_space_info_->size("CORRELATED");
    nactv_ = mo_space_info_->size("ACTIVE");

    actv_sym_ = mo_space_info_->symmetry("ACTIVE");

    core_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->corr_absolute_mo("ACTIVE");
    virt_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");

    label_to_mos_.clear();
    label_to_mos_["c"] = core_mos_;
    label_to_mos_["a"] = actv_mos_;
    label_to_mos_["v"] = virt_mos_;

    // in Pitzer ordering
    corr_mos_rel_.clear();
    corr_mos_rel_.reserve(ncmo_);
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0, offset = nfrzcpi_[h]; i < ncmopi_[h]; ++i) {
            corr_mos_rel_.push_back(std::make_pair(h, i + offset));
        }
    }

    // in Pitzer ordering
    corr_mos_rel_space_.resize(ncmo_);
    for (std::string space : {"c", "a", "v"}) {
        const auto& mos = label_to_mos_[space];
        for (size_t p = 0, size = mos.size(); p < size; ++p) {
            corr_mos_rel_space_[mos[p]] = std::make_pair(space, p);
        }
    }

    //    core_mos_abs_ = mo_space_info_->absolute_mo("RESTRICTED_DOCC");
    //    actv_mos_abs_ = mo_space_info_->absolute_mo("ACTIVE");
    //    core_mos_rel_ = mo_space_info_->get_relative_mo("RESTRICTED_DOCC");
    //    actv_mos_rel_ = mo_space_info_->get_relative_mo("ACTIVE");
    //    virt_mos_rel_ = mo_space_info_->get_relative_mo("RESTRICTED_UOCC");

    //    restricted_uocc_dim_ = mo_space_info_->dimension("RESTRICTED_UOCC");
    //    inactive_docc_dim_ = mo_space_info_->dimension("INACTIVE_DOCC");

    //    nmo_ = mo_space_info_->size("ALL");
    //    nrdocc_ = rdocc_mos_.size();
    //    nruocc_ = ruocc_mos_.size();
    //    nfdocc_ = mo_space_info_->size("FROZEN_DOCC");
}

void CASSCF_NEW::read_options() {
    print_ = options_->get_int("PRINT");
    debug_print_ = options_->get_bool("CASSCF_DEBUG_PRINTING");

    int_type_ = options_->get_str("INT_TYPE");

    maxiter_ = options_->get_int("CASSCF_MAXITER");
    e_conv_ = options_->get_double("CASSCF_E_CONVERGENCE");
    g_conv_ = options_->get_double("CASSCF_G_CONVERGENCE");

    // DIIS options
    do_diis_ = options_->get_bool("CASSCF_DO_DIIS");
    diis_freq_ = options_->get_int("CASSCF_DIIS_FREQ");
    diis_start_ = options_->get_int("CASSCF_DIIS_START");
    diis_max_vec_ = options_->get_int("CASSCF_DIIS_MAX_VEC");

    // CI update options
    ci_type_ = options_->get_str("CASSCF_CI_SOLVER");
    ci_freq_ = options_->get_int("CASSCF_CI_FREQ");

    // rotations
    max_rot_ = options_->get_double("CASSCF_MAX_ROTATION");
    internal_rot_ = options_->get_bool("CASSCF_INTERNAL_ROT");

    zero_rots_.clear();
    auto zero_rots = options_->get_gen_list("CASSCF_ZERO_ROT");

    //    if (zero_rots.size() != 0) {
    //        size_t npairs = zero_rots.size();
    //        for (size_t i = 0; i < npairs; ++i) {
    //            py::list pair = zero_rots[i];
    //            if (pair.size() != 3) {
    //                psi::outfile->Printf("\n  Error: invalid input of CASSCF_ZERO_ROT.");
    //                psi::outfile->Printf("\n  Each entry should take an array of three numbers.");
    //                throw std::runtime_error("Invalid input of CASSCF_ZERO_ROT");
    //            }

    //            size_t irrep = py::cast<size_t>(pair[0]);
    //            if (irrep >= nirrep_ or irrep < 0) {
    //                psi::outfile->Printf("\n  Error: invalid irrep in CASSCF_ZERO_ROT.");
    //                psi::outfile->Printf("\n  Check the input irrep (start from 0) not to exceed
    //                %d",
    //                                     nirrep_ - 1);
    //                throw std::runtime_error("Invalid irrep in CASSCF_ZERO_ROT");
    //            }

    //            size_t i1 = py::cast<size_t>(pair[1]) - 1;
    //            size_t i2 = py::cast<size_t>(pair[2]) - 1;
    //            size_t n = nmo_dim_[irrep];
    //            if (i1 >= n or i1 < 0 or i2 >= n or i2 < 0) {
    //                psi::outfile->Printf("\n  Error: invalid orbital indices in
    //                CASSCF_ZERO_ROT."); psi::outfile->Printf("\n  The input orbital indices (start
    //                from 1) should not to "
    //                                     "exceed %d (number of orbitals in irrep %d)",
    //                                     n, irrep);
    //                throw std::runtime_error("Invalid orbital indices in CASSCF_ZERO_ROT");
    //            }

    //            size_t offset = 0, h = 0;
    //            while (h < irrep) {
    //                offset += nmo_dim_[h];
    //                h++;
    //            }

    //            zero_rots_.push_back(std::make_pair(i1 + offset, i2 + offset));
    //        }
    //    }
}

void CASSCF_NEW::print_options() {
    // fill in information
    std::vector<std::pair<std::string, int>> info_int{{"Printing level", print_},
                                                      {"Max number of iterations", maxiter_},
                                                      {"Frequency of doing CI", ci_freq_}};

    std::vector<std::pair<std::string, double>> info_double{{"Energy convergence", e_conv_},
                                                            {"Gradient convergence", g_conv_},
                                                            {"Max value for rotation", max_rot_}};

    std::vector<std::pair<std::string, std::string>> info_string{{"Integral type", int_type_},
                                                                 {"CI solver type", ci_type_}};

    std::vector<std::pair<std::string, bool>> info_bool{
        {"Debug printing", debug_print_}, {"Include internal rotations", internal_rot_}};

    if (do_diis_) {
        info_int.push_back({"DIIS start", diis_start_});
        info_int.push_back({"Max DIIS vectors", diis_max_vec_});
        info_int.push_back({"Frequency of DIIS extrapolation", diis_freq_});
    }

    // print some information
    print_selected_options("Calculation Information", info_string, info_bool, info_double,
                           info_int);
}

void CASSCF_NEW::nonredundant_pairs() {
    // compute the number of nonredundant pairs
    auto cross_pairs = [&](const psi::Dimension& dim1, const psi::Dimension& dim2) {
        std::vector<size_t> out(nirrep_);
        for (int h = 0; h < nirrep_; ++h) {
            out[h] = dim1[h] * dim2[h];
        }
        return out;
    };

    auto ct = psi::Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> spaces{"RESTRICTED_DOCC", "ACTIVE", "RESTRICTED_UOCC"};
    std::map<std::pair<int, int>, std::vector<size_t>> npairs;

    for (int i = 0; i < 3; ++i) {
        auto dim1 = mo_space_info_->dimension(spaces[i]);
        for (int j = i + 1; j < 3; ++j) {
            auto dim2 = mo_space_info_->dimension(spaces[j]);
            npairs[std::make_pair(i, j)] = cross_pairs(dim1, dim2);
        }
    }

    if (internal_rot_) {
        auto actv_dim = mo_space_info_->dimension("ACTIVE");
        std::vector<size_t> active(nirrep_);
        for (int h = 0; h < nirrep_; ++h) {
            size_t n = actv_dim[h];
            active[h] = n * (n - 1) / 2;
        }
        npairs[std::make_pair(1, 1)] = active;
    }

    print_h2("Independent Orbital Rotations");
    outfile->Printf("\n    %-33s", "ORBITAL SPACES");
    for (int h = 0; h < nirrep_; ++h) {
        outfile->Printf("  %4s", ct.gamma(h).symbol());
    }
    outfile->Printf("\n    %s", std::string(33 + nirrep_ * 6, '-').c_str());

    for (const auto& key_value : npairs) {
        const auto& key = key_value.first;
        auto block1 = key.first;
        auto block2 = key.second;
        outfile->Printf("\n    %15s / %15s", spaces[block1].c_str(), spaces[block2].c_str());

        const auto& value = key_value.second;
        for (int h = 0; h < nirrep_; ++h) {
            outfile->Printf("  %4zu", value[h]);
        }
    }
    outfile->Printf("\n    %s\n", std::string(33 + nirrep_ * 6, '-').c_str());
}

void CASSCF_NEW::setup_JK() {
    local_timer jk_timer;

    auto basis_set = ints_->wfn()->basisset();

    if (int_type_.find("DF") != std::string::npos) {
        if (options_->get_str("SCF_TYPE") == "DF") {
            JK_ = JK::build_JK(basis_set, ints_->wfn()->get_basisset("DF_BASIS_SCF"),
                               psi::Process::environment.options, "MEM_DF");
        } else {
            throw psi::PSIEXCEPTION("Inconsistent DF type between Psi4 and Forte");
        }
    } else if (int_type_ == "CHOLESKY") {
        psi::Options& options = psi::Process::environment.options;
        CDJK* jk = new CDJK(basis_set, options_->get_double("CHOLESKY_TOLERANCE"));

        if (options["INTS_TOLERANCE"].has_changed())
            jk->set_cutoff(options.get_double("INTS_TOLERANCE"));
        if (options["SCREENING"].has_changed())
            jk->set_csam(options.get_str("SCREENING") == "CSAM");
        if (options["PRINT"].has_changed())
            jk->set_print(options.get_int("PRINT"));
        if (options["DEBUG"].has_changed())
            jk->set_debug(options.get_int("DEBUG"));
        if (options["BENCH"].has_changed())
            jk->set_bench(options.get_int("BENCH"));
        if (options["DF_INTS_IO"].has_changed())
            jk->set_df_ints_io(options.get_str("DF_INTS_IO"));
        jk->set_condition(options.get_double("DF_FITTING_CONDITION"));
        if (options["DF_INTS_NUM_THREADS"].has_changed())
            jk->set_df_ints_num_threads(options.get_int("DF_INTS_NUM_THREADS"));

        JK_ = std::shared_ptr<JK>(jk);
    } else if (int_type_ == "CONVENTIONAL") {
        JK_ = JK::build_JK(basis_set, psi::BasisSet::zero_ao_basis_set(),
                           psi::Process::environment.options, "PK");
    }

    JK_->set_memory(psi::Process::environment.get_memory() * 0.85);
    JK_->initialize();
    JK_->C_left().clear();
    JK_->C_right().clear();

    if (print_ > 1)
        outfile->Printf("  Initializing JK took %.3f seconds.", jk_timer.get());
}

void CASSCF_NEW::setup_ambit() {
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    BlockedTensor::add_mo_space("c", "i,j", core_mos_, NoSpin);
    BlockedTensor::add_mo_space("a", "t,u,v,w,y,x,z", actv_mos_, NoSpin);
    BlockedTensor::add_mo_space("v", "a,b", virt_mos_, NoSpin);

    BlockedTensor::add_composite_mo_space("o", "k,l", {"c", "a"});
    BlockedTensor::add_composite_mo_space("g", "p,q,r,s", {"c", "a", "v"});
}

void CASSCF_NEW::init_tensors() {
    // save a copy of initial MO
    C0_ = ints_->wfn()->Ca()->clone();

    // save a copy of AO OEI
    H_ao_ = ints_->wfn()->H()->clone();

    // MO OEI
    H_mo_ = std::make_shared<psi::Matrix>("Hbare_MO", nmopi_, nmopi_);

    // Fock matrices
    Fd_.resize(ncmo_);
    Fock_ = std::make_shared<psi::Matrix>("Fock_MO", nmopi_, nmopi_);
    F_closed_ = std::make_shared<psi::Matrix>("Fock_inactive", nmopi_, nmopi_);
    F_active_ = std::make_shared<psi::Matrix>("Fock_active", nmopi_, nmopi_);

    auto tensor_type = ambit::CoreTensor;
    Fc_ = ambit::BlockedTensor::build(tensor_type, "Fc", {"gg"});
    F_ = ambit::BlockedTensor::build(tensor_type, "F", {"gg"});

    // two-electron integrals
    V_ = ambit::BlockedTensor::build(tensor_type, "V", {"gaaa"});

    // 1-RDM and 2-RDM
    D1_ = ambit::BlockedTensor::build(tensor_type, "1RDM", {"aa"});
    D2_ = ambit::BlockedTensor::build(tensor_type, "2RDM", {"aaaa"});
    rdm1_ = std::make_shared<psi::Matrix>("1RDM", nactvpi_, nactvpi_);

    // orbital gradients related
    std::vector<std::string> g_blocks{"ac", "vo"};
    if (internal_rot_) {
        g_blocks.push_back("aa");
        Guu_ = ambit::BlockedTensor::build(CoreTensor, "Guu", {"aa"});
        Guv_ = ambit::BlockedTensor::build(CoreTensor, "Guv", {"aa"});
        jk_internal_ = ambit::BlockedTensor::build(CoreTensor, "tei_internal", {"aaa"});
        d2_internal_ = ambit::BlockedTensor::build(CoreTensor, "rdm_internal", {"aaa"});
    }
    A_ = ambit::BlockedTensor::build(tensor_type, "A", {"ca", "ao", "vo"});
    g_ = ambit::BlockedTensor::build(tensor_type, "g", g_blocks);
    h_diag_ = ambit::BlockedTensor::build(tensor_type, "h_diag", g_blocks);
}

double CASSCF_NEW::compute_energy() {
    C_ = C0_->clone();
    build_mo_integrals();

    diagonalize_hamiltonian();

    build_fock();

    compute_orbital_grad();

    compute_orbital_hess_diag();

    return 0.0;

    // Provide a nice summary at the end for iterations
    std::vector<int> iter_con;

    //    // DIIS options
    //    double diis_gradient_norm = options_->get_double("CASSCF_DIIS_NORM");

    //    // CI update options
    //    bool ci_step = options_->get_bool("CASSCF_CI_STEP");

    //    psi::Dimension nhole_dim = mo_space_info_->dimension("GENERALIZED HOLE");
    //    psi::Dimension npart_dim = mo_space_info_->dimension("GENERALIZED PARTICLE");
    //    psi::SharedMatrix S(new psi::Matrix("Orbital Rotation", nirrep_, nhole_dim, npart_dim));
    //    psi::SharedMatrix Sstep;

    //    // Setup the DIIS manager
    //    auto diis_manager = std::make_shared<DIISManager>(
    //        diis_max_vec_, "MCSCF DIIS", DIISManager::OldestAdded, DIISManager::InCore);
    //    diis_manager->set_error_vector_size(1, DIISEntry::Matrix, S.get());
    //    diis_manager->set_vector_size(1, DIISEntry::Matrix, S.get());

    //    int diis_count = 0;

    //    E_casscf_ = 0.0;
    //    double E_casscf_old = 0.0, Ediff = 0.0;

    //    psi::SharedMatrix Ca = wfn_->Ca();

    //    print_h2("CASSCF_NEW Iteration");
    //    outfile->Printf("\n  iter    ||g||           Delta_E            E_CASSCF_NEW CONV_TYPE");

    //    for (int iter = 1; iter <= maxiter_; iter++) {
    //        local_timer casscf_total_iter;

    //        local_timer trans_ints_timer;
    //        build_tei_from_ao(Ca);
    //        if (print_ > 1) {
    //            outfile->Printf("\n\n  Transform Integrals takes %8.8f s.",
    //            trans_ints_timer.get());
    //        }
    //        iter_con.push_back(iter);

    //        // Perform a CASCI
    //        E_casscf_old = E_casscf_;
    //        if (print_ > 0) {
    //            std::string ci_type = options_->get_str("CASSCF_CI_SOLVER");
    //            outfile->Printf("\n\n  Performing a CAS with %s", ci_type.c_str());
    //        }

    //        local_timer cas_timer;

    //        if (iter == 1 or iter % ci_freq_ == 0) {
    //            diagonalize_hamiltonian();
    //        } else {
    //            compute_reference_energy();
    //        }

    //        if (print_ > 0) {
    //            outfile->Printf("\n\n CAS took %8.6f seconds.", cas_timer.get());
    //        }

    //        CASSCFOrbitalOptimizer orbital_optimizer(gamma1_, gamma2_, tei_gaaa_, options_,
    //                                                 mo_space_info_, ints_);

    //        orbital_optimizer.set_frozen_one_body(F_frozen_core_);
    //        orbital_optimizer.set_symmmetry_mo(Ca);
    //        orbital_optimizer.one_body(Hcore_->clone());
    //        if (print_ > 0) {
    //            orbital_optimizer.set_print_timings(true);
    //        }
    //        orbital_optimizer.set_jk(JK_);
    //        orbital_optimizer.update();
    //        double g_norm = orbital_optimizer.orbital_gradient_norm();

    //        Ediff = E_casscf_ - E_casscf_old;
    //        if (iter > 1 && std::fabs(Ediff) < e_conv_ && g_norm < g_conv_) {

    //            outfile->Printf("\n  %4d   %10.12f   %10.12f   %10.12f  %10.6f s", iter, g_norm,
    //            Ediff,
    //                            E_casscf_, casscf_total_iter.get());

    //            outfile->Printf("\n\n  A miracle has come to pass: CASSCF iterations have
    //            converged."); break;
    //        }

    //        Sstep = orbital_optimizer.approx_solve();

    //        // Max rotation
    //        double maxS = Sstep->absmax();
    //        if (maxS > max_rot_) {
    //            Sstep->scale(max_rot_ / maxS);
    //        }

    //        // Add step to overall rotation
    //        S->add(Sstep);

    //        // TODO:  Add options controlled.  Iteration and g_norm
    //        if (do_diis_ and (iter >= diis_start_ or g_norm < diis_gradient_norm)) {
    //            diis_manager->add_entry(2, Sstep.get(), S.get());
    //            diis_count++;
    //        }

    //        if (do_diis_ and iter > diis_start_ and (diis_count % diis_freq_ == 0)) {
    //            diis_manager->extrapolate(1, S.get());
    //        }
    //        psi::SharedMatrix Cp = orbital_optimizer.rotate_orbitals(C0_, S);

    //        // update MO coefficients
    //        Ca->copy(Cp);

    //        std::string diis_start_label = "";
    //        if (do_diis_ and (iter > diis_start_ or g_norm < diis_gradient_norm)) {
    //            diis_start_label = "DIIS";
    //        }
    //        outfile->Printf("\n %4d %14.12f %18.12f %18.12f %6.1f s  %4s ~", iter, g_norm, Ediff,
    //                        E_casscf_, casscf_total_iter.get(), diis_start_label.c_str());
    //    }

    //    diis_manager->delete_diis_file();
    //    diis_manager.reset();

    //    // if(casscf_debug_print_)
    //    //{
    //    //    overlap_orbitals(this->Ca(), C_start);
    //    //}
    //    //    if (options_->get_bool("MONITOR_SA_SOLUTION")) {
    //    //        overlap_coefficients();
    //    //    }

    //    outfile->Printf("\n\n @ Final CASSCF_NEW Energy = %20.15f\n", E_casscf_);
    //    if (iter_con.size() == size_t(maxiter_) && maxiter_ > 1) {
    //        outfile->Printf("\n CASSCF_NEW did not converge");
    //        throw psi::PSIEXCEPTION("CASSCF_NEW did not converge.");
    //    }

    //    // semicanonicalize orbitals
    //    auto U = semicanonicalize(Ca);
    //    auto Ca_semi = linalg::doublet(Ca, U, false, false);
    //    Ca_semi->set_name(Ca->name());

    //    // restransform integrals if derivatives are needed
    //    if (options_->get_str("DERTYPE") == "FIRST") {
    //        print_h2("Final Integral Transformation for CASSCF_NEW Gradients");
    //        ints_->update_orbitals(Ca_semi, Ca_semi);

    //        // diagonalize the Hamiltonian one last time
    //        diagonalize_hamiltonian();
    //    } else {
    //        wfn_->Ca()->copy(Ca_semi);
    //    }

    //    psi::Process::environment.globals["CURRENT ENERGY"] = E_casscf_;
    //    psi::Process::environment.globals["CASSCF_ENERGY"] = E_casscf_;

    //    return E_casscf_;
}

void CASSCF_NEW::build_mo_integrals() {
    // form closed-shell Fock matrix
    build_fock_inactive();

    // bare one-electron integrals
    H_mo_ = H_ao_->clone();
    H_mo_->transform(C_);
    H_mo_->set_name("Hbare_MO");

    // compute the closed-shell energy
    compute_energy_closed();

    // form the MO 2e-integrals
    build_tei_from_ao();
}

void CASSCF_NEW::compute_energy_closed() {
    e_frozen_ = 0.0, e_closed_ = 0.0;
    for (int h = 0; h < nirrep_; ++h) {

        for (int i = 0; i < ndoccpi_[h]; ++i) {
            e_closed_ += F_closed_->get(h, i, i) + H_mo_->get(h, i, i);
        }

        for (int i = 0; i < nfrzcpi_[h]; ++i) {
            e_frozen_ += F_closed_->get(h, i, i) + H_mo_->get(h, i, i);
        }
    }
}

void CASSCF_NEW::build_tei_from_ao() {
    // This function will do an integral transformation using the JK builder,
    // and return the integrals of type <px|uy> = (pu|xy).
    timer_on("Build (pu|xy) integrals");

    // Transform C matrix to C1 symmetry (JK will do this anyway)
    psi::SharedMatrix aotoso = ints_->wfn()->aotoso();
    auto C_nosym = std::make_shared<psi::Matrix>(nso_, ncmo_);

    // Transform from the SO to the AO basis for the C matrix
    // MO in Pitzer ordering and only keep the non-frozen MOs
    for (int h = 0, index = 0; h < nirrep_; ++h) {
        for (int i = 0, offset = nfrzcpi_[h]; i < ncmopi_[h]; ++i) {
            int nao = nso_, nso = nsopi_[h];

            if (!nso)
                continue;

            C_DGEMV('N', nao, nso, 1.0, aotoso->pointer(h)[0], nso, &C_->pointer(h)[0][i + offset],
                    nmopi_[h], 0.0, &C_nosym->pointer()[0][index], ncmo_);

            index += 1;
        }
    }

    // set up the active part of the C matrix
    auto Cact = std::make_shared<psi::Matrix>("Cact", nso_, nactv_);
    std::vector<std::shared_ptr<psi::Matrix>> Cact_vec(nactv_);

    for (size_t x = 0; x < nactv_; ++x) {
        psi::SharedVector Ca_nosym_vec = C_nosym->get_column(0, actv_mos_[x]);
        Cact->set_column(0, x, Ca_nosym_vec);

        std::string name = "Cact slice " + std::to_string(x);
        auto temp = std::make_shared<psi::Matrix>(name, nso_, 1);
        temp->set_column(0, 0, Ca_nosym_vec);
        Cact_vec[x] = temp;
    }

    // The following type of integrals are needed:
    // (pu|xy) = C_{Mp}^T C_{Nu} C_{Rx}^T C_{Sy} (MN|RS)
    //         = C_{Mp}^T C_{Nu} J_{MN}^{xy}
    //         = C_{Mp}^T J_{MN}^{xy} C_{Nu}

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

    // second-half transformation and fill in to BlockedTensor
    size_t nactv2 = nactv_ * nactv_;
    size_t nactv3 = nactv2 * nactv_;

    for (size_t x = 0, offset = 0; x < nactv_; ++x) {
        offset += x;
        for (size_t y = x; y < nactv_; ++y) {
            std::shared_ptr<psi::Matrix> J = JK_->J()[x * nactv_ + y - offset];
            auto half_trans = psi::linalg::triplet(C_nosym, J, Cact, true, false, false);

            for (size_t p = 0; p < ncmo_; ++p) {
                // grab the block data
                std::string p_space = corr_mos_rel_space_[p].first;
                std::string block = p_space + "aaa";

                size_t np = corr_mos_rel_space_[p].second;
                auto& data = V_.block(block).data();

                for (size_t u = 0; u < nactv_; ++u) {
                    double value = half_trans->get(p, u);

                    data[np * nactv3 + u * nactv2 + x * nactv_ + y] = value;
                    data[np * nactv3 + u * nactv2 + y * nactv_ + x] = value;
                }
            }
        }
    }
}

void CASSCF_NEW::build_fock_inactive() {
    // Implementation Notes (in AO basis)
    // F_frozen = D_{uv}^{frozen} * (2 * (uv|rs) - (us|rv))
    // F_restricted = D_{uv}^{restricted} * (2 * (uv|rs) - (us|rv))
    // F_inactive = Hcore + F_frozen + F_restricted
    // D_{uv}^{frozen} = \sum_{i}^{frozen} C_{ui} * C_{vi}
    // D_{uv}^{restricted} = \sum_{i}^{restricted} C_{ui} * C_{vi}

    // grab part of Ca for inactive docc
    auto Cdocc = std::make_shared<psi::Matrix>("C_INACTIVE", nirrep_, nsopi_, ndoccpi_);
    for (int h = 0; h < nirrep_; h++) {
        for (int i = 0; i < ndoccpi_[h]; i++) {
            Cdocc->set_column(h, i, C_->get_column(h, i));
        }
    }

    // JK build
    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();

    JK_->set_do_K(true);
    Cl.clear();
    Cr.clear();
    Cl.push_back(Cdocc); // Cr is the same as Cl
    JK_->compute();

    F_closed_->copy(JK_->J()[0]);
    F_closed_->scale(2.0);
    F_closed_->subtract(JK_->K()[0]);

    F_closed_->add(H_ao_);
    F_closed_->transform(C_);

    // put it in Ambit BlockedTensor format
    format_fock(F_closed_, Fc_);

    if (debug_print_) {
        F_closed_->print();
    }
}

void CASSCF_NEW::build_fock(bool rebuild_inactive) {
    if (rebuild_inactive) {
        build_fock_inactive();
    }

    build_fock_active();

    Fock_->copy(F_closed_);
    Fock_->add(F_active_);

    format_fock(Fock_, F_);

    // fill in diagonal Fock in Pitzer ordering
    for (const std::string& space : {"c", "a", "v"}) {
        std::string block = space + space;
        auto mos = label_to_mos_[space];
        for (size_t i = 0, size = mos.size(); i < size; ++i) {
            Fd_[mos[i]] = F_.block(block).data()[i * size + i];
        }
    }
}

void CASSCF_NEW::build_fock_active() {
    // Implementation Notes (in AO basis)
    // F_active = D_{uv}^{active} * ( (uv|rs) - 0.5 * (us|rv) )
    // D_{uv}^{active} = \sum_{xy}^{active} C_{ux} * C_{vy} * Gamma1_{xy}

    // grab part of Ca for active
    auto Cactv = std::make_shared<psi::Matrix>("C_ACTIVE", nirrep_, nsopi_, nactvpi_);
    for (int h = 0; h < nirrep_; h++) {
        int offset = nfrzcpi_[h] + ndoccpi_[h];
        for (int i = 0; i < nactvpi_[h]; i++) {
            Cactv->set_column(h, i, C_->get_column(h, i + offset));
        }
    }

    // dress Cactv by one-density, which will the C_right for JK
    auto Cactv_dressed = linalg::doublet(Cactv, rdm1_, false, false);

    // JK build
    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();

    JK_->set_do_K(true);
    Cl.clear();
    Cr.clear();
    Cl.push_back(Cactv);
    Cr.push_back(Cactv_dressed);
    JK_->compute();

    F_active_->copy(JK_->K()[0]);
    F_active_->scale(-0.5);
    F_active_->add(JK_->J()[0]);

    // transform to MO
    F_active_->transform(C_);

    if (debug_print_) {
        F_active_->print();
    }
}

void CASSCF_NEW::format_fock(psi::SharedMatrix Fock, ambit::BlockedTensor F) {
    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        auto irrep_index_pair1 = corr_mos_rel_[i[0]];
        auto irrep_index_pair2 = corr_mos_rel_[i[1]];

        int h1 = irrep_index_pair1.first;
        int h2 = irrep_index_pair2.first;

        if (h1 == h2) {
            auto p = irrep_index_pair1.second;
            auto q = irrep_index_pair2.second;
            value = Fock->get(h1, p, q);
        } else {
            value = 0.0;
        }
    });
}

void CASSCF_NEW::compute_reference_energy() {
    // compute energy given that all useful MO integrals are available
    energy_ = e_nuc_ + e_closed_;
    energy_ += Fc_["uv"] * D1_["uv"];
    energy_ += 0.5 * V_["uvxy"] * D2_["uvxy"];

    if (debug_print_) {
        outfile->Printf("\n  Frozen-core energy   %20.15f", e_frozen_);
        outfile->Printf("\n  Closed-shell energy  %20.15f", e_closed_);
    }
}

void CASSCF_NEW::diagonalize_hamiltonian() {
    // prepare integrals for active-space solver
    auto fci_ints = std::make_shared<ActiveSpaceIntegrals>(ints_, actv_mos_, actv_sym_, core_mos_);

    // change to antisymmetrized TEI in physists' notation
    auto actv_ab = ambit::Tensor::build(CoreTensor, "tei_actv_aa", std::vector<size_t>(4, nactv_));
    actv_ab("pqrs") = V_.block("aaaa")("prqs");

    auto actv_aa = ambit::Tensor::build(CoreTensor, "tei_actv_aa", std::vector<size_t>(4, nactv_));
    actv_aa.copy(actv_ab);
    actv_aa("uvxy") -= actv_ab("uvyx");

    fci_ints->set_active_integrals(actv_aa, actv_ab, actv_aa);

    auto& oei = Fc_.block("aa").data();
    fci_ints->set_restricted_one_body_operator(oei, oei);

    fci_ints->set_scalar_energy(e_closed_ - e_frozen_);

    // diagonalize Hamiltonian
    auto state_map = to_state_nroots_map(state_weights_map_);
    auto active_space_solver = make_active_space_solver(ci_type_, state_map, scf_info_,
                                                        mo_space_info_, fci_ints, options_);
    active_space_solver->set_print(print_);
    const auto state_energies_map = active_space_solver->compute_energy();

    // compute RDMs
    auto rdms = active_space_solver->compute_average_rdms(state_weights_map_, 2);
    energy_ = compute_average_state_energy(state_energies_map, state_weights_map_);

    D1_.block("aa").copy(rdms.g1a());
    D1_.block("aa")("pq") += rdms.g1b()("pq");
    format_1rdm();

    // change to chemists' notation
    D2_.block("aaaa")("pqrs") = rdms.SFg2()("prqs");
    D2_.block("aaaa")("pqrs") += rdms.SFg2()("qrps");
    D2_.scale(0.5);
}

void CASSCF_NEW::format_1rdm() {
    rdm1_->zero();
    const auto& d1_data = D1_.block("aa").data();

    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        for (int u = 0; u < nactvpi_[h]; ++u) {
            size_t nu = u + offset;
            for (int v = 0; v < nactvpi_[h]; ++v) {
                rdm1_->set(h, u, v, d1_data[nu * nactv_ + v + offset]);
            }
        }
        offset += nactvpi_[h];
    }

    if (debug_print_) {
        rdm1_->print();
    }
}

void CASSCF_NEW::compute_orbital_grad() {
    // build orbital Lagrangian
    A_["ri"] = 2.0 * F_["ri"];
    A_["ru"] = Fc_["rt"] * D1_["tu"];
    A_["ru"] += V_["rtvw"] * D2_["tuvw"];

    // build orbital gradients
    g_["pq"] = 2.0 * A_["pq"];
    g_["pq"] -= 2.0 * A_["qp"];
}

void CASSCF_NEW::compute_orbital_hess_diag() {
    // modified diagonal Hessian from Theor. Chem. Acc. 97, 88-95 (1997)

    // virtual-core block
    h_diag_.block("vc").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = label_to_mos_["v"][i[0]];
        auto i1 = label_to_mos_["c"][i[1]];
        value = 4.0 * (Fd_[i0] - Fd_[i1]);
    });

    // virtual-active block
    auto& d1_data = D1_.block("aa").data();
    auto& a_data = A_.block("aa").data();

    h_diag_.block("va").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = label_to_mos_["v"][i[0]];
        auto i1 = i[1] * nactv_ + i[1];
        value = 2.0 * (Fd_[i0] * d1_data[i1] - a_data[i1]);
    });

    // active-core block
    h_diag_.block("ac").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = label_to_mos_["a"][i[0]];
        auto i1 = label_to_mos_["c"][i[1]];
        auto i1p = i[0] * nactv_ + i[0];
        value = 4.0 * (Fd_[i0] - Fd_[i1]);
        value += 2.0 * (Fd_[i1] * d1_data[i1p] - a_data[i1p]);
    });

    // active-active block [see SI of J. Chem. Phys. 152, 074102 (2020)]
    if (internal_rot_) {
        size_t nactv2 = nactv_ * nactv_;
        size_t nactv3 = nactv2 * nactv_;

        auto& fc_data = Fc_.block("aa").data();
        auto& v_data = V_.block("aaaa").data();
        auto& d2_data = D2_.block("aaaa").data();

        // two types of G: G^{uu}_{vv} and G^{uv}_{vu}

        // (uu|xy)
        jk_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto idx = i[0] * nactv3 + i[0] * nactv2 + i[1] * nactv_ + i[2];
            value = v_data[idx];
        });

        // D_{vv,xy}
        d2_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto idx = i[0] * nactv3 + i[0] * nactv2 + i[1] * nactv_ + i[2];
            value = d2_data[idx];
        });

        Guu_["uv"] = jk_internal_["uxy"] * d2_internal_["vxy"];

        // (ux|uy)
        jk_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto idx = i[0] * nactv3 + i[1] * nactv2 + i[0] * nactv_ + i[2];
            value = v_data[idx];
        });

        // D_{vx,vy}
        d2_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto idx = i[0] * nactv3 + i[1] * nactv2 + i[0] * nactv_ + i[2];
            value = d2_data[idx];
        });

        Guu_["uv"] += 2.0 * jk_internal_["uxy"] * d2_internal_["vxy"];

        Guu_.block("aa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto i0 = i[0] * nactv_ + i[0];
            auto i1 = i[1] * nactv_ + i[1];
            value += fc_data[i0] * d1_data[i1];
        });

        Guv_["uv"] = Fc_["uv"] * D1_["vu"];
        Guv_["uv"] += V_["uvxy"] * D2_["vuxy"];
        Guv_["uv"] += 2.0 * V_["uxvy"] * D2_["vxuy"];

        // build diagonal Hessian
        h_diag_["uv"] = 2.0 * Guu_["uv"];
        h_diag_["uv"] += 2.0 * Guu_["vu"];
        h_diag_["uv"] -= 2.0 * Guv_["uv"];
        h_diag_["uv"] -= 2.0 * Guv_["vu"];

        h_diag_.block("aa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto i0 = i[0] * nactv_ + i[0];
            auto i1 = i[1] * nactv_ + i[1];
            value -= 2.0 * (a_data[i0] + a_data[i1]);
        });
    }
}

// void CASSCF_NEW::diagonalize_hamiltonian() {
//    // perform a CAS-CI with the active given in the input
//    std::shared_ptr<ActiveSpaceIntegrals> fci_ints = get_ci_integrals();

//    std::string casscf_ci_type = options_->get_str("CASSCF_CI_SOLVER");

//    auto state_map = to_state_nroots_map(state_weights_map_);
//    auto active_space_solver = make_active_space_solver(casscf_ci_type, state_map, scf_info_,
//                                                        mo_space_info_, fci_ints, options_);
//    active_space_solver->set_print(print_);
//    const auto state_energies_map = active_space_solver->compute_energy();
//    cas_ref_ = active_space_solver->compute_average_rdms(state_weights_map_, 2);
//    E_casscf_ = compute_average_state_energy(state_energies_map, state_weights_map_);

//    // Compute 1-RDM
//    gamma1_ = cas_ref_.g1a().clone();
//    gamma1_("ij") += cas_ref_.g1b()("ij");

//    // Compute 2-RDM
//    gamma2_ = cas_ref_.SFg2();
//}

// std::shared_ptr<psi::Matrix> CASSCF_NEW::set_frozen_core_orbitals() {
//    auto Ca = wfn_->Ca();
//    auto C_core = std::make_shared<psi::Matrix>("C_core", nirrep_, nsopi_, frozen_docc_dim_);

//    // Need to get the frozen block of the C matrix
//    for (size_t h = 0; h < nirrep_; h++) {
//        for (int i = 0; i < frozen_docc_dim_[h]; i++) {
//            C_core->set_column(h, i, Ca->get_column(h, i));
//        }
//    }

//    JK_->set_do_K(true);
//    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
//    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();

//    Cl.clear();
//    Cr.clear();
//    Cl.push_back(C_core); // Cr is the same as Cl

//    JK_->compute();

//    psi::SharedMatrix F_core = JK_->J()[0];
//    psi::SharedMatrix K_core = JK_->K()[0];

//    F_core->scale(2.0);
//    F_core->subtract(K_core);

//    return F_core;
//}

// std::shared_ptr<ActiveSpaceIntegrals> CASSCF_NEW::get_ci_integrals() {
//    std::vector<int> actv_sym = mo_space_info_->symmetry("ACTIVE");
//    auto fci_ints = std::make_shared<ActiveSpaceIntegrals>(ints_, actv_mos_, actv_sym,
//    rdocc_mos_);

//    if (!(options_->get_bool("RESTRICTED_DOCC_JK"))) {
//        fci_ints->set_active_integrals_and_restricted_docc();
//    } else {
//        auto active_aa = ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsAA",
//                                              {nactv_, nactv_, nactv_, nactv_});
//        auto active_ab = ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsAB",
//                                              {nactv_, nactv_, nactv_, nactv_});
//        const std::vector<double>& tei_paaa_data = tei_gaaa_.data();

//        size_t nactv2 = nactv_ * nactv_;
//        size_t nactv3 = nactv2 * nactv_;

//        active_ab.iterate([&](const std::vector<size_t>& i, double& value) {
//            value = tei_paaa_data[actv_mos_[i[0]] * nactv3 + i[1] * nactv2 + i[2] * nactv_ +
//            i[3]];
//        });

//        active_aa.copy(active_ab);
//        active_aa("u,v,x,y") -= active_ab("u, v, y, x");

//        fci_ints->set_active_integrals(active_aa, active_ab, active_aa);
//        if (debug_print_) {
//            outfile->Printf("\n\n  tei_active_aa: %8.8f, tei_active_ab: %8.8f", active_aa.norm(2),
//                            active_ab.norm(2));
//        }

//        auto oei = compute_restricted_docc_operator();
//        fci_ints->set_restricted_one_body_operator(oei, oei);
//        fci_ints->set_scalar_energy(scalar_energy_);
//    }

//    return fci_ints;
//}

// std::vector<double> CASSCF_NEW::compute_restricted_docc_operator() {
//    auto Ca = wfn_->Ca();

//    double Edocc = 0.0;                         // energy from restricted docc
//    double Efrzc = ints_->frozen_core_energy(); // energy from frozen docc
//    scalar_energy_ = ints_->scalar();           // scalar energy from the Integral class

//    // bare one-electron integrals
//    std::shared_ptr<psi::Matrix> Hcore = Hcore_->clone();
//    Hcore->transform(Ca);

//    // one-electron integrals dressed by inactive orbitals
//    std::vector<double> oei(nactv_ * nactv_, 0.0);
//    // one-electron integrals in SharedMatrix format, set to MO Hcore by default
//    std::shared_ptr<psi::Matrix> oei_shared_matrix = Hcore;

//    // special case when there is no inactive docc
//    if (nrdocc_ + nfdocc_ != 0) {
//        // compute inactive Fock
//        build_fock_inactive(Ca);
//        oei_shared_matrix = Fclosed_;

//        // compute energy from inactive docc
//        for (size_t h = 0; h < nirrep_; h++) {
//            for (int rd = 0; rd < inactive_docc_dim_[h]; rd++) {
//                Edocc += Hcore->get(h, rd, rd) + Fclosed_->get(h, rd, rd);
//            }
//        }

//        // Edocc includes frozen-core energy and should be subtracted
//        scalar_energy_ += Edocc - Efrzc;
//    }

//    // fill in oei data
//    for (size_t u = 0; u < nactv_; ++u) {
//        size_t h = actv_mos_rel_[u].first;   // irrep
//        size_t nu = actv_mos_rel_[u].second; // index

//        for (size_t v = 0; v < nactv_; ++v) {
//            if (actv_mos_rel_[v].first != h)
//                continue;

//            size_t nv = actv_mos_rel_[v].second;

//            oei[u * nactv_ + v] = oei_shared_matrix->get(h, nu, nv);
//        }
//    }

//    if (debug_print_) {
//        for (size_t u = 0; u < nactv_; u++) {
//            for (size_t v = 0; v < nactv_; v++) {
//                outfile->Printf("\n  oei(%d, %d) = %8.8f", u, v, oei[u * nactv_ + v]);
//            }
//        }

//        outfile->Printf("\n Frozen Core Energy = %8.8f", Efrzc);
//        outfile->Printf("\n Restricted Energy = %8.8f", Edocc - Efrzc);
//        outfile->Printf("\n Scalar Energy = %8.8f", scalar_energy_);
//    }

//    return oei;
//}

// std::shared_ptr<psi::Matrix> CASSCF_NEW::semicanonicalize(std::shared_ptr<psi::Matrix> Ca) {
//    print_h2("Semi-canonicalize CASSCF_NEW Orbitals");

//    // build generalized Fock matrix
//    outfile->Printf("\n    Building Fock matrix  ...");
//    build_fock(Ca);
//    outfile->Printf(" Done.");

//    // unitary rotation matrix for output
//    auto U = std::make_shared<psi::Matrix>("U_CAS_SEMI", nmo_dim_, nmo_dim_);
//    U->identity();

//    // diagonalize three sub-blocks (restricted_docc, active, restricted_uocc)
//    std::vector<psi::Dimension> mos_dim{restricted_docc_dim_, active_dim_, restricted_uocc_dim_};
//    std::vector<psi::Dimension> mos_offsets{frozen_docc_dim_, inactive_docc_dim_,
//                                            inactive_docc_dim_ + active_dim_};
//    for (int i = 0; i < 3; ++i) {
//        outfile->Printf("\n    Diagonalizing block " + std::to_string(i) + " ...");

//        auto dim = mos_dim[i];
//        auto offset_dim = mos_offsets[i];

//        auto Fsub = std::make_shared<psi::Matrix>("Fsub_" + std::to_string(i), dim, dim);

//        for (size_t h = 0; h < nirrep_; ++h) {
//            for (int p = 0; p < dim[h]; ++p) {
//                size_t np = p + offset_dim[h];
//                for (int q = 0; q < dim[h]; ++q) {
//                    size_t nq = q + offset_dim[h];
//                    Fsub->set(h, p, q, Fock_->get(h, np, nq));
//                }
//            }
//        }

//        // test off-diagonal elements to decide if need to diagonalize this block
//        auto Fsub_od = Fsub->clone();
//        Fsub_od->zero_diagonal();

//        double Fsub_max = Fsub_od->absmax();
//        double Fsub_norm = std::sqrt(Fsub_od->sum_of_squares());

//        double threshold_max = 0.1 * g_conv_;
//        if (int_type_ == "CHOLESKY") {
//            double cd_tlr = options_->get_double("CHOLESKY_TOLERANCE");
//            threshold_max = (threshold_max < 0.5 * cd_tlr) ? 0.5 * cd_tlr : threshold_max;
//        }
//        double threshold_rms = std::sqrt(dim.sum() * (dim.sum() - 1) / 2.0) * threshold_max;

//        // diagonalize
//        if (Fsub_max > threshold_max or Fsub_norm > threshold_rms) {
//            auto Usub = std::make_shared<psi::Matrix>("Usub_" + std::to_string(i), dim, dim);
//            auto Esub = std::make_shared<psi::Vector>("Esub_" + std::to_string(i), dim);
//            Fsub->diagonalize(Usub, Esub);

//            // fill in data
//            for (size_t h = 0; h < nirrep_; ++h) {
//                for (int p = 0; p < dim[h]; ++p) {
//                    size_t np = p + offset_dim[h];
//                    for (int q = 0; q < dim[h]; ++q) {
//                        size_t nq = q + offset_dim[h];
//                        U->set(h, np, nq, Usub->get(h, p, q));
//                    }
//                }
//            }
//        } // end if need to diagonalize

//        outfile->Printf(" Done.");
//    } // end sub block

//    return U;
//}

// void CASSCF_NEW::overlap_orbitals(const psi::SharedMatrix& C_old, const psi::SharedMatrix& C_new)
// {
//    psi::SharedMatrix S_orbitals(new psi::Matrix("Overlap", nsopi_, nsopi_));
//    psi::SharedMatrix S_basis = wfn_->S();
//    S_orbitals = psi::linalg::triplet(C_old, S_basis, C_new, true, false, false);
//    S_orbitals->set_name("C^T S C (Overlap)");
//    for (size_t h = 0; h < nirrep_; h++) {
//        for (int i = 0; i < S_basis->rowspi(h); i++) {
//            if (std::fabs(S_basis->get(h, i, i) - 1.0000000) > 1e-6) {
//                //    S_basis->get_row(h, i)->print();
//            }
//        }
//    }
//}

std::unique_ptr<CASSCF_NEW>
make_casscf_new(const std::map<StateInfo, std::vector<double>>& state_weight_map,
                std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ForteIntegrals> ints) {
    return std::make_unique<CASSCF_NEW>(state_weight_map, options, mo_space_info, scf_info, ints);
}

} // namespace forte
