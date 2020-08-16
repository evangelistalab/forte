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
#include "helpers/lbfgs/lbfgs.h"
#include "helpers/timer.h"
#include "sci/aci.h"
#include "fci/fci_solver.h"
#include "base_classes/active_space_solver.h"
#include "base_classes/rdms.h"

#include "sci/fci_mo.h"
#include "orbital-helpers/mp2_nos.h"
#include "orbital-helpers/orbitaloptimizer.h"
#include "orbital-helpers/semi_canonicalize.h"

#include "integrals/integrals.h"
#include "casscf/casscf_orb_grad.h"
#include "casscf/casscf_new.h"
#include "helpers/lbfgs/lbfgs_param.h"
#include "helpers/lbfgs/rosenbrock.h"

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

    // frozen-core energy
    e_frozen_ = ints_->frozen_core_energy();

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
}

void CASSCF_NEW::read_options() {
    print_ = options_->get_int("PRINT");
    debug_print_ = options_->get_bool("CASSCF_DEBUG_PRINTING");

    int_type_ = options_->get_str("INT_TYPE");

    maxiter_ = options_->get_int("CASSCF_MAXITER");
    e_conv_ = options_->get_double("CASSCF_E_CONVERGENCE");
    g_conv_ = options_->get_double("CASSCF_G_CONVERGENCE");

    orb_type_redundant_ = options_->get_str("CASSCF_ORB_TYPE");

    // DIIS options
    diis_freq_ = options_->get_int("CASSCF_DIIS_FREQ");
    diis_start_ = options_->get_int("CASSCF_DIIS_START");
    diis_max_vec_ = options_->get_int("CASSCF_DIIS_MAX_VEC");
    diis_min_vec_ = options_->get_int("CASSCF_DIIS_MIN_VEC");
    do_diis_ = (diis_start_ < 1) ? false : true;

    // CI update options
    ci_type_ = options_->get_str("CASSCF_CI_SOLVER");
    ci_freq_ = options_->get_int("CASSCF_CI_FREQ");

    // rotations
    max_rot_ = options_->get_double("CASSCF_MAX_ROTATION");
    internal_rot_ = options_->get_bool("CASSCF_INTERNAL_ROT");

    zero_rots_.resize(nirrep_);
    auto zero_rots = options_->get_gen_list("CASSCF_ZERO_ROT");

    if (zero_rots.size() != 0) {
        size_t npairs = zero_rots.size();
        for (size_t i = 0; i < npairs; ++i) {
            py::list pair = zero_rots[i];
            if (pair.size() != 3) {
                outfile->Printf("\n  Error: invalid input of CASSCF_ZERO_ROT.");
                outfile->Printf("\n  Each entry should take an array of three numbers.");
                throw std::runtime_error("Invalid input of CASSCF_ZERO_ROT");
            }

            int irrep = py::cast<int>(pair[0]);
            if (irrep >= nirrep_ or irrep < 0) {
                outfile->Printf("\n  Error: invalid irrep in CASSCF_ZERO_ROT.");
                outfile->Printf("\n  Check the input irrep (start from 0) not to exceed %d",
                                nirrep_ - 1);
                throw std::runtime_error("Invalid irrep in CASSCF_ZERO_ROT");
            }

            size_t i1 = py::cast<size_t>(pair[1]) - 1;
            size_t i2 = py::cast<size_t>(pair[2]) - 1;
            size_t n = nmopi_[irrep];
            if (i1 >= n or i1 < 0 or i2 >= n or i2 < 0) {
                outfile->Printf("\n  Error: invalid orbital indices in CASSCF_ZERO_ROT.");
                outfile->Printf("\n The input orbital indices (start from 1) should not exceed %d "
                                "(number of orbitals in irrep %d)",
                                n, irrep);
                throw std::runtime_error("Invalid orbital indices in CASSCF_ZERO_ROT");
            }

            zero_rots_[irrep][i1].emplace(i2);
            zero_rots_[irrep][i2].emplace(i1);
        }
    }
}

void CASSCF_NEW::print_options() {
    // fill in information
    std::vector<std::pair<std::string, int>> info_int{{"Printing level", print_},
                                                      {"Max number of iterations", maxiter_},
                                                      {"Frequency of doing CI", ci_freq_}};

    std::vector<std::pair<std::string, double>> info_double{{"Energy convergence", e_conv_},
                                                            {"Gradient convergence", g_conv_},
                                                            {"Max value for rotation", max_rot_}};

    std::vector<std::pair<std::string, std::string>> info_string{
        {"Integral type", int_type_},
        {"CI solver type", ci_type_},
        {"Orbital type", orb_type_redundant_}};

    std::vector<std::pair<std::string, bool>> info_bool{
        {"Include internal rotations", internal_rot_}, {"Debug printing", debug_print_}};

    if (do_diis_) {
        info_int.push_back({"DIIS start", diis_start_});
        info_int.push_back({"Min DIIS vectors", diis_min_vec_});
        info_int.push_back({"Max DIIS vectors", diis_max_vec_});
        info_int.push_back({"Frequency of DIIS extrapolation", diis_freq_});
    }

    // print some information
    print_selected_options("Calculation Information", info_string, info_bool, info_double,
                           info_int);
}

void CASSCF_NEW::nonredundant_pairs() {
    // prepare indices for rotation pairs
    rot_mos_irrep_.clear();
    rot_mos_block_.clear();

    std::map<std::string, std::vector<int>> nrots{{"vc", std::vector<int>(nirrep_, 0)},
                                                  {"va", std::vector<int>(nirrep_, 0)},
                                                  {"ac", std::vector<int>(nirrep_, 0)}};

    for (const std::string& block : {"vc", "va", "ac"}) {
        const auto& mos1 = label_to_mos_[block.substr(0, 1)];
        const auto& mos2 = label_to_mos_[block.substr(1, 1)];

        for (int i = 0, si = mos1.size(); i < si; ++i) {
            int hi = corr_mos_rel_[mos1[i]].first;
            auto ni = corr_mos_rel_[mos1[i]].second;

            for (int j = 0, sj = mos2.size(); j < sj; ++j) {
                int hj = corr_mos_rel_[mos2[j]].first;
                auto nj = corr_mos_rel_[mos2[j]].second;

                if (hi == hj) {
                    if (zero_rots_[hi].find(ni) != zero_rots_[hi].end()) {
                        if (zero_rots_[hi][ni].find(nj) != zero_rots_[hi][ni].end())
                            break;
                    }
                    rot_mos_irrep_.push_back(std::make_tuple(hi, ni, nj));
                    rot_mos_block_.push_back(std::make_tuple(block, i, j));
                    nrots[block][hi] += 1;
                }
            }
        }
    }

    if (internal_rot_) {
        nrots["aa"] = std::vector<int>(nirrep_, 0);

        const auto& mos = label_to_mos_["a"];
        for (int i = 0, s = mos.size(); i < s; ++i) {
            int hi = corr_mos_rel_[mos[i]].first;
            auto ni = corr_mos_rel_[mos[i]].second;

            for (int j = i + 1; j < s; ++j) {
                int hj = corr_mos_rel_[mos[j]].first;
                auto nj = corr_mos_rel_[mos[j]].second;

                if (hi == hj) {
                    if (zero_rots_[hi].find(ni) != zero_rots_[hi].end()) {
                        if (zero_rots_[hi][ni].find(nj) != zero_rots_[hi][ni].end())
                            break;
                    }
                    rot_mos_irrep_.push_back(std::make_tuple(hi, nj, ni));
                    rot_mos_block_.push_back(std::make_tuple("aa", j, i));
                    nrots["aa"][hi] += 1;
                }
            }
        }
    }

    nrot_ = rot_mos_irrep_.size();

    // printing
    auto ct = psi::Process::environment.molecule()->point_group()->char_table();
    std::map<std::string, std::string> space_map{
        {"c", "RESTRICTED_DOCC"}, {"a", "ACTIVE"}, {"v", "RESTRICTED_UOCC"}};

    print_h2("Independent Orbital Rotations");
    outfile->Printf("\n    %-33s", "ORBITAL SPACES");
    for (int h = 0; h < nirrep_; ++h) {
        outfile->Printf("  %4s", ct.gamma(h).symbol());
    }
    outfile->Printf("\n    %s", std::string(33 + nirrep_ * 6, '-').c_str());

    for (const auto& key_value : nrots) {
        const auto& key = key_value.first;
        auto block1 = space_map[key.substr(0, 1)];
        auto block2 = space_map[key.substr(1, 1)];
        outfile->Printf("\n    %15s / %15s", block1.c_str(), block2.c_str());

        const auto& value = key_value.second;
        for (int h = 0; h < nirrep_; ++h) {
            outfile->Printf("  %4zu", value[h]);
        }
    }
    outfile->Printf("\n    %s", std::string(33 + nirrep_ * 6, '-').c_str());
}

void CASSCF_NEW::setup_JK() {
    local_timer jk_timer;
    print_h2("Initialize JK Builder", "==>", "<==\n");

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
    A_ = ambit::BlockedTensor::build(tensor_type, "A", {"ca", "ao", "vo"});

    std::vector<std::string> g_blocks{"ac", "vo"};
    if (internal_rot_) {
        g_blocks.push_back("aa");
        Guu_ = ambit::BlockedTensor::build(CoreTensor, "Guu", {"aa"});
        Guv_ = ambit::BlockedTensor::build(CoreTensor, "Guv", {"aa"});
        jk_internal_ = ambit::BlockedTensor::build(CoreTensor, "tei_internal", {"aaa"});
        d2_internal_ = ambit::BlockedTensor::build(CoreTensor, "rdm_internal", {"aaa"});
    }
    g_ = ambit::BlockedTensor::build(tensor_type, "g", g_blocks);
    h_diag_ = ambit::BlockedTensor::build(tensor_type, "h_diag", g_blocks);

    grad_ = std::make_shared<psi::Vector>("Gradient Vector", nrot_);
    hess_diag_ = std::make_shared<psi::Vector>("Diagonal Hessian", nrot_);
}

double CASSCF_NEW::compute_energy() {
    // initial orbitals and rotations
    C_ = C0_->clone();

    R_ = std::make_shared<psi::Matrix>("Orbital Rotation", nmopi_, nmopi_);
    dR_ = std::make_shared<psi::Matrix>("Orbital Rotation Update", nmopi_, nmopi_);

    R_v_ = std::make_shared<psi::Vector>("R", nrot_);
    auto R_v_old = std::make_shared<psi::Vector>("Rold", nrot_);
    dR_v_ = std::make_shared<psi::Vector>("dR", nrot_);

    auto U = std::make_shared<psi::Matrix>("Orthogonal Transformation", nmopi_, nmopi_);

    if (do_diis_) {
        diis_manager_ = std::make_shared<DIISManager>(
            diis_max_vec_, "CASSCF DIIS", DIISManager::OldestAdded, DIISManager::InCore);
        diis_manager_->set_error_vector_size(1, DIISEntry::Vector, dR_v_.get());
        diis_manager_->set_vector_size(1, DIISEntry::Vector, R_v_.get());
    }

//    LBFGS lbfgs(nrot_);

    std::vector<double> e_history;
    std::vector<double> g_history;

    CASSCF_ORB_GRAD cas_grad(options_, mo_space_info_, ints_);

    LBFGS_PARAM lbfgs_param;
    lbfgs_param.epsilon = g_conv_;
    lbfgs_param.maxiter = 20;
    lbfgs_param.print = 3;
    lbfgs_param.max_dir = max_rot_;
    lbfgs_param.step_length_method = LBFGS_PARAM::STEP_LENGTH_METHOD::MAX_CORRECTION;

    LBFGS lbfgs(lbfgs_param);

    for (int macro = 1; macro <= maxiter_; ++macro) {
        auto fci_ints = cas_grad.active_space_ints();

        auto state_map = to_state_nroots_map(state_weights_map_);
        auto active_space_solver = make_active_space_solver(ci_type_, state_map, scf_info_,
                                                            mo_space_info_, fci_ints, options_);
        active_space_solver->set_print(print_);
        const auto state_energies_map = active_space_solver->compute_energy();

        auto rdms = active_space_solver->compute_average_rdms(state_weights_map_, 2);
        energy_ = compute_average_state_energy(state_energies_map, state_weights_map_);

        cas_grad.set_rdms(rdms);

        lbfgs.minimize(cas_grad, R_v_);

        C_ = cas_grad.Ca()->clone();
        double g_norm = cas_grad.grad_norm();
        double e_delta = (macro > 1) ? energy_ - e_history[macro - 2] : energy_;
        outfile->Printf("\n    Iter.      Current Energy  Energy Diff.    Orb. Grad.");
        outfile->Printf("\n    %4d   %18.12f  %12.4e  %12.4e", macro, energy_, e_delta, g_norm);
        e_history.push_back(energy_);
        g_history.push_back(g_norm);

        if (std::fabs(e_delta) < e_conv_ and g_norm < g_conv_) {
            outfile->Printf("\n    A miracle has come to pass: MCSCF iterations have converged!");
            break;
        }

        lbfgs.reset();
    }

//    // start iteration
//    for (int iter = 1; iter <= maxiter_; ++iter) {
//        build_mo_integrals();

//        if (iter == 1 or (iter % ci_freq_ == 0)) {
//            diagonalize_hamiltonian();
//        } else {
//            compute_reference_energy();
//        }

//        build_fock();

//        compute_orbital_grad();

//        // now test convergence
//        double e_delta = (iter > 1) ? energy_ - e_history[iter - 2] : energy_;
//        double g_norm = grad_->norm();
//        outfile->Printf("\n    Iter.      Current Energy  Energy Diff.    Orb. Grad.");
//        outfile->Printf("\n    %4d   %18.12f  %12.4e  %12.4e", iter, energy_, e_delta, g_norm);
//        e_history.push_back(energy_);
//        g_history.push_back(g_norm);

//        if (std::fabs(e_delta) < e_conv_ and g_norm < g_conv_) {
//            outfile->Printf("\n    A miracle has come to pass: MCSCF iterations have converged!");
//            break;
//        }

//        // if not converged, update orbitals
//        compute_orbital_hess_diag();
//        lbfgs.set_hess_diag(hess_diag_);
//        lbfgs.reset();

//        dR_v_ = lbfgs.compute_correction(R_v_, grad_);

//        // scale dR
//        double dr_max = 0.0;
//        for (size_t i = 0; i < nrot_; ++i) {
//            double v = std::fabs(dR_v_->get(i));
//            if (v > dr_max)
//                dr_max = v;
//        }
//        if (dr_max > max_rot_)
//            dR_v_->scale(max_rot_ / dr_max);

//        reshape_rot_update();

//        // add to R matrix for next iteration
//        R_->add(dR_);
//        R_v_->add(dR_v_);

//        if (do_diis_ and iter >= diis_start_) {
//            diis_manager_->add_entry(2, dR_.get(), R_.get());
//            outfile->Printf("  S");

//            if ((iter - diis_start_) % diis_freq_ == 0 and
//                diis_manager_->subspace_size() > diis_min_vec_) {
//                diis_manager_->extrapolate(1, R_.get());
//                outfile->Printf("/E");
//            }
//        }

//        // compute unitary matrix U = exp(R)
//        U->copy(R_);
//        U->expm(3);

//        // update orbitals
//        C_ = psi::linalg::doublet(C0_, U, false, false);
//        C_->set_name(C0_->name());

//        if (debug_print_) {
//            dR_->print();
//            R_->print();
//            C_->print();
//        }

//        if (iter % ci_freq_ == 0)
//            diis_manager_->reset_subspace();
//    }
//    diis_manager_->reset_subspace();

    // print summary
    print_h2("MCSCF Iteration Summary");
    outfile->Printf("\n    Iter.      Current Energy  Energy Diff.    Orb. Grad.");
    outfile->Printf("\n    -----------------------------------------------------");
    for (int i = 0, size = e_history.size(); i < size; ++i) {
        double e = e_history[i];
        double e_delta = (i == 0) ? 0.0 : e_history[i] - e_history[i - 1];
        double g = g_history[i];
        outfile->Printf("\n    %4d   %18.12f  %12.4e  %12.4e", i + 1, e, e_delta, g);
    }
    outfile->Printf("\n    -----------------------------------------------------");

    // fix orbitals for redundant pairs
    auto U_re = canonicalize();
    C_ = psi::linalg::doublet(C_, U_re, false, false);
    C_->set_name(C0_->name());

    // rediagonalize Hamiltonian
    build_mo_integrals();
    diagonalize_hamiltonian();

    // pass to wave function
    ints_->wfn()->Ca()->copy(C_);
    ints_->wfn()->Cb()->copy(C_);

    return energy_;
}

void CASSCF_NEW::build_mo_integrals() {
    // form closed-shell Fock matrix
    build_fock_inactive();

    // bare one-electron integrals
    build_oei_from_ao();

    // compute the closed-shell energy
    compute_energy_closed();

    // form the MO 2e-integrals
    build_tei_from_ao();
}

void CASSCF_NEW::build_oei_from_ao() {
    H_mo_ = H_ao_->clone();
    H_mo_->transform(C_);
    H_mo_->set_name("Hbare_MO");

    if (debug_print_) {
        H_mo_->print();
    }
}

void CASSCF_NEW::compute_energy_closed() {
    e_closed_ = 0.0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < ndoccpi_[h]; ++i) {
            e_closed_ += F_closed_->get(h, i, i) + H_mo_->get(h, i, i);
        }
    }

    if (debug_print_) {
        outfile->Printf("\n  Frozen-core energy   %20.15f", e_frozen_);
        outfile->Printf("\n  Closed-shell energy  %20.15f", e_closed_);
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

    timer_off("Build (pu|xy) integrals");
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
        for (int i = 0, offset = ndoccpi_[h]; i < nactvpi_[h]; i++) {
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
        outfile->Printf("\n  Reference energy     %20.15f", energy_);
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

    // reshape and format to SharedVector
    reshape_rot_ambit(g_, grad_);

    if (debug_print_) {
        grad_->print();
    }
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

        // G^{uu}_{vv}

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

        // G^{uv}_{vu}
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

    // reshape and format to SharedVector
    reshape_rot_ambit(h_diag_, hess_diag_);

    if (debug_print_) {
        hess_diag_->print();
    }
}

void CASSCF_NEW::reshape_rot_ambit(ambit::BlockedTensor bt, psi::SharedVector sv) {
    size_t vec_size = sv->dimpi().sum();
    if (vec_size != nrot_) {
        std::runtime_error("Inconsistent size between SharedVector and number of rotaitons");
    }

    for (size_t n = 0; n < nrot_; ++n) {
        std::string block;
        int i, j;
        std::tie(block, i, j) = rot_mos_block_[n];
        auto tensor = bt.block(block);
        auto dim1 = tensor.dim(1);

        sv->set(n, tensor.data()[i * dim1 + j]);
    }
}

void CASSCF_NEW::reshape_rot_update() {
    for (size_t n = 0; n < nrot_; ++n) {
        int h, i, j;
        std::tie(h, i, j) = rot_mos_irrep_[n];

        dR_->set(h, i, j, dR_v_->get(n));
        dR_->set(h, j, i, -dR_v_->get(n));
    }
}

std::shared_ptr<psi::Matrix> CASSCF_NEW::canonicalize() {
    print_h2("Canonicalize Orbitals (" + orb_type_redundant_ + ")");

    // unitary rotation matrix for output
    auto U = std::make_shared<psi::Matrix>("U_redundant", nmopi_, nmopi_);
    U->identity();

    // diagonalize sub-blocks of Fock
    auto ncorepi = mo_space_info_->dimension("RESTRICTED_DOCC");
    auto nvirtpi = mo_space_info_->dimension("RESTRICTED_UOCC");
    std::vector<psi::Dimension> mos_dim{ncorepi, nvirtpi};
    std::vector<psi::Dimension> mos_offsets{nfrzcpi_, ndoccpi_ + nactvpi_};
    std::vector<std::string> names{"RESTRICTED_DOCC", "RESTRICTED_UOCC"};

    if (orb_type_redundant_ == "CANONICAL") {
        mos_dim.push_back(nactvpi_);
        mos_offsets.push_back(ndoccpi_);
        names.push_back("ACTIVE");
    }

    for (int i = 0, size = mos_dim.size(); i < size; ++i) {
        std::string block_name = "Diagonalizing Fock block " + names[i] + " ...";
        outfile->Printf("\n    %-44s", block_name.c_str());

        auto dim = mos_dim[i];
        auto offset_dim = mos_offsets[i];

        auto Fsub = std::make_shared<psi::Matrix>("Fsub_" + names[i], dim, dim);

        for (int h = 0; h < nirrep_; ++h) {
            for (int p = 0; p < dim[h]; ++p) {
                size_t np = p + offset_dim[h];
                for (int q = 0; q < dim[h]; ++q) {
                    size_t nq = q + offset_dim[h];
                    Fsub->set(h, p, q, Fock_->get(h, np, nq));
                }
            }
        }

        // test off-diagonal elements to decide if need to diagonalize this block
        auto Fsub_od = Fsub->clone();
        Fsub_od->zero_diagonal();

        double Fsub_max = Fsub_od->absmax();
        double Fsub_norm = std::sqrt(Fsub_od->sum_of_squares());

        double threshold_max = 0.1 * g_conv_;
        if (int_type_ == "CHOLESKY") {
            double cd_tlr = options_->get_double("CHOLESKY_TOLERANCE");
            threshold_max = (threshold_max < 0.5 * cd_tlr) ? 0.5 * cd_tlr : threshold_max;
        }
        double threshold_rms = std::sqrt(dim.sum() * (dim.sum() - 1) / 2.0) * threshold_max;

        // diagonalize
        if (Fsub_max > threshold_max or Fsub_norm > threshold_rms) {
            auto Usub = std::make_shared<psi::Matrix>("Usub_" + names[i], dim, dim);
            auto Esub = std::make_shared<psi::Vector>("Esub_" + names[i], dim);
            Fsub->diagonalize(Usub, Esub);

            // fill in data
            for (int h = 0; h < nirrep_; ++h) {
                for (int p = 0; p < dim[h]; ++p) {
                    size_t np = p + offset_dim[h];
                    for (int q = 0; q < dim[h]; ++q) {
                        size_t nq = q + offset_dim[h];
                        U->set(h, np, nq, Usub->get(h, p, q));
                    }
                }
            }
        } // end if need to diagonalize

        outfile->Printf(" Done.");
    } // end sub block

    // natural orbitals
    if (orb_type_redundant_ == "NATURAL") {
        std::string block_name = "Diagonalizing 1-RDM ...";
        outfile->Printf("\n    %-44s", block_name.c_str());

        auto A = std::make_shared<psi::Matrix>("O_actv", nactvpi_, nactvpi_);

        D1_.citerate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>&, const double& value) {
                auto irrep_index_pair1 = corr_mos_rel_[i[0]];
                auto irrep_index_pair2 = corr_mos_rel_[i[1]];

                int h1 = irrep_index_pair1.first;
                int h2 = irrep_index_pair2.first;

                if (h1 == h2) {
                    auto offset = ndoccpi_[h1];
                    auto p = irrep_index_pair1.second - offset;
                    auto q = irrep_index_pair2.second - offset;
                    A->set(h1, p, q, value);
                }
            });

        auto Usub = std::make_shared<psi::Matrix>("Usub_ACTIVE", nactvpi_, nactvpi_);
        auto Esub = std::make_shared<psi::Vector>("Esub_ACTIVE", nactvpi_);
        A->diagonalize(Usub, Esub, descending);

        // fill in data
        for (int h = 0; h < nirrep_; ++h) {
            for (int p = 0; p < nactvpi_[h]; ++p) {
                size_t np = p + ndoccpi_[h];
                for (int q = 0; q < nactvpi_[h]; ++q) {
                    size_t nq = q + ndoccpi_[h];
                    U->set(h, np, nq, Usub->get(h, p, q));
                }
            }
        }

        outfile->Printf(" Done.");
    }

    return U;
}

std::unique_ptr<CASSCF_NEW>
make_casscf_new(const std::map<StateInfo, std::vector<double>>& state_weight_map,
                std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ForteIntegrals> ints) {
    return std::make_unique<CASSCF_NEW>(state_weight_map, options, mo_space_info, scf_info, ints);
}

} // namespace forte
