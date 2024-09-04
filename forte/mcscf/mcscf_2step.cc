/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "ambit/tensor.h"

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libdiis/diismanager.h"

#include "base_classes/rdms.h"
#include "integrals/integrals.h"
#include "integrals/active_space_integrals.h"
#include "integrals/make_integrals.h"
#include "helpers/printing.h"
#include "helpers/lbfgs/lbfgs.h"
#include "helpers/lbfgs/lbfgs_param.h"
#include "orbital-helpers/semi_canonicalize.h"

#include "gradient_tpdm/backtransform_tpdm.h"
#include "mcscf/mcscf_orb_grad.h"
#include "mcscf/mcscf_2step.h"

using namespace ambit;

namespace forte {

MCSCF_2STEP::MCSCF_2STEP(std::shared_ptr<ActiveSpaceSolver> as_solver,
                         const std::map<StateInfo, std::vector<double>>& state_weights_map,
                         std::shared_ptr<ForteOptions> options,
                         std::shared_ptr<MOSpaceInfo> mo_space_info,
                         std::shared_ptr<forte::SCFInfo> scf_info,
                         std::shared_ptr<ForteIntegrals> ints)
    : as_solver_(as_solver), state_weights_map_(state_weights_map), options_(options),
      mo_space_info_(mo_space_info), scf_info_(scf_info), ints_(ints) {
    startup();
}

void MCSCF_2STEP::startup() {
    print_method_banner({"Multi-Configurational Self Consistent Field",
                         "Two-Step Approximate Second-Order AO Algorithm",
                         "written by Chenyang Li, Kevin P. Hannon, and Shuhe Wang"});

    // read and print options
    read_options();
    print_options();
}

void MCSCF_2STEP::read_options() {
    print_ = int_to_print_level(options_->get_int("PRINT"));
    debug_print_ = options_->get_bool("MCSCF_DEBUG_PRINTING");

    int_type_ = options_->get_str("INT_TYPE");

    der_type_ = options_->get_str("DERTYPE");
    if (der_type_ == "FIRST" and ints_->integral_type() == Custom)
        throw std::runtime_error("MCSCF energy gradient not available for CUSTOM integrals!");

    maxiter_ = options_->get_int("MCSCF_MAXITER");
    micro_maxiter_ = options_->get_int("MCSCF_MICRO_MAXITER");

    e_conv_ = options_->get_double("MCSCF_E_CONVERGENCE");
    g_conv_ = options_->get_double("MCSCF_G_CONVERGENCE");

    orb_type_redundant_ = options_->get_str("MCSCF_FINAL_ORBITAL");

    ci_type_ = options_->get_str("ACTIVE_SPACE_SOLVER");
    if (ci_type_ == "") {
        throw std::runtime_error("ACTIVE_SPACE_SOLVER is not specified!");
    }
    mci_maxiter_ = options_->get_int("MCSCF_MCI_MAXITER");
    if (ci_type_ == "BLOCK2")
        mci_maxiter_ = options_->get_int("BLOCK2_N_TOTAL_SWEEPS");

    opt_orbs_ = not options_->get_bool("MCSCF_NO_ORBOPT");
    max_rot_ = options_->get_double("MCSCF_MAX_ROTATION");
    internal_rot_ = options_->get_bool("MCSCF_INTERNAL_ROT");

    // DIIS options
    diis_freq_ = options_->get_int("MCSCF_DIIS_FREQ");
    diis_start_ = options_->get_int("MCSCF_DIIS_START");
    diis_max_vec_ = options_->get_int("MCSCF_DIIS_MAX_VEC");
    diis_min_vec_ = options_->get_int("MCSCF_DIIS_MIN_VEC");
    do_diis_ = diis_start_ >= 1;
}

void MCSCF_2STEP::print_options() {
    // fill in information
    std::vector<std::pair<std::string, int>> info_int{
        {"Max number of macro iter.", maxiter_},
        {"Max number of micro iter. for orbitals", micro_maxiter_},
        {"Max number of micro iter. for CI", mci_maxiter_}};

    std::vector<std::pair<std::string, std::string>> info_string;

    if (do_diis_) {
        info_int.emplace_back("DIIS start", diis_start_);
        info_int.emplace_back("Min DIIS vectors", diis_min_vec_);
        info_int.emplace_back("Max DIIS vectors", diis_max_vec_);
        info_int.emplace_back("Frequency of DIIS extrapolation", diis_freq_);
    }

    table_printer printer;
    printer.add_int_data(info_int);
    printer.add_double_data({{"Energy convergence", e_conv_},
                             {"Gradient convergence", g_conv_},
                             {"Max value for rotation", max_rot_}});
    printer.add_string_data({{"Print level", to_string(print_)},
                             {"Integral type", int_type_},
                             {"CI solver type", ci_type_},
                             {"Final orbital type", orb_type_redundant_},
                             {"Derivative type", der_type_}});
    printer.add_bool_data({{"Optimize orbitals", opt_orbs_},
                           {"Include internal rotations", internal_rot_},
                           {"Debug printing", debug_print_}});

    std::string table = printer.get_table("MCSCF Calculation Information");
    psi::outfile->Printf("%s", table.c_str());
}

double MCSCF_2STEP::compute_energy() {
    // pass energy to Psi4 environment
    auto pass_energy_to_psi4 = [&](bool converged = true) {
        psi::Process::environment.globals["CURRENT ENERGY"] = energy_;
        if (converged)
            psi::Process::environment.globals["MCSCF ENERGY"] = energy_;
    };

    // throw no convergence error
    auto throw_convergence_error = [&](bool sr = false) {
        std::stringstream msg;
        msg << (sr ? "SCF" : "MCSCF") << " did not converge in " << maxiter_ << " iterations!";
        psi::outfile->Printf("\n  %s", msg.str().c_str());
        psi::outfile->Printf("\n  Please increase MCSCF_MAXITER!");
        if (options_->get_bool("MCSCF_DIE_IF_NOT_CONVERGED")) {
            psi::outfile->Printf(
                "\n  This error may be ignored by setting MCSCF_DIE_IF_NOT_CONVERGED.");
            throw std::runtime_error(msg.str());
        }
    };

    // prepare for orbital gradients
    const bool ignore_frozen = options_->get_bool("MCSCF_IGNORE_FROZEN_ORBS");
    MCSCF_ORB_GRAD cas_grad(options_, mo_space_info_, ints_, ignore_frozen);
    auto nrot = cas_grad.nrot();
    auto dG = std::make_shared<psi::Vector>("dG", nrot);

    // set up initial guess for rotation matrix (R = 0)
    auto R = std::make_shared<psi::Vector>("R", nrot);

    // check if there is no orbital optimization
    bool no_orb_opt = (!opt_orbs_ or !nrot);

    // convergence for final CI
    auto r_conv = options_->get_double("R_CONVERGENCE");
    auto as_maxiter = options_->get_int("DL_MAXITER");
    if (ci_type_ == "BLOCK2")
        as_maxiter = options_->get_int("BLOCK2_N_TOTAL_SWEEPS");

    auto active_space_ints = cas_grad.active_space_ints();
    as_solver_->set_active_space_integrals(active_space_ints);

    as_solver_->set_print(PrintLevel::Default);
    as_solver_->set_e_convergence(e_conv_);
    as_solver_->set_r_convergence(r_conv);
    as_solver_->set_maxiter(no_orb_opt ? as_maxiter : mci_maxiter_);

    // initial CI and resulting RDMs
    const auto state_energies_map = as_solver_->compute_energy();
    auto e_c = compute_average_state_energy(state_energies_map, state_weights_map_);

    auto rdms = as_solver_->compute_average_rdms(state_weights_map_, 2, RDMsType::spin_free);
    cas_grad.set_rdms(rdms);
    cas_grad.evaluate(R, dG);

    // Case 1: if there is no orbital optimization
    if (no_orb_opt) {
        energy_ = e_c;
        pass_energy_to_psi4();
        if (der_type_ == "FIRST" and options_->get_str("CORRELATION_SOLVER") == "NONE") {
            cas_grad.compute_nuclear_gradient();
        }
        return energy_;
    }

    // set up L-BFGS solver and its parameters for micro iteration
    auto lbfgs_param = std::make_shared<LBFGS_PARAM>();
    lbfgs_param->epsilon = g_conv_;
    lbfgs_param->print = debug_print_ ? 5 : static_cast<int>(print_);
    lbfgs_param->max_dir = max_rot_;
    lbfgs_param->step_length_method = LBFGS_PARAM::STEP_LENGTH_METHOD::MAX_CORRECTION;
    LBFGS lbfgs(lbfgs_param);

    bool converged = false;

    if (is_single_reference()) { // Case 2: if there is only 1 determinant
        lbfgs_param->maxiter = micro_maxiter_ > maxiter_ ? micro_maxiter_ : maxiter_;
        maxiter_ = lbfgs_param->maxiter;

        print_h2("Single-Reference Orbital Optimization");
        cas_grad.set_rdms(rdms);
        energy_ = lbfgs.minimize(cas_grad, R);
        converged = lbfgs.converged();
        pass_energy_to_psi4(converged);

        if (converged) {
            psi::outfile->Printf("\n\n  SCF converged in %d iterations!", lbfgs.iter());
            psi::outfile->Printf("\n  @ Final energy: %.15f", energy_);
        }
    } else { // Case 3: multi-determinant SCF
        // DIIS extrapolation for macro iteration
        psi::DIISManager diis_manager(do_diis_ ? diis_max_vec_ : 0, "MCSCF DIIS",
                                      psi::DIISManager::RemovalPolicy::OldestAdded,
                                      psi::DIISManager::StoragePolicy::OnDisk);
        if (do_diis_) {
            diis_manager.set_error_vector_size(dG.get());
            diis_manager.set_vector_size(R.get());
        }

        // CI solver set up
        bool restart = (ci_type_ == "FCI" or ci_type_ == "DETCI");
        as_solver_->set_maxiter(restart ? mci_maxiter_ : as_maxiter);

        // CI convergence criteria along the way
        double dl_e_conv = 5.0e-7;
        double dl_r_conv = 8.0e-5;

        // start iterations
        lbfgs_param->maxiter = micro_maxiter_;
        int bad_count = 0;
        bool skip_de_conv = (ci_type_.find("DMRG") != std::string::npos or
                             ci_type_.find("BLOCK2") != std::string::npos);

        std::vector<MCSCF_HISTORY> history;

        print_h2("MCSCF Iterations");
        std::string dash1 = std::string(30, '-');
        std::string dash2 = std::string(88, '-');
        psi::outfile->Printf("\n                      Energy CI                    Energy Orbital");
        psi::outfile->Printf("\n           %s  %s", dash1.c_str(), dash1.c_str());
        psi::outfile->Printf(
            "\n    Iter.        Total Energy       Delta        Total Energy       "
            "Delta  Orb. Grad.  Micro");
        psi::outfile->Printf("\n    %s", dash2.c_str());

        for (int macro = 1; macro <= maxiter_; ++macro) {
            // optimize orbitals
            cas_grad.set_rdms(rdms);

            if (print_ >= PrintLevel::Verbose)
                print_h2("Optimizing Orbitals for Current RDMs");

            double e_o = lbfgs.minimize(cas_grad, R);
            energy_ = e_o;

            // info for orbital optimization
            dG->subtract(*lbfgs.g());
            double g_rms = dG->rms();
            dG->copy(*lbfgs.g());

            int n_micro = lbfgs.iter();
            char o_conv = lbfgs.converged() ? 'Y' : 'N';

            // save data for this macro iteration
            MCSCF_HISTORY hist(e_c, e_o, g_rms, n_micro);
            history.push_back(hist);

            double de = e_o - e_c;
            double de_o = (macro > 1) ? e_o - history[macro - 2].e_o : e_o;
            double de_c = (macro > 1) ? e_c - history[macro - 2].e_c : e_c;

            // print data of this iteration
            if (print_ >= PrintLevel::Verbose) {
                std::string title = "Iter.        Total Energy       Delta        Total Energy     "
                                    "  Delta  Orb. Grad.  Micro";
                if (do_diis_)
                    title += "  DIIS";
                psi::outfile->Printf("\n\n    %s", title.c_str());
            }
            psi::outfile->Printf("\n    %4d %20.12f %11.4e%20.12f %11.4e  %10.4e %4d/%c", macro,
                                 e_c, de_c, e_o, de_o, g_rms, n_micro, o_conv);

            // test convergence
            if (macro == 1 and lbfgs.converged() and std::fabs(de) < e_conv_) {
                psi::outfile->Printf("\n\n  Initial orbitals are already converged!");
                converged = true;
                break;
            }

            bool is_de_conv = skip_de_conv or std::fabs(de) < e_conv_;
            bool is_e_conv = std::fabs(de_c) < e_conv_ and std::fabs(de_o) < e_conv_;
            bool is_g_conv = g_rms < g_conv_ or lbfgs.converged();
            bool is_diis_conv = !do_diis_ or macro < diis_start_ + diis_min_vec_ or
                                diis_manager.subspace_size() > 1;
            if (is_de_conv and is_e_conv and is_g_conv and is_diis_conv) {
                psi::outfile->Printf("\n    %s", dash2.c_str());
                psi::outfile->Printf(
                    "\n\n  A miracle has come to pass: MCSCF iterations have converged!");
                converged = true;
                break;
            }

            // nail down results for DMRG
            if (ci_type_ == "BLOCK2" or ci_type_ == "DMRG") {
                if (std::fabs(de_c) < 1.0e-2 or g_rms < 1.0e-3) {
                    options_->set_bool("READ_ACTIVE_WFN_GUESS", true);
                    mci_maxiter_ = 14;
                    // focus on the last bond dimension
                    if (ci_type_ == "BLOCK2") {
                        auto nsweeps = options_->get_int_list("BLOCK2_SWEEP_N_SWEEPS");
                        auto bond_dims = options_->get_int_list("BLOCK2_SWEEP_BOND_DIMS");
                        auto noises = options_->get_double_list("BLOCK2_SWEEP_NOISES");
                        auto dltols = options_->get_double_list("BLOCK2_SWEEP_DAVIDSON_TOLS");
                        if (bond_dims.size() == 0) {
                            options_->set_int_list("BLOCK2_SWEEP_BOND_DIMS", {500});
                            options_->set_int_list("BLOCK2_SWEEP_N_SWEEPS", {10});
                            options_->set_double_list("BLOCK2_SWEEP_NOISES", {0.0});
                            options_->set_double_list("BLOCK2_SWEEP_DAVIDSON_TOLS", {1.0e-8});
                        } else {
                            auto bond_dim = bond_dims.back();
                            options_->set_int_list("BLOCK2_SWEEP_BOND_DIMS",
                                                   {bond_dim, bond_dim, bond_dim});
                            options_->set_int_list("BLOCK2_SWEEP_N_SWEEPS", {4, 4, 6});
                            options_->set_double_list("BLOCK2_SWEEP_NOISES", {1.0e-6, 1.0e-7, 0.0});
                            options_->set_double_list("BLOCK2_SWEEP_DAVIDSON_TOLS",
                                                      {1.0e-7, 1.0e-8, 1.0e-9});
                        }
                    } else {
                        auto nsweeps = options_->get_int_list("DMRG_SWEEP_MAX_SWEEPS");
                        auto bond_dims = options_->get_int_list("DMRG_SWEEP_STATES");
                        auto noises = options_->get_double_list("DMRG_SWEEP_NOISE_PREFAC");
                        auto dltols = options_->get_double_list("DMRG_SWEEP_DVDSON_RTOL");
                        auto etols = options_->get_double_list("DMRG_SWEEP_ENERGY_CONV");
                        auto bond_dim = bond_dims.back();
                        options_->set_int_list("DMRG_SWEEP_MAX_SWEEPS",
                                               {bond_dim, bond_dim, bond_dim});
                        options_->set_int_list("DMRG_SWEEP_MAX_SWEEPS", {4, 4, 6});
                        options_->set_double_list("DMRG_SWEEP_NOISE_PREFAC", {1.0e-2, 5.0e-3, 0.0});
                        options_->set_double_list("DMRG_SWEEP_DVDSON_RTOL",
                                                  {1.0e-5, 1.0e-6, 1.0e-7});
                        options_->set_double_list("DMRG_SWEEP_ENERGY_CONV",
                                                  {1.0e-6, 1.0e-7, 1.0e-8});
                    }
                } else {
                    as_solver_->set_maxiter(++mci_maxiter_);
                }
            }

            // DIIS for orbitals
            if (do_diis_) {
                if (macro >= diis_start_) {
                    // reset DIIS if current orbital update unreasonable
                    bool reset_diis = false;
                    if (de_o > 0.0 or de_c > 0.0 or (g_rms / history[macro - 2].g_rms > 2.0))
                        ++bad_count;
                    if (bad_count > 5) {
                        reset_diis = true;
                        bad_count = 0;
                    }
                    if (reset_diis) {
                        psi::outfile->Printf("   R/");
                        diis_manager.reset_subspace();
                    } else {
                        psi::outfile->Printf("   ");
                    }

                    diis_manager.add_entry(dG.get(), R.get());
                    psi::outfile->Printf("S");
                }

                if ((macro - diis_start_) % diis_freq_ == 0 and
                    diis_manager.subspace_size() > diis_min_vec_) {
                    diis_manager.extrapolate(R.get());
                    psi::outfile->Printf("/E");

                    // update the actual integrals for CI, skip gradient computation
                    cas_grad.evaluate(R, dG, false);
                }
            }

            // adjust CI convergence
            dl_e_conv = 0.02 * std::fabs(de) < e_conv_ ? e_conv_ : 0.02 * std::fabs(de);
            dl_r_conv = 0.005 * g_rms < r_conv ? r_conv : 0.005 * g_rms;

            // solve the CI problem
            auto fci_ints = cas_grad.active_space_ints();
            auto print_level = debug_print_ ? PrintLevel::Debug
                                            : (print_ >= PrintLevel::Verbose ? PrintLevel::Verbose
                                                                             : PrintLevel::Quiet);
            e_c = diagonalize_hamiltonian(as_solver_, fci_ints,
                                          {print_level, dl_e_conv, dl_r_conv, false});
            rdms = as_solver_->compute_average_rdms(state_weights_map_, 2, RDMsType::spin_free);
        }

        diis_manager.reset_subspace();
        diis_manager.delete_diis_file();
    }

    // perform final CI using converged orbitals
    if (print_ >= PrintLevel::Default)
        psi::outfile->Printf("\n\n  Performing final CI Calculation using converged orbitals");

    energy_ =
        diagonalize_hamiltonian(as_solver_, cas_grad.active_space_ints(),
                                {print_, e_conv_, r_conv, options_->get_bool("DUMP_ACTIVE_WFN")});

    if (ints_->integral_type() != Custom) {
        auto final_orbs = options_->get_str("MCSCF_FINAL_ORBITAL");

        if (final_orbs != "UNSPECIFIED" or der_type_ == "FIRST") {
            // fix orbitals for redundant pairs
            rdms = as_solver_->compute_average_rdms(state_weights_map_, 1, RDMsType::spin_free);
            auto F = cas_grad.fock(rdms);
            ints_->set_fock_matrix(F, F);

            auto inactive_mix = options_->get_bool("SEMI_CANONICAL_MIX_INACTIVE");
            auto active_mix = options_->get_bool("SEMI_CANONICAL_MIX_ACTIVE");

            // if we do not freeze the core, we need to set the inactive_mix flag to make sure
            // the core orbitals are canonicalized together with the active orbitals
            inactive_mix = ignore_frozen;

            psi::outfile->Printf("\n  Canonicalizing final MCSCF orbitals");
            SemiCanonical semi(mo_space_info_, ints_, options_, inactive_mix, active_mix);
            semi.semicanonicalize(rdms, false, final_orbs == "NATURAL", false);

            cas_grad.canonicalize_final(semi.Ua());
        }

        // pass to wave function
        auto Ca = cas_grad.Ca();
        ints_->wfn()->Ca()->copy(Ca);
        ints_->wfn()->Cb()->copy(Ca);

        // throw error if not converged
        if (not converged)
            throw_convergence_error();

        // for nuclear gradient
        if (der_type_ == "FIRST" and options_->get_str("CORRELATION_SOLVER") == "NONE") {
            // TODO: remove this re-diagonalization if CI transformation is impelementd
            if (not is_single_reference()) {
                diagonalize_hamiltonian(
                    as_solver_, cas_grad.active_space_ints(),
                    {PrintLevel::Quiet, e_conv_, r_conv, options_->get_bool("DUMP_ACTIVE_WFN")});
            }

            // recompute gradient due to canonicalization
            rdms = as_solver_->compute_average_rdms(state_weights_map_, 2, RDMsType::spin_free);
            cas_grad.set_rdms(rdms);
            cas_grad.evaluate(R, dG);

            // compute densities used for nuclear gradient
            cas_grad.compute_nuclear_gradient();
        }
    } else {
        // throw error if not converged
        if (not converged)
            throw_convergence_error();
    }

    return energy_;
}

bool MCSCF_2STEP::is_single_reference() {
    auto nactv = mo_space_info_->size("ACTIVE");
    auto nclosed_electrons = mo_space_info_->size("INACTIVE_DOCC");

    if (state_weights_map_.size() == 1) {
        for (const auto& [state, _] : state_weights_map_) {
            auto na = state.na() - nclosed_electrons;
            auto nb = state.nb() - nclosed_electrons;

            // no electrons in active
            if (na == 0 and nb == 0)
                return true;

            // fully occupied active
            if (na == nactv and nb == nactv)
                return true;

            // high-spin open-shell
            size_t nd = (static_cast<int>(na) - static_cast<int>(nb)) > 0 ? na - nb : nb - na;
            if (nd == nactv)
                return true;
        }
    }

    return false;
}

double
MCSCF_2STEP::diagonalize_hamiltonian(std::shared_ptr<ActiveSpaceSolver>& as_solver_,
                                     std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                     const std::tuple<PrintLevel, double, double, bool>& params) {
    const auto& [print, e_conv, r_conv, dump_wfn] = params;

    as_solver_->set_print(print);
    as_solver_->set_e_convergence(e_conv);
    as_solver_->set_r_convergence(r_conv);
    as_solver_->set_active_space_integrals(fci_ints);

    const auto state_energies_map = as_solver_->compute_energy();

    if (dump_wfn)
        as_solver_->dump_wave_function();

    return compute_average_state_energy(state_energies_map, state_weights_map_);
}

bool MCSCF_2STEP::test_history(const std::vector<MCSCF_HISTORY>& history, const int& n_samples) {
    if (n_samples < 6)
        return true;

    int hist_size = static_cast<int>(history.size());
    int n = hist_size > n_samples ? n_samples : hist_size;
    int offset = hist_size - n;

    std::vector<double> delta_e(n - 1);
    std::vector<double> delta_g(n - 1);
    for (int i = 1; i < n; ++i) {
        delta_e[i - 1] = history[i + offset].e_o - history[i - 1 + offset].e_o;
        delta_g[i - 1] = history[i + offset].g_rms - history[i - 1 + offset].g_rms;
    }

    double max_dg = *std::max_element(delta_g.begin(), delta_g.end(), [](double a, double b) {
        return std::fabs(a) < std::fabs(b);
    });
    double max_de = *std::max_element(delta_e.begin(), delta_e.end(), [](double a, double b) {
        return std::fabs(a) < std::fabs(b);
    });

    if (max_dg < 10 * g_conv_ and max_de < 10 * e_conv_)
        return true;

    int e_decrease_count = 0, g_decrease_count = 0;
    int e_trap_count = 0, g_trap_count = 0;
    for (int i = 0; i < n - 1; ++i) {
        e_decrease_count += delta_e[i] < 0 ? 1 : 0;
        g_decrease_count += delta_g[i] < 0 ? 1 : 0;

        double qe = std::fabs(delta_e[i]) / max_de;
        double qg = std::fabs(delta_g[i]) / max_dg;
        e_trap_count += (qe < 1.5 and qe > 0.8) ? 1 : 0;
        g_trap_count += (qg < 1.5 and qg > 0.8) ? 1 : 0;
    }

    int more_than_half = n / 2 + 1;
    if (e_decrease_count > more_than_half or g_decrease_count > more_than_half) {
        return true;
    } else {
        if (e_trap_count < more_than_half and g_trap_count < more_than_half)
            return true;
    }

    return false;
}

void MCSCF_2STEP::print_macro_iteration(const std::vector<MCSCF_HISTORY>& history) {
    print_h2("MCSCF Iteration Summary");
    std::string dash1 = std::string(30, '-');
    std::string dash2 = std::string(88, '-');
    psi::outfile->Printf("\n                      Energy CI                    Energy Orbital");
    psi::outfile->Printf("\n           %s  %s", dash1.c_str(), dash1.c_str());
    psi::outfile->Printf("\n    Iter.        Total Energy       Delta        Total Energy       "
                         "Delta  Orb. Grad.  Micro");
    psi::outfile->Printf("\n    %s", dash2.c_str());
    for (int i = 0, size = static_cast<int>(history.size()); i < size; ++i) {
        double e_c = history[i].e_c;
        double e_o = history[i].e_o;
        double de_c = (i == 0) ? 0.0 : history[i].e_c - history[i - 1].e_c;
        double de_o = (i == 0) ? 0.0 : history[i].e_o - history[i - 1].e_o;
        double g = history[i].g_rms;
        int n = history[i].n_micro;
        psi::outfile->Printf("\n    %4d %20.12f %11.4e%20.12f %11.4e  %10.4e  %4d", i + 1, e_c,
                             de_c, e_o, de_o, g, n);
    }
    psi::outfile->Printf("\n    %s", dash2.c_str());
    psi::Process::environment.globals["CURRENT ENERGY"] = history.back().e_o;
    psi::Process::environment.globals["MCSCF ENERGY"] = history.back().e_o;
}

std::unique_ptr<MCSCF_2STEP>
make_mcscf_two_step(std::shared_ptr<ActiveSpaceSolver> as_solver,
                    const std::map<StateInfo, std::vector<double>>& state_weight_map,
                    std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                    std::shared_ptr<MOSpaceInfo> mo_space_info,
                    std::shared_ptr<ForteIntegrals> ints) {
    return std::make_unique<MCSCF_2STEP>(as_solver, state_weight_map, options, mo_space_info,
                                         scf_info, ints);
}

} // namespace forte
