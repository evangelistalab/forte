/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "casscf/casscf_orb_grad.h"
#include "casscf/mcscf_2step.h"

using namespace ambit;

namespace forte {

MCSCF_2STEP::MCSCF_2STEP(const std::map<StateInfo, std::vector<double>>& state_weights_map,
                         std::shared_ptr<ForteOptions> options,
                         std::shared_ptr<MOSpaceInfo> mo_space_info,
                         std::shared_ptr<forte::SCFInfo> scf_info,
                         std::shared_ptr<ForteIntegrals> ints)
    : state_weights_map_(state_weights_map), options_(options), mo_space_info_(mo_space_info),
      scf_info_(scf_info), ints_(ints) {
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
    print_ = options_->get_int("PRINT");
    debug_print_ = options_->get_bool("CASSCF_DEBUG_PRINTING");

    int_type_ = options_->get_str("INT_TYPE");

    der_type_ = options_->get_str("DERTYPE");
    if (der_type_ == "FIRST" and ints_->integral_type() == Custom)
        throw std::runtime_error("MCSCF energy gradient not available for CUSTOM integrals!");

    maxiter_ = options_->get_int("CASSCF_MAXITER");
    micro_maxiter_ = options_->get_int("CASSCF_MICRO_MAXITER");
    micro_miniter_ = options_->get_int("CASSCF_MICRO_MINITER");
    if (micro_maxiter_ < micro_miniter_)
        micro_miniter_ = micro_maxiter_;

    e_conv_ = options_->get_double("CASSCF_E_CONVERGENCE");
    g_conv_ = options_->get_double("CASSCF_G_CONVERGENCE");

    orb_type_redundant_ = options_->get_str("CASSCF_FINAL_ORBITAL");

    ci_type_ = options_->get_str("CASSCF_CI_SOLVER");

    opt_orbs_ = not options_->get_bool("CASSCF_NO_ORBOPT");
    max_rot_ = options_->get_double("CASSCF_MAX_ROTATION");
    internal_rot_ = options_->get_bool("CASSCF_INTERNAL_ROT");

    // DIIS options
    diis_freq_ = options_->get_int("CASSCF_DIIS_FREQ");
    diis_start_ = options_->get_int("CASSCF_DIIS_START");
    diis_max_vec_ = options_->get_int("CASSCF_DIIS_MAX_VEC");
    diis_min_vec_ = options_->get_int("CASSCF_DIIS_MIN_VEC");
    do_diis_ = diis_start_ >= 1;
}

void MCSCF_2STEP::print_options() {
    // fill in information
    std::vector<std::pair<std::string, int>> info_int{
        {"Printing level", print_},
        {"Max number of macro iterations", maxiter_},
        {"Max number of micro iterations", micro_maxiter_},
        {"Min number of micro iterations", micro_miniter_}};

    std::vector<std::pair<std::string, double>> info_double{{"Energy convergence", e_conv_},
                                                            {"Gradient convergence", g_conv_},
                                                            {"Max value for rotation", max_rot_}};

    std::vector<std::pair<std::string, std::string>> info_string{
        {"Integral type", int_type_},
        {"CI solver type", ci_type_},
        {"Final orbital type", orb_type_redundant_},
        {"Derivative type", der_type_}};

    std::vector<std::pair<std::string, bool>> info_bool{
        {"Optimize orbitals", opt_orbs_},
        {"Include internal rotations", internal_rot_},
        {"Debug printing", debug_print_}};

    if (do_diis_) {
        info_int.emplace_back("DIIS start", diis_start_);
        info_int.emplace_back("Min DIIS vectors", diis_min_vec_);
        info_int.emplace_back("Max DIIS vectors", diis_max_vec_);
        info_int.emplace_back("Frequency of DIIS extrapolation", diis_freq_);
    }

    // print some information
    print_selected_options("Calculation Information", info_string, info_bool, info_double,
                           info_int);
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
        psi::outfile->Printf("\n  Please increase CASSCF_MAXITER!");
        if (options_->get_bool("CASSCF_DIE_IF_NOT_CONVERGED")) {
            psi::outfile->Printf(
                "\n  This error may be ignored by setting CASSCF_DIE_IF_NOT_CONVERGED.");
            throw std::runtime_error(msg.str());
        }
    };

    // prepare for orbital gradients
    CASSCF_ORB_GRAD cas_grad(options_, mo_space_info_, ints_);
    auto nrot = cas_grad.nrot();
    auto dG = std::make_shared<psi::Vector>("dG", nrot);

    // set up initial guess for rotation matrix (R = 0)
    auto R = std::make_shared<psi::Vector>("R", nrot);

    bool no_orb_opt = (!opt_orbs_ or !nrot);

    // convergence for final CI
    auto r_conv = options_->get_double("R_CONVERGENCE");
    auto as_maxiter = options_->get_int("DL_MAXITER");

    auto as_solver =
        make_active_space_solver(ci_type_, to_state_nroots_map(state_weights_map_), scf_info_,
                                 mo_space_info_, cas_grad.active_space_ints(), options_);
    as_solver->set_print(print_);
    as_solver->set_e_convergence(e_conv_);
    as_solver->set_r_convergence(no_orb_opt ? r_conv : 1.0e-2);
    as_solver->set_maxiter(no_orb_opt ? as_maxiter : 15);
    as_solver->set_die_if_not_converged(no_orb_opt);

    // initial CI and resulting RDMs
    const auto state_energies_map = as_solver->compute_energy();
    auto e_c = compute_average_state_energy(state_energies_map, state_weights_map_);
    auto rdms = as_solver->compute_average_rdms(state_weights_map_, 2, RDMsType::spin_free);
    cas_grad.set_rdms(rdms);
    cas_grad.evaluate(R, dG);

    // Case 1: if there is no orbital optimization
    if (no_orb_opt) {
        energy_ = e_c;
        pass_energy_to_psi4();
        if (der_type_ == "FIRST") {
            cas_grad.compute_nuclear_gradient();
        }
        return energy_;
    }

    // set up L-BFGS solver and its parameters for micro iteration
    auto lbfgs_param = std::make_shared<LBFGS_PARAM>();
    lbfgs_param->epsilon = g_conv_;
    lbfgs_param->print = debug_print_ ? 5 : print_;
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
        bool restart = (ci_type_ == "FCI" or ci_type_ == "DETCI" or ci_type_ == "CAS");
        as_solver->set_restart(restart);
        as_solver->set_die_if_not_converged(false);
        as_solver->set_maxiter(restart ? 15 : as_maxiter);

        // CI convergence criteria along the way
        double dl_e_conv = 5.0e-7;
        double dl_r_conv = 8.0e-5;

        // start iterations
        lbfgs_param->maxiter = micro_miniter_;
        bool skip_de_conv = (ci_type_.find("DMRG") != std::string::npos);
        std::vector<CASSCF_HISTORY> history;

        for (int macro = 1; macro <= maxiter_; ++macro) {
            // optimize orbitals
            cas_grad.set_rdms(rdms);

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
            CASSCF_HISTORY hist(e_c, e_o, g_rms, n_micro);
            history.push_back(hist);

            double de = e_o - e_c;
            double de_o = (macro > 1) ? e_o - history[macro - 2].e_o : e_o;
            double de_c = (macro > 1) ? e_c - history[macro - 2].e_c : e_c;

            // print data of this iteration
            print_h2("MCSCF Macro Iter. " + std::to_string(macro));
            std::string title = "         Energy CI (  Delta E  )"
                                "         Energy Opt. (  Delta E  )"
                                "  E_OPT - E_CI   Orbital RMS  Micro";
            if (do_diis_)
                title += "  DIIS";
            psi::outfile->Printf("\n    %s", title.c_str());
            psi::outfile->Printf("\n    %18.12f (%11.4e)  %18.12f (%11.4e)  %12.4e  %12.4e %4d/%c",
                                 e_c, de_c, e_o, de_o, de, g_rms, n_micro, o_conv);

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
                psi::outfile->Printf(
                    "\n\n  A miracle has come to pass: MCSCF iterations have converged!");
                converged = true;
                break;
            }

            // test history
            bool reset_diis = false;
            int increase_lbfgs = 0;
            if (macro > 6) {
                if (macro > maxiter_ * 3 / 2) {
                    int n_samples = 20 < maxiter_ / 2 ? 20 : maxiter_ / 2;
                    if (not test_history(history, n_samples)) {
                        psi::outfile->Printf(
                            "\n\n  MCSCF does not seem to be converging! Quitting!");
                        break;
                    }
                }
                if (not test_history(history, 5 + macro / 6)) {
                    reset_diis = true;
                    increase_lbfgs = (lbfgs_param->maxiter + 2) < micro_maxiter_ ? 2 : 0;
                } else {
                    increase_lbfgs = lbfgs_param->maxiter > micro_miniter_ ? -1 : 0;
                }
            }

            // adjust max number micro iterations
            lbfgs_param->maxiter += increase_lbfgs;

            // DIIS for orbitals
            if (do_diis_) {
                if (macro >= diis_start_) {
                    // reset DIIS if current orbital update unreasonable
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
            auto print_level = debug_print_ ? 5 : print_;
            e_c = diagonalize_hamiltonian(as_solver, fci_ints,
                                          {print_level, dl_e_conv, dl_r_conv, false});
            rdms = as_solver->compute_average_rdms(state_weights_map_, 2, RDMsType::spin_free);
        }

        diis_manager.reset_subspace();
        diis_manager.delete_diis_file();

        // print summary
        print_macro_iteration(history);
    }

    // perform final CI using converged orbitals
    energy_ =
        diagonalize_hamiltonian(as_solver, cas_grad.active_space_ints(),
                                {print_, e_conv_, r_conv, options_->get_bool("DUMP_ACTIVE_WFN")});

    if (ints_->integral_type() != Custom) {
        auto final_orbs = options_->get_str("CASSCF_FINAL_ORBITAL");

        if (final_orbs != "UNSPECIFIED" or der_type_ == "FIRST") {
            // fix orbitals for redundant pairs
            rdms = as_solver->compute_average_rdms(state_weights_map_, 1, RDMsType::spin_free);
            auto F = cas_grad.fock(rdms);
            ints_->set_fock_matrix(F, F);

            SemiCanonical semi(mo_space_info_, ints_, options_);
            semi.semicanonicalize(rdms, false, final_orbs == "NATURAL", false);

            cas_grad.canonicalize_final(semi.Ua());

            // TODO: need to implement the transformation of CI coefficients due to orbital changes
        }

        // pass to wave function
        auto Ca = cas_grad.Ca();
        ints_->wfn()->Ca()->copy(Ca);
        ints_->wfn()->Cb()->copy(Ca);

        // throw error if not converged
        if (not converged)
            throw_convergence_error();

        // for nuclear gradient
        if (der_type_ == "FIRST") {
            // TODO: remove this re-diagonalization if CI transformation is impelementd
            if (not is_single_reference()) {
                diagonalize_hamiltonian(
                    as_solver, cas_grad.active_space_ints(),
                    {print_, e_conv_, r_conv, options_->get_bool("DUMP_ACTIVE_WFN")});
            }

            // recompute gradient due to canonicalization
            rdms = as_solver->compute_average_rdms(state_weights_map_, 2, RDMsType::spin_free);
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
    auto nclosed_electrons = 2 * mo_space_info_->size("INACTIVE_DOCC");

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

double MCSCF_2STEP::diagonalize_hamiltonian(std::shared_ptr<ActiveSpaceSolver>& as_solver,
                                            std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                            const std::tuple<int, double, double, bool>& params) {
    const auto& [print, e_conv, r_conv, dump_wfn] = params;
    as_solver->set_print(print);
    as_solver->set_e_convergence(e_conv);
    as_solver->set_r_convergence(r_conv);
    as_solver->set_die_if_not_converged(false);
    as_solver->set_active_space_integrals(fci_ints);

    const auto state_energies_map = as_solver->compute_energy();

    if (dump_wfn)
        as_solver->dump_wave_function();

    return compute_average_state_energy(state_energies_map, state_weights_map_);
}

bool MCSCF_2STEP::test_history(const std::vector<CASSCF_HISTORY>& history, const int& n_samples) {
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

void MCSCF_2STEP::print_macro_iteration(const std::vector<CASSCF_HISTORY>& history) {
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
make_mcscf_two_step(const std::map<StateInfo, std::vector<double>>& state_weight_map,
                    std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                    std::shared_ptr<MOSpaceInfo> mo_space_info,
                    std::shared_ptr<ForteIntegrals> ints) {
    return std::make_unique<MCSCF_2STEP>(state_weight_map, options, mo_space_info, scf_info, ints);
}

} // namespace forte
