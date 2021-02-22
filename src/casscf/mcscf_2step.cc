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

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libdiis/diisentry.h"
#include "psi4/libdiis/diismanager.h"

#include "base_classes/rdms.h"
#include "integrals/integrals.h"
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
    do_diis_ = (diis_start_ < 1) ? false : true;
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
        info_int.push_back({"DIIS start", diis_start_});
        info_int.push_back({"Min DIIS vectors", diis_min_vec_});
        info_int.push_back({"Max DIIS vectors", diis_max_vec_});
        info_int.push_back({"Frequency of DIIS extrapolation", diis_freq_});
    }

    // print some information
    print_selected_options("Calculation Information", info_string, info_bool, info_double,
                           info_int);
}

double MCSCF_2STEP::compute_energy() {
    // prepare for orbital gradients
    CASSCF_ORB_GRAD cas_grad(options_, mo_space_info_, ints_);
    auto nrot = cas_grad.nrot();
    auto dG = std::make_shared<psi::Vector>("dG", nrot);

    // set up initial guess for rotation matrix (R = 0)
    auto R = std::make_shared<psi::Vector>("R", nrot);
    auto dR = std::make_shared<psi::Vector>();

    // directly return if no orbital optimization
    double r_conv = options_->get_double("R_CONVERGENCE");
    std::unique_ptr<ActiveSpaceSolver> as_solver;

    if (not opt_orbs_ or nrot == 0) {
        std::tie(as_solver, energy_) = diagonalize_hamiltonian(
            cas_grad.active_space_ints(), {print_, e_conv_, r_conv, false, false});
        auto rdms = as_solver->compute_average_rdms(state_weights_map_, 2);
        cas_grad.set_rdms(rdms);
        cas_grad.evaluate(R, dG);
        if (der_type_ == "FIRST") {
            cas_grad.compute_nuclear_gradient();
        }
        return energy_;
    }

    // DIIS extropolation for macro iteration
    psi::DIISManager diis_manager(do_diis_ ? diis_max_vec_ : 0, "MCSCF DIIS",
                                  psi::DIISManager::OldestAdded, psi::DIISManager::OnDisk);
    if (do_diis_) {
        dR = std::make_shared<psi::Vector>("dR", nrot);
        diis_manager.set_error_vector_size(1, psi::DIISEntry::Vector, dR.get());
        diis_manager.set_vector_size(1, psi::DIISEntry::Vector, R.get());
    }

    // set up L-BFGS solver and its parameters for micro iteration
    auto lbfgs_param = std::make_shared<LBFGS_PARAM>();
    lbfgs_param->maxiter = micro_miniter_;
    lbfgs_param->print = debug_print_ ? 5 : print_;
    lbfgs_param->max_dir = max_rot_;
    lbfgs_param->step_length_method = LBFGS_PARAM::STEP_LENGTH_METHOD::MAX_CORRECTION;

    LBFGS lbfgs(lbfgs_param);

    // CI convergence criteria along the way
    double dl_e_conv = nrot ? 1.0e-6 : e_conv_;
    double dl_r_conv = nrot ? 5.0e-4 : r_conv;

    // start iterations
    bool converged = false;
    bool sr = mo_space_info_->size("ACTIVE") == 0;
    double e_c;
    RDMs rdms;
    std::vector<CASSCF_HISTORY> history;
    bool dump_wfn = ci_type_ == "DETCI";

    for (int macro = 1; macro <= maxiter_; ++macro) {
        // solve CI problem
        if (macro == 1 or (not sr)) {
            auto fci_ints = cas_grad.active_space_ints();
            auto print_level = debug_print_ ? print_ : 0;
            bool read_wfn_guess = dump_wfn and macro != 1;
            std::tie(as_solver, e_c) = diagonalize_hamiltonian(
                fci_ints, {print_level, dl_e_conv, dl_r_conv, read_wfn_guess, dump_wfn});
            rdms = as_solver->compute_average_rdms(state_weights_map_, 2);
        }
        double de_c = (macro > 1) ? e_c - history[macro - 2].e_c : e_c;

        // optimize orbitals
        cas_grad.set_rdms(rdms);
        if (macro > 1) {
            double epsilon = std::fabs(de_c) * 0.1;
            epsilon = epsilon > 1.0e-6 ? 1.0e-6 : epsilon;
            lbfgs_param->epsilon = epsilon < g_conv_ ? g_conv_ : epsilon;
        }

        print_h2("Optimizing Orbitals for Current RDMs");
        double e_o = lbfgs.minimize(cas_grad, R);

        // info for orbital optimization
        dG->subtract(lbfgs.g());
        double g_rms = dG->rms();
        dG->copy(*lbfgs.g());

        int n_micro = lbfgs.iter();
        char o_conv = lbfgs.converged() ? 'Y' : 'N';

        // save data for this macro iteration
        CASSCF_HISTORY hist(e_c, e_o, g_rms, n_micro);

        double de = e_o - e_c;
        double de_o = (macro > 1) ? e_o - history[macro - 2].e_o : e_o;

        history.push_back(hist);

        // print data of this iteration
        print_h2("MCSCF Macro Iter. " + std::to_string(macro));
        std::string title = "         Energy CI (  Delta E  )         Energy Opt. (  Delta E  )  "
                            "E_OPT - E_CI   Orbital RMS  Micro";
        if (do_diis_)
            title += "  DIIS";
        psi::outfile->Printf("\n    %s", title.c_str());
        psi::outfile->Printf("\n    %18.12f (%11.4e)  %18.12f (%11.4e)  %12.4e  %12.4e %4d/%c", e_c,
                             de_c, e_o, de_o, de, g_rms, n_micro, o_conv);

        // test convergence
        bool is_e_conv =
            std::fabs(de) < e_conv_ and std::fabs(de_c) < e_conv_ and std::fabs(de_o) < e_conv_;
        bool is_g_conv = g_rms < g_conv_ or lbfgs.converged();
        // at convergence, DIIS should not be just reset
        bool is_diis_conv = do_diis_ ? (diis_manager.subspace_size() > 1) : true;
        if (is_e_conv and is_g_conv and is_diis_conv) {
            std::string msg = "A miracle has come to pass: MCSCF iterations have converged!";
            psi::outfile->Printf("\n\n  %s", msg.c_str());
            energy_ = e_o;
            converged = true;
            break;
        }

        // set convergence thresholds for Davidson-Liu solver
        if (macro > 1) {
            dl_e_conv = 0.1 * std::fabs(de);
            dl_r_conv = 0.5 * std::sqrt(dl_e_conv);
            if (0.01 * g_rms < g_conv_ or dl_e_conv < e_conv_) {
                dl_e_conv = e_conv_;
                dl_r_conv = r_conv;
            }
        }

        // DIIS for orbitals
        if (do_diis_) {
            if (macro >= diis_start_) {
                // reset DIIS if current orbital update unreasonable
                if (de_c > 0.0 or (de > 0.0 and de_o > 0.0)) {
                    psi::outfile->Printf("   R/");
                    diis_manager.reset_subspace();
                } else {
                    psi::outfile->Printf("   ");
                }

                dR->subtract(R);
                dR->scale(-1.0);

                diis_manager.add_entry(2, dR.get(), R.get());
                psi::outfile->Printf("S");
            }

            if ((macro - diis_start_) % diis_freq_ == 0 and
                diis_manager.subspace_size() > diis_min_vec_) {
                diis_manager.extrapolate(1, R.get());
                psi::outfile->Printf("/E");

                // update the actual integrals for CI, skip gradient computation
                cas_grad.evaluate(R, dG, false);
            }

            dR->copy(*R);
        }

        // increase micro iterations if energy goes up
        int inc = 0;
        int& nit = lbfgs_param->maxiter;

        if (de > 0.0 and nit <= micro_maxiter_ - 2)
            inc = 2;

        if (macro >= 4) {
            double de_o1 = history[macro - 2].e_o - history[macro - 3].e_o;
            double de_o2 = history[macro - 3].e_o - history[macro - 4].e_o;
            if (de_o > 0.0 and de_o1 > 0.0 and de_o2 > 0.0 and nit <= micro_maxiter_ - 5)
                inc = 5;

            if (de < 0.0 and de_o < 0.0 and de_o1 < 0.0 and nit > micro_miniter_ + 2)
                inc -= 1;
        }

        nit += inc;
    }

    diis_manager.reset_subspace();
    diis_manager.delete_diis_file();

    // print summary
    print_macro_iteration(history);

    // function to throw not converging error
    auto throw_converence_error = [&]() {
        if (not converged) {
            auto m = std::to_string(maxiter_);
            throw std::runtime_error("MCSCF did not converge in " + m + " iterations!");
        }
    };

    if (ints_->integral_type() != Custom) {
        // fix orbitals for redundant pairs
        auto F = cas_grad.fock();
        ints_->set_fock_matrix(F, F);

        SemiCanonical semi(mo_space_info_, ints_, options_);
        semi.semicanonicalize(rdms, 1, false, false);

        cas_grad.canonicalize_final(semi.Ua());

        // rediagonalize Hamiltonian
        auto fci_ints = cas_grad.active_space_ints();
        auto dump_wfn_new = dump_wfn and options_->get_bool("DUMP_ACTIVE_WFN");
        std::tie(as_solver, energy_) = diagonalize_hamiltonian(
            fci_ints, {print_, dl_e_conv, dl_r_conv, dump_wfn, dump_wfn_new});

        // pass to wave function
        auto Ca = cas_grad.Ca();
        ints_->wfn()->Ca()->copy(Ca);
        ints_->wfn()->Cb()->copy(Ca);

        // throw error if not converged
        throw_converence_error();

        // for nuclear gradient
        if (der_type_ == "FIRST") {
            // recompute gradient due to canonicalization
            rdms = as_solver->compute_average_rdms(state_weights_map_, 2);
            cas_grad.set_rdms(rdms);
            cas_grad.evaluate(R, dG);

            // compute densities used for nuclear gradient
            cas_grad.compute_nuclear_gradient();
        }
    } else {
        // throw error if not converged
        throw_converence_error();
    }

    return energy_;
}

std::tuple<std::unique_ptr<ActiveSpaceSolver>, double>
MCSCF_2STEP::diagonalize_hamiltonian(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                     const std::tuple<int, double, double, bool, bool>& params) {
    auto state_map = to_state_nroots_map(state_weights_map_);
    auto active_space_solver = make_active_space_solver(ci_type_, state_map, scf_info_,
                                                        mo_space_info_, fci_ints, options_);

    int print;
    double e_conv, r_conv;
    bool read_wfn_guess, dump_wfn;
    std::tie(print, e_conv, r_conv, read_wfn_guess, dump_wfn) = params;

    active_space_solver->set_print(print);
    active_space_solver->set_e_convergence(e_conv);
    active_space_solver->set_r_convergence(r_conv);
    active_space_solver->set_read_initial_guess(read_wfn_guess);

    const auto state_energies_map = active_space_solver->compute_energy();

    if (dump_wfn)
        active_space_solver->dump_wave_function();

    double e = compute_average_state_energy(state_energies_map, state_weights_map_);

    return {std::move(active_space_solver), e};
}

void MCSCF_2STEP::print_macro_iteration(std::vector<CASSCF_HISTORY>& history) {
    print_h2("MCSCF Iteration Summary");
    std::string dash1 = std::string(30, '-').c_str();
    std::string dash2 = std::string(88, '-').c_str();
    psi::outfile->Printf("\n                      Energy CI                    Energy Orbital");
    psi::outfile->Printf("\n           %s  %s", dash1.c_str(), dash1.c_str());
    psi::outfile->Printf("\n    Iter.        Total Energy       Delta        Total Energy       "
                         "Delta  Orb. Grad.  Micro");
    psi::outfile->Printf("\n    %s", dash2.c_str());
    for (int i = 0, size = history.size(); i < size; ++i) {
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
