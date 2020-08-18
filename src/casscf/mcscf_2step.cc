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
                         "Two-Step Approx. Second-Order AO Algorithm",
                         "written by Chenyang Li and Kevin P. Hannon"});

    // read and print options
    read_options();
    print_options();
}

void MCSCF_2STEP::read_options() {
    print_ = options_->get_int("PRINT");
    debug_print_ = options_->get_bool("CASSCF_DEBUG_PRINTING");

    int_type_ = options_->get_str("INT_TYPE");

    maxiter_ = options_->get_int("CASSCF_MAXITER");
    micro_maxiter_ = options_->get_int("CASSCF_MICRO_MAXITER");

    e_conv_ = options_->get_double("CASSCF_E_CONVERGENCE");
    g_conv_ = options_->get_double("CASSCF_G_CONVERGENCE");

    orb_type_redundant_ = options_->get_str("CASSCF_FINAL_ORBITAL");

    ci_type_ = options_->get_str("CASSCF_CI_SOLVER");

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
        {"Max number of micro iterations", micro_maxiter_}};

    std::vector<std::pair<std::string, double>> info_double{{"Energy convergence", e_conv_},
                                                            {"Gradient convergence", g_conv_},
                                                            {"Max value for rotation", max_rot_}};

    std::vector<std::pair<std::string, std::string>> info_string{
        {"Integral type", int_type_},
        {"CI solver type", ci_type_},
        {"Final orbital type", orb_type_redundant_}};

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

double MCSCF_2STEP::compute_energy() {
    struct CASSCF_HISTORY {
        double e_c;   // energy from CI
        double e_o;   // energy after orbital optimization
        double g_rms; // RMS of gradient vector
        int n_micro;  // number of micro iteration
    };

    std::vector<CASSCF_HISTORY> history;

    // prepare for orbital gradients
    CASSCF_ORB_GRAD cas_grad(options_, mo_space_info_, ints_);
    auto dG = std::make_shared<psi::Vector>("dG", cas_grad.nrot());

    // set up initial guess for rotation matrix (R = 0)
    auto R = std::make_shared<psi::Vector>("R", cas_grad.nrot());
    auto dR = std::make_shared<psi::Vector>();

    // DIIS extropolation for macro iteration
    auto diis_manager =
        std::make_shared<psi::DIISManager>(do_diis_ ? diis_max_vec_ : 0, "MCSCF DIIS",
                                           psi::DIISManager::OldestAdded, psi::DIISManager::InCore);
    if (do_diis_) {
        dR = std::make_shared<psi::Vector>("dR", cas_grad.nrot());

        diis_manager->set_error_vector_size(1, psi::DIISEntry::Vector, dR.get());
        diis_manager->set_vector_size(1, psi::DIISEntry::Vector, R.get());
    }

    // set up L-BFGS solver and its parameters for micro iteration
    auto lbfgs_param = std::make_shared<LBFGS_PARAM>();
    lbfgs_param->epsilon = g_conv_;
    int micro_maxiter_0 = micro_maxiter_ > 5 ? 6 : micro_maxiter_;
    lbfgs_param->maxiter = micro_maxiter_0;
    lbfgs_param->m = micro_maxiter_0;
    lbfgs_param->print = debug_print_ ? 5 : print_;
    lbfgs_param->max_dir = max_rot_;
    lbfgs_param->step_length_method = LBFGS_PARAM::STEP_LENGTH_METHOD::MAX_CORRECTION;

    LBFGS lbfgs(lbfgs_param);

    // start iterations
    bool converged = false;
    bool sr = mo_space_info_->size("ACTIVE") == 0;
    double e_c;

    for (int macro = 1; macro <= maxiter_; ++macro) {
        // solve CI problem
        RDMs rdms;
        if (macro == 1 or (not sr)) {
            auto fci_ints = cas_grad.active_space_ints();
            auto as_solver = diagonalize_hamiltonian(fci_ints, debug_print_ ? print_ : 0, e_c);
            rdms = as_solver->compute_average_rdms(state_weights_map_, 2);
        }

        // optimize orbitals
        cas_grad.set_rdms(rdms);
        double e_o = lbfgs.minimize(cas_grad, R);

        // info for orbital optimization
        dG->subtract(lbfgs.g());
        double g_rms = dG->rms();
        dG->copy(*lbfgs.g());

        int n_micro = lbfgs.iter();
        char o_conv = lbfgs.converged() ? 'Y' : 'N';

        // save data for this macro iteration
        CASSCF_HISTORY hist;
        hist.e_c = e_c;
        hist.e_o = e_o;
        hist.g_rms = g_rms;
        hist.n_micro = n_micro;

        double de = e_o - e_c;
        double de_c = (macro > 1) ? e_c - history[macro - 2].e_c : e_c;
        double de_o = (macro > 1) ? e_o - history[macro - 2].e_o : e_o;

        history.push_back(hist);

        // print data of this iteration
        print_h2("MCSCF Macro Iter. " + std::to_string(macro));
        std::string title = "         Energy CI (  Delta E  )         Energy Opt. (  Delta E  )  "
                            "E_OPT - E_CI   Orbital RMS  Micro";
        if (do_diis_)
            title += "  DIIS";
        psi::outfile->Printf("\n    %s", title.c_str());
        psi::outfile->Printf("\n    %18.12f (%11.4e)  %18.12f (%11.4e)  %12.4e  %12.4e %4d/%c",
                             e_c, de_c, e_o, de_o, de, g_rms, n_micro, o_conv);

        // test convergence
        bool is_e_conv =
            std::fabs(de) < e_conv_ and std::fabs(de_c) < e_conv_ and std::fabs(de_o) < e_conv_;
        bool is_g_conv = g_rms < g_conv_ or lbfgs.converged();
        if (is_e_conv and is_g_conv) {
            std::string msg = "A miracle has come to pass: MCSCF iterations have converged!";
            psi::outfile->Printf("\n\n  %s", msg.c_str());
            energy_ = e_o;
            converged = true;
            break;
        }

        // DIIS for orbitals
        if (do_diis_) {
            dR->subtract(R);
            dR->scale(-1.0);

            if (macro >= diis_start_) {
                diis_manager->add_entry(2, dR.get(), R.get());
                psi::outfile->Printf("   S");
            }

            if ((macro - diis_start_) % diis_freq_ == 0 and
                diis_manager->subspace_size() > diis_min_vec_) {
                diis_manager->extrapolate(1, R.get());
                psi::outfile->Printf("/E");
            }

            dR->copy(*R);
        }

        // increase micro iterations if energy goes up
        if (macro > 4) {
            int inc = 0;
            int& nit = lbfgs_param->maxiter;

            if (de > 0.0 and nit <= micro_maxiter_ - 2)
                inc = 2;

            double de_o1 = history[macro - 2].e_o - history[macro - 3].e_o;
            double de_o2 = history[macro - 3].e_o - history[macro - 4].e_o;
            if (de_o > 0.0 and de_o1 > 0.0 and de_o2 > 0.0 and nit <= micro_maxiter_ - 5)
                inc = 5;

            if (de < 0.0 and de_o < 0.0 and de_o1 < 0.0 and nit > micro_maxiter_0)
                inc -= 1;

            if (inc > 0)
                diis_manager->reset_subspace();

            nit += inc;
        }
    }

    // print summary
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

    // fix orbitals for redundant pairs
    cas_grad.canonicalize_final();

    // rediagonalize Hamiltonian
    auto fci_ints = cas_grad.active_space_ints();
    auto as_solver = diagonalize_hamiltonian(fci_ints, print_, energy_);

    // pass to wave function
    ints_->wfn()->Ca()->copy(cas_grad.Ca());
    ints_->wfn()->Cb()->copy(cas_grad.Ca());

    // throw error if not converged
    if (not converged) {
        throw std::runtime_error("MCSCF did not converge in " + std::to_string(maxiter_) +
                                 " iterations!");
    }

    return energy_;
}

std::unique_ptr<ActiveSpaceSolver>
MCSCF_2STEP::diagonalize_hamiltonian(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                     const int print, double& e_c) {
    auto state_map = to_state_nroots_map(state_weights_map_);
    auto active_space_solver = make_active_space_solver(ci_type_, state_map, scf_info_,
                                                        mo_space_info_, fci_ints, options_);
    active_space_solver->set_print(print);
    const auto state_energies_map = active_space_solver->compute_energy();

    // TODO: need to save CI vectors and dump to file and let solver read them

    e_c = compute_average_state_energy(state_energies_map, state_weights_map_);

    return active_space_solver;
}

std::unique_ptr<MCSCF_2STEP>
make_mcscf_two_step(const std::map<StateInfo, std::vector<double>>& state_weight_map,
                    std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                    std::shared_ptr<MOSpaceInfo> mo_space_info,
                    std::shared_ptr<ForteIntegrals> ints) {
    return std::make_unique<MCSCF_2STEP>(state_weight_map, options, mo_space_info, scf_info, ints);
}

} // namespace forte
