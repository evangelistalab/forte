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

#include "lambda-ci.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <unordered_map>

#include "mini-boost/boost/format.hpp"
#include "mini-boost/boost/timer.hpp"

#include <libciomr/libciomr.h>
#include <libmints/molecule.h>
#include <libpsio/psio.h>
#include <libpsio/psio.hpp>
#include <libqt/qt.h>

#include "cartographer.h"
#include "dynamic_bitset_determinant.h"
#include "ex-aci.h"
#include "sparse_ci_solver.h"
#include "string_determinant.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

/**
 * Template to store 3-index quantity of any type
 * I use it to store c_I, symmetry label, and ordered labels
 * for determinants
 **/
template <typename a, typename b, typename c>
using oVector = std::vector<std::pair<a, std::pair<b, c>>>;

/**
 * Used for initializing a vector of pairs of any type
 */
template <typename a, typename b> using pVector = std::vector<std::pair<a, b>>;

inline double clamp(double x, double a, double b)

{
    return x < a ? a : (x > b ? b : x);
}

/**
 * @brief smootherstep
 * @param edge0
 * @param edge1
 * @param x
 * @return
 *
 * This is a smooth step function that is
 * 0.0 for x <= edge0
 * 1.0 for x >= edge1
 */
inline double smootherstep(double edge0, double edge1, double x) {
    if (edge1 == edge0) {
        return x <= edge0 ? 0.0 : 1.0;
    }
    // Scale, and clamp x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    // Evaluate polynomial
    return x * x * x * (x * (x * 6. - 15.) + 10.);
}

EX_ACI::EX_ACI(boost::shared_ptr<Wavefunction> wfn, Options& options, ForteIntegrals* ints)
    : Wavefunction(options, _default_psio_lib_), options_(options), ints_(ints) {
    // Copy the wavefunction information
    copy(wfn);

    startup();
    print_info();
}

void EX_ACI::startup() {

    // Connect the integrals to the determinant class
    StringDeterminant::set_ints(ints_);
    DynamicBitsetDeterminant::set_ints(ints_);

    // The number of correlated molecular orbitals
    ncmo_ = ints_->ncmo();
    ncmopi_ = ints_->ncmopi();

    // Number of correlated electrons
    ncel_ = 0;
    for (int h = 0; h < nirrep_; ++h) {
        ncel_ += 2 * doccpi_[h] + soccpi_[h];
    }
    outfile->Printf("\n  Number of electrons: %d", ncel_);

    // Overwrite the frozen orbitals arrays
    frzcpi_ = ints_->frzcpi();
    frzvpi_ = ints_->frzvpi();

    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();

    // Create the array with mo symmetry and compute the number of frozen orbitals
    nfrzc_ = 0;
    for (int h = 0; h < nirrep_; ++h) {
        nfrzc_ += frzcpi_[h];
        for (int p = 0; p < ncmopi_[h]; ++p) {
            mo_symmetry_.push_back(h);
        }
    }

    outfile->Printf("\n  There are %d frozen orbitals.", nfrzc_);

    // Collect information about the reference wavefunction
    wavefunction_multiplicity_ = 1;
    if (options_["MULTIPLICITY"].has_changed()) {
        wavefunction_multiplicity_ = options_.get_int("MULTIPLICITY");
    }
    wavefunction_symmetry_ = 0;
    if (options_["ROOT_SYM"].has_changed()) {
        wavefunction_symmetry_ = options_.get_int("ROOT_SYM");
    }

    // Build the reference determinant with correct symmetry
    reference_determinant_ = StringDeterminant(get_occupation());

    outfile->Printf("\n  The reference determinant is:\n");
    reference_determinant_.print();

    // Read options
    nroot_ = options_.get_int("NROOT");

    tau_p_ = options_.get_double("TAUP");
    tau_q_ = options_.get_double("TAUQ");

    do_smooth_ = options_.get_bool("SMOOTH");
    smooth_threshold_ = options_.get_double("SMOOTH_THRESHOLD");

    spin_tol_ = options_.get_double("SPIN_TOL");
    // set the initial S^2 guess as input multiplicity
    for (int n = 0; n < nroot_; ++n) {
        root_spin_vec_.push_back(
            std::make_pair((wavefunction_multiplicity_ - 1.0) / 2.0, wavefunction_multiplicity_ - 1.0));
    }

    perturb_select_ = options_.get_bool("PERTURB_SELECT");
    pq_function_ = options_.get_str("PQ_FUNCTION");
    q_rel_ = options_.get_bool("Q_REL");
    q_reference_ = options_.get_str("Q_REFERENCE");
    ex_alg_ = options_.get_str("EXCITED_ALGORITHM");
    post_root_ = max(nroot_, options_.get_int("POST_ROOT"));
    post_diagonalize_ = options_.get_bool("POST_DIAGONALIZE");
    form_1_RDM_ = options_.get_bool("1_RDM");
    do_guess_ = options_.get_bool("LAMBDA_GUESS");
    spin_complete_ = options_.get_bool("ENFORCE_SPIN_COMPLETE");

    aimed_selection_ = false;
    energy_selection_ = false;
    if (options_.get_str("SELECT_TYPE") == "AIMED_AMP") {
        aimed_selection_ = true;
        energy_selection_ = false;
    } else if (options_.get_str("SELECT_TYPE") == "AIMED_ENERGY") {
        aimed_selection_ = true;
        energy_selection_ = true;
    } else if (options_.get_str("SELECT_TYPE") == "ENERGY") {
        aimed_selection_ = false;
        energy_selection_ = true;
    } else if (options_.get_str("SELECT_TYPE") == "AMP") {
        aimed_selection_ = false;
        energy_selection_ = false;
    }
}

EX_ACI::~EX_ACI() {}

std::vector<int> EX_ACI::get_occupation() {

    std::vector<int> occupation(2 * ncmo_, 0);

    // Get reference type
    std::string ref_type = options_.get_str("REFERENCE");
    outfile->Printf("\n  Using %s reference.\n", ref_type.c_str());

    // nsym denotes the number of electrons needed to assign symmetry and multiplicity
    int nsym = wavefunction_multiplicity_ - 1;
    int orb_sym = wavefunction_symmetry_;

    if (wavefunction_multiplicity_ == 1) {
        nsym = 2;
    }

    // Grab an ordered list of orbital energies, symmetry labels, and Pitzer-indices
    oVector<double, int, int> labeled_orb_en;
    oVector<double, int, int> labeled_orb_en_alfa;
    oVector<double, int, int> labeled_orb_en_beta;

    if (ref_type == "RHF" or ref_type == "RKS" or ref_type == "ROHF") {
        labeled_orb_en = sym_labeled_orbitals("RHF");
    } else if (ref_type == "UHF" or ref_type == "UKS") {
        labeled_orb_en_alfa = sym_labeled_orbitals("ALFA");
        labeled_orb_en_beta = sym_labeled_orbitals("BETA");
    }

    // For a restricted reference
    if (ref_type == "RHF" or ref_type == "RKS" or ref_type == "ROHF") {

        // Build initial reference determinant from restricted reference
        for (size_t i = 0; i < nalpha() - nfrzc_; ++i) {
            occupation[labeled_orb_en[i].second.second] = 1;
        }
        for (size_t i = 0; i < nbeta() - nfrzc_; ++i) {
            occupation[ncmo_ + labeled_orb_en[i].second.second] = 1;
        }

        // Loop over as many outer-shell electrons needed to get correct symmetry
        for (int k = 1; k <= nsym;) {

            bool add = false;

            // remove electron from highest-energy docc
            occupation[labeled_orb_en[nalpha() - k - nfrzc_].second.second] = 0;
            outfile->Printf("\n  Electron removed from %d, out of %d",
                            labeled_orb_en[nalpha() - k - nfrzc_].second.second, ncel_);

            // Determine proper symmetry for new occupation
            orb_sym = wavefunction_symmetry_;

            if (wavefunction_multiplicity_ == 1) {
                orb_sym =
                    direct_sym_product(labeled_orb_en[nalpha() - 1 - nfrzc_].second.first, orb_sym);
            } else {
                for (int i = 1; i <= nsym; ++i) {
                    orb_sym = direct_sym_product(labeled_orb_en[nalpha() - i - nfrzc_].second.first,
                                                 orb_sym);
                }
                orb_sym =
                    direct_sym_product(labeled_orb_en[nalpha() - k - nfrzc_].second.first, orb_sym);
            }

            outfile->Printf("\n  Need orbital of symmetry %d", orb_sym);

            // add electron to lowest-energy orbital of proper symmetry
            // Loop from current occupation to max MO until correct orbital is reached
            for (int i = nalpha() - k - nfrzc_; i < ncmo_ - nfrzc_; ++i) {
                if (orb_sym == labeled_orb_en[i].second.first and
                    occupation[labeled_orb_en[i].second.second] != 1) {
                    occupation[labeled_orb_en[i].second.second] = 1;
                    outfile->Printf("\n  Added electron to %d", labeled_orb_en[i].second.second);
                    add = true;
                    break;
                }
            }

            // If a new occupation could not be created, add the electron back and remove a
            // different one
            if (!add) {
                occupation[labeled_orb_en[nalpha() - k - nfrzc_].second.second] = 1;
                outfile->Printf(
                    "\n  No orbital of %d symmetry available! Putting electron back. \n", orb_sym);
                ++k;
            } else {
                break;
            }
        }

    }
    // For an unrestricted reference
    else if (ref_type == "UHF" or ref_type == "UKS") {

        // Make the reference
        // For singlets, this will be "ground-state", closed-shell

        for (size_t i = 0; i < nalpha() - nfrzc_; ++i) {
            occupation[labeled_orb_en_alfa[i].second.second] = 1;
        }
        for (size_t i = 0; i < nbeta() - nfrzc_; ++i) {
            occupation[ncmo_ + labeled_orb_en_beta[i].second.second] = 1;
        }
        if (nalpha() >= nbeta()) {

            for (int k = 1; k < nsym;) {

                bool add = false;
                // remove electron from highest-energy docc
                occupation[labeled_orb_en_alfa[nalpha() - k - nfrzc_].second.second] = 0;
                outfile->Printf("\n  Electron removed from %d, out of %d",
                                labeled_orb_en_alfa[nalpha() - k - nfrzc_].second.second, ncel_);

                // Determine proper symmetry for new occupation
                orb_sym = wavefunction_symmetry_;

                if (wavefunction_multiplicity_ == 1) {
                    orb_sym = direct_sym_product(
                        labeled_orb_en_alfa[nalpha() - 1 - nfrzc_].second.first, orb_sym);
                } else {
                    for (int i = 1; i <= nsym; ++i) {
                        orb_sym = direct_sym_product(
                            labeled_orb_en_alfa[nalpha() - i - nfrzc_].second.first, orb_sym);
                    }
                    orb_sym = direct_sym_product(
                        labeled_orb_en_alfa[nalpha() - k - nfrzc_].second.first, orb_sym);
                }

                outfile->Printf("\n  Need orbital of symmetry %d", orb_sym);

                // add electron to lowest-energy orbital of proper symmetry
                for (int i = nalpha() - k - nfrzc_; i < ncmo_; ++i) {
                    if (orb_sym == labeled_orb_en_alfa[i].second.first and
                        occupation[labeled_orb_en_alfa[i].second.second] != 1) {
                        occupation[labeled_orb_en_alfa[i].second.second] = 1;
                        outfile->Printf("\n  Added electron to %d",
                                        labeled_orb_en_alfa[i].second.second);
                        add = true;
                        break;
                    }
                }

                // If a new occupation could not be created, add the electron back and remove a
                // different one
                if (!add) {
                    occupation[labeled_orb_en_alfa[nalpha() - k - nfrzc_].second.second] = 1;
                    outfile->Printf(
                        "\n  No orbital of %d symmetry available! Putting electron back. \n",
                        orb_sym);
                    ++k;
                } else {
                    break;
                }
            }
        }

        if (nalpha() < nbeta()) {

            for (int k = 1; k < nsym;) {

                bool add = false;
                // remove electron from highest-energy docc
                occupation[labeled_orb_en_beta[nbeta() - k - nfrzc_].second.second] = 0;
                outfile->Printf("\n  Electron removed from %d, out of %d",
                                labeled_orb_en_beta[nbeta() - k - nfrzc_].second.second, ncel_);

                // Determine proper symmetry for new occupation
                orb_sym = wavefunction_symmetry_;

                if (wavefunction_multiplicity_ == 1) {
                    orb_sym = direct_sym_product(
                        labeled_orb_en_beta[nbeta() - 1 - nfrzc_].second.first, orb_sym);
                } else {
                    for (int i = 1; i <= nsym; ++i) {
                        orb_sym = direct_sym_product(
                            labeled_orb_en_beta[nbeta() - i - nfrzc_].second.first, orb_sym);
                    }
                    orb_sym = direct_sym_product(
                        labeled_orb_en_beta[nbeta() - k - nfrzc_].second.first, orb_sym);
                }

                outfile->Printf("\n  Need orbital of symmetry %d", orb_sym);

                // add electron to lowest-energy orbital of proper symmetry
                for (int i = nbeta() - k - nfrzc_; i < ncmo_; ++i) {
                    if (orb_sym == labeled_orb_en_beta[i].second.first and
                        occupation[labeled_orb_en_beta[i].second.second] != 1) {
                        occupation[labeled_orb_en_beta[i].second.second] = 1;
                        outfile->Printf("\n  Added electron to %d",
                                        labeled_orb_en_beta[i].second.second);
                        add = true;
                        break;
                    }
                }

                // If a new occupation could not be created, add the electron back and remove a
                // different one
                if (!add) {
                    occupation[labeled_orb_en_beta[nbeta() - k - nfrzc_].second.second] = 1;
                    outfile->Printf(
                        "\n  No orbital of %d symmetry available! Putting electron back. \n",
                        orb_sym);
                    ++k;
                } else {
                    break;
                }
            }
        }
    }

    return occupation;
}

void EX_ACI::print_info() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{
        {"Symmetry", wavefunction_symmetry_},
        {"Number of roots", nroot_},
        {"Root used for properties", options_.get_int("ROOT")}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"P-threshold", tau_p_},
        {"Q-threshold", tau_q_},
        {"Convergence threshold", options_.get_double("E_CONVERGENCE")}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Determinant selection criterion",
         energy_selection_ ? "Second-order Energy" : "First-order Coefficients"},
        {"Selection criterion", aimed_selection_ ? "Aimed selection" : "Threshold"},
        {"Parameter type", perturb_select_ ? "PT" : "Non-PT"},
        {"PQ Function", options_.get_str("PQ_FUNCTION")},
        {"Q Type", q_rel_ ? "Relative Energy" : "Absolute Energy"}};
    //    {"Number of electrons",nel},
    //    {"Number of correlated alpha electrons",nalpha_},
    //    {"Number of correlated beta electrons",nbeta_},
    //    {"Number of restricted docc electrons",rdoccpi_.sum()},
    //    {"Charge",charge},
    //    {"Multiplicity",multiplicity},

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    outfile->Printf("\n  %s", string(52, '-').c_str());
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-40s   %5d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-39s %8.2e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-39s %s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    outfile->Printf("\n  %s", string(52, '-').c_str());
}

double EX_ACI::compute_energy() {
    ForteTimer t_iamrcisd;
    outfile->Printf("\n\n  Iterative Adaptive CI (v 2.0)");

    SharedMatrix H;
    SharedMatrix P_evecs;
    SharedMatrix PQ_evecs;
    SharedVector P_evals;
    SharedVector PQ_evals;

    // Use the reference determinant as a starting point
    std::vector<bool> alfa_bits = reference_determinant_.get_alfa_bits_vector_bool();
    std::vector<bool> beta_bits = reference_determinant_.get_beta_bits_vector_bool();
    DynamicBitsetDeterminant bs_det(alfa_bits, beta_bits);
    P_space_.push_back(bs_det);

    if (alfa_bits != beta_bits) {
        DynamicBitsetDeterminant ref2 = bs_det;
        ref2.spin_flip();
        P_space_.push_back(ref2);
    }

    P_space_map_[bs_det] = 1;

    if (do_guess_) {
        form_initial_space(P_space_, nroot_);
    }

    outfile->Printf("\n  The model space contains %zu determinants", P_space_.size());

    double old_avg_energy = reference_determinant_.energy() + nuclear_repulsion_energy_;
    double new_avg_energy = 0.0;

    std::vector<std::vector<double>> energy_history;
    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);

    int spin_projection = options_.get_int("SPIN_PROJECTION");

    int root;
    int maxcycle = 20;
    for (cycle_ = 0; cycle_ < maxcycle; ++cycle_) {
        // Step 1. Diagonalize the Hamiltonian in the P space

        outfile->Printf("\n\n  Cycle %3d", cycle_);
        outfile->Printf("\n  Initial P space dimension: %zu", P_space_.size());

        if (spin_complete_) {
            check_spin_completeness(P_space_);
            outfile->Printf("\n  %s: %zu determinants", "Spin-complete dimension of the P space",
                            P_space_.size());
        }

        // Set the roots as the lowest possible as initial guess
        int num_ref_roots = std::min(nroot_, int(P_space_.size()));

        // save the dimension of the previous iteration
        size_t PQ_space_init = PQ_space_.size();

        if (options_.get_str("DIAG_ALGORITHM") == "DAVIDSONLIST") {
            sparse_solver.diagonalize_hamiltonian(P_space_, P_evals, P_evecs, num_ref_roots,
                                                  DavidsonLiuList);
        } else {
            sparse_solver.diagonalize_hamiltonian(P_space_, P_evals, P_evecs, num_ref_roots,
                                                  DavidsonLiuSparse);
        }

        // Use projection to ensure P space is spin pure
        // Compute spin contamination
        // spins is a vector of (S, S^2, #I used to compute spin, %det space)
        //

        auto spins = compute_spin(P_space_, P_evecs, num_ref_roots);
        if (spin_projection == 1 or spin_projection == 3) {
            double spin_contam = 0.0;
            for (int n = 0; n < num_ref_roots; ++n) {
                spin_contam += spins[n].first.second;
            }
            spin_contam /= static_cast<double>(num_ref_roots);

            if (spin_contam >= spin_tol_) {
                outfile->Printf("\n  Average spin contamination per root is %1.5f", spin_contam);
                spin_transform(P_space_, P_evecs, num_ref_roots);
                P_evecs->zero();
                P_evecs = PQ_spin_evecs_->clone();
                sparse_solver.compute_H_expectation_val(P_space_, P_evals, P_evecs, num_ref_roots,
                                                        DavidsonLiuList);
            } else {
                outfile->Printf(
                    "\n  Average spin contamination (%1.6f) is less than tolerance (%1.6f)",
                    spin_contam, spin_tol_);
                outfile->Printf("\n  No need to perform spin projection.");
            }
        } else {
            outfile->Printf("\n  Not performing spin projection");
        }

        spins.clear();
        spins = compute_spin(P_space_, P_evecs, num_ref_roots);

        // Print the energy
        outfile->Printf("\n");
        for (int i = 0; i < num_ref_roots; ++i) {
            double abs_energy = P_evals->get(i) + nuclear_repulsion_energy_;
            double exc_energy = pc_hartree2ev * (P_evals->get(i) - P_evals->get(0));
            outfile->Printf(
                "\n    P-space  CI Energy Root %3d        = %.12f Eh = %8.4f eV  S^2 = %1.5f ",
                i + 1, abs_energy, exc_energy, spins[i].first.second);
        }
        outfile->Printf("\n");

        // Step 2. Find determinants in the Q space
        find_q_space(num_ref_roots, P_evals, P_evecs);

        if (spin_complete_) {
            check_spin_completeness(PQ_space_);
            outfile->Printf("\n  Spin-complete dimension of the PQ space: %zu", PQ_space_.size());
        }

        // Step 3. Diagonalize the Hamiltonian in the P + Q space
        if (options_.get_str("DIAG_ALGORITHM") == "DAVIDSONLIST") {
            sparse_solver.diagonalize_hamiltonian(PQ_space_, PQ_evals, PQ_evecs, num_ref_roots,
                                                  DavidsonLiuList);
        } else {
            sparse_solver.diagonalize_hamiltonian(PQ_space_, PQ_evals, PQ_evecs, num_ref_roots,
                                                  DavidsonLiuSparse);
        }

        auto pq_spins = compute_spin(PQ_space_, PQ_evecs, num_ref_roots);

        if (spin_projection == 1 or spin_projection == 3) {
            double spin_contam = 0.0;
            for (size_t n = 0; n < num_ref_roots; ++n) {
                spin_contam += pq_spins[n].first.second;
            }
            spin_contam /= static_cast<double>(num_ref_roots);

            if (spin_contam >= spin_tol_) {
                outfile->Printf("\n  Average spin contamination (%1.5f) is above tolerance (%1.5f)",
                                spin_contam, spin_tol_);
                spin_transform(PQ_space_, PQ_evecs, num_ref_roots);
                PQ_evecs->zero();
                PQ_evecs = PQ_spin_evecs_->clone();
                sparse_solver.compute_H_expectation_val(PQ_space_, PQ_evals, PQ_evecs,
                                                        num_ref_roots, DavidsonLiuList);
            } else {
                outfile->Printf(
                    "\n  Average spin contamination (%1.6f) is less than tolerance (%1.6f)",
                    spin_contam, spin_tol_);
                outfile->Printf("\n  No need to perform spin projection.");
            }
        } else {
            outfile->Printf("\n  Not performing spin projection");
        }
        spins.clear();
        spins = compute_spin(PQ_space_, PQ_evecs, num_ref_roots);

        // Print the energy
        outfile->Printf("\n");
        for (int i = 0; i < num_ref_roots; ++i) {
            double abs_energy = PQ_evals->get(i) + nuclear_repulsion_energy_;
            double exc_energy = pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
            outfile->Printf(
                "\n    PQ-space CI Energy Root %3d        = %.12f Eh = %8.4f eV  S^2 = %3.5f",
                i + 1, abs_energy, exc_energy, spins[i].first.second);
            outfile->Printf("\n    PQ-space CI Energy + EPT2 Root %3d = %.12f Eh = %8.4f eV", i + 1,
                            abs_energy + multistate_pt2_energy_correction_[i],
                            exc_energy +
                                pc_hartree2ev * (multistate_pt2_energy_correction_[i] -
                                                 multistate_pt2_energy_correction_[0]));
        }
        outfile->Printf("\n");

        // get final dimension of P space
        int PQ_space_final = PQ_space_.size();
        outfile->Printf("\n PQ space dimension difference (current - previous) : %d \n",
                        PQ_space_final - PQ_space_init);

        // Step 4. Check convergence and break if needed
        if (check_convergence(energy_history, PQ_evals)) {
            break;
        }

        // Step 5. Check if the procedure is stuck
        bool stuck = check_stuck(energy_history, PQ_evals);
        if (stuck) {
            outfile->Printf("\n  The procedure is stuck! Printing final energy (but be careful).");
            break;
        }
        print_wfn(PQ_space_, PQ_evecs, num_ref_roots);

        // Step 6. Prune the P + Q space to get an updated P space
        prune_q_space(PQ_space_, P_space_, P_space_map_, PQ_evecs, num_ref_roots);

        // Print information about the wave function
        // print_wfn(PQ_space_,PQ_evecs,num_ref_roots);

    } // end cycle

    if (spin_projection == 2 or spin_projection == 3) {
        double spin_contam = 0.0;
        for (int n = 0; n < nroot_; ++n) {
            spin_contam += root_spin_vec_[n].first;
        }
        spin_contam /= static_cast<double>(nroot_);

        if (spin_contam >= spin_tol_) {
            outfile->Printf("\n  Average spin contamination per root is %1.5f", spin_contam);
            spin_transform(P_space_, P_evecs, nroot_);
            P_evecs->zero();
            P_evecs = PQ_spin_evecs_->clone();
            sparse_solver.compute_H_expectation_val(P_space_, P_evals, P_evecs, nroot_,
                                                    DavidsonLiuList);
        } else {
            outfile->Printf("\n  Average spin contamination (%1.6f) is less than tolerance",
                            spin_contam);
            outfile->Printf("\n  No need to perform spin projection.");
        }
    }

    // Do Hamiltonian smoothing
    if (do_smooth_) {
        smooth_hamiltonian(P_space_, P_evals, P_evecs, nroot_);
    }

    // Re-diagonalize H, solving for more roots
    if (post_diagonalize_) {
        root = nroot_;
        sparse_solver.diagonalize_hamiltonian(PQ_space_, PQ_evals, PQ_evecs, post_root_,
                                              DavidsonLiuSparse);
        outfile->Printf(" \n  Re-diagonalizing the Hamiltonian with %zu roots.\n", post_root_);
        outfile->Printf(" \n  WARNING: EPT2 is meaningless for roots %zu and higher. I'm not even "
                        "printing them.",
                        root + 1);
        nroot_ = post_root_;
    }

    outfile->Printf("\n\n  ==> Post-Iterations <==\n");

    outfile->Printf("\n  Printing Wavefunction Information:");
    print_wfn(PQ_space_, PQ_evecs, nroot_);
    outfile->Printf("\n\n     Order       # of Dets        Total |c^2|  ");
    outfile->Printf("\n   ---------   -------------   -----------------  ");

    wfn_analyzer(PQ_space_, PQ_evecs, nroot_);
    for (int i = 0; i < nroot_; ++i) {
        double abs_energy = PQ_evals->get(i) + nuclear_repulsion_energy_;
        double exc_energy = pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
        outfile->Printf("\n  * Adaptive-CI Energy Root %3d        = %.12f Eh = %8.4f eV", i + 1,
                        abs_energy, exc_energy);
        if (post_diagonalize_ == false) {
            outfile->Printf("\n  * Adaptive-CI Energy Root %3d + EPT2 = %.12f Eh = %8.4f eV", i + 1,
                            abs_energy + multistate_pt2_energy_correction_[i],
                            exc_energy +
                                pc_hartree2ev * (multistate_pt2_energy_correction_[i] -
                                                 multistate_pt2_energy_correction_[0]));
        } else if (post_diagonalize_ and i < root) {
            outfile->Printf("\n  * Adaptive-CI Energy Root %3d + EPT2 = %.12f Eh = %8.4f eV", i + 1,
                            abs_energy + multistate_pt2_energy_correction_[i],
                            exc_energy +
                                pc_hartree2ev * (multistate_pt2_energy_correction_[i] -
                                                 multistate_pt2_energy_correction_[0]));
        }
    }

    if (form_1_RDM_) {
        // Print 1D RDM
        SharedMatrix Dalpha(new Matrix("Dalpha", nmo_, nmo_));
        SharedMatrix Dbeta(new Matrix("Dbeta", nmo_, nmo_));

        compute_1rdm(Dalpha, Dbeta, PQ_space_, PQ_evecs, nroot_);
        diagonalize_order nMatz = evals_only_descending;
        D1_->print();

        SharedVector no_occnum(new Vector("Natural Orbital occupation Numbers", nmo_));
        SharedMatrix NOevecs(new Matrix("NO Evecs", nmo_, nmo_));
        D1_->diagonalize(NOevecs, no_occnum, nMatz);
        no_occnum->print();

        std::vector<int> active_space(nirrep_);
        for (size_t p = 0, maxp = D1_->nrow(); p < maxp; ++p) {
            if ((D1_->get(p, p) >= 0.02) and (D1_->get(p, p) <= 1.98)) {
                // if( (no_occnum->get(p)  >= 0.02) and (no_occnum->get(p) <= 1.98 ) ){
                active_space[mo_symmetry_[p]] += 1;
            } else {
                continue;
            }
        }

        outfile->Printf("\n  Suggested active space from ACI:  \n  [");
        for (int i = 0; i < nirrep_; ++i)
            outfile->Printf(" %d", active_space[i]);
        outfile->Printf(" ]");
    }

    outfile->Printf("\n\n  %s: %f s", "Adaptive-CI (bitset) ran in ", t_iamrcisd.elapsed());
    outfile->Printf("\n\n  %s: %d", "Saving information for root", options_.get_int("ROOT") + 1);

    double root_energy = PQ_evals->get(options_.get_int("ROOT")) + nuclear_repulsion_energy_;
    double root_energy_pt2 =
        root_energy + multistate_pt2_energy_correction_[options_.get_int("ROOT")];
    Process::environment.globals["CURRENT ENERGY"] = root_energy;
    Process::environment.globals["EX-ACI ENERGY"] = root_energy;
    Process::environment.globals["EX-ACI+PT2 ENERGY"] = root_energy_pt2;

    return PQ_evals->get(options_.get_int("ROOT")) + nuclear_repulsion_energy_;
}

void EX_ACI::find_q_space(int nroot, SharedVector evals, SharedMatrix evecs) {
    // Find the SD space out of the reference
    std::vector<DynamicBitsetDeterminant> sd_dets_vec;
    std::map<DynamicBitsetDeterminant, int> new_dets_map;

    ForteTimer t_ms_build;

    // This hash saves the determinant coupling to the model space eigenfunction
    std::map<DynamicBitsetDeterminant, std::vector<double>> V_hash;

    for (size_t I = 0, max_I = P_space_.size(); I < max_I; ++I) {
        auto& det = P_space_[I];
        generate_excited_determinants(nroot, I, evecs, det, V_hash);
    }
    outfile->Printf("\n  %s: %zu determinants", "Dimension of the SD space", V_hash.size());
    outfile->Printf("\n  %s: %f s\n", "Time spent building the model space", t_ms_build.elapsed());

    // This will contain all the determinants
    PQ_space_.clear();

    // Add the P-space determinants and zero the hash
    for (size_t J = 0, max_J = P_space_.size(); J < max_J; ++J) {
        PQ_space_.push_back(P_space_[J]);
        V_hash.erase(P_space_[J]);
    }

    ForteTimer t_ms_screen;

    using bsmap_it = std::map<DynamicBitsetDeterminant, std::vector<double>>::const_iterator;
    pVector<double, double> C1(nroot, std::make_pair(0.0, 0.0));
    pVector<double, double> E2(nroot, std::make_pair(0.0, 0.0));
    std::vector<double> V(nroot, 0.0);
    DynamicBitsetDeterminant det;
    pVector<double, DynamicBitsetDeterminant> sorted_dets;
    std::vector<double> ept2(nroot, 0.0);
    double criteria;
    print_warning_ = false;

    // Check the coupling between the reference and the SD space
    for (const auto& it : V_hash) {
        double EI = it.first.energy();
        // Loop over roots
        // The tau_q parameter type is chosen here ( keyword bool "perturb_select" )
        for (int n = 0; n < nroot; ++n) {
            det = it.first;
            V[n] = it.second[n];

            double C1_I =
                perturb_select_
                    ? -V[n] / (EI - evals->get(n))
                    : (((EI - evals->get(n)) / 2.0) -
                       sqrt(std::pow(((EI - evals->get(n)) / 2.0), 2.0) + std::pow(V[n], 2.0))) /
                          V[n];
            double E2_I =
                perturb_select_
                    ? -V[n] * V[n] / (EI - evals->get(n))
                    : ((EI - evals->get(n)) / 2.0) -
                          sqrt(std::pow(((EI - evals->get(n)) / 2.0), 2.0) + std::pow(V[n], 2.0));

            C1[n] = std::make_pair(std::fabs(C1_I), C1_I);
            E2[n] = std::make_pair(std::fabs(E2_I), E2_I);
        }
        // make q space in a number of ways with C1 and E1 as input, produces PQ_space
        if (ex_alg_ == "STATE_AVERAGE" and nroot_ != 1) {
            criteria = average_q_values(nroot, C1, E2);
        } else {
            criteria = root_select(nroot, C1, E2);
        }

        if (aimed_selection_) {
            sorted_dets.push_back(std::make_pair(criteria, it.first));
        } else {
            if (std::fabs(criteria) > tau_q_) {
                PQ_space_.push_back(it.first);
            } else {
                for (int n = 0; n < nroot; ++n) {
                    ept2[n] += E2[n].second;
                }
            }
        }
    } // end loop over determinants

    if (ex_alg_ == "STATE_AVERAGE" and print_warning_) {
        outfile->Printf("\n  WARNING: There are not enough roots with the correct S^2 to compute "
                        "dE2! You should increase nroot.");
        outfile->Printf("\n  Setting q_rel = false for this iteration.\n");
    }

    if (aimed_selection_) {
        std::sort(sorted_dets.begin(), sorted_dets.end());
        double sum = 0.0;
        double E2_I = 0.0;
        for (size_t I = 0, max_I = sorted_dets.size(); I < max_I; ++I) {
            const DynamicBitsetDeterminant& det = sorted_dets[I].second;
            if (sum + sorted_dets[I].first < tau_q_) {
                sum += sorted_dets[I].first;
                double EI = det.energy();
                const auto& V_vec = V_hash[det];
                for (int n = 0; n < nroot; ++n) {
                    double V = V_vec[n];
                    E2_I = perturb_select_ ? -V * V / (EI - evals->get(n))
                                           : ((EI - evals->get(n)) / 2.0) -
                                                 sqrt(std::pow(((EI - evals->get(n)) / 2.0), 2.0) +
                                                      std::pow(V, 2.0));
                    ept2[n] += E2_I;
                }
            } else {
                PQ_space_.push_back(sorted_dets[I].second);
            }
        }
    }

    multistate_pt2_energy_correction_ = ept2;

    outfile->Printf("\n  %s: %zu determinants", "Dimension of the P + Q space", PQ_space_.size());
    outfile->Printf("\n  %s: %f s", "Time spent screening the model space", t_ms_screen.elapsed());
}

double EX_ACI::average_q_values(int nroot, pVector<double, double> C1, pVector<double, double> E2) {
    // f_E2 and f_C1 will store the selected function of the chosen q-criteria
    std::pair<double, double> f_C1;
    std::pair<double, double> f_E2;

    // Make vector of pairs for ∆e_n,0
    pVector<double, double> dE2(nroot, std::make_pair(0.0, 0.0));

    // Compute a determinant's effect on ground state or adjacent state transition
    q_rel_ = options_.get_bool("Q_REL");

    if (q_rel_ == true and nroot > 1) {
        if (q_reference_ == "GS") {
            for (int n = 0; n < nroot; ++n) {
                dE2[n] =
                    std::make_pair(std::fabs(E2[n].first - E2[0].first), E2[n].second - E2[0].second);
            }
        }
        if (q_reference_ == "ADJACENT") {
            for (int n = 1; n < nroot; ++n) {
                dE2[n] = std::make_pair(std::fabs(E2[n].first - E2[n - 1].first),
                                   E2[n].second - E2[n - 1].second);
            }
        }
    } else if (q_rel_ == true and nroot == 1) {
        q_rel_ = false;
    }

    // Choose the function of couplings for each root.
    // If nroot = 1, choose the max
    if (pq_function_ == "MAX" or nroot == 1) {
        f_C1 = *std::max_element(C1.begin(), C1.end());
        f_E2 = q_rel_ and (nroot != 1) ? *std::max_element(dE2.begin(), dE2.end())
                                       : *std::max_element(E2.begin(), E2.end());
    } else if (pq_function_ == "AVERAGE") {
        double C1_average = 0.0;
        double E2_average = 0.0;
        double dE2_average = 0.0;
        double dim_inv = 1.0 / nroot;
        for (int n = 0; n < nroot; ++n) {
            C1_average += C1[n].first * dim_inv;
            E2_average += E2[n].first * dim_inv;
        }
        if (q_rel_) {
            double inv_d = 1.0 / (nroot - 1.0);
            for (int n = 1; n < nroot; ++n) {
                dE2_average += dE2[n].first * inv_d;
            }
        }
        f_C1 = std::make_pair(C1_average, 0);
        f_E2 = q_rel_ ? std::make_pair(dE2_average, 0) : std::make_pair(E2_average, 0);
    } else {
        throw PSIEXCEPTION(options_.get_str("PQ_FUNCTION") + " is not a valid option");
    }

    double select_value = 0.0;
    if (aimed_selection_) {
        select_value = energy_selection_ ? f_E2.first : std::pow(f_C1.first, 2.0);
    } else {
        select_value = energy_selection_ ? f_E2.first : f_C1.first;
    }
    return select_value;
}

double EX_ACI::root_select(int nroot, pVector<double, double> C1, pVector<double, double> E2) {
    double select_value;
    ref_root_ = options_.get_int("ROOT");

    if (ref_root_ + 1 > nroot_) {
        throw PSIEXCEPTION(
            "Your selection is not a valid reference option. Check ROOT in options.");
    }

    if (nroot == 1) {
        ref_root_ = 0;
    }

    if (aimed_selection_) {
        select_value = energy_selection_ ? E2[ref_root_].first : std::pow(C1[ref_root_].first, 2.0);
    } else {
        select_value = energy_selection_ ? E2[ref_root_].first : C1[ref_root_].first;
    }

    return select_value;
}

void EX_ACI::find_q_space_single_root(int nroot, SharedVector evals, SharedMatrix evecs) {
    // Find the SD space out of the reference
    std::vector<DynamicBitsetDeterminant> sd_dets_vec;
    std::map<DynamicBitsetDeterminant, int> new_dets_map;

    ForteTimer t_ms_build;

    // This hash saves the determinant coupling to the model space eigenfunction
    std::map<DynamicBitsetDeterminant, double> V_map;

    for (size_t I = 0, max_I = P_space_.size(); I < max_I; ++I) {
        DynamicBitsetDeterminant& det = P_space_[I];
        generate_excited_determinants_single_root(nroot, I, evecs, det, V_map);
    }
    outfile->Printf("\n  %s: %zu determinants", "Dimension of the SD space", V_map.size());
    outfile->Printf("\n  %s: %f s\n", "Time spent building the model space", t_ms_build.elapsed());

    // This will contain all the determinants
    PQ_space_.clear();

    // Add the P-space determinants to PQ and remove them from the hash
    for (size_t J = 0, max_J = P_space_.size(); J < max_J; ++J) {
        PQ_space_.push_back(P_space_[J]);
        V_map.erase(P_space_[J]);
    }

    ForteTimer t_ms_screen;

    using bsmap_it = std::map<DynamicBitsetDeterminant, std::vector<double>>::const_iterator;
    std::vector<std::pair<double, double>> C1(nroot_, std::make_pair(0.0, 0.0));
    std::vector<std::pair<double, double>> E2(nroot_, std::make_pair(0.0, 0.0));
    std::vector<double> ept2(nroot_, 0.0);

    std::vector<std::pair<double, DynamicBitsetDeterminant>> sorted_dets;

    // Check the coupling between the reference and the SD space
    for (const auto& det_V : V_map) {
        double EI = det_V.first.energy();
        for (int n = 0; n < nroot; ++n) {
            double V = det_V.second;
            double C1_I = -V / (EI - evals->get(n));
            double E2_I = -V * V / (EI - evals->get(n));

            C1[n] = std::make_pair(std::fabs(C1_I), C1_I);
            E2[n] = std::make_pair(std::fabs(E2_I), E2_I);
        }

        std::pair<double, double> max_C1 = *std::max_element(C1.begin(), C1.end());
        std::pair<double, double> max_E2 = *std::max_element(E2.begin(), E2.end());

        if (aimed_selection_) {
            double aimed_value = energy_selection_ ? max_E2.first : std::pow(max_C1.first, 2.0);
            sorted_dets.push_back(std::make_pair(aimed_value, det_V.first));
        } else {
            double select_value = energy_selection_ ? max_E2.first : max_C1.first;
            if (std::fabs(select_value) > tau_q_) {
                PQ_space_.push_back(det_V.first);
            } else {
                for (int n = 0; n < nroot; ++n) {
                    ept2[n] += E2[n].second;
                }
            }
        }
    }

    if (aimed_selection_) {
        // Sort the CI coefficients in ascending order
        std::sort(sorted_dets.begin(), sorted_dets.end());

        double sum = 0.0;
        for (size_t I = 0, max_I = sorted_dets.size(); I < max_I; ++I) {
            const DynamicBitsetDeterminant& det = sorted_dets[I].second;
            if (sum + sorted_dets[I].first < tau_q_) {
                sum += sorted_dets[I].first;
                double EI = det.energy();
                const double V = V_map[det];
                for (int n = 0; n < nroot; ++n) {
                    double E2_I = -V * V / (EI - evals->get(n));
                    ept2[n] += E2_I;
                }
            } else {
                PQ_space_.push_back(sorted_dets[I].second);
            }
        }
    }

    multistate_pt2_energy_correction_ = ept2;

    outfile->Printf("\n  %s: %zu determinants", "Dimension of the P + Q space", PQ_space_.size());
    outfile->Printf("\n  %s: %f s", "Time spent screening the model space", t_ms_screen.elapsed());
}

void EX_ACI::generate_excited_determinants_single_root(
    int nroot, int I, SharedMatrix evecs, DynamicBitsetDeterminant& det,
    std::map<DynamicBitsetDeterminant, double>& V_hash) {
    std::vector<int> aocc = det.get_alfa_occ();
    std::vector<int> bocc = det.get_beta_occ();
    std::vector<int> avir = det.get_alfa_vir();
    std::vector<int> bvir = det.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta = bocc.size();
    int nvalpha = avir.size();
    int nvbeta = bvir.size();

    int n = 0;
    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a) {
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                DynamicBitsetDeterminant new_det(det);
                new_det.set_alfa_bit(ii, false);
                new_det.set_alfa_bit(aa, true);
                double HIJ = det.slater_rules(new_det);
                V_hash[new_det] += HIJ * evecs->get(I, n);
            }
        }
    }

    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a) {
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                DynamicBitsetDeterminant new_det(det);
                new_det.set_beta_bit(ii, false);
                new_det.set_beta_bit(aa, true);
                double HIJ = det.slater_rules(new_det);
                V_hash[new_det] += HIJ * evecs->get(I, n);
            }
        }
    }

    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = i + 1; j < noalpha; ++j) {
            int jj = aocc[j];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                for (int b = a + 1; b < nvalpha; ++b) {
                    int bb = avir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == 0) {
                        DynamicBitsetDeterminant new_det(det);
                        new_det.set_alfa_bit(ii, false);
                        new_det.set_alfa_bit(jj, false);
                        new_det.set_alfa_bit(aa, true);
                        new_det.set_alfa_bit(bb, true);

                        double HIJ = ints_->aptei_aa(ii, jj, aa, bb);

                        // grap the alpha bits of both determinants
                        const boost::dynamic_bitset<>& Ia = det.alfa_bits();
                        const boost::dynamic_bitset<>& Ja = new_det.alfa_bits();

                        // compute the sign of the matrix element
                        HIJ *= DynamicBitsetDeterminant::SlaterSign(Ia, ii) *
                               DynamicBitsetDeterminant::SlaterSign(Ia, jj) *
                               DynamicBitsetDeterminant::SlaterSign(Ja, aa) *
                               DynamicBitsetDeterminant::SlaterSign(Ja, bb);

                        V_hash[new_det] += HIJ * evecs->get(I, n);
                    }
                }
            }
        }
    }

    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = 0; j < nobeta; ++j) {
            int jj = bocc[j];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                for (int b = 0; b < nvbeta; ++b) {
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == 0) {
                        DynamicBitsetDeterminant new_det(det);
                        new_det.set_alfa_bit(ii, false);
                        new_det.set_beta_bit(jj, false);
                        new_det.set_alfa_bit(aa, true);
                        new_det.set_beta_bit(bb, true);

                        double HIJ = ints_->aptei_ab(ii, jj, aa, bb);

                        // grap the alpha bits of both determinants
                        const boost::dynamic_bitset<>& Ia = det.alfa_bits();
                        const boost::dynamic_bitset<>& Ib = det.beta_bits();
                        const boost::dynamic_bitset<>& Ja = new_det.alfa_bits();
                        const boost::dynamic_bitset<>& Jb = new_det.beta_bits();

                        // compute the sign of the matrix element
                        HIJ *= DynamicBitsetDeterminant::SlaterSign(Ia, ii) *
                               DynamicBitsetDeterminant::SlaterSign(Ib, jj) *
                               DynamicBitsetDeterminant::SlaterSign(Ja, aa) *
                               DynamicBitsetDeterminant::SlaterSign(Jb, bb);

                        V_hash[new_det] += HIJ * evecs->get(I, n);
                    }
                }
            }
        }
    }
    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int j = i + 1; j < nobeta; ++j) {
            int jj = bocc[j];
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                for (int b = a + 1; b < nvbeta; ++b) {
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^
                         (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == 0) {
                        DynamicBitsetDeterminant new_det(det);
                        new_det.set_beta_bit(ii, false);
                        new_det.set_beta_bit(jj, false);
                        new_det.set_beta_bit(aa, true);
                        new_det.set_beta_bit(bb, true);

                        double HIJ = ints_->aptei_bb(ii, jj, aa, bb);

                        // grap the alpha bits of both determinants
                        const boost::dynamic_bitset<>& Ib = det.beta_bits();
                        const boost::dynamic_bitset<>& Jb = new_det.beta_bits();

                        // compute the sign of the matrix element
                        HIJ *= DynamicBitsetDeterminant::SlaterSign(Ib, ii) *
                               DynamicBitsetDeterminant::SlaterSign(Ib, jj) *
                               DynamicBitsetDeterminant::SlaterSign(Jb, aa) *
                               DynamicBitsetDeterminant::SlaterSign(Jb, bb);

                        V_hash[new_det] += HIJ * evecs->get(I, n);
                    }
                }
            }
        }
    }
}

void EX_ACI::generate_excited_determinants(
    int nroot, int I, SharedMatrix evecs, DynamicBitsetDeterminant& det,
    std::map<DynamicBitsetDeterminant, std::vector<double>>& V_hash) {
    std::vector<int> aocc = det.get_alfa_occ();
    std::vector<int> bocc = det.get_beta_occ();
    std::vector<int> avir = det.get_alfa_vir();
    std::vector<int> bvir = det.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta = bocc.size();
    int nvalpha = avir.size();
    int nvbeta = bvir.size();

    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a) {
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                DynamicBitsetDeterminant new_det(det);
                new_det.set_alfa_bit(ii, false);
                new_det.set_alfa_bit(aa, true);
                if (P_space_map_.find(new_det) == P_space_map_.end()) {
                    double HIJ = det.slater_rules(new_det);
                    if (V_hash.count(new_det) == 0) {
                        V_hash[new_det] = std::vector<double>(nroot);
                    }
                    for (int n = 0; n < nroot; ++n) {
                        V_hash[new_det][n] += HIJ * evecs->get(I, n);
                    }
                }
            }
        }
    }

    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a) {
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                DynamicBitsetDeterminant new_det(det);
                new_det.set_beta_bit(ii, false);
                new_det.set_beta_bit(aa, true);
                if (P_space_map_.find(new_det) == P_space_map_.end()) {
                    double HIJ = det.slater_rules(new_det);
                    if (V_hash.count(new_det) == 0) {
                        V_hash[new_det] = std::vector<double>(nroot);
                    }
                    for (int n = 0; n < nroot; ++n) {
                        V_hash[new_det][n] += HIJ * evecs->get(I, n);
                    }
                }
            }
        }
    }

    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = i + 1; j < noalpha; ++j) {
            int jj = aocc[j];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                for (int b = a + 1; b < nvalpha; ++b) {
                    int bb = avir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == 0) {
                        DynamicBitsetDeterminant new_det(det);
                        new_det.set_alfa_bit(ii, false);
                        new_det.set_alfa_bit(jj, false);
                        new_det.set_alfa_bit(aa, true);
                        new_det.set_alfa_bit(bb, true);
                        if (P_space_map_.find(new_det) == P_space_map_.end()) {
                            double HIJ = det.slater_rules(new_det);
                            if (V_hash.count(new_det) == 0) {
                                V_hash[new_det] = std::vector<double>(nroot);
                            }
                            for (int n = 0; n < nroot; ++n) {
                                V_hash[new_det][n] += HIJ * evecs->get(I, n);
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = 0; j < nobeta; ++j) {
            int jj = bocc[j];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                for (int b = 0; b < nvbeta; ++b) {
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == 0) {
                        DynamicBitsetDeterminant new_det(det);
                        new_det.set_alfa_bit(ii, false);
                        new_det.set_beta_bit(jj, false);
                        new_det.set_alfa_bit(aa, true);
                        new_det.set_beta_bit(bb, true);
                        if (P_space_map_.find(new_det) == P_space_map_.end()) {
                            double HIJ = det.slater_rules(new_det);
                            if (V_hash.count(new_det) == 0) {
                                V_hash[new_det] = std::vector<double>(nroot);
                            }
                            for (int n = 0; n < nroot; ++n) {
                                V_hash[new_det][n] += HIJ * evecs->get(I, n);
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int j = i + 1; j < nobeta; ++j) {
            int jj = bocc[j];
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                for (int b = a + 1; b < nvbeta; ++b) {
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^
                         (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == 0) {
                        DynamicBitsetDeterminant new_det(det);
                        new_det.set_beta_bit(ii, false);
                        new_det.set_beta_bit(jj, false);
                        new_det.set_beta_bit(aa, true);
                        new_det.set_beta_bit(bb, true);
                        if (P_space_map_.find(new_det) == P_space_map_.end()) {
                            double HIJ = det.slater_rules(new_det);
                            if (V_hash.count(new_det) == 0) {
                                V_hash[new_det] = std::vector<double>(nroot);
                            }
                            for (int n = 0; n < nroot; ++n) {
                                V_hash[new_det][n] += HIJ * evecs->get(I, n);
                            }
                        }
                    }
                }
            }
        }
    }
}

void EX_ACI::generate_pair_excited_determinants(
    int nroot, int I, SharedMatrix evecs, DynamicBitsetDeterminant& det,
    std::map<DynamicBitsetDeterminant, std::vector<double>>& V_hash) {
    std::vector<int> aocc = det.get_alfa_occ();
    std::vector<int> bocc = det.get_beta_occ();
    std::vector<int> avir = det.get_alfa_vir();
    std::vector<int> bvir = det.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta = bocc.size();
    int nvalpha = avir.size();
    int nvbeta = bvir.size();

    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a) {
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                DynamicBitsetDeterminant new_det(det);
                new_det.set_alfa_bit(ii, false);
                new_det.set_alfa_bit(aa, true);
                if (P_space_map_.find(new_det) == P_space_map_.end()) {
                    double HIJ = det.slater_rules(new_det);
                    if (V_hash.count(new_det) == 0) {
                        V_hash[new_det] = std::vector<double>(nroot);
                    }
                    for (int n = 0; n < nroot; ++n) {
                        V_hash[new_det][n] += HIJ * evecs->get(I, n);
                    }
                }
            }
        }
    }

    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a) {
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                DynamicBitsetDeterminant new_det(det);
                new_det.set_beta_bit(ii, false);
                new_det.set_beta_bit(aa, true);
                if (P_space_map_.find(new_det) == P_space_map_.end()) {
                    double HIJ = det.slater_rules(new_det);
                    if (V_hash.count(new_det) == 0) {
                        V_hash[new_det] = std::vector<double>(nroot);
                    }
                    for (int n = 0; n < nroot; ++n) {
                        V_hash[new_det][n] += HIJ * evecs->get(I, n);
                    }
                }
            }
        }
    }

    // Generate a/b pair excitations

    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        if (det.get_beta_bit(ii)) {
            int jj = ii;
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                if (not det.get_beta_bit(aa)) {
                    int bb = aa;
                    DynamicBitsetDeterminant new_det(det);
                    new_det.set_alfa_bit(ii, false);
                    new_det.set_beta_bit(jj, false);
                    new_det.set_alfa_bit(aa, true);
                    new_det.set_beta_bit(bb, true);
                    if (P_space_map_.find(new_det) == P_space_map_.end()) {
                        double HIJ = det.slater_rules(new_det);
                        if (V_hash.count(new_det) == 0) {
                            V_hash[new_det] = std::vector<double>(nroot);
                        }
                        for (int n = 0; n < nroot; ++n) {
                            V_hash[new_det][n] += HIJ * evecs->get(I, n);
                        }
                    }
                }
            }
        }
    }
}

bool EX_ACI::check_convergence(std::vector<std::vector<double>>& energy_history,
                               SharedVector evals) {
    int nroot = evals->dim();

    if (energy_history.size() == 0) {
        std::vector<double> new_energies;
        for (int n = 0; n < nroot; ++n) {
            double state_n_energy = evals->get(n) + nuclear_repulsion_energy_;
            new_energies.push_back(state_n_energy);
        }
        energy_history.push_back(new_energies);
        return false;
    }

    double old_avg_energy = 0.0;
    double new_avg_energy = 0.0;

    std::vector<double> new_energies;
    std::vector<double> old_energies = energy_history[energy_history.size() - 1];

    // Only average over roots with correct S^2
    for (int n = 0; n < nroot; ++n) {
        double state_n_energy = evals->get(n) + nuclear_repulsion_energy_;
        new_energies.push_back(state_n_energy);
        new_avg_energy += state_n_energy;
        old_avg_energy += old_energies[n];
    }

    old_avg_energy /= static_cast<double>(nroot);
    new_avg_energy /= static_cast<double>(nroot);

    energy_history.push_back(new_energies);

    // Check for convergence
    return (std::fabs(new_avg_energy - old_avg_energy) < options_.get_double("E_CONVERGENCE"));
}

bool EX_ACI::check_stuck(std::vector<std::vector<double>>& energy_history, SharedVector evals) {
    int nroot = evals->dim();
    if (cycle_ < 3) {
        return false;
    } else {
        std::vector<double> av_energies;
        av_energies.clear();

        for (int i = 0; i < cycle_; ++i) {
            double energy = 0.0;
            for (int n = 0; n < nroot; ++n) {
                energy += energy_history[i][n] / static_cast<double>(nroot);
            }
            av_energies.push_back(energy);
        }

        if (std::fabs(av_energies[cycle_ - 1] - av_energies[cycle_ - 3]) <
                options_.get_double("E_CONVERGENCE") and
            std::fabs(av_energies[cycle_ - 2] - av_energies[cycle_ - 4]) <
                options_.get_double("E_CONVERGENCE")) {
            return true;
        } else {
            return false;
        }
    }
}

void EX_ACI::prune_q_space(std::vector<DynamicBitsetDeterminant>& large_space,
                           std::vector<DynamicBitsetDeterminant>& pruned_space,
                           std::map<DynamicBitsetDeterminant, int>& pruned_space_map,
                           SharedMatrix evecs, int nroot) {
    // Select the new reference space using the sorted CI coefficients
    pruned_space.clear();
    pruned_space_map.clear();

    // Create a vector that stores the absolute value of the CI coefficients
    // Use a function of the CI coefficients for each root as the criteria
    // This function will be the same one used when selecting the PQ space
    pVector<double, size_t> dm_det_list;
    for (size_t I = 0, max = large_space.size(); I < max; ++I) {
        double criteria = 0.0;
        for (int n = 0; n < nroot; ++n) {
            if (pq_function_ == "MAX") {
                criteria = std::max(criteria, std::fabs(evecs->get(I, n)));
            } else if (pq_function_ == "AVERAGE") {
                criteria += std::fabs(evecs->get(I, n));
            }
        }
        criteria /= static_cast<double>(nroot);
        dm_det_list.push_back(std::make_pair(criteria, I));
    }

    // Decide which determinants will go in pruned_space
    // Include all determinants such that
    // sum_I |C_I|^2 < tau_p, where the sum runs over all the excluded determinants
    if (aimed_selection_) {
        // Sort the CI coefficients in ascending order
        outfile->Printf("  AIMED SELECTION \n");
        std::sort(dm_det_list.begin(), dm_det_list.end());

        double sum = 0.0;
        for (size_t I = 0; I < large_space.size(); ++I) {
            double dsum = std::pow(dm_det_list[I].first, 2.0);
            if (sum + dsum < tau_p_) {
                sum += dsum;
            } else {
                pruned_space.push_back(large_space[dm_det_list[I].second]);
                pruned_space_map[large_space[dm_det_list[I].second]] = 1;
            }
        }
    }
    // Include all determinants such that |C_I| > tau_p
    else {

        std::sort(dm_det_list.begin(), dm_det_list.end());
        for (size_t I = 0; I < large_space.size(); ++I) {
            if (dm_det_list[I].first > tau_p_) {
                pruned_space.push_back(large_space[dm_det_list[I].second]);
                pruned_space_map[large_space[dm_det_list[I].second]] = 1;
            }
        }
    }
}

void EX_ACI::smooth_hamiltonian(std::vector<DynamicBitsetDeterminant>& space, SharedVector evals,
                                SharedMatrix evecs, int nroot) {
    size_t ndets = space.size();

    SharedMatrix H(new Matrix("H-smooth", ndets, ndets));

    SharedMatrix F(new Matrix("F-smooth", ndets, ndets));

    // Build the smoothed Hamiltonian
    for (int I = 0; I < ndets; ++I) {
        for (int J = 0; J < ndets; ++J) {
            double CI = evecs->get(I, 0);
            double CJ = evecs->get(J, 0);
            double HIJ = space[I].slater_rules(space[J]);
            double factorI = smootherstep(tau_p_ * tau_p_, smooth_threshold_, CI * CI);
            double factorJ = smootherstep(tau_p_ * tau_p_, smooth_threshold_, CJ * CJ);
            if (I != J) {
                HIJ *= factorI * factorJ;
                F->set(I, J, factorI * factorJ);
            }
            H->set(I, J, HIJ);
        }
    }

    evecs->print();
    H->print();
    F->print();

    SharedMatrix evecs_s(new Matrix("C-smooth", ndets, ndets));
    SharedVector evals_s(new Vector("lambda-smooth", ndets));

    H->diagonalize(evecs_s, evals_s);

    outfile->Printf("\n  * sAdaptive-CI Energy Root %3d        = %.12f Eh", 1,
                    evals_s->get(0) + nuclear_repulsion_energy_);
}

pVector<std::pair<double, double>, std::pair<size_t, double>>
EX_ACI::compute_spin(std::vector<DynamicBitsetDeterminant> space, SharedMatrix evecs, int nroot) {
    double norm;
    double S2;
    double S;
    pVector<std::pair<double, double>, std::pair<size_t, double>> spin_vec;
    // pVector<double,size_t> det_weight;

    for (int n = 0; n < nroot; ++n) {
        // Compute the expectation value of the spin
        size_t max_sample = 1000;
        size_t max_I = 0;
        double sum_weight = 0.0;
        pVector<double, size_t> det_weight;

        for (size_t I = 0, maxI = space.size(); I < maxI; ++I) {
            det_weight.push_back(std::make_pair(evecs->get(I, n), I));
        }

        // Don't require the determinants to be pre-sorted
        std::sort(det_weight.begin(), det_weight.end());
        std::reverse(det_weight.begin(), det_weight.end());

        const double wfn_threshold = (space.size() < 10) ? 1.00 : 0.95;
        for (size_t I = 0, max = space.size(); I < max; ++I) {
            if ((sum_weight < wfn_threshold) and (I < max_sample)) {
                sum_weight += std::pow(det_weight[I].first, 2.0);
                max_I++;
            } else if (std::fabs(det_weight[I].first - det_weight[I - 1].first) < 1.0e-6) {
                // Special case, if there are several equivalent determinants
                sum_weight += std::pow(det_weight[I].first, 2.0);
                max_I++;
            } else {
                break;
            }
        }

        S2 = 0.0;
        norm = 0.0;
        for (int sI = 0; sI < max_I; ++sI) {
            size_t I = det_weight[sI].second;
            for (int sJ = 0; sJ < max_I; ++sJ) {
                size_t J = det_weight[sJ].second;
                if (std::fabs(evecs->get(I, n) * evecs->get(J, n)) > 1.0e-12) {
                    const double S2IJ = space[I].spin2(space[J]);
                    S2 += evecs->get(I, n) * evecs->get(J, n) * S2IJ;
                }
            }
            norm += std::pow(evecs->get(I, n), 2.0);
        }

        S2 /= norm;
        S2 = std::fabs(S2);
        S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
        spin_vec.push_back(std::make_pair(make_pair(S, S2), std::make_pair(max_I, sum_weight)));
    }
    return spin_vec;
}

void EX_ACI::print_wfn(std::vector<DynamicBitsetDeterminant> space, SharedMatrix evecs, int nroot) {
    pVector<double, size_t> det_weight;
    pVector<std::pair<double, double>, std::pair<size_t, double>> spins;
    double sum_weight;
    double S2;
    double S;
    size_t max_I;

    std::vector<std::string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet", "sextet",
                                   "septet", "octet", "nonet", "decaet"});
    string state_label;

    for (int n = 0; n < nroot; ++n) {
        det_weight.clear();
        outfile->Printf("\n\n  Most important contributions to root %3d:", n);

        for (size_t I = 0, max = space.size(); I < max; ++I) {
            det_weight.push_back(std::make_pair(std::fabs(evecs->get(I, n)), I));
        }
        std::sort(det_weight.begin(), det_weight.end());
        std::reverse(det_weight.begin(), det_weight.end());
        size_t max_dets = std::min(10, evecs->nrow());
        for (size_t I = 0; I < max_dets; ++I) {
            outfile->Printf("\n  %3zu  %9.6f %.9f  %10zu %s", I,
                            evecs->get(det_weight[I].second, n),
                            det_weight[I].first * det_weight[I].first, det_weight[I].second,
                            space[det_weight[I].second].str().c_str());
        }

        spins = compute_spin(space, evecs, nroot);
        S = spins[n].first.first;
        S2 = spins[n].first.second;
        max_I = spins[n].second.first;
        sum_weight = spins[n].second.second;

        state_label = s2_labels[std::round(S * 2.0)];
        outfile->Printf("\n\n  Spin State for root %zu: S^2 = %5.3f, S = %5.3f, %s (from %zu "
                        "determinants,%3.2f%)",
                        n, S2, S, state_label.c_str(), max_I, 100.0 * sum_weight);
        root_spin_vec_.clear();
        root_spin_vec_[n] = std::make_pair(S, S2);
    }
}

int EX_ACI::direct_sym_product(int sym1, int sym2) {

    /*Create a matrix for direct products of Abelian symmetry groups
    *
    * This matrix is 8x8, but it works for molecules of both D2H and
    * C2V symmetry
    *
    * Due to properties of groups, direct_sym_product(a,b) solves both
    *
    * a (x) b = ?
    * and
    * a (x) ? = b
    */

    boost::shared_ptr<Matrix> dp(new Matrix("dp", 8, 8));

    for (int p = 0; p < 2; ++p) {
        for (int q = 0; q < 2; ++q) {
            if (p != q) {
                dp->set(p, q, 1);
            }
        }
    }

    for (int p = 2; p < 4; ++p) {
        for (int q = 0; q < 2; ++q) {
            dp->set(p, q, dp->get(p - 2, q) + 2);
            dp->set(q, p, dp->get(p - 2, q) + 2);
        }
    }

    for (int p = 2; p < 4; ++p) {
        for (int q = 2; q < 4; ++q) {
            dp->set(p, q, dp->get(p - 2, q - 2));
        }
    }

    for (int p = 4; p < 8; ++p) {
        for (int q = 0; q < 4; ++q) {
            dp->set(p, q, dp->get(p - 4, q) + 4);
            dp->set(q, p, dp->get(p - 4, q) + 4);
        }
    }

    for (int p = 4; p < 8; ++p) {
        for (int q = 4; q < 8; ++q) {
            dp->set(p, q, dp->get(p - 4, q - 4));
        }
    }

    return dp->get(sym1, sym2);
}

void EX_ACI::wfn_analyzer(std::vector<DynamicBitsetDeterminant> det_space, SharedMatrix evecs,
                          int nroot) {
    for (int n = 0; n < nroot; ++n) {
        pVector<size_t, double> excitation_counter(1 + (1 + cycle_) * 2);
        pVector<double, size_t> det_weight;
        for (size_t I = 0; I < det_space.size(); ++I) {
            det_weight.push_back(std::make_pair(std::fabs(evecs->get(I, n)), I));
        }

        std::sort(det_weight.begin(), det_weight.end());
        std::reverse(det_weight.begin(), det_weight.end());

        DynamicBitsetDeterminant ref;
        ref.copy(det_space[det_weight[0].second]);

        auto alfa_bits = ref.alfa_bits();
        auto beta_bits = ref.beta_bits();

        for (size_t I = 0, max = det_space.size(); I < max; ++I) {
            int ndiff = 0;
            auto ex_alfa_bits = det_space[det_weight[I].second].alfa_bits();
            auto ex_beta_bits = det_space[det_weight[I].second].beta_bits();

            // Compute number of differences in both alpha and beta strings wrt ref
            for (size_t a = 0, max = alfa_bits.size(); a < max; ++a) {
                if (alfa_bits[a] != ex_alfa_bits[a]) {
                    ++ndiff;
                }
                if (beta_bits[a] != ex_beta_bits[a]) {
                    ++ndiff;
                }
            }
            ndiff /= 2;
            excitation_counter[ndiff] = std::make_pair(excitation_counter[ndiff].first + 1,
                                                       excitation_counter[ndiff].second +
                                                           std::pow(det_weight[I].first, 2.0));
        }
        int order = 0;
        size_t det = 0;
        for (auto i : excitation_counter) {
            outfile->Printf("\n      %2d          %8zu           %.11f", order, i.first, i.second);
            det += i.first;
            if (det == det_space.size())
                break;
            ++order;
        }
        outfile->Printf("\n\n  Highest-order exitation searched:     %zu  \n",
                        excitation_counter.size() - 1);
    }
}

oVector<double, int, int> EX_ACI::sym_labeled_orbitals(std::string type) {
    oVector<double, int, int> labeled_orb;

    if (type == "RHF" or type == "ROHF" or type == "ALFA") {

        // Create a vector of orbital energy and index pairs (Pitzer ordered)
        pVector<double, int> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; h++) {
            for (int a = 0; a < ncmopi_[h]; a++) {
                orb_e.push_back(std::make_pair(epsilon_a_->get(h, a + frzcpi_[h]), a + cumidx));
            }
            cumidx += ncmopi_[h];
        }

        // Create a vector that stores the orbital energy, symmetry, and Pitzer-ordered index
        for (int a = 0; a < ncmo_; ++a) {
            labeled_orb.push_back(
                std::make_pair(orb_e[a].first, std::make_pair(mo_symmetry_[a], orb_e[a].second)));
        }

        // Order by energy, low to high
        std::sort(labeled_orb.begin(), labeled_orb.end());
    }

    if (type == "BETA") {
        // Create a vector of orbital energy and index pairs (Pitzer ordered)
        pVector<double, int> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; h++) {
            for (int a = 0; a < ncmopi_[h] - frzcpi_[h]; a++) {
                orb_e.push_back(std::make_pair(epsilon_b_->get(h, a + frzcpi_[h]), a + cumidx));
            }
            cumidx += (ncmopi_[h]);
        }

        // Create a vector that stores the orbital energy, symmetry, and Pitzer-ordered index
        for (int a = 0; a < ncmo_; ++a) {
            labeled_orb.push_back(
                std::make_pair(orb_e[a].first, std::make_pair(mo_symmetry_[a], orb_e[a].second)));
        }

        // Order by energy, low to high
        std::sort(labeled_orb.begin(), labeled_orb.end());
    }

    for (int i = 0; i < ncmo_; ++i) {
        outfile->Printf("\n %f    %d    %d", labeled_orb[i].first, labeled_orb[i].second.first,
                        labeled_orb[i].second.second);
    }

    return labeled_orb;
}

void EX_ACI::compute_1rdm(SharedMatrix A, SharedMatrix B,
                          std::vector<DynamicBitsetDeterminant> det_space, SharedMatrix evecs,
                          int nroot) {

    // Make a vector of indices for core and active orbitals
    int ncmopi = 0;
    std::vector<size_t> idx_a;
    std::vector<size_t> idx_c;
    for (int h = 0; h < nirrep_; ++h) {
        for (size_t i = 0; i < nmopi_[h]; ++i) {
            size_t idx = i + ncmopi;
            if (i < frzcpi_[h]) {
                idx_c.push_back(idx);
            }
            if (i >= frzcpi_[h] and i < (frzcpi_[h] + ncmopi_[h])) {
                idx_a.push_back(idx);
            }
        }
        ncmopi += nmopi_[h];
    }

    // Occupy frozen core with 1.0
    for (size_t p = 0; p < nfrzc_; ++p) {
        size_t np = idx_c[p];
        A->set(np, np, 1.0);
        B->set(np, np, 1.0);
    }

    // Populate active indices
    for (size_t p = 0; p < ncmo_; ++p) {
        size_t np = idx_a[p];
        for (size_t q = p; q < ncmo_; ++q) {
            size_t nq = idx_a[q];

            if ((mo_symmetry_[p] ^ mo_symmetry_[q]) != 0)
                continue;

            // Loop over determinants
            for (size_t I = 0, max = det_space.size(); I < max; ++I) {
                DynamicBitsetDeterminant Ja, Jb;
                double C_I = evecs->get(I, 0);
                double a = 1.0, b = 1.0;

                a *= OneOP(det_space[I], Ja, 0, p, q) * C_I;
                b *= OneOP(det_space[I], Jb, 1, p, q) * C_I;

                for (size_t J = 0, mJ = det_space.size(); J < mJ; ++J) {
                    double C_J = evecs->get(J, 0);
                    A->add(np, nq, a * (det_space[J] == Ja) * C_J);
                    B->add(np, nq, b * (det_space[J] == Jb) * C_J);
                }
            }
            A->set(nq, np, A->get(np, nq));
            B->set(nq, np, B->get(np, nq));
        }
    }

    double trace = 0.0;
    D1_ = A->clone();
    for (int p = 0; p < nmo_; ++p) {
        for (int q = 0; q < nmo_; ++q) {
            D1_->add(p, q, B->get(p, q));
            if (p == q)
                trace += D1_->get(p, q);
        }
    }
    outfile->Printf("\n\n  Trace of 1-RDM is %6.3f\n", trace);
}
double EX_ACI::OneOP(const DynamicBitsetDeterminant& J, DynamicBitsetDeterminant& Jnew,
                     const bool sp, const size_t& p, const size_t& q) {
    timer_on("1PO");
    DynamicBitsetDeterminant tmp = J;

    double sign = 1.0;

    if (sp == false) {
        if (tmp.get_alfa_bit(q)) {
            sign *= CheckSign(tmp.get_alfa_occ(), q);
            tmp.set_alfa_bit(q, 0);
        } else {
            timer_off("1PO");
            return 0.0;
        }

        if (!tmp.get_alfa_bit(p)) {
            sign *= CheckSign(tmp.get_alfa_occ(), p);
            tmp.set_alfa_bit(p, 1);
            Jnew.copy(tmp);
            timer_off("1PO");
            return sign;
        } else {
            timer_off("1PO");
            return 0.0;
        }
        Jnew.print();
    } else {
        if (tmp.get_beta_bit(q)) {
            sign *= CheckSign(tmp.get_beta_occ(), q);
            tmp.set_beta_bit(q, 0);
        } else {
            timer_off("1PO");
            return 0.0;
        }

        if (!tmp.get_beta_bit(p)) {
            sign *= CheckSign(tmp.get_beta_occ(), p);
            tmp.set_beta_bit(p, 1);
            Jnew.copy(tmp);
            timer_off("1PO");
            return sign;
        } else {
            timer_off("1PO");
            return 0.0;
        }
        Jnew.print();
    }
}

double EX_ACI::CheckSign(std::vector<int> I, const int& n) {
    timer_on("Check Sign");
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (I[i])
            ++count;
    }
    timer_off("Check Sign");
    return pow(-1.0, count % 2);
}

// make this call other functions eg, lambda, casincrement, cas
void EX_ACI::form_initial_space(std::vector<DynamicBitsetDeterminant> P_space, int nroot) {
    int converge = 0;
    size_t count = 0;
    double thresh = options_.get_double("LAMBDA_THRESH");
    outfile->Printf(
        "\n  Forming initial space from lowest-energy determinants within %1.1f Hartree", thresh);
    std::vector<DynamicBitsetDeterminant> P_space_init = P_space;
    //	double e_min = P_space_init[0].energy();

    while (converge < 2) {
        std::map<DynamicBitsetDeterminant, double> V_map;
        for (auto& I : P_space) {

            // Add all P space determinants so that no duplicates are created
            V_map[I] = I.energy();

            // Generate all singly excited determinants and store their energy
            std::vector<int> aocc = I.get_alfa_occ();
            std::vector<int> bocc = I.get_beta_occ();
            std::vector<int> avir = I.get_alfa_vir();
            std::vector<int> bvir = I.get_beta_vir();

            int noalpha = aocc.size();
            int nobeta = bocc.size();
            int nvalpha = avir.size();
            int nvbeta = bvir.size();

            // Generate aa excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        DynamicBitsetDeterminant new_det(I);
                        new_det.set_alfa_bit(ii, false);
                        new_det.set_alfa_bit(aa, true);
                        if (V_map.count(new_det) == 0) {
                            double EI = new_det.energy();
                            V_map[new_det] = EI;
                        }
                    }
                }
            }

            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        DynamicBitsetDeterminant new_det(I);
                        new_det.set_beta_bit(ii, false);
                        new_det.set_beta_bit(aa, true);
                        if (V_map.count(new_det) == 0) {
                            double EI = new_det.energy();
                            V_map[new_det] = EI;
                        }
                    }
                }
            }
        } // End loop over determinants

        pVector<double, DynamicBitsetDeterminant> det_map;
        det_map.clear();

        for (auto& S : V_map) {
            det_map.push_back(std::make_pair(S.second, S.first));
        }

        // Sort determinants in ascending order
        std::sort(det_map.begin(), det_map.end());
        size_t old_dim = P_space.size();
        P_space.clear();

        double e_min = det_map[0].first;
        for (auto& I : det_map) {
            double diff = I.first - e_min;
            if (diff < thresh) {
                P_space.push_back(I.second);
                // outfile->Printf("\n  Det # %zu = %4.16f", num, I.first);
            } else {
                continue;
            }
        }

        count++;
        outfile->Printf("\n  Order %zu determinants: %zu", count, P_space.size() - old_dim);

        if (P_space.size() == old_dim) {
            converge++;
        } else {
            continue;
        }
    }

    P_space_.clear();
    P_space_ = P_space;
}

void EX_ACI::add_spin_pair(std::vector<DynamicBitsetDeterminant> det_space) {
    std::vector<size_t> single_idx;
    // Get indices of lonely determinants
    for (size_t I = 0, maxI = det_space.size(); I < maxI; ++I) {
        for (size_t J = 0, maxJ = det_space.size(); J < maxJ; ++J) {
            if (det_space[I].get_alfa_occ() == det_space[J].get_beta_occ() and
                det_space[I].get_beta_occ() == det_space[J].get_alfa_occ()) {
                break;
            } else if (J == det_space.size() - 1) {
                //   outfile->Printf("\n  Det at index %zu has no spin-partner", I);
                single_idx.push_back(I);
            } else {
                continue;
            }
        }
    }

    // Give them a partner
    for (auto& I : single_idx) {
        DynamicBitsetDeterminant new_det = det_space[I];
        new_det.spin_flip();
        PQ_space_.push_back(new_det);
    }
}

/* Spin_transform builds the S2 matrix in
 * the determinant basis, diagonalizes it,
 * and transforms the cI coefficients with
 * the eigenvectors.
 */

void EX_ACI::spin_transform(std::vector<DynamicBitsetDeterminant> det_space, SharedMatrix cI,
                            int nroot) {

    outfile->Printf("\n  Performing Spin Projection...");
    Timer timer;

    size_t det_size = det_space.size();
    SharedMatrix S2(new Matrix("S^2", det_size, det_size));

    // Build S^2
    for (size_t I = 0; I < det_size; ++I) {
        for (size_t J = 0; J <= I; ++J) {
            S2->set(I, J, det_space[I].spin2(det_space[J]));
            S2->set(J, I, S2->get(I, J));
        }
    }

    SharedMatrix T(new Matrix("T", det_size, det_size));
    SharedVector Evals(new Vector("Evals", det_size));

    // Diagonalize S^2, evals will be in ascending order
    // Evecs will be in the same order as evals
    S2->diagonalize(T, Evals);

    // Count the number of CSFs with correct spin
    // and get their indices wrt columns in T
    size_t csf_num = 0;
    size_t csf_idx = 0;
    for (size_t l = 0; l < det_size; ++l) {
        if (std::fabs(Evals->get(l) -
                      (0.25 * (wavefunction_multiplicity_ * wavefunction_multiplicity_ - 1.0))) <=
            0.01) {
            csf_num++;
        } else if (csf_num == 0) {
            csf_idx++;
        } else {
            continue;
        }
    }

    SharedMatrix C_trans(new Matrix("C_trans", det_size, nroot));
    outfile->Printf("\n  csf_num: %zu \n", csf_num);

    SharedMatrix C(new Matrix("c", det_size, nroot));
    C->gemm('t', 'n', csf_num, nroot, det_size, 1.0, T, det_size, cI, nroot, 0.0, nroot);
    C_trans->gemm('n', 'n', det_size, nroot, csf_num, 1.0, T, det_size, C, nroot, 0.0, nroot);

    // Normalize transformed vectors
    for (size_t n = 0; n < nroot; ++n) {
        double denom = 0.0;
        for (size_t I = 0; I < det_size; ++I) {
            denom += std::pow(C_trans->get(I, n), 2.0);
        }
        denom = std::sqrt(1.0 / denom);
        C_trans->scale_column(0, n, denom);
    }
    PQ_spin_evecs_.reset(new Matrix("PQ_spin_evecs_", det_size, nroot));
    PQ_spin_evecs_ = C_trans->clone();

    outfile->Printf("\n  Time spent performing spin transformation: %6.6f s", timer.get());
}

void EX_ACI::check_spin_completeness(std::vector<DynamicBitsetDeterminant>& det_space) {
    std::map<DynamicBitsetDeterminant, bool> det_map;

    // Add all determinants to the map, assume set is mostly spin complete
    for (auto& I : det_space) {
        det_map[I] = true;
    }

    size_t ndet = 0;

    // Loop over determinants
    size_t max = det_space.size();
    for (size_t I = 0; I < max; ++I) {
        // Loop over mos
        for (size_t i = 0; i < ncmo_; ++i) {
            for (size_t j = 0; j < ncmo_; ++j) {
                if (det_space[I].get_alfa_bit(i) == det_space[I].get_beta_bit(j) and
                    det_space[I].get_alfa_bit(i) == 1 and det_space[I].get_alfa_bit(j) == 0 and
                    det_space[I].get_beta_bit(i) == 0) {

                    DynamicBitsetDeterminant det(det_space[I]);
                    det.set_alfa_bit(i, det_space[I].get_beta_bit(i));
                    det.set_beta_bit(j, det_space[I].get_alfa_bit(j));
                    det.set_alfa_bit(j, det_space[I].get_beta_bit(j));
                    det.set_beta_bit(i, det_space[I].get_alfa_bit(i));

                    if (det_map.count(det) == 0) {
                        det_space.push_back(det);
                        det_map[det] = false;
                        ndet++;
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }
        }
    }
    if (ndet > 0) {
        outfile->Printf("\n  Determinant space is spin incomplete!");
        outfile->Printf("\n  %zu more determinants are needed.", ndet);
    } else {
        outfile->Printf("\n  Determinant space is spin complete");
    }
}
}
} // EndNamespaces
