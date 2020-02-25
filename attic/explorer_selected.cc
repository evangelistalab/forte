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

#include "mini-boost/boost/format.hpp"
#include "mini-boost/boost/timer.hpp"

#include "physconst.h"
#include <libciomr/libciomr.h>
#include <libqt/qt.h>

#include "cartographer.h"
#include "lambda-ci.h"
#include "string_determinant.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

/**
 * Diagonalize the
 */
void LambdaCI::diagonalize_selected_space(std::shared_ptr<ForteOptions> options) {
    outfile->Printf("\n\n  Diagonalizing the Hamiltonian in the model space (Lambda = %.2f Eh)\n",
                    space_m_threshold_);

    // 1) Build the Hamiltonian
    ForteTimer t_hbuild;
    SharedMatrix H_m = build_model_space_hamiltonian(options);
    outfile->Printf("\n  Time spent building H model       = %f s", t_hbuild.elapsed());

    // 2) Setup stuff necessary to diagonalize the Hamiltonian
    int ndets_m = H_m->nrow();
    int nroots = ndets_m;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
        nroots = std::min(options.get_int("NROOT"), ndets_m);
    }
    SharedMatrix evecs_m(new Matrix("U", ndets_m, nroots));
    SharedVector evals_m(new Vector("e", nroots));

    // 3) Diagonalize the model space Hamiltonian
    ForteTimer t_hdiag;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
        outfile->Printf("\n  Using the Davidson-Liu algorithm.");
        davidson_liu(H_m, evals_m, evecs_m, nroots);
    } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
        outfile->Printf("\n  Performing full diagonalization.");
        H_m->diagonalize(evecs_m, evals_m);
    }
    outfile->Printf("\n  Time spent diagonalizing H        = %f s", t_hdiag.elapsed());

    // 4) Print the energy
    int nroots_print = std::min(nroots, 25);
    for (int i = 0; i < nroots_print; ++i) {
        outfile->Printf("\n  Small CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                        evals_m->get(i), pc_hartree2ev * (evals_m->get(i) - evals_m->get(0)));
    }

    double significant_threshold = 0.001;
    double significant_wave_function = 0.95;
    for (int i = 0; i < nroots_print; ++i) {
        outfile->Printf(
            "\n  The most important determinants (%.0f%% of the wave functions) for root %d:",
            100.0 * significant_wave_function, i + 1);
        // Identify all contributions with |C_J| > significant_threshold
        double** C_mat = evecs_m->pointer();
        std::vector<std::pair<double, int>> C_J_sorted;
        for (int J = 0; J < ndets_m; ++J) {
            if (std::fabs(C_mat[J][i]) > significant_threshold) {
                C_J_sorted.push_back(std::make_pair(std::fabs(C_mat[J][i]), J));
            }
        }
        // Sort them and print
        std::sort(C_J_sorted.begin(), C_J_sorted.end(), std::greater<std::pair<double, int>>());
        double cum_wfn = 0.0;
        for (size_t I = 0, max_I = C_J_sorted.size(); I < max_I; ++I) {
            int J = C_J_sorted[I].second;
            outfile->Printf("\n %3ld   %+9.6f   %9.6f   %.6f   %d", I, C_mat[J][i],
                            C_mat[J][i] * C_mat[J][i], H_m->get(J, J), J);
            cum_wfn += C_mat[J][i] * C_mat[J][i];
            if (cum_wfn > significant_wave_function)
                break;
        }
    }

    int num_roots = options.get_int("NROOT");
    outfile->Printf(
        "\n\n  Building a selected Hamiltonian using the criterium by Roth (kappa) for %d roots",
        num_roots);
    SharedMatrix H = build_select_hamiltonian_roth(options, evals_m, evecs_m);

    // 3) Setup stuff necessary to diagonalize the Hamiltonian
    int ndets = H->nrow();
    SharedMatrix evecs(new Matrix("U", ndets, nroots));
    SharedVector evals(new Vector("e", nroots));

    // 4) Diagonalize the Hamiltonian
    ForteTimer t_hdiag_large;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
        outfile->Printf("\n  Using the Davidson-Liu algorithm.");
        davidson_liu(H, evals, evecs, nroots);
    } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
        outfile->Printf("\n  Performing full diagonalization.");
        H->diagonalize(evecs, evals);
    }
    outfile->Printf("\n  Time spent diagonalizing H        = %f s", t_hdiag_large.elapsed());

    // 5) Print the energy
    for (int i = 0; i < nroots_print; ++i) {
        outfile->Printf("\n  Adaptive CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                        evals->get(i), pc_hartree2ev * (evals->get(i) - evals->get(0)));
        outfile->Printf("\n  Adaptive CI Energy + EPT2 Root %3d = %.12f = %.12f + %.12f", i + 1,
                        evals->get(i) + multistate_pt2_energy_correction_[i], evals->get(i),
                        multistate_pt2_energy_correction_[i]);
    }

    // 6) Print the major contributions to the eigenvector
    for (int i = 0; i < nroots_print; ++i) {
        outfile->Printf(
            "\n  The most important determinants (%.0f%% of the wave functions) for root %d:",
            100.0 * significant_wave_function, i + 1);
        // Identify all contributions with |C_J| > significant_threshold
        double** C_mat = evecs->pointer();
        std::vector<std::pair<double, int>> C_J_sorted;
        for (int J = 0; J < ndets; ++J) {
            if (std::fabs(C_mat[J][i]) > significant_threshold) {
                C_J_sorted.push_back(std::make_pair(std::fabs(C_mat[J][i]), J));
            }
        }
        // Sort them and print
        std::sort(C_J_sorted.begin(), C_J_sorted.end(), std::greater<std::pair<double, int>>());
        double cum_wfn = 0.0;
        for (size_t I = 0, max_I = C_J_sorted.size(); I < max_I; ++I) {
            int J = C_J_sorted[I].second;
            outfile->Printf("\n %3ld   %+9.6f   %9.6f   %.6f   %d", I, C_mat[J][i],
                            C_mat[J][i] * C_mat[J][i], H->get(J, J), J);
            cum_wfn += C_mat[J][i] * C_mat[J][i];
            if (cum_wfn > significant_wave_function)
                break;
        }
    }
}

/**
 * Build the Hamiltonian matrix for all determinants that fall in the model space.
 * It assumes that the determinants are stored in increasing energetic order.
 * @param ndets
 * @return a SharedMatrix object that contains the Hamiltonian
 */
SharedMatrix LambdaCI::build_model_space_hamiltonian(Options& options) {
    int ntot_dets = static_cast<int>(determinants_.size());

    // the number of determinants used to form the Hamiltonian matrix
    int ndets = 0;

    // Determine the size of the Hamiltonian matrix
    if (options.get_str("H_TYPE") == "FIXED_SIZE") {
        ndets = std::min(options.get_int("NDETS"), ntot_dets);
        outfile->Printf(
            "\n  Building the model space Hamiltonian using the first %d determinants\n", ndets);
        outfile->Printf("\n  The energy range spanned is [%f,%f]\n", determinants_[0].get<0>(),
                        determinants_[ndets - 1].get<0>());
    } else if (options.get_str("H_TYPE") == "FIXED_ENERGY") {
        double E0 = determinants_[0].get<0>();
        for (int I = 0; I < ntot_dets; ++I) {
            double EI = determinants_[I].get<0>();
            if (EI - E0 > space_m_threshold_) {
                break;
            }
            ndets++;
        }
        outfile->Printf("\n\n  Building the model Hamiltonian using determinants with excitation "
                        "energy less than %f Eh",
                        space_m_threshold_);
        outfile->Printf("\n  This requires a total of %d determinants", ndets);
        int max_ndets_fixed_energy = options.get_int("MAX_NDETS");
        if (ndets > max_ndets_fixed_energy) {
            outfile->Printf(
                "\n\n  WARNING: the number of determinants required to build the Hamiltonian (%d)\n"
                "  exceeds the maximum number allowed (%d).  Reducing the size of H.\n\n",
                ndets, max_ndets_fixed_energy);
            ndets = max_ndets_fixed_energy;
        }
    }

    SharedMatrix H(new Matrix("Hamiltonian Matrix", ndets, ndets));

// Form the Hamiltonian matrix
#pragma omp parallel for schedule(dynamic)
    for (int I = 0; I < ndets; ++I) {
        boost::tuple<double, int, int, int, int>& determinantI = determinants_[I];
        const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
        for (int J = I + 1; J < ndets; ++J) {
            boost::tuple<double, int, int, int, int>& determinantJ = determinants_[J];
            const int J_class_a = determinantJ.get<1>(); // std::get<1>(determinantI);
            const int Jsa = determinantJ.get<2>();       // std::get<1>(determinantI);
            const int J_class_b = determinantJ.get<3>(); // std::get<2>(determinantI);
            const int Jsb = determinantJ.get<4>();       // std::get<2>(determinantI);
            const double HIJ = StringDeterminant::SlaterRules(
                vec_astr_symm_[I_class_a][Isa].get<2>(), vec_bstr_symm_[I_class_b][Isb].get<2>(),
                vec_astr_symm_[J_class_a][Jsa].get<2>(), vec_bstr_symm_[J_class_b][Jsb].get<2>());
            H->set(I, J, HIJ);
            H->set(J, I, HIJ);
        }
        H->set(I, I, determinantI.get<0>());
    }
    return H;
}

/**
 * Build the Hamiltonian matrix for all determinants that fall in the model space.
 * It assumes that the determinants are stored in increasing energetic order.
 * @param ndets
 * @return a SharedMatrix object that contains the Hamiltonian
 */
SharedMatrix LambdaCI::build_select_hamiltonian_roth(Options& options, SharedVector evals,
                                                     SharedMatrix evecs) {
    int ntot_dets = static_cast<int>(determinants_.size());

    // Find out which determinants will be included
    std::vector<int> selected_dets;
    int ndets_m = evecs->nrow();
    int ndets_i = 0;

    for (int J = 0; J < ndets_m; ++J)
        selected_dets.push_back(J);

    bool aimed_selection = false;
    bool energy_select = false;
    if (options.get_str("SELECT_TYPE") == "AIMED_AMP") {
        aimed_selection = true;
        energy_select = false;
    } else if (options.get_str("SELECT_TYPE") == "AIMED_ENERGY") {
        aimed_selection = true;
        energy_select = true;
    } else if (options.get_str("SELECT_TYPE") == "ENERGY") {
        aimed_selection = false;
        energy_select = true;
    } else if (options.get_str("SELECT_TYPE") == "AMP") {
        aimed_selection = false;
        energy_select = false;
    }

    if (energy_select) {
        outfile->Printf("\n  Building a selected Hamiltonian using the energy criterium");
    } else {
        outfile->Printf("\n  Building a selected Hamiltonian using the amplitude criterium");
    }

    int nroot = options.get_int("NROOT");
    std::vector<double> V_q(nroot, 0.0);
    std::vector<double> t_q(nroot, 0.0);
    std::vector<std::pair<double, double>> kappa_q(nroot, std::make_pair(0.0, 0.0));
    std::vector<std::pair<double, double>> chi_q(nroot, std::make_pair(0.0, 0.0));
    std::vector<double> ept2(nroot, 0.0);

    std::vector<std::pair<double, size_t>> aimed_selection_vec;
    double aimed_selection_sum = 0.0;

    //    #pragma omp parallel for schedule(dynamic)
    for (int I = ndets_m; I < ntot_dets; ++I) {
        boost::tuple<double, int, int, int, int>& determinantI = determinants_[I];
        const double EI = determinantI.get<0>();     // std::get<0>(determinantI);
        const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
        for (int n = 0; n < nroot; ++n) {
            V_q[n] = 0.0;
            t_q[n] = 0.0;
        }
        for (int J = 0; J < ndets_m; ++J) {
            boost::tuple<double, int, int, int, int>& determinantJ = determinants_[J];
            const double EJ = determinantJ.get<0>();     // std::get<0>(determinantJ);
            const int J_class_a = determinantJ.get<1>(); // std::get<1>(determinantJ);
            const int Jsa = determinantJ.get<2>();       // std::get<1>(determinantJ);
            const int J_class_b = determinantJ.get<3>(); // std::get<2>(determinantJ);
            const int Jsb = determinantJ.get<4>();       // std::get<2>(determinantJ);
            const double HIJ = StringDeterminant::SlaterRules(
                vec_astr_symm_[I_class_a][Isa].get<2>(), vec_bstr_symm_[I_class_b][Isb].get<2>(),
                vec_astr_symm_[J_class_a][Jsa].get<2>(), vec_bstr_symm_[J_class_b][Jsb].get<2>());
            for (int n = 0; n < nroot; ++n) {
                V_q[n] += evecs->get(J, n) * HIJ;
            }
        }
        for (int n = 0; n < nroot; ++n) {
            double kappa = -V_q[n] / (EI - evals->get(n));
            double chi = -V_q[n] * V_q[n] / (EI - evals->get(n));
            kappa_q[n] = std::make_pair(std::fabs(kappa), kappa);
            chi_q[n] = std::make_pair(std::fabs(chi), chi);
        }

        //        double kappa =  - V / (EI - E);
        //        double chi = - V * V / (EI - E);
        std::pair<double, double> max_kappa = *std::max_element(kappa_q.begin(), kappa_q.end());
        std::pair<double, double> max_chi = *std::max_element(chi_q.begin(), chi_q.end());

        double selection_value = energy_select ? max_chi.first : max_kappa.first;

        // Do not select now, just store the determinant index and the selection criterion
        if (aimed_selection) {
            if (energy_select) {
                aimed_selection_vec.push_back(std::make_pair(selection_value, I));
                aimed_selection_sum += selection_value;
            } else {
                aimed_selection_vec.push_back(std::make_pair(selection_value * selection_value, I));
                aimed_selection_sum += selection_value * selection_value;
            }
        } else {
            if (std::fabs(selection_value) > t2_threshold_) {
                //            #pragma omp critical
                selected_dets.push_back(I);
                ndets_i += 1;
            } else {
                for (int n = 0; n < nroot; ++n)
                    ept2[n] += chi_q[n].second;
            }
        }
    }

    if (aimed_selection) {
        std::sort(aimed_selection_vec.begin(), aimed_selection_vec.end());
        std::reverse(aimed_selection_vec.begin(), aimed_selection_vec.end());
        size_t maxI = aimed_selection_vec.size();

        outfile->Printf("\n  Initial value of sigma in the aimed selection = %24.14f",
                        aimed_selection_sum);
        for (size_t I = 0; I < maxI; ++I) {
            if (aimed_selection_sum > t2_threshold_) {
                selected_dets.push_back(aimed_selection_vec[I].second);
                aimed_selection_sum -= aimed_selection_vec[I].first;
                ndets_i += 1;
            } else {
                break;
            }
        }
        outfile->Printf("\n  Final value of sigma in the aimed selection   = %24.14f",
                        aimed_selection_sum);
        outfile->Printf("\n  Selected %zu determinants", selected_dets.size());
    }

    multistate_pt2_energy_correction_ = ept2;

    // the number of determinants used to form the Hamiltonian matrix
    int ndets = ndets_i + ndets_m;
    outfile->Printf("\n\n  %d total states: %d (main) + %d (intermediate)", ntot_dets, ndets_m,
                    ndets_i);
    outfile->Printf(
        "\n  %d states were discarded because the coupling to the main space is less than %f muE_h",
        ntot_dets - ndets_m - ndets_i, t2_threshold_ * 1000000.0);

    SharedMatrix H(new Matrix("Hamiltonian Matrix", ndets, ndets));
// Form the Hamiltonian matrix
#pragma omp parallel for schedule(dynamic)
    for (int I = 0; I < ndets; ++I) {
        boost::tuple<double, int, int, int, int>& determinantI = determinants_[selected_dets[I]];
        const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
        for (int J = I + 1; J < ndets; ++J) {
            boost::tuple<double, int, int, int, int>& determinantJ =
                determinants_[selected_dets[J]];
            const int J_class_a = determinantJ.get<1>(); // std::get<1>(determinantI);
            const int Jsa = determinantJ.get<2>();       // std::get<1>(determinantI);
            const int J_class_b = determinantJ.get<3>(); // std::get<2>(determinantI);
            const int Jsb = determinantJ.get<4>();       // std::get<2>(determinantI);
            const double HIJ = StringDeterminant::SlaterRules(
                vec_astr_symm_[I_class_a][Isa].get<2>(), vec_bstr_symm_[I_class_b][Isb].get<2>(),
                vec_astr_symm_[J_class_a][Jsa].get<2>(), vec_bstr_symm_[J_class_b][Jsb].get<2>());
            H->set(I, J, HIJ);
            H->set(J, I, HIJ);
        }
        H->set(I, I, determinantI.get<0>());
    }

    return H;
}

void LambdaCI::diagonalize_renormalized_space(std::shared_ptr<ForteOptions> options) {
    outfile->Printf("\n\n  Diagonalizing the Hamiltonian in the model space");
    outfile->Printf("\n  using a renormalization procedure (Lambda = %.2f Eh)\n",
                    space_m_threshold_);

    int nroot = options.get_int("NROOT");

    bool energy_select = (options.get_str("SELECT_TYPE") == "ENERGY");

    int renomalization_steps = options.get_int("RENORMALIZATION_STEPS");
    double delta_lambda = space_i_threshold_ / static_cast<double>(renomalization_steps);

    int ntot_dets = static_cast<int>(determinants_.size());
    size_t search_from = 0;
    double E0 = determinants_[0].get<0>();
    std::vector<size_t> selected_dets;
    SharedMatrix H;
    SharedMatrix evecs;
    SharedVector evals;

    std::vector<double> ept2(nroot, 0.0);

    for (int step = 0; step < renomalization_steps; ++step) {
        size_t num_selected_dets = selected_dets.size();

        // Find all the determinants within the range [0,DL)
        double min_range = static_cast<double>(step) * delta_lambda;
        double max_range = min_range + delta_lambda;

        ForteTimer t_select;
        outfile->Printf("\n\n  Finding dets in the range : [%f,%f), starting from %zu", min_range,
                        max_range, search_from);
        std::vector<size_t> det_in_range;
        for (int I = search_from; I < ntot_dets; ++I) {
            double EI = determinants_[I].get<0>();
            if (EI - E0 >= max_range) {
                search_from = I;
                break;
            } else if (EI - E0 >= min_range) {
                det_in_range.push_back(I);
            }
        }
        size_t num_det_in_range = det_in_range.size();
        outfile->Printf("\n  Found %zu", num_det_in_range);

        std::vector<double> V_q(nroot, 0.0);
        std::vector<double> t_q(nroot, 0.0);
        std::vector<std::pair<double, double>> kappa_q(nroot, std::make_pair(0.0, 0.0));
        std::vector<std::pair<double, double>> chi_q(nroot, std::make_pair(0.0, 0.0));

        std::vector<size_t> selected_test_dets = selected_dets;
        if (step != 0) {
            for (size_t I = 0; I < num_det_in_range; ++I) {
                boost::tuple<double, int, int, int, int>& determinantI =
                    determinants_[det_in_range[I]];
                const double EI = determinantI.get<0>();     // std::get<0>(determinantI);
                const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
                const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
                const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
                const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
                for (int n = 0; n < nroot; ++n) {
                    V_q[n] = 0.0;
                    t_q[n] = 0.0;
                }
                for (int J = 0; J < num_selected_dets; ++J) {
                    boost::tuple<double, int, int, int, int>& determinantJ =
                        determinants_[selected_dets[J]];
                    const double EJ = determinantJ.get<0>();     // std::get<0>(determinantJ);
                    const int J_class_a = determinantJ.get<1>(); // std::get<1>(determinantJ);
                    const int Jsa = determinantJ.get<2>();       // std::get<1>(determinantJ);
                    const int J_class_b = determinantJ.get<3>(); // std::get<2>(determinantJ);
                    const int Jsb = determinantJ.get<4>();       // std::get<2>(determinantJ);
                    const double HIJ =
                        StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),
                                                       vec_bstr_symm_[I_class_b][Isb].get<2>(),
                                                       vec_astr_symm_[J_class_a][Jsa].get<2>(),
                                                       vec_bstr_symm_[J_class_b][Jsb].get<2>());
                    for (int n = 0; n < nroot; ++n) {
                        V_q[n] += evecs->get(J, n) * HIJ;
                    }
                }
                for (int n = 0; n < nroot; ++n) {
                    double kappa = -V_q[n] / (EI - evals->get(n));
                    double chi = -V_q[n] * V_q[n] / (EI - evals->get(n));
                    kappa_q[n] = std::make_pair(std::fabs(kappa), kappa);
                    chi_q[n] = std::make_pair(std::fabs(chi), chi);
                }

                std::pair<double, double> max_kappa =
                    *std::max_element(kappa_q.begin(), kappa_q.end());
                std::pair<double, double> max_chi = *std::max_element(chi_q.begin(), chi_q.end());

                double selection_value = energy_select ? max_chi.first : max_kappa.first;

                // Do not select now, just store the determinant index and the selection criterion
                if (std::fabs(selection_value) > t2_threshold_) {
                    //            #pragma omp critical
                    selected_test_dets.push_back(det_in_range[I]);
                } else {
                    for (int n = 0; n < nroot; ++n)
                        ept2[n] += chi_q[n].second;
                }
            }
        } else {
            for (size_t n = 0, maxn = det_in_range.size(); n < maxn; ++n) {
                selected_test_dets.push_back(det_in_range[n]);
            }
        }
        outfile->Printf(".  Of these %zu passed a screening.  Total dets. %zu.",
                        selected_test_dets.size() - selected_dets.size(),
                        selected_test_dets.size());
        outfile->Printf("\n  Time spent selecting the new dets = %f s", t_select.elapsed());

        multistate_pt2_energy_correction_ = ept2;

        size_t num_selected_test_dets = selected_test_dets.size();
        H.reset(new Matrix("Hamiltonian Matrix", num_selected_test_dets, num_selected_test_dets));

        evecs.reset(new Matrix("U", num_selected_test_dets, nroot));
        evals.reset(new Vector("e", nroot));

        // Form the Hamiltonian matrix
        ForteTimer t_h;
#pragma omp parallel for schedule(dynamic)
        for (size_t I = 0; I < num_selected_test_dets; ++I) {
            boost::tuple<double, int, int, int, int>& determinantI =
                determinants_[selected_test_dets[I]];
            const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
            const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
            const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
            const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
            for (size_t J = I + 1; J < num_selected_test_dets; ++J) {
                boost::tuple<double, int, int, int, int>& determinantJ =
                    determinants_[selected_test_dets[J]];
                const int J_class_a = determinantJ.get<1>(); // std::get<1>(determinantI);
                const int Jsa = determinantJ.get<2>();       // std::get<1>(determinantI);
                const int J_class_b = determinantJ.get<3>(); // std::get<2>(determinantI);
                const int Jsb = determinantJ.get<4>();       // std::get<2>(determinantI);
                const double HIJ =
                    StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),
                                                   vec_bstr_symm_[I_class_b][Isb].get<2>(),
                                                   vec_astr_symm_[J_class_a][Jsa].get<2>(),
                                                   vec_bstr_symm_[J_class_b][Jsb].get<2>());
                H->set(I, J, HIJ);
                H->set(J, I, HIJ);
            }
            H->set(I, I, determinantI.get<0>());
        }
        outfile->Printf("\n  Time spent forming H              = %f s", t_h.elapsed());

        // 4) Diagonalize the Hamiltonian
        ForteTimer t_hdiag_large;
        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
            outfile->Printf("\n  Using the Davidson-Liu algorithm.");
            davidson_liu(H, evals, evecs, nroot);
        } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
            outfile->Printf("\n  Performing full diagonalization.");
            H->diagonalize(evecs, evals);
        }

        outfile->Printf("\n  Time spent diagonalizing H        = %f s", t_hdiag_large.elapsed());
        // 5) Print the energy
        for (int i = 0; i < nroot; ++i) {
            outfile->Printf("\n  Adaptive CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                            evals->get(i), pc_hartree2ev * (evals->get(i) - evals->get(0)));
            outfile->Printf("\n  Adaptive CI Energy + EPT2 Root %3d = %.12f = %.12f + %.12f", i + 1,
                            evals->get(i) + multistate_pt2_energy_correction_[i], evals->get(i),
                            multistate_pt2_energy_correction_[i]);
        }

        size_t num_added = 0;
        for (size_t I = num_selected_dets; I < num_selected_test_dets; ++I) {
            bool keep = true;
            for (int n = 0; n < nroot; ++n) {
                if (std::fabs(evecs->get(I, n)) > t2_threshold_)
                    keep = true;
            }
            if (keep) {
                selected_dets.push_back(selected_test_dets[I]);
                num_added += 1;
            }
        }
        outfile->Printf("\n  After diagonalization added %zu determinants", num_added);
    }

    size_t num_selected_dets = selected_dets.size();

    //    // 4) Print the energy
    //    int nroots_print = std::min(nroot,25);
    //    for (int i = 0; i < nroots_print; ++ i){
    //        outfile->Printf("\n  Small CI Energy Root %3d = %.12f Eh = %8.4f eV",i +
    //        1,evals->get(i),pc_hartree2ev * (evals->get(i) - evals->get(0)));
    //    }
    // 5) Print the energy
    for (int i = 0; i < nroot; ++i) {
        outfile->Printf("\n  Adaptive CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                        evals->get(i), pc_hartree2ev * (evals->get(i) - evals->get(0)));
        outfile->Printf("\n  Adaptive CI Energy + EPT2 Root %3d = %.12f = %.12f + %.12f", i + 1,
                        evals->get(i) + multistate_pt2_energy_correction_[i], evals->get(i),
                        multistate_pt2_energy_correction_[i]);
    }

    double significant_threshold = 0.001;
    double significant_wave_function = 0.95;
    for (int i = 0; i < nroot; ++i) {
        outfile->Printf(
            "\n  The most important determinants (%.0f%% of the wave functions) for root %d:",
            100.0 * significant_wave_function, i + 1);
        // Identify all contributions with |C_J| > significant_threshold
        double** C_mat = evecs->pointer();
        std::vector<std::pair<double, int>> C_J_sorted;
        for (int J = 0; J < num_selected_dets; ++J) {
            if (std::fabs(C_mat[J][i]) > significant_threshold) {
                C_J_sorted.push_back(std::make_pair(std::fabs(C_mat[J][i]), J));
            }
        }
        // Sort them and print
        std::sort(C_J_sorted.begin(), C_J_sorted.end(), std::greater<std::pair<double, int>>());
        double cum_wfn = 0.0;
        for (size_t I = 0, max_I = C_J_sorted.size(); I < max_I; ++I) {
            int J = C_J_sorted[I].second;
            outfile->Printf("\n %3ld   %+9.6f   %9.6f   %.6f   %d", I, C_mat[J][i],
                            C_mat[J][i] * C_mat[J][i], H->get(J, J), J);
            cum_wfn += C_mat[J][i] * C_mat[J][i];
            if (cum_wfn > significant_wave_function)
                break;
        }
    }

    //    int num_roots = options.get_int("NROOT");
    //    outfile->Printf("\n\n  Building a selected Hamiltonian using the criterium by Roth (kappa)
    //    for %d roots",num_roots);
    //    SharedMatrix H = build_select_hamiltonian_roth(options,evals_m,evecs_m);

    //    // 3) Setup stuff necessary to diagonalize the Hamiltonian
    //    int ndets = H->nrow();
    //    SharedMatrix evecs(new Matrix("U",ndets,nroots));
    //    SharedVector evals(new Vector("e",nroots));

    //    // 4) Diagonalize the Hamiltonian
    //    ForteTimer t_hdiag_large;
    //    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
    //        outfile->Printf("\n  Using the Davidson-Liu algorithm.");
    //        davidson_liu(H,evals,evecs,nroots);
    //    }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
    //        outfile->Printf("\n  Performing full diagonalization.");
    //        H->diagonalize(evecs,evals);
    //    }
    //    outfile->Printf("\n  Time spent diagonalizing H        = %f s",t_hdiag_large.elapsed());
    //

    //    // 5) Print the energy
    //    for (int i = 0; i < nroots_print; ++ i){
    //        outfile->Printf("\n  Adaptive CI Energy Root %3d = %.12f Eh = %8.4f eV",i +
    //        1,evals->get(i),pc_hartree2ev * (evals->get(i) - evals->get(0)));
    //        outfile->Printf("\n  Adaptive CI Energy + EPT2 Root %3d = %.12f = %.12f + %.12f",i +
    //        1,evals->get(i) + multistate_pt2_energy_correction_[i],
    //                evals->get(i),multistate_pt2_energy_correction_[i]);
    //    }

    //    // 6) Print the major contributions to the eigenvector
    //    for (int i = 0; i < nroots_print; ++ i){
    //        outfile->Printf("\n  The most important determinants (%.0f%% of the wave functions)
    //        for root %d:",100.0 * significant_wave_function,i + 1);
    //        // Identify all contributions with |C_J| > significant_threshold
    //        double** C_mat = evecs->pointer();
    //        std::vector<std::pair<double,int> > C_J_sorted;
    //        for (int J = 0; J < ndets; ++J){
    //            if (std::fabs(C_mat[J][i]) > significant_threshold){
    //                C_J_sorted.push_back(std::make_pair(std::fabs(C_mat[J][i]),J));
    //            }
    //        }
    //        // Sort them and print
    //        std::sort(C_J_sorted.begin(),C_J_sorted.end(),std::greater<std::pair<double,int> >());
    //        double cum_wfn = 0.0;
    //        for (size_t I = 0, max_I = C_J_sorted.size(); I < max_I; ++I){
    //            int J = C_J_sorted[I].second;
    //            outfile->Printf("\n %3ld   %+9.6f   %9.6f   %.6f   %d",I,C_mat[J][i],C_mat[J][i] *
    //            C_mat[J][i],H->get(J,J),J);
    //            cum_wfn += C_mat[J][i] * C_mat[J][i];
    //            if (cum_wfn > significant_wave_function) break;
    //        }
    //    }
}

void LambdaCI::diagonalize_renormalized_fixed_space(std::shared_ptr<ForteOptions> options) {

    int nroot = options.get_int("NROOT");
    size_t ren_ndets = options.get_int("REN_MAX_NDETS");

    outfile->Printf("\n\n  Diagonalizing the Hamiltonian in the model space");
    outfile->Printf("\n  using a renormalization procedure keeping %zu determinants\n", ren_ndets);

    bool energy_select = (options.get_str("SELECT_TYPE") == "ENERGY");

    size_t renomalization_steps = options.get_int("RENORMALIZATION_STEPS");
    double delta_lambda = space_i_threshold_ / static_cast<double>(renomalization_steps);

    size_t ntot_dets = static_cast<int>(determinants_.size());
    size_t search_from = 0;
    double E0 = determinants_[0].get<0>();
    std::vector<size_t> selected_dets;
    SharedMatrix H;
    SharedMatrix evecs;
    SharedVector evals;

    std::vector<double> ept2(nroot, 0.0);

    size_t dets_per_step = ren_ndets / 2;
    renomalization_steps = ntot_dets / dets_per_step;

    size_t mat_size = dets_per_step + ren_ndets;
    H.reset(new Matrix("Hamiltonian Matrix", mat_size, mat_size));
    evecs.reset(new Matrix("U", mat_size, nroot));
    evals.reset(new Vector("e", nroot));

    outfile->Printf("\n\n  Determinants added each step: %zu", dets_per_step);
    outfile->Printf("\n\n  Number of steps             : %zu", renomalization_steps);

    //    for (int step = 0; step < renomalization_steps; ++step){
    for (int step = 0; step < renomalization_steps + 1; ++step) {
        ForteTimer t_select;

        size_t num_selected_dets = selected_dets.size();

        std::vector<size_t> det_in_range;
        double min_range = determinants_[search_from].get<0>();
        for (size_t I = search_from; I < std::min(search_from + dets_per_step, ntot_dets); ++I) {
            det_in_range.push_back(I);
            double EI = determinants_[I].get<0>();
        }
        size_t num_det_in_range = det_in_range.size();
        search_from += num_det_in_range;
        double max_range = determinants_[search_from - 1].get<0>();

        outfile->Printf("\n\n  Adding dets in the range : [%f,%f), starting from %zu",
                        min_range - determinants_[0].get<0>(),
                        max_range - determinants_[0].get<0>(), search_from);

        //        // Find all the determinants within the range [0,DL)
        //        double min_range = static_cast<double>(step) * delta_lambda;
        //        double max_range = min_range + delta_lambda;

        //        outfile->Printf("\n\n  Finding dets in the range : [%f,%f), starting from
        //        %zu",min_range,max_range,search_from);
        //        std::vector<size_t> det_in_range;
        //        for (int I = search_from; I < ntot_dets; ++I){
        //            double EI = determinants_[I].get<0>();
        //            if (EI - E0 >= max_range){
        //                search_from = I;
        //                break;
        //            }else if(EI - E0 >= min_range){
        //                det_in_range.push_back(I);
        //            }
        //        }
        //        size_t num_det_in_range = det_in_range.size();
        //        outfile->Printf("\n  Found %zu",num_det_in_range);

        std::vector<double> V_q(nroot, 0.0);
        std::vector<double> t_q(nroot, 0.0);
        std::vector<std::pair<double, double>> kappa_q(nroot, std::make_pair(0.0, 0.0));
        std::vector<std::pair<double, double>> chi_q(nroot, std::make_pair(0.0, 0.0));

        std::vector<size_t> selected_test_dets = selected_dets;
        if (step != 0) {
            for (size_t I = 0; I < num_det_in_range; ++I) {
                boost::tuple<double, int, int, int, int>& determinantI =
                    determinants_[det_in_range[I]];
                const double EI = determinantI.get<0>();     // std::get<0>(determinantI);
                const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
                const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
                const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
                const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
                for (int n = 0; n < nroot; ++n) {
                    V_q[n] = 0.0;
                    t_q[n] = 0.0;
                }
                for (int J = 0; J < num_selected_dets; ++J) {
                    boost::tuple<double, int, int, int, int>& determinantJ =
                        determinants_[selected_dets[J]];
                    const double EJ = determinantJ.get<0>();     // std::get<0>(determinantJ);
                    const int J_class_a = determinantJ.get<1>(); // std::get<1>(determinantJ);
                    const int Jsa = determinantJ.get<2>();       // std::get<1>(determinantJ);
                    const int J_class_b = determinantJ.get<3>(); // std::get<2>(determinantJ);
                    const int Jsb = determinantJ.get<4>();       // std::get<2>(determinantJ);
                    const double HIJ =
                        StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),
                                                       vec_bstr_symm_[I_class_b][Isb].get<2>(),
                                                       vec_astr_symm_[J_class_a][Jsa].get<2>(),
                                                       vec_bstr_symm_[J_class_b][Jsb].get<2>());
                    for (int n = 0; n < nroot; ++n) {
                        V_q[n] += evecs->get(J, n) * HIJ;
                    }
                }
                for (int n = 0; n < nroot; ++n) {
                    double kappa = -V_q[n] / (EI - evals->get(n));
                    double chi = -V_q[n] * V_q[n] / (EI - evals->get(n));
                    kappa_q[n] = std::make_pair(std::fabs(kappa), kappa);
                    chi_q[n] = std::make_pair(std::fabs(chi), chi);
                }

                std::pair<double, double> max_kappa =
                    *std::max_element(kappa_q.begin(), kappa_q.end());
                std::pair<double, double> max_chi = *std::max_element(chi_q.begin(), chi_q.end());

                double selection_value = energy_select ? max_chi.first : max_kappa.first;

                // Do not select now, just store the determinant index and the selection criterion
                if (std::fabs(selection_value) > t2_threshold_) {
                    //            #pragma omp critical
                    selected_test_dets.push_back(det_in_range[I]);
                } else {
                    for (int n = 0; n < nroot; ++n)
                        ept2[n] += chi_q[n].second;
                }
            }
        } else {
            for (size_t n = 0, maxn = det_in_range.size(); n < maxn; ++n) {
                selected_test_dets.push_back(det_in_range[n]);
            }
        }
        outfile->Printf(".  Of these %zu passed a screening.  Total dets. %zu.",
                        selected_test_dets.size() - selected_dets.size(),
                        selected_test_dets.size());
        outfile->Printf("\n  Time spent selecting the new dets = %f s", t_select.elapsed());

        multistate_pt2_energy_correction_ = ept2;

        size_t num_selected_test_dets = selected_test_dets.size();

        //        H.reset(new Matrix("Hamiltonian
        //        Matrix",num_selected_test_dets,num_selected_test_dets));
        //        evecs.reset(new Matrix("U",num_selected_test_dets,nroot));
        //        evals.reset(new Vector("e",nroot));

        H->zero();
        evecs->zero();
        evals->zero();
        // Form the Hamiltonian matrix
        ForteTimer t_h;
#pragma omp parallel for schedule(dynamic)
        for (size_t I = 0; I < num_selected_test_dets; ++I) {
            boost::tuple<double, int, int, int, int>& determinantI =
                determinants_[selected_test_dets[I]];
            const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
            const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
            const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
            const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
            for (size_t J = I + 1; J < num_selected_test_dets; ++J) {
                boost::tuple<double, int, int, int, int>& determinantJ =
                    determinants_[selected_test_dets[J]];
                const int J_class_a = determinantJ.get<1>(); // std::get<1>(determinantI);
                const int Jsa = determinantJ.get<2>();       // std::get<1>(determinantI);
                const int J_class_b = determinantJ.get<3>(); // std::get<2>(determinantI);
                const int Jsb = determinantJ.get<4>();       // std::get<2>(determinantI);
                const double HIJ =
                    StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),
                                                   vec_bstr_symm_[I_class_b][Isb].get<2>(),
                                                   vec_astr_symm_[J_class_a][Jsa].get<2>(),
                                                   vec_bstr_symm_[J_class_b][Jsb].get<2>());
                H->set(I, J, HIJ);
                H->set(J, I, HIJ);
            }
            H->set(I, I, determinantI.get<0>());
        }
        outfile->Printf("\n  Time spent forming H              = %f s", t_h.elapsed());

        // 4) Diagonalize the Hamiltonian
        ForteTimer t_hdiag_large;
        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
            outfile->Printf("\n  Using the Davidson-Liu algorithm.");
            davidson_liu(H, evals, evecs, nroot);
        } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
            outfile->Printf("\n  Performing full diagonalization.");
            H->diagonalize(evecs, evals);
        }

        outfile->Printf("\n  Time spent diagonalizing H        = %f s", t_hdiag_large.elapsed());
        // 5) Print the energy
        for (int i = 0; i < nroot; ++i) {
            outfile->Printf("\n  Ren. step CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                            evals->get(i), pc_hartree2ev * (evals->get(i) - evals->get(0)));
            outfile->Printf("\n  Ren. step CI Energy + EPT2 Root %3d = %.12f = %.12f + %.12f",
                            i + 1, evals->get(i) + multistate_pt2_energy_correction_[i],
                            evals->get(i), multistate_pt2_energy_correction_[i]);
        }

        std::vector<std::pair<double, size_t>> dm_det_list;

        for (size_t I = 0; I < num_selected_test_dets; ++I) {
            double max_dm = 0.0;
            for (int n = 0; n < nroot; ++n) {
                max_dm = std::max(max_dm, std::fabs(evecs->get(I, n)));
            }
            dm_det_list.push_back(std::make_pair(max_dm, selected_test_dets[I]));
        }

        std::sort(dm_det_list.begin(), dm_det_list.end());
        std::reverse(dm_det_list.begin(), dm_det_list.end());

        selected_dets.clear();
        for (size_t I = 0, max_I = std::min(dm_det_list.size(), ren_ndets); I < max_I; ++I) {
            selected_dets.push_back(dm_det_list[I].second);
        }
        outfile->Printf("\n  After diagonalization there are %zu determinants",
                        selected_dets.size());
    }

    size_t num_selected_dets = selected_dets.size();

    //    // 4) Print the energy
    //    int nroots_print = std::min(nroot,25);
    //    for (int i = 0; i < nroots_print; ++ i){
    //        outfile->Printf("\n  Small CI Energy Root %3d = %.12f Eh = %8.4f eV",i +
    //        1,evals->get(i),pc_hartree2ev * (evals->get(i) - evals->get(0)));
    //    }
    // 5) Print the energy
    for (int i = 0; i < nroot; ++i) {
        outfile->Printf("\n  Adaptive CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                        evals->get(i), pc_hartree2ev * (evals->get(i) - evals->get(0)));
        outfile->Printf("\n  Adaptive CI Energy + EPT2 Root %3d = %.12f = %.12f + %.12f", i + 1,
                        evals->get(i) + multistate_pt2_energy_correction_[i], evals->get(i),
                        multistate_pt2_energy_correction_[i]);
    }

    double significant_threshold = 0.001;
    double significant_wave_function = 0.95;
    for (int i = 0; i < nroot; ++i) {
        outfile->Printf(
            "\n  The most important determinants (%.0f%% of the wave functions) for root %d:",
            100.0 * significant_wave_function, i + 1);
        // Identify all contributions with |C_J| > significant_threshold
        double** C_mat = evecs->pointer();
        std::vector<std::pair<double, int>> C_J_sorted;
        for (int J = 0; J < num_selected_dets; ++J) {
            if (std::fabs(C_mat[J][i]) > significant_threshold) {
                C_J_sorted.push_back(std::make_pair(std::fabs(C_mat[J][i]), J));
            }
        }
        // Sort them and print
        std::sort(C_J_sorted.begin(), C_J_sorted.end(), std::greater<std::pair<double, int>>());
        double cum_wfn = 0.0;
        for (size_t I = 0, max_I = C_J_sorted.size(); I < max_I; ++I) {
            int J = C_J_sorted[I].second;
            outfile->Printf("\n %3ld   %+9.6f   %9.6f   %.6f   %d", I, C_mat[J][i],
                            C_mat[J][i] * C_mat[J][i], H->get(J, J), J);
            cum_wfn += C_mat[J][i] * C_mat[J][i];
            if (cum_wfn > significant_wave_function)
                break;
        }
    }

    //    int num_roots = options.get_int("NROOT");
    //    outfile->Printf("\n\n  Building a selected Hamiltonian using the criterium by Roth (kappa)
    //    for %d roots",num_roots);
    //    SharedMatrix H = build_select_hamiltonian_roth(options,evals_m,evecs_m);

    //    // 3) Setup stuff necessary to diagonalize the Hamiltonian
    //    int ndets = H->nrow();
    //    SharedMatrix evecs(new Matrix("U",ndets,nroots));
    //    SharedVector evals(new Vector("e",nroots));

    //    // 4) Diagonalize the Hamiltonian
    //    ForteTimer t_hdiag_large;
    //    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
    //        outfile->Printf("\n  Using the Davidson-Liu algorithm.");
    //        davidson_liu(H,evals,evecs,nroots);
    //    }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
    //        outfile->Printf("\n  Performing full diagonalization.");
    //        H->diagonalize(evecs,evals);
    //    }
    //    outfile->Printf("\n  Time spent diagonalizing H        = %f s",t_hdiag_large.elapsed());
    //

    //    // 5) Print the energy
    //    for (int i = 0; i < nroots_print; ++ i){
    //        outfile->Printf("\n  Adaptive CI Energy Root %3d = %.12f Eh = %8.4f eV",i +
    //        1,evals->get(i),pc_hartree2ev * (evals->get(i) - evals->get(0)));
    //        outfile->Printf("\n  Adaptive CI Energy + EPT2 Root %3d = %.12f = %.12f + %.12f",i +
    //        1,evals->get(i) + multistate_pt2_energy_correction_[i],
    //                evals->get(i),multistate_pt2_energy_correction_[i]);
    //    }

    //    // 6) Print the major contributions to the eigenvector
    //    for (int i = 0; i < nroots_print; ++ i){
    //        outfile->Printf("\n  The most important determinants (%.0f%% of the wave functions)
    //        for root %d:",100.0 * significant_wave_function,i + 1);
    //        // Identify all contributions with |C_J| > significant_threshold
    //        double** C_mat = evecs->pointer();
    //        std::vector<std::pair<double,int> > C_J_sorted;
    //        for (int J = 0; J < ndets; ++J){
    //            if (std::fabs(C_mat[J][i]) > significant_threshold){
    //                C_J_sorted.push_back(std::make_pair(std::fabs(C_mat[J][i]),J));
    //            }
    //        }
    //        // Sort them and print
    //        std::sort(C_J_sorted.begin(),C_J_sorted.end(),std::greater<std::pair<double,int> >());
    //        double cum_wfn = 0.0;
    //        for (size_t I = 0, max_I = C_J_sorted.size(); I < max_I; ++I){
    //            int J = C_J_sorted[I].second;
    //            outfile->Printf("\n %3ld   %+9.6f   %9.6f   %.6f   %d",I,C_mat[J][i],C_mat[J][i] *
    //            C_mat[J][i],H->get(J,J),J);
    //            cum_wfn += C_mat[J][i] * C_mat[J][i];
    //            if (cum_wfn > significant_wave_function) break;
    //        }
    //    }
}
}
} // EndNamespaces
