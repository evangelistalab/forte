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

#define BIGNUM 1E100
#define MAXIT 500

int david2(double** A, int N, int M, double* eps, double** v, double cutoff, int print);

namespace psi {
namespace forte {

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

/**
 * Diagonalize the Hamiltonian in the model and intermediate space, the external space is ignored
 */
void LambdaCI::diagonalize_p_space(std::shared_ptr<ForteOptions> options) {
    outfile->Printf("\n\n  Diagonalizing the Hamiltonian in the model + intermediate space\n");

    // 1) Build the Hamiltonian
    ForteTimer t_hbuild;
    SharedMatrix H = build_hamiltonian_parallel(options);
    H->print();
    outfile->Printf("\n  Time spent building H             = %f s", t_hbuild.elapsed());

    // 2) Smooth out the couplings of the model and intermediate space
    ForteTimer t_hsmooth;

    if (options.get_bool("SELECT")) {
        select_important_hamiltonian(H);
    }

    if (options.get_bool("SMOOTH")) {
        smooth_hamiltonian(H);
    }

    outfile->Printf("\n  Time spent smoothing H            = %f s", t_hsmooth.elapsed());

    // 3) Setup stuff necessary to diagonalize the Hamiltonian
    int ndets = H->nrow();
    int nroots = ndets;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
        nroots = std::min(options.get_int("NROOT"), ndets);
    }
    SharedMatrix evecs(new Matrix("U", ndets, nroots));
    SharedVector evals(new Vector("e", nroots));

    // 4) Diagonalize the Hamiltonian
    ForteTimer t_hdiag;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
        outfile->Printf("\n  Using the Davidson-Liu algorithm.");
        davidson_liu(H, evals, evecs, nroots);
    } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
        outfile->Printf("\n  Performing full diagonalization.");
        H->diagonalize(evecs, evals);
    }
    outfile->Printf("\n  Time spent diagonalizing H        = %f s", t_hdiag.elapsed());

    // Set some environment variables
    Process::environment.globals["LAMBDA-CI ENERGY"] = evals->get(options_.get_int("ROOT"));

    print_results(evecs, evals, nroots);
}

void LambdaCI::print_results(SharedMatrix evecs, SharedVector evals, int nroots) {
    std::vector<std::string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet",
                                        "sextet", "septet", "octet", "nonet"});

    int nroots_print = std::min(nroots, 25);

    for (int i = 0; i < nroots_print; ++i) {
        // Find the most significant contributions to this root
        size_t ndets = evecs->nrow();
        std::vector<std::pair<double, int>> C_J_sorted;

        double significant_threshold = 0.0005;
        double significant_wave_function = 0.99999;

        double** C_mat = evecs->pointer();
        for (int J = 0; J < ndets; ++J) {
            if (std::fabs(C_mat[J][i]) > significant_threshold) {
                C_J_sorted.push_back(std::make_pair(std::fabs(C_mat[J][i]), J));
            }
        }

        // Sort them and
        int num_sig = 0;
        std::sort(C_J_sorted.begin(), C_J_sorted.end(), std::greater<std::pair<double, int>>());
        double cum_wfn = 0.0;
        for (size_t I = 0, max_I = C_J_sorted.size(); I < max_I; ++I) {
            int J = C_J_sorted[I].second;
            cum_wfn += C_mat[J][i] * C_mat[J][i];
            num_sig++;
            if (cum_wfn > significant_wave_function)
                break;
        }
        //        outfile->Printf("\nAnalysis on %d out of %zu sorted (%zu
        //        total)",num_sig,C_J_sorted.size(),determinants_.size());

        double norm = 0.0;
        double S2 = 0.0;
        for (int sI = 0; sI < num_sig; ++sI) {
            int I = C_J_sorted[sI].second;
            boost::tuple<double, int, int, int, int>& determinantI = determinants_[I];
            const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
            const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
            const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
            const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
            for (int sJ = 0; sJ < num_sig; ++sJ) {
                int J = C_J_sorted[sJ].second;
                boost::tuple<double, int, int, int, int>& determinantJ = determinants_[J];
                const int J_class_a = determinantJ.get<1>(); // std::get<1>(determinantI);
                const int Jsa = determinantJ.get<2>();       // std::get<1>(determinantI);
                const int J_class_b = determinantJ.get<3>(); // std::get<2>(determinantI);
                const int Jsb = determinantJ.get<4>();       // std::get<2>(determinantI);
                if (std::fabs(C_mat[I][i] * C_mat[J][i]) > 1.0e-12) {
                    const double S2IJ =
                        StringDeterminant::Spin2(vec_astr_symm_[I_class_a][Isa].get<2>(),
                                                 vec_bstr_symm_[I_class_b][Isb].get<2>(),
                                                 vec_astr_symm_[J_class_a][Jsa].get<2>(),
                                                 vec_bstr_symm_[J_class_b][Jsb].get<2>());
                    //                    outfile->Printf("\nI = %d J = %d S2IJ = %.12f, C_I C_J =
                    //                    %.12f",I,J,S2IJ,C_mat[I][i] * C_mat[J][i]);
                    S2 += C_mat[I][i] * S2IJ * C_mat[J][i];
                }
            }
            norm += C_mat[I][i] * C_mat[I][i];
        }
        S2 /= norm;
        double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
        std::string state_label = s2_labels[std::round(S * 2.0)];
        outfile->Printf(
            "\n  Adaptive CI Energy Root %3d = %20.12f Eh = %8.4f eV (S^2 = %5.3f, S = %5.3f, %s)",
            i + 1, evals->get(i), pc_hartree2ev * (evals->get(i) - evals->get(0)), S2, S,
            state_label.c_str());
    }

    // 6) Print the major contributions to the eigenvector
    double significant_threshold = 0.001;
    double significant_wave_function = 0.95;
    for (int i = 0; i < nroots_print; ++i) {
        outfile->Printf(
            "\n\n  => Root %3d <=\n\n  Determinants contribution to %.0f%% of the wave function:",
            i + 1, 100.0 * significant_wave_function);
        // Identify all contributions with |C_J| > significant_threshold
        double** C_mat = evecs->pointer();
        std::vector<std::pair<double, int>> C_J_sorted;
        size_t ndets = evecs->nrow();
        for (int J = 0; J < ndets; ++J) {
            if (std::fabs(C_mat[J][i]) > significant_threshold) {
                C_J_sorted.push_back(std::make_pair(std::fabs(C_mat[J][i]), J));
            }
        }
        // Sort them and print
        std::sort(C_J_sorted.begin(), C_J_sorted.end(), std::greater<std::pair<double, int>>());
        double cum_wfn = 0.0;
        int num_sig = 0;
        for (size_t I = 0, max_I = C_J_sorted.size(); I < max_I; ++I) {
            int J = C_J_sorted[I].second;
            outfile->Printf("\n %3ld   %+9.6f   %9.6f   %d", I, C_mat[J][i],
                            C_mat[J][i] * C_mat[J][i], J);
            cum_wfn += C_mat[J][i] * C_mat[J][i];
            num_sig++;
            if (cum_wfn > significant_wave_function)
                break;
        }

        // Compute the density matrices
        std::fill(Da_.begin(), Da_.end(), 0.0);
        std::fill(Db_.begin(), Db_.end(), 0.0);
        double norm = 0.0;
        for (int I = 0; I < ndets; ++I) {
            boost::tuple<double, int, int, int, int>& determinantI = determinants_[I];
            const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
            const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
            const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
            const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
            double w = C_mat[I][i] * C_mat[I][i];
            if (w > 1.0e-12) {
                StringDeterminant::SlaterdiagOPDM(vec_astr_symm_[I_class_a][Isa].get<2>(),
                                                  vec_bstr_symm_[I_class_b][Isb].get<2>(), Da_, Db_,
                                                  w);
            }
            norm += C_mat[I][i] * C_mat[I][i];
        }
        //        outfile->Printf("\n  2-norm of the CI vector: %f",norm);
        for (int p = 0; p < ncmo_; ++p) {
            Da_[p] /= norm;
            Db_[p] /= norm;
        }
        outfile->Printf("\n\n  Occupation numbers");
        double na = 0.0;
        double nb = 0.0;
        for (int h = 0, p = 0; h < nirrep_; ++h) {
            for (int n = 0; n < ncmopi_[h]; ++n) {
                outfile->Printf("\n  %4d  %1d  %4d   %5.3f    %5.3f", p + 1, h, n, Da_[p], Db_[p]);
                na += Da_[p];
                nb += Db_[p];
                p += 1;
            }
        }
        outfile->Printf("\n  Total number of alpha/beta electrons: %f/%f", na, nb);
    }
}

/**
 * Diagonalize the
 */
void LambdaCI::diagonalize_p_space_lowdin(std::shared_ptr<ForteOptions> options) {
    outfile->Printf("\n\n  Diagonalizing the Hamiltonian in the P space with Lowdin's "
                    "contributions from the Q space\n");
    int root = 0;
    double E = 1.0e100;
    double delta_E = 1.0e10;
    for (int cycle = 0; cycle < 20; ++cycle) {
        // 1) Build the Hamiltonian
        ForteTimer t_hbuild;
        SharedMatrix H = build_hamiltonian_parallel(options);
        outfile->Printf("\n  Time spent building H             = %f s", t_hbuild.elapsed());

        // 2) Add the Lowding contribution to H
        ForteTimer t_hbuild_lowdin;
        lowdin_hamiltonian(H, E);
        outfile->Printf("\n  Time spent on Lowding corrections = %f s", t_hbuild_lowdin.elapsed());

        // 3) Smooth out the couplings of the model and intermediate space
        ForteTimer t_hsmooth;
        smooth_hamiltonian(H);
        outfile->Printf("\n  Time spent smoothing H            = %f s", t_hsmooth.elapsed());

        // 4) Setup stuff necessary to diagonalize the Hamiltonian
        int ndets = H->nrow();
        int nroots = ndets;

        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
            nroots = std::min(options.get_int("NROOT"), ndets);
        }

        SharedMatrix evecs(new Matrix("U", ndets, nroots));
        SharedVector evals(new Vector("e", nroots));

        // 5) Diagonalize the Hamiltonian
        ForteTimer t_hdiag;
        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
            outfile->Printf("\n  Using the Davidson-Liu algorithm.");
            davidson_liu(H, evals, evecs, nroots);
        } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
            outfile->Printf("\n  Performing full diagonalization.");
            H->diagonalize(evecs, evals);
        }
        outfile->Printf("\n  Time spent diagonalizing H        = %f s", t_hdiag.elapsed());

        // 5) Print the energy
        delta_E = evals->get(root) - E;
        E = evals->get(root);
        outfile->Printf("\n  Cycle %3d  E= %.12f  DE = %.12f", cycle, evals->get(root),
                        std::fabs(delta_E) > 10.0 ? 0 : delta_E);

        if (std::fabs(delta_E) < options.get_double("E_CONVERGENCE")) {
            outfile->Printf("\n\n  Adaptive CI Energy Root %3d = %.12f Eh", root + 1,
                            evals->get(root));
            outfile->Printf("\n  Lowdin iterations converged!\n");
            break;
        }
    }
    // evaluate_perturbative_corrections(evals,evecs);
}

/**
 * Build the Hamiltonian matrix from the list of determinants.
 * It assumes that the determinants are stored in increasing energetic order.
 * @param ndets
 * @return a SharedMatrix object that contains the Hamiltonian
 */
SharedMatrix LambdaCI::build_hamiltonian(Options& options) {
    int ntot_dets = static_cast<int>(determinants_.size());

    // the number of determinants used to form the Hamiltonian matrix
    int ndets = 0;

    // Determine the size of the Hamiltonian matrix
    if (options.get_str("H_TYPE") == "FIXED_SIZE") {
        ndets = std::min(options.get_int("NDETS"), ntot_dets);
        outfile->Printf("\n  Building the Hamiltonian using the first %d determinants\n", ndets);
        outfile->Printf("\n  The energy range spanned is [%f,%f]\n", determinants_[0].get<0>(),
                        determinants_[ndets - 1].get<0>());
    } else if (options.get_str("H_TYPE") == "FIXED_ENERGY") {
        outfile->Printf("\n\n  Building the Hamiltonian using determinants with excitation energy "
                        "less than %f Eh",
                        determinant_threshold_);
        int max_ndets_fixed_energy = options.get_int("MAX_NDETS");
        ndets = std::min(max_ndets_fixed_energy, ntot_dets);
        if (ndets == max_ndets_fixed_energy) {
            outfile->Printf(
                "\n\n  WARNING: the number of determinants used to build the Hamiltonian\n"
                "  exceeds the maximum number allowed (%d).  Reducing the size of H.\n\n",
                max_ndets_fixed_energy);
        }
    }

    SharedMatrix H(new Matrix("Hamiltonian Matrix", ndets, ndets));

    // Form the Hamiltonian matrix
    StringDeterminant detI(reference_determinant_);
    StringDeterminant detJ(reference_determinant_);

    for (int I = 0; I < ndets; ++I) {
        boost::tuple<double, int, int, int, int>& determinantI = determinants_[I];
        int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
        int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
        int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
        int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
        detI.set_bits(vec_astr_symm_[I_class_a][Isa].get<2>(),
                      vec_bstr_symm_[I_class_b][Isb].get<2>());
        for (int J = I + 1; J < ndets; ++J) {
            boost::tuple<double, int, int, int, int>& determinantJ = determinants_[J];
            int J_class_a = determinantJ.get<1>(); // std::get<1>(determinantI);
            int Jsa = determinantJ.get<2>();       // std::get<1>(determinantI);
            int J_class_b = determinantJ.get<3>(); // std::get<2>(determinantI);
            int Jsb = determinantJ.get<4>();       // std::get<2>(determinantI);
            detJ.set_bits(vec_astr_symm_[J_class_a][Jsa].get<2>(),
                          vec_bstr_symm_[J_class_b][Jsb].get<2>());
            double HIJ = detI.slater_rules(detJ);
            H->set(I, J, HIJ);
            H->set(J, I, HIJ);
        }
        H->set(I, I, determinantI.get<0>());
    }

    return H;
}

/**
 * Build the Hamiltonian matrix from the list of determinants.
 * It assumes that the determinants are stored in increasing energetic order.
 * @param ndets
 * @return a SharedMatrix object that contains the Hamiltonian
 */
SharedMatrix LambdaCI::build_hamiltonian_parallel(Options& options) {
    int ntot_dets = static_cast<int>(determinants_.size());

    // the number of determinants used to form the Hamiltonian matrix
    int ndets = 0;

    // Determine the size of the Hamiltonian matrix
    if (options.get_str("H_TYPE") == "FIXED_SIZE") {
        ndets = std::min(options.get_int("NDETS"), ntot_dets);
        outfile->Printf("\n  Building the Hamiltonian using the first %d determinants\n", ndets);
        outfile->Printf("\n  The energy range spanned is [%f,%f]\n", determinants_[0].get<0>(),
                        determinants_[ndets - 1].get<0>());
    } else if (options.get_str("H_TYPE") == "FIXED_ENERGY") {
        double E0 = determinants_[0].get<0>();
        for (int I = 0; I < ntot_dets; ++I) {
            double EI = determinants_[I].get<0>();
            if (EI - E0 > space_i_threshold_) {
                break;
            }
            ndets++;
        }
        outfile->Printf("\n\n  Building the Hamiltonian using determinants with excitation energy "
                        "less than %f Eh",
                        space_i_threshold_);
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
            //        for (int J = I + 1; J < ndets; ++J){
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
        const double HII = nuclear_repulsion_energy_ +
                           StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),
                                                          vec_bstr_symm_[I_class_b][Isb].get<2>(),
                                                          vec_astr_symm_[I_class_a][Isa].get<2>(),
                                                          vec_bstr_symm_[I_class_b][Isb].get<2>());
        H->set(I, I, HII);
    }
    return H;
}

void LambdaCI::smooth_hamiltonian(SharedMatrix H) {
    int ndets = H->nrow();

    // Partition the Hamiltonian into main and intermediate model space
    int ndets_model = 0;
    for (int I = 0; I < ndets; ++I) {
        if (H->get(I, I) - H->get(0, 0) > space_m_threshold_) {
            break;
        }
        ndets_model++;
    }
    outfile->Printf("\n\n  The model space of dimension %d will be split into %d (main) + %d "
                    "(intermediate) states",
                    ndets, ndets_model, ndets - ndets_model);
    for (int I = 0; I < ndets; ++I) {
        for (int J = 0; J < ndets; ++J) {
            if (I != J) {
                double HIJ = H->get(I, J);
                double EI = H->get(I, I);
                double EJ = H->get(J, J);
                double EI0 = EI - H->get(0, 0);
                double EJ0 = EJ - H->get(0, 0);
                double factorI = 1.0 - smootherstep(space_m_threshold_, space_i_threshold_, EI0);
                double factorJ = 1.0 - smootherstep(space_m_threshold_, space_i_threshold_, EJ0);
                H->set(I, J, factorI * factorJ * HIJ);
            }
        }
    }
}

void LambdaCI::select_important_hamiltonian(SharedMatrix H) {
    int ndets = H->nrow();

    // Partition the Hamiltonian into main and intermediate model space
    int ndets_model = 0;
    for (int I = 0; I < ndets; ++I) {
        if (H->get(I, I) - H->get(0, 0) > space_m_threshold_) {
            break;
        }
        ndets_model++;
    }
    outfile->Printf("\n\n  The model space of dimension %d will be split into %d (main) + %d "
                    "(intermediate) states",
                    ndets, ndets_model, ndets - ndets_model);

    // Check if any of the determinants in the model space has a large overlap with the model space
    size_t ndiscarded = 0;
    for (int J = ndets_model; J < ndets; ++J) {
        bool is_important = false;
        for (int I = 0; I < ndets_model; ++I) {
            double HIJ = H->get(I, J);
            double EI = H->get(I, I);
            double EJ = H->get(J, J);
            double T2 = std::pow(HIJ / (EI - EJ), 2.0);
            if (T2 > t2_threshold_) {
                is_important = true;
            }
        }
        if (not is_important) {
            // Eliminate this determinant
            for (int I = 0; I < ndets; ++I) {
                H->set(I, J, 0);
                H->set(J, I, 0);
            }
            ndiscarded += 1;
        }
    }
    outfile->Printf("\n\n  %ld states were discarded because the coupling to the main space is "
                    "less than %f muE_h",
                    ndiscarded, t2_threshold_ * 1000000.0);
}

void LambdaCI::evaluate_perturbative_corrections(SharedVector evals, SharedMatrix evecs) {
    int root = 0;
    double E_0 = evals->get(root);

    int ntot_dets = static_cast<int>(determinants_.size());
    int ndets_p = evecs->nrow();

    outfile->Printf("\n\n  Computing a second-order PT correction from the external (%d) to the "
                    "model space (%d)",
                    ntot_dets - ndets_p, ndets_p);

    // Model space - external space 2nd order correction
    double E_2_PQ = 0.0;
#pragma omp parallel for schedule(dynamic)
    for (int A = ndets_p; A < ntot_dets; ++A) {
        boost::tuple<double, int, int, int, int>& determinantA = determinants_[A];
        const double EA = determinantA.get<0>();
        const int A_class_a = determinantA.get<1>(); // std::get<1>(determinantI);
        const int Asa = determinantA.get<2>();       // std::get<1>(determinantI);
        const int A_class_b = determinantA.get<3>(); // std::get<2>(determinantI);
        const int Asb = determinantA.get<4>();       // std::get<2>(determinantI);
        double coupling = 0.0;
        for (int I = 0; I < ndets_p; ++I) {
            boost::tuple<double, int, int, int, int>& determinantI = determinants_[I];
            const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
            const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
            const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
            const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
            const double HIA = StringDeterminant::SlaterRules(
                vec_astr_symm_[I_class_a][Isa].get<2>(), vec_bstr_symm_[I_class_b][Isb].get<2>(),
                vec_astr_symm_[A_class_a][Asa].get<2>(), vec_bstr_symm_[A_class_b][Asb].get<2>());
            coupling += evecs->get(I, root) * HIA;
        }
        E_2_PQ -= coupling * coupling / (EA - E_0);
    }
    outfile->Printf("\n\n Adaptive CI + PT2 Energy Root %3d = %.12f Eh", root + 1, E_0 + E_2_PQ);
}

void LambdaCI::lowdin_hamiltonian(SharedMatrix H, double E) {
    int ntot_dets = static_cast<int>(determinants_.size());
    int ndets_p = H->nrow();

    outfile->Printf("\n\n  Computing a second-order PT correction from the external (%d) to the "
                    "model space (%d)",
                    ntot_dets - ndets_p, ndets_p);

#pragma omp parallel for schedule(dynamic)
    for (int I = 0; I < ndets_p; ++I) {
        boost::tuple<double, int, int, int, int>& determinantI = determinants_[I];
        const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
        for (int J = I; J < ndets_p; ++J) {
            boost::tuple<double, int, int, int, int>& determinantJ = determinants_[J];
            const int J_class_a = determinantJ.get<1>(); // std::get<1>(determinantI);
            const int Jsa = determinantJ.get<2>();       // std::get<1>(determinantI);
            const int J_class_b = determinantJ.get<3>(); // std::get<2>(determinantI);
            const int Jsb = determinantJ.get<4>();       // std::get<2>(determinantI);
            double coupling = 0.0;
            for (int A = ndets_p; A < ntot_dets; ++A) {
                boost::tuple<double, int, int, int, int>& determinantA = determinants_[A];
                const double EA = determinantA.get<0>();
                const int A_class_a = determinantA.get<1>(); // std::get<1>(determinantI);
                const int Asa = determinantA.get<2>();       // std::get<1>(determinantI);
                const int A_class_b = determinantA.get<3>(); // std::get<2>(determinantI);
                const int Asb = determinantA.get<4>();       // std::get<2>(determinantI);
                const double HIA =
                    StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),
                                                   vec_bstr_symm_[I_class_b][Isb].get<2>(),
                                                   vec_astr_symm_[A_class_a][Asa].get<2>(),
                                                   vec_bstr_symm_[A_class_b][Asb].get<2>());
                const double HJA =
                    StringDeterminant::SlaterRules(vec_astr_symm_[J_class_a][Jsa].get<2>(),
                                                   vec_bstr_symm_[J_class_b][Jsb].get<2>(),
                                                   vec_astr_symm_[A_class_a][Asa].get<2>(),
                                                   vec_bstr_symm_[A_class_b][Asb].get<2>());
                coupling += HIA * HJA / (E - EA);
            }
            double HIJ = H->get(I, J);
            H->set(I, J, HIJ + coupling);
            H->set(J, I, HIJ + coupling);
        }
    }
}

/**
 * Diagonalize the Hamiltonian in the model and intermediate space, the external space is ignored
 */
void LambdaCI::diagonalize_p_space_direct(std::shared_ptr<ForteOptions> options) {
    outfile->Printf("\n\n  Diagonalizing the Hamiltonian in the model + intermediate space\n");

    // 1) Build the Hamiltonian
    ForteTimer t_hbuild;
    std::vector<std::vector<std::pair<int, double>>> H_sparse = build_hamiltonian_direct(options);

    outfile->Printf("\n  Time spent building H             = %f s", t_hbuild.elapsed());

    // 2) Smooth out the couplings of the model and intermediate space
    ForteTimer t_hsmooth;

    ////    if (options.get_bool("SELECT")){
    ////        select_important_hamiltonian(H);
    ////    }

    ////    if (options.get_bool("SMOOTH")){
    ////        smooth_hamiltonian(H);
    ////    }

    //    outfile->Printf("\n  Time spent smoothing H            = %f s",t_hsmooth.elapsed());
    //

    // 3) Setup stuff necessary to diagonalize the Hamiltonian
    int ndets = H_sparse.size();
    int nroots = ndets;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
        nroots = std::min(options.get_int("NROOT"), ndets);
    }
    SharedMatrix evecs(new Matrix("U", ndets, nroots));
    SharedVector evals(new Vector("e", nroots));

    // 4) Diagonalize the Hamiltonian
    ForteTimer t_hdiag;
    outfile->Printf("\n  Using the Davidson-Liu algorithm.");
    davidson_liu_sparse(H_sparse, evals, evecs, nroots);
    outfile->Printf("\n  Time spent diagonalizing H        = %f s", t_hdiag.elapsed());

    // Set some environment variables
    Process::environment.globals["LAMBDA-CI ENERGY"] = evals->get(options.get_int("ROOT"));

    print_results(evecs, evals, nroots);
}

/**
 * Build the Hamiltonian matrix from the list of determinants.
 * It assumes that the determinants are stored in increasing energetic order.
 * @param ndets
 * @return a SharedMatrix object that contains the Hamiltonian
 */
std::vector<std::vector<std::pair<int, double>>>
LambdaCI::build_hamiltonian_direct(Options& options) {
    int ntot_dets = static_cast<int>(determinants_.size());

    // the number of determinants used to form the Hamiltonian matrix
    int ndets = 0;

    // Determine the size of the Hamiltonian matrix
    if (options.get_str("H_TYPE") == "FIXED_SIZE") {
        ndets = std::min(options.get_int("NDETS"), ntot_dets);
        outfile->Printf("\n  Building the Hamiltonian using the first %d determinants\n", ndets);
        outfile->Printf("\n  The energy range spanned is [%f,%f]\n", determinants_[0].get<0>(),
                        determinants_[ndets - 1].get<0>());
    } else if (options.get_str("H_TYPE") == "FIXED_ENERGY") {
        double E0 = determinants_[0].get<0>();
        for (int I = 0; I < ntot_dets; ++I) {
            double EI = determinants_[I].get<0>();
            if (EI - E0 > space_i_threshold_) {
                break;
            }
            ndets++;
        }
        outfile->Printf("\n\n  Building the Hamiltonian using determinants with excitation energy "
                        "less than %f Eh",
                        space_i_threshold_);
        outfile->Printf("\n  This requires a total of %d determinants", ndets);
    }

    std::vector<std::vector<std::pair<int, double>>> H_sparse;

    size_t num_nonzero = 0;
    // Form the Hamiltonian matrix
    //    #pragma omp parallel for schedule(dynamic)
    for (int I = 0; I < ndets; ++I) {
        std::vector<std::pair<int, double>> H_row;
        boost::tuple<double, int, int, int, int>& determinantI = determinants_[I];
        const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
        const double HII = nuclear_repulsion_energy_ +
                           StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),
                                                          vec_bstr_symm_[I_class_b][Isb].get<2>(),
                                                          vec_astr_symm_[I_class_a][Isa].get<2>(),
                                                          vec_bstr_symm_[I_class_b][Isb].get<2>());
        H_row.push_back(std::make_pair(I, determinantI.get<0>() /*HII*/));
        for (int J = 0; J < ndets; ++J) {
            if (I != J) {
                boost::tuple<double, int, int, int, int>& determinantJ = determinants_[J];
                const int J_class_a = determinantJ.get<1>(); // std::get<1>(determinantI);
                const int Jsa = determinantJ.get<2>();       // std::get<1>(determinantI);
                const int J_class_b = determinantJ.get<3>(); // std::get<2>(determinantI);
                const int Jsb = determinantJ.get<4>();       // std::get<2>(determinantI);
                const double HIJ =
                    StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),
                                                   vec_bstr_symm_[I_class_b][Isb].get<2>(),
                                                   vec_astr_symm_[J_class_a][Jsa].get<2>(),
                                                   vec_bstr_symm_[J_class_b][Jsb].get<2>());
                if (std::fabs(HIJ) >= 1.0e-12) {
                    H_row.push_back(std::make_pair(J, HIJ));
                    num_nonzero += 1;
                }
            }
        }
        //        #pragma omp critical
        H_sparse.push_back(H_row);
    }

    outfile->Printf("\n  %zu nonzero elements out of %zu (%e)", num_nonzero, size_t(ndets * ndets),
                    double(num_nonzero) / double(ndets * ndets));
    return H_sparse;
}

void LambdaCI::davidson_liu(SharedMatrix H, SharedVector Eigenvalues, SharedMatrix Eigenvectors,
                            int nroots) {
    david2(H->pointer(), H->nrow(), nroots, Eigenvalues->pointer(), Eigenvectors->pointer(),
           1.0e-10, 0);
}

bool LambdaCI::davidson_liu_sparse(std::vector<std::vector<std::pair<int, double>>> H_sparse,
                                   SharedVector Eigenvalues, SharedMatrix Eigenvectors,
                                   int nroots) {
    //    david2(H->pointer(),H->nrow(),nroots,Eigenvalues->pointer(),Eigenvectors->pointer(),1.0e-10,0);

    int N = static_cast<int>(H_sparse.size());
    int M = nroots;
    double* eps = Eigenvalues->pointer();
    double** v = Eigenvectors->pointer();
    double cutoff = 1.0e-10;
    int print = 0;

    int i, j, k, L, I;
    double minimum;
    int min_pos, numf, iter, *conv, converged, maxdim, skip_check;
    int *small2big, init_dim;
    int smart_guess = 1;
    double *Adiag, **b, **bnew, **sigma, **G;
    double *lambda, **alpha, **f, *lambda_old;
    double norm, denom, diff;

    maxdim = 8 * M; // Set it back to the original value (8)

    b = block_matrix(maxdim, N);          /* current set of guess vectors,
                               stored by row */
    bnew = block_matrix(M, N);            /* guess vectors formed from old vectors,
                               stored by row*/
    sigma = block_matrix(N, maxdim);      /* sigma vectors, stored by column */
    G = block_matrix(maxdim, maxdim);     /* Davidson mini-Hamitonian */
    f = block_matrix(maxdim, N);          /* residual eigenvectors, stored by row */
    alpha = block_matrix(maxdim, maxdim); /* eigenvectors of G */
    lambda = init_array(maxdim);          /* eigenvalues of G */
    lambda_old = init_array(maxdim);      /* approximate roots from previous
                               iteration */

    if (smart_guess) { /* Use eigenvectors of a sub-matrix as initial guesses */

        if (N > 7 * M)
            init_dim = 7 * M;
        else
            init_dim = M;
        Adiag = init_array(N);
        small2big = init_int_array(7 * M);
        for (i = 0; i < N; i++) {
            Adiag[i] = H_sparse[i][0].second;
        }
        for (i = 0; i < init_dim; i++) {
            minimum = Adiag[0];
            min_pos = 0;
            for (j = 1; j < N; j++)
                if (Adiag[j] < minimum) {
                    minimum = Adiag[j];
                    min_pos = j;
                    small2big[i] = j;
                }

            Adiag[min_pos] = BIGNUM;
            lambda_old[i] = minimum;
        }
        for (i = 0; i < init_dim; i++) {
            for (j = 0; j < init_dim; j++) {
                std::vector<std::pair<int, double>>& H_row = H_sparse[small2big[i]];
                size_t maxc = H_row.size();
                for (size_t c = 0; c < maxc; ++c) {
                    if (H_row[c].first == small2big[j]) {
                        G[i][j] = H_row[c].second;
                        break;
                    }
                }
                // G[i][j] = A[small2big[i]][small2big[j]];
            }
        }

        sq_rsp(init_dim, init_dim, G, lambda, 1, alpha, 1e-12);

        for (i = 0; i < init_dim; i++) {
            for (j = 0; j < init_dim; j++)
                b[i][small2big[j]] = alpha[j][i];
        }

        free(Adiag);
        free(small2big);
    } else { /* Use unit vectors as initial guesses */
        Adiag = init_array(N);
        //        for(i=0; i < N; i++) { Adiag[i] = A[i][i]; }
        for (i = 0; i < N; i++) {
            Adiag[i] = H_sparse[i][0].second;
        }
        for (i = 0; i < M; i++) {
            minimum = Adiag[0];
            min_pos = 0;
            for (j = 1; j < N; j++)
                if (Adiag[j] < minimum) {
                    minimum = Adiag[j];
                    min_pos = j;
                }

            b[i][min_pos] = 1.0;
            Adiag[min_pos] = BIGNUM;
            lambda_old[i] = minimum;
        }
        free(Adiag);
    }

    L = init_dim;
    iter = 0;
    converged = 0;
    conv = init_int_array(M); /* boolean array for convergence of each
                       root */
    while (converged < M && iter < MAXIT) {

        skip_check = 0;
        if (print)
            printf("\niter = %d\n", iter);

        /* form mini-matrix */
        for (int J = 0; J < N; ++J) {
            for (int r = 0; r < maxdim; ++r) {
                sigma[J][r] = 0.0;
            }
            std::vector<std::pair<int, double>>& H_row = H_sparse[J];
            size_t maxc = H_row.size();
            for (int c = 0; c < maxc; ++c) {
                int K = H_row[c].first;
                double HJK = H_row[c].second;
                for (int r = 0; r < L; ++r) {
                    sigma[J][r] += HJK * b[r][K];
                }
            }
        }
        //        C_DGEMM('n','t', N, L, N, 1.0, &(A[0][0]), N, &(b[0][0]), N,
        //                0.0, &(sigma[0][0]), maxdim);
        C_DGEMM('n', 'n', L, L, N, 1.0, &(b[0][0]), N, &(sigma[0][0]), maxdim, 0.0, &(G[0][0]),
                maxdim);

        /* diagonalize mini-matrix */
        sq_rsp(L, L, G, lambda, 1, alpha, 1e-12);

        /* form preconditioned residue vectors */
        for (k = 0; k < M; k++)
            for (I = 0; I < N; I++) {
                f[k][I] = 0.0;
                for (i = 0; i < L; i++) {
                    f[k][I] += alpha[i][k] * (sigma[I][i] - lambda[k] * b[i][I]);
                }
                denom = lambda[k] - H_sparse[I][0].second; // A[I][I];
                if (std::fabs(denom) > 1e-6)
                    f[k][I] /= denom;
                else
                    f[k][I] = 0.0;
            }

        /* normalize each residual */
        for (k = 0; k < M; k++) {
            norm = 0.0;
            for (I = 0; I < N; I++) {
                norm += f[k][I] * f[k][I];
            }
            norm = sqrt(norm);
            for (I = 0; I < N; I++) {
                f[k][I] /= norm;
                if (norm > 1e-6)
                    f[k][I] /= norm;
                else
                    f[k][I] = 0.0;
            }
        }

        /* schmidt orthogonalize the f[k] against the set of b[i] and add
           new vectors */
        for (k = 0, numf = 0; k < M; k++)
            if (schmidt_add(b, L, N, f[k])) {
                L++;
                numf++;
            }

        /* If L is close to maxdim, collapse to one guess per root */
        if (maxdim - L < M) {
            if (print) {
                printf("Subspace too large: maxdim = %d, L = %d\n", maxdim, L);
                printf("Collapsing eigenvectors.\n");
            }
            for (i = 0; i < M; i++) {
                memset((void*)bnew[i], 0, N * sizeof(double));
                for (j = 0; j < L; j++) {
                    for (k = 0; k < N; k++) {
                        bnew[i][k] += alpha[j][i] * b[j][k];
                    }
                }
            }

            /* orthonormalize the new vectors */
            /* copy new vectors into place */
            for (i = 0; i < M; i++) {
                norm = 0.0;
                // Project out the orthonormal vectors
                for (j = 0; j < i; ++j) {
                    double proj = 0.0;
                    for (k = 0; k < N; k++) {
                        proj += b[j][k] * bnew[i][k];
                    }
                    for (k = 0; k < N; k++) {
                        bnew[i][k] -= proj * b[j][k];
                    }
                }
                for (k = 0; k < N; k++) {
                    norm += bnew[i][k] * bnew[i][k];
                }
                norm = std::sqrt(norm);
                for (k = 0; k < N; k++) {
                    b[i][k] = bnew[i][k] / norm;
                }
            }

            skip_check = 1;

            L = M;
        }

        /* check convergence on all roots */
        if (!skip_check) {
            converged = 0;
            zero_int_array(conv, M);
            if (print) {
                printf("Root      Eigenvalue       Delta  Converged?\n");
                printf("---- -------------------- ------- ----------\n");
            }
            for (k = 0; k < M; k++) {
                diff = std::fabs(lambda[k] - lambda_old[k]);
                if (diff < cutoff) {
                    conv[k] = 1;
                    converged++;
                }
                lambda_old[k] = lambda[k];
                if (print) {
                    printf("%3d  %20.14f %4.3e    %1s\n", k, lambda[k], diff,
                           conv[k] == 1 ? "Y" : "N");
                }
            }
        }

        iter++;
    }

    /* generate final eigenvalues and eigenvectors */
    // if(converged == M) {
    for (i = 0; i < M; i++) {
        eps[i] = lambda[i];
        for (I = 0; I < N; I++) {
            v[I][i] = 0.0;
        }
        for (j = 0; j < L; j++) {
            for (I = 0; I < N; I++) {
                v[I][i] += alpha[j][i] * b[j][I];
            }
        }
        // Normalize v
        norm = 0.0;
        for (I = 0; I < N; I++) {
            norm += v[I][i] * v[I][i];
        }
        norm = std::sqrt(norm);
        for (I = 0; I < N; I++) {
            v[I][i] /= norm;
        }
    }
    if (print)
        printf("Davidson algorithm converged in %d iterations.\n", iter);
    //    }

    free(conv);
    free_block(b);
    free_block(bnew);
    free_block(sigma);
    free_block(G);
    free_block(f);
    free_block(alpha);
    free(lambda);
    free(lambda_old);

    return converged;
}
}
} // EndNamespaces

// SharedMatrix R(new Matrix("R",n,L));
// SharedMatrix G(new Matrix("G",L,L));
// SharedVector Rho(new Matrix("G Eigenvalues",L,L));
// Hb->gemm(false,false,1.0,H,b,0.0);
// G->gemm(true,false,1.0,b,Hb,0.0);

// Rho->set_diagonal(rho);
// R->gemm(false,false,1.0,Hb,alpha,0.0);
// R->gemm(false,false,-1.0,b,alpha,1.0);

//// BEGIN DEBUGGING
//// Write the Hamiltonian to disk
// outfile->Printf("\n\n  WRITING FILE TO DISK...");
//
// std::ofstream of("ham.dat", ios::binary | ios::out);
// of.write(reinterpret_cast<char*>(&ndets),sizeof(int));
// double** H_mat = H->pointer();
// of.write(reinterpret_cast<char*>(&(H_mat[0][0])),ndets * ndets * sizeof(double));
// of.close();
// outfile->Printf(" DONE.");
//
//// END DEBUGGING
