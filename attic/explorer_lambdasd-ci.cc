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
void LambdaCI::lambda_mrcisd(std::shared_ptr<ForteOptions> options) {
    outfile->Printf("\n\n  Lambda-MRCISD");

    int nroot = options.get_int("NROOT");

    double selection_threshold = t2_threshold_;

    outfile->Printf("\n\n  Diagonalizing the Hamiltonian in the model space (Lambda = %.2f Eh)\n",
                    space_m_threshold_);

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

    SharedMatrix H;
    SharedMatrix evecs;
    SharedVector evals;

    // 1) Build the Hamiltonian using the StringDeterminant representation
    std::vector<StringDeterminant> ref_space;
    std::map<StringDeterminant, int> ref_space_map;

    for (size_t I = 0, maxI = determinants_.size(); I < maxI; ++I) {
        boost::tuple<double, int, int, int, int>& determinantI = determinants_[I];
        const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
        StringDeterminant det(vec_astr_symm_[I_class_a][Isa].get<2>(),
                              vec_bstr_symm_[I_class_b][Isb].get<2>());
        ref_space.push_back(det);
        ref_space_map[det] = 1;
    }

    size_t dim_ref_space = ref_space.size();

    outfile->Printf("\n  The model space contains %zu determinants", dim_ref_space);

    H.reset(new Matrix("Hamiltonian Matrix", dim_ref_space, dim_ref_space));
    evecs.reset(new Matrix("U", dim_ref_space, nroot));
    evals.reset(new Vector("e", nroot));

    ForteTimer t_h_build;
#pragma omp parallel for schedule(dynamic)
    for (size_t I = 0; I < dim_ref_space; ++I) {
        const StringDeterminant& detI = ref_space[I];
        for (size_t J = I; J < dim_ref_space; ++J) {
            const StringDeterminant& detJ = ref_space[J];
            double HIJ = detI.slater_rules(detJ);
            H->set(I, J, HIJ);
            H->set(J, I, HIJ);
        }
    }
    outfile->Printf("\n  Time spent building H               = %f s", t_h_build.elapsed());

    // 4) Diagonalize the Hamiltonian
    ForteTimer t_hdiag_large;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
        outfile->Printf("\n  Using the Davidson-Liu algorithm.");
        davidson_liu(H, evals, evecs, nroot);
    } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
        outfile->Printf("\n  Performing full diagonalization.");
        H->diagonalize(evecs, evals);
    }

    outfile->Printf("\n  Time spent diagonalizing H          = %f s", t_hdiag_large.elapsed());

    // 5) Print the energy
    for (int i = 0; i < nroot; ++i) {
        outfile->Printf("\n  Ren. step CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                        evals->get(i) + nuclear_repulsion_energy_,
                        pc_hartree2ev * (evals->get(i) - evals->get(0)));
        //        outfile->Printf("\n  Ren. step CI Energy + EPT2 Root %3d = %.12f = %.12f +
        //        %.12f",i + 1,evals->get(i) + multistate_pt2_energy_correction_[i],
        //                evals->get(i),multistate_pt2_energy_correction_[i]);
    }

    int nmo = reference_determinant_.nmo();
    //    size_t nfrzc = frzc_.size();
    //    size_t nfrzv = frzv_.size();

    std::vector<int> aocc(nalpha_);
    std::vector<int> bocc(nbeta_);
    std::vector<int> avir(ncmo_ - nalpha_);
    std::vector<int> bvir(ncmo_ - nbeta_);

    int noalpha = nalpha_;
    int nobeta = nbeta_;
    int nvalpha = ncmo_ - nalpha_;
    int nvbeta = ncmo_ - nbeta_;

    // Find the SD space out of the reference
    std::vector<StringDeterminant> sd_dets_vec;
    std::map<StringDeterminant, int> new_dets_map;
    ForteTimer t_ms_build;

    for (size_t I = 0, max_I = ref_space_map.size(); I < max_I; ++I) {
        const StringDeterminant& det = ref_space[I];
        for (int p = 0, i = 0, a = 0; p < ncmo_; ++p) {
            if (det.get_alfa_bit(p)) {
                //                if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                aocc[i] = p;
                i++;
                //                }
            } else {
                //                if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                avir[a] = p;
                a++;
                //                }
            }
        }
        for (int p = 0, i = 0, a = 0; p < ncmo_; ++p) {
            if (det.get_beta_bit(p)) {
                //                if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                bocc[i] = p;
                i++;
                //                }
            } else {
                //                if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                bvir[a] = p;
                a++;
                //                }
            }
        }

        // Generate aa excitations
        for (int i = 0; i < noalpha; ++i) {
            int ii = aocc[i];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_) {
                    StringDeterminant new_det(det);
                    new_det.set_alfa_bit(ii, false);
                    new_det.set_alfa_bit(aa, true);
                    if (ref_space_map.find(new_det) == ref_space_map.end()) {
                        sd_dets_vec.push_back(new_det);
                    }
                }
            }
        }

        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_) {
                    StringDeterminant new_det(det);
                    new_det.set_beta_bit(ii, false);
                    new_det.set_beta_bit(aa, true);
                    if (ref_space_map.find(new_det) == ref_space_map.end()) {
                        sd_dets_vec.push_back(new_det);
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
                             mo_symmetry_[bb]) == wavefunction_symmetry_) {
                            StringDeterminant new_det(det);
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(jj, false);
                            new_det.set_alfa_bit(aa, true);
                            new_det.set_alfa_bit(bb, true);
                            if (ref_space_map.find(new_det) == ref_space_map.end()) {
                                sd_dets_vec.push_back(new_det);
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
                             mo_symmetry_[bb]) == wavefunction_symmetry_) {
                            StringDeterminant new_det(det);
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_beta_bit(jj, false);
                            new_det.set_alfa_bit(aa, true);
                            new_det.set_beta_bit(bb, true);
                            if (ref_space_map.find(new_det) == ref_space_map.end()) {
                                sd_dets_vec.push_back(new_det);
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
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                             mo_symmetry_[bb]) == wavefunction_symmetry_) {
                            StringDeterminant new_det(det);
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(jj, false);
                            new_det.set_beta_bit(aa, true);
                            new_det.set_beta_bit(bb, true);
                            if (ref_space_map.find(new_det) == ref_space_map.end()) {
                                sd_dets_vec.push_back(new_det);
                            }
                        }
                    }
                }
            }
        }
    }

    outfile->Printf("\n  The SD excitation space has dimension: %zu", sd_dets_vec.size());

    ForteTimer t_ms_screen;

    sort(sd_dets_vec.begin(), sd_dets_vec.end());
    sd_dets_vec.erase(unique(sd_dets_vec.begin(), sd_dets_vec.end()), sd_dets_vec.end());

    outfile->Printf("\n  The SD excitation space has dimension: %zu (unique)", sd_dets_vec.size());
    outfile->Printf("\n  Time spent building the model space = %f s", t_ms_build.elapsed());

    // This will contain all the determinants
    std::vector<StringDeterminant> ref_sd_dets;
    for (size_t J = 0, max_J = ref_space.size(); J < max_J; ++J) {
        ref_sd_dets.push_back(ref_space[J]);
    }

    // Check the coupling between the reference and the SD space
    std::vector<std::pair<double, size_t>> new_dets_importance_vec;

    std::vector<double> V(nroot, 0.0);
    std::vector<std::pair<double, double>> C1(nroot, std::make_pair(0.0, 0.0));
    std::vector<std::pair<double, double>> E2(nroot, std::make_pair(0.0, 0.0));
    std::vector<double> ept2(nroot, 0.0);

    double aimed_selection_sum = 0.0;

    for (size_t I = 0, max_I = sd_dets_vec.size(); I < max_I; ++I) {
        double EI = sd_dets_vec[I].energy();
        for (int n = 0; n < nroot; ++n) {
            V[n] = 0;
        }
        for (size_t J = 0, max_J = ref_space.size(); J < max_J; ++J) {
            double HIJ = sd_dets_vec[I].slater_rules(ref_space[J]);
            for (int n = 0; n < nroot; ++n) {
                V[n] += evecs->get(J, n) * HIJ;
            }
        }
        for (int n = 0; n < nroot; ++n) {
            double C1_I = -V[n] / (EI - evals->get(n));
            double E2_I = -V[n] * V[n] / (EI - evals->get(n));
            C1[n] = std::make_pair(std::fabs(C1_I), C1_I);
            E2[n] = std::make_pair(std::fabs(E2_I), E2_I);
        }

        //        double C1 = std::fabs(V / (EI - evals->get(0)));
        //        double E2 = std::fabs(V * V / (EI - evals->get(0)));

        //        double select_value = (energy_select ? E2 : C1);

        std::pair<double, double> max_C1 = *std::max_element(C1.begin(), C1.end());
        std::pair<double, double> max_E2 = *std::max_element(E2.begin(), E2.end());

        double select_value = energy_select ? max_E2.first : max_C1.first;

        // Do not select now, just store the determinant index and the selection criterion
        if (aimed_selection) {
            if (energy_select) {
                new_dets_importance_vec.push_back(std::make_pair(select_value, I));
                aimed_selection_sum += select_value;
            } else {
                new_dets_importance_vec.push_back(std::make_pair(select_value * select_value, I));
                aimed_selection_sum += select_value * select_value;
            }
        } else {
            if (std::fabs(select_value) > t2_threshold_) {
                new_dets_importance_vec.push_back(std::make_pair(select_value, I));
            } else {
                for (int n = 0; n < nroot; ++n)
                    ept2[n] += E2[n].second;
            }
        }
    }

    if (aimed_selection) {
        std::sort(new_dets_importance_vec.begin(), new_dets_importance_vec.end());
        std::reverse(new_dets_importance_vec.begin(), new_dets_importance_vec.end());
        size_t maxI = new_dets_importance_vec.size();
        outfile->Printf("\n  The SD space will be generated using the aimed scheme (%s)",
                        energy_select ? "energy" : "amplitude");
        outfile->Printf("\n  Initial value of sigma in the aimed selection = %24.14f",
                        aimed_selection_sum);
        for (size_t I = 0; I < maxI; ++I) {
            if (aimed_selection_sum > t2_threshold_) {
                ref_sd_dets.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
                aimed_selection_sum -= new_dets_importance_vec[I].first;
            } else {
                break;
            }
        }
        outfile->Printf("\n  Final value of sigma in the aimed selection   = %24.14f",
                        aimed_selection_sum);
        outfile->Printf("\n  Selected %zu determinants", ref_sd_dets.size() - ref_space.size());
    } else {
        outfile->Printf("\n  The SD space will be generated by screening (%s)",
                        energy_select ? "energy" : "amplitude");
        size_t maxI = new_dets_importance_vec.size();
        for (size_t I = 0; I < maxI; ++I) {
            ref_sd_dets.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
        }
    }

    multistate_pt2_energy_correction_ = ept2;

    size_t dim_ref_sd_dets = ref_sd_dets.size();

    outfile->Printf("\n  After screening the Lambda-CISD space contains %zu determinants",
                    dim_ref_sd_dets);
    outfile->Printf("\n  Time spent screening the model space = %f s", t_ms_screen.elapsed());

    evecs.reset(new Matrix("U", dim_ref_sd_dets, nroot));
    evals.reset(new Vector("e", nroot));
    // Full algorithm
    if (options.get_str("ENERGY_TYPE") == "LMRCISD") {
        H.reset(new Matrix("Hamiltonian Matrix", dim_ref_sd_dets, dim_ref_sd_dets));

        ForteTimer t_h_build2;
#pragma omp parallel for schedule(dynamic)
        for (size_t I = 0; I < dim_ref_sd_dets; ++I) {
            const StringDeterminant& detI = ref_sd_dets[I];
            for (size_t J = I; J < dim_ref_sd_dets; ++J) {
                const StringDeterminant& detJ = ref_sd_dets[J];
                double HIJ = detI.slater_rules(detJ);
                if (I == J)
                    HIJ += nuclear_repulsion_energy_;
                H->set(I, J, HIJ);
                H->set(J, I, HIJ);
            }
        }
        outfile->Printf("\n  Time spent building H               = %f s", t_h_build2.elapsed());

        // 4) Diagonalize the Hamiltonian
        ForteTimer t_hdiag_large2;
        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
            outfile->Printf("\n  Using the Davidson-Liu algorithm.");
            davidson_liu(H, evals, evecs, nroot);
        } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
            outfile->Printf("\n  Performing full diagonalization.");
            H->diagonalize(evecs, evals);
        }

        outfile->Printf("\n  Time spent diagonalizing H          = %f s", t_hdiag_large2.elapsed());

    }
    // Sparse algorithm
    else {
        ForteTimer t_h_build2;
        std::vector<std::vector<std::pair<int, double>>> H_sparse;

        size_t num_nonzero = 0;
        // Form the Hamiltonian matrix
        for (size_t I = 0; I < dim_ref_sd_dets; ++I) {
            std::vector<std::pair<int, double>> H_row;
            const StringDeterminant& detI = ref_sd_dets[I];
            double HII = detI.slater_rules(detI) + nuclear_repulsion_energy_;
            H_row.push_back(std::make_pair(int(I), HII));
            for (size_t J = 0; J < dim_ref_sd_dets; ++J) {
                if (I != J) {
                    const StringDeterminant& detJ = ref_sd_dets[J];
                    double HIJ = detI.slater_rules(detJ);
                    if (std::fabs(HIJ) >= 1.0e-12) {
                        H_row.push_back(std::make_pair(int(J), HIJ));
                        num_nonzero += 1;
                    }
                }
            }
            H_sparse.push_back(H_row);
        }

        outfile->Printf("\n  %ld nonzero elements out of %ld (%e)", num_nonzero,
                        size_t(dim_ref_sd_dets * dim_ref_sd_dets),
                        double(num_nonzero) / double(dim_ref_sd_dets * dim_ref_sd_dets));
        outfile->Printf("\n  Time spent building H               = %f s", t_h_build2.elapsed());

        // 4) Diagonalize the Hamiltonian
        ForteTimer t_hdiag_large2;
        outfile->Printf("\n  Using the Davidson-Liu algorithm.");
        davidson_liu_sparse(H_sparse, evals, evecs, nroot);
        outfile->Printf("\n  Time spent diagonalizing H          = %f s", t_hdiag_large2.elapsed());
    }
    outfile->Printf("\n  Finished building H");

    outfile->Printf("\n\n  => Lambda+SD-CI <=\n");
    // 5) Print the energy
    for (int i = 0; i < nroot; ++i) {
        outfile->Printf("\n  Adaptive CI Energy Root %3d        = %20.12f Eh = %8.4f eV", i + 1,
                        evals->get(i), pc_hartree2ev * (evals->get(i) - evals->get(0)));
        outfile->Printf("\n  Adaptive CI Energy + EPT2 Root %3d = %20.12f Eh = %8.4f eV", i + 1,
                        evals->get(i) + multistate_pt2_energy_correction_[i],
                        pc_hartree2ev *
                            (evals->get(i) - evals->get(0) + multistate_pt2_energy_correction_[i] -
                             multistate_pt2_energy_correction_[0]));
    }

    // Set some environment variables
    Process::environment.globals["LAMBDA+SD-CI ENERGY"] = evals->get(options_.get_int("ROOT"));

    print_results_lambda_sd_ci(ref_sd_dets, evecs, evals, nroot);
}

void LambdaCI::print_results_lambda_sd_ci(vector<StringDeterminant>& determinants,
                                          SharedMatrix evecs, SharedVector evals, int nroots) {
    std::vector<std::string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet",
                                        "sextet", "septet", "octet", "nonet"});

    int nroots_print = std::min(nroots, 25);

    outfile->Printf("\n\n  => Summary of results <=\n");

    for (int i = 0; i < nroots_print; ++i) {
        // Find the most significant contributions to this root
        size_t ndets = evecs->nrow();
        std::vector<std::pair<double, int>> C_J_sorted;

        double significant_threshold = 0.00001;
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
        outfile->Printf("\nAnalysis on %d out of %zu sorted (%zu total)", num_sig,
                        C_J_sorted.size(), determinants.size());

        double norm = 0.0;
        double S2 = 0.0;
        for (int sI = 0; sI < num_sig; ++sI) {
            int I = C_J_sorted[sI].second;
            for (int sJ = 0; sJ < num_sig; ++sJ) {
                int J = C_J_sorted[sJ].second;
                if (std::fabs(C_mat[I][i] * C_mat[J][i]) > 1.0e-12) {
                    const double S2IJ = determinants[I].spin2(determinants[J]);
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
            double w = C_mat[I][i] * C_mat[I][i];
            if (w > 1.0e-12) {
                determinants[I].diag_opdm(Da_, Db_, w);
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
void LambdaCI::lambda_mrcis(std::shared_ptr<ForteOptions> options) {
    outfile->Printf("\n\n  Lambda-MRCIS");

    int nroot = options.get_int("NROOT");

    double selection_threshold = t2_threshold_;

    outfile->Printf("\n\n  Diagonalizing the Hamiltonian in the model space (Lambda = %.2f Eh)\n",
                    space_m_threshold_);

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

    SharedMatrix H;
    SharedMatrix evecs;
    SharedVector evals;

    // 1) Build the Hamiltonian using the StringDeterminant representation
    std::vector<StringDeterminant> ref_space;
    std::map<StringDeterminant, int> ref_space_map;

    for (size_t I = 0, maxI = determinants_.size(); I < maxI; ++I) {
        boost::tuple<double, int, int, int, int>& determinantI = determinants_[I];
        const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
        StringDeterminant det(vec_astr_symm_[I_class_a][Isa].get<2>(),
                              vec_bstr_symm_[I_class_b][Isb].get<2>());
        ref_space.push_back(det);
        ref_space_map[det] = 1;
    }

    size_t dim_ref_space = ref_space.size();

    outfile->Printf("\n  The model space contains %zu determinants", dim_ref_space);

    H.reset(new Matrix("Hamiltonian Matrix", dim_ref_space, dim_ref_space));
    evecs.reset(new Matrix("U", dim_ref_space, nroot));
    evals.reset(new Vector("e", nroot));

    ForteTimer t_h_build;
#pragma omp parallel for schedule(dynamic)
    for (size_t I = 0; I < dim_ref_space; ++I) {
        const StringDeterminant& detI = ref_space[I];
        for (size_t J = I; J < dim_ref_space; ++J) {
            const StringDeterminant& detJ = ref_space[J];
            double HIJ = detI.slater_rules(detJ);
            H->set(I, J, HIJ);
            H->set(J, I, HIJ);
        }
    }
    outfile->Printf("\n  Time spent building H               = %f s", t_h_build.elapsed());

    // 4) Diagonalize the Hamiltonian
    ForteTimer t_hdiag_large;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
        outfile->Printf("\n  Using the Davidson-Liu algorithm.");
        davidson_liu(H, evals, evecs, nroot);
    } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
        outfile->Printf("\n  Performing full diagonalization.");
        H->diagonalize(evecs, evals);
    }

    outfile->Printf("\n  Time spent diagonalizing H          = %f s", t_hdiag_large.elapsed());

    // 5) Print the energy
    for (int i = 0; i < nroot; ++i) {
        outfile->Printf("\n  Ren. step CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                        evals->get(i) + nuclear_repulsion_energy_,
                        pc_hartree2ev * (evals->get(i) - evals->get(0)));
        //        outfile->Printf("\n  Ren. step CI Energy + EPT2 Root %3d = %.12f = %.12f +
        //        %.12f",i + 1,evals->get(i) + multistate_pt2_energy_correction_[i],
        //                evals->get(i),multistate_pt2_energy_correction_[i]);
    }

    int nmo = reference_determinant_.nmo();
    //    size_t nfrzc = frzc_.size();
    //    size_t nfrzv = frzv_.size();

    std::vector<int> aocc(nalpha_);
    std::vector<int> bocc(nbeta_);
    std::vector<int> avir(ncmo_ - nalpha_);
    std::vector<int> bvir(ncmo_ - nbeta_);

    int noalpha = nalpha_;
    int nobeta = nbeta_;
    int nvalpha = ncmo_ - nalpha_;
    int nvbeta = ncmo_ - nbeta_;

    // Find the SD space out of the reference
    std::vector<StringDeterminant> sd_dets_vec;
    std::map<StringDeterminant, int> new_dets_map;
    ForteTimer t_ms_build;

    for (size_t I = 0, max_I = ref_space_map.size(); I < max_I; ++I) {
        const StringDeterminant& det = ref_space[I];
        for (int p = 0, i = 0, a = 0; p < ncmo_; ++p) {
            if (det.get_alfa_bit(p)) {
                //                if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                aocc[i] = p;
                i++;
                //                }
            } else {
                //                if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                avir[a] = p;
                a++;
                //                }
            }
        }
        for (int p = 0, i = 0, a = 0; p < ncmo_; ++p) {
            if (det.get_beta_bit(p)) {
                //                if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                bocc[i] = p;
                i++;
                //                }
            } else {
                //                if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                bvir[a] = p;
                a++;
                //                }
            }
        }

        // Generate alpha single excitations
        for (int i = 0; i < noalpha; ++i) {
            int ii = aocc[i];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_) {
                    StringDeterminant new_det(det);
                    new_det.set_alfa_bit(ii, false);
                    new_det.set_alfa_bit(aa, true);
                    if (ref_space_map.find(new_det) == ref_space_map.end()) {
                        sd_dets_vec.push_back(new_det);
                    }
                }
            }
        }
        // Generate beta single excitations
        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_) {
                    StringDeterminant new_det(det);
                    new_det.set_beta_bit(ii, false);
                    new_det.set_beta_bit(aa, true);
                    if (ref_space_map.find(new_det) == ref_space_map.end()) {
                        sd_dets_vec.push_back(new_det);
                    }
                }
            }
        }
    }

    outfile->Printf("\n  The S excitation space has dimension: %zu", sd_dets_vec.size());

    ForteTimer t_ms_screen;

    sort(sd_dets_vec.begin(), sd_dets_vec.end());
    sd_dets_vec.erase(unique(sd_dets_vec.begin(), sd_dets_vec.end()), sd_dets_vec.end());

    outfile->Printf("\n  The S excitation space has dimension: %zu (unique)", sd_dets_vec.size());
    outfile->Printf("\n  Time spent building the model space = %f s", t_ms_build.elapsed());

    // This will contain all the determinants
    std::vector<StringDeterminant> ref_sd_dets;
    for (size_t J = 0, max_J = ref_space.size(); J < max_J; ++J) {
        ref_sd_dets.push_back(ref_space[J]);
    }

    // Check the coupling between the reference and the SD space
    std::vector<std::pair<double, size_t>> new_dets_importance_vec;

    std::vector<double> V(nroot, 0.0);
    std::vector<std::pair<double, double>> C1(nroot, std::make_pair(0.0, 0.0));
    std::vector<std::pair<double, double>> E2(nroot, std::make_pair(0.0, 0.0));
    std::vector<double> ept2(nroot, 0.0);

    double aimed_selection_sum = 0.0;

    for (size_t I = 0, max_I = sd_dets_vec.size(); I < max_I; ++I) {
        double EI = sd_dets_vec[I].energy();
        for (int n = 0; n < nroot; ++n) {
            V[n] = 0;
        }
        for (size_t J = 0, max_J = ref_space.size(); J < max_J; ++J) {
            double HIJ = sd_dets_vec[I].slater_rules(ref_space[J]);
            for (int n = 0; n < nroot; ++n) {
                V[n] += evecs->get(J, n) * HIJ;
            }
        }
        for (int n = 0; n < nroot; ++n) {
            double C1_I = -V[n] / (EI - evals->get(n));
            double E2_I = -V[n] * V[n] / (EI - evals->get(n));
            C1[n] = std::make_pair(std::fabs(C1_I), C1_I);
            E2[n] = std::make_pair(std::fabs(E2_I), E2_I);
        }

        //        double C1 = std::fabs(V / (EI - evals->get(0)));
        //        double E2 = std::fabs(V * V / (EI - evals->get(0)));

        //        double select_value = (energy_select ? E2 : C1);

        std::pair<double, double> max_C1 = *std::max_element(C1.begin(), C1.end());
        std::pair<double, double> max_E2 = *std::max_element(E2.begin(), E2.end());

        double select_value = energy_select ? max_E2.first : max_C1.first;

        // Do not select now, just store the determinant index and the selection criterion
        if (aimed_selection) {
            if (energy_select) {
                new_dets_importance_vec.push_back(std::make_pair(select_value, I));
                aimed_selection_sum += select_value;
            } else {
                new_dets_importance_vec.push_back(std::make_pair(select_value * select_value, I));
                aimed_selection_sum += select_value * select_value;
            }
        } else {
            if (std::fabs(select_value) >= t2_threshold_) {
                new_dets_importance_vec.push_back(std::make_pair(select_value, I));
            } else {
                for (int n = 0; n < nroot; ++n)
                    ept2[n] += E2[n].second;
            }
        }
    }

    if (aimed_selection) {
        std::sort(new_dets_importance_vec.begin(), new_dets_importance_vec.end());
        std::reverse(new_dets_importance_vec.begin(), new_dets_importance_vec.end());
        size_t maxI = new_dets_importance_vec.size();
        outfile->Printf("\n  The S space will be generated using the aimed scheme (%s)",
                        energy_select ? "energy" : "amplitude");
        outfile->Printf("\n  Initial value of sigma in the aimed selection = %24.14f",
                        aimed_selection_sum);
        for (size_t I = 0; I < maxI; ++I) {
            if (aimed_selection_sum > t2_threshold_) {
                ref_sd_dets.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
                aimed_selection_sum -= new_dets_importance_vec[I].first;
            } else {
                break;
            }
        }
        outfile->Printf("\n  Final value of sigma in the aimed selection   = %24.14f",
                        aimed_selection_sum);
        outfile->Printf("\n  Selected %zu determinants", ref_sd_dets.size() - ref_space.size());
    } else {
        outfile->Printf("\n  The S space will be generated by screening (%s)",
                        energy_select ? "energy" : "amplitude");
        size_t maxI = new_dets_importance_vec.size();
        for (size_t I = 0; I < maxI; ++I) {
            ref_sd_dets.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
        }
    }

    multistate_pt2_energy_correction_ = ept2;

    size_t dim_ref_sd_dets = ref_sd_dets.size();

    outfile->Printf("\n  After screening, the Lambda+S-CI space contains %zu determinants",
                    dim_ref_sd_dets);
    outfile->Printf("\n  Time spent screening the model space = %f s", t_ms_screen.elapsed());

    evecs.reset(new Matrix("U", dim_ref_sd_dets, nroot));
    evals.reset(new Vector("e", nroot));
    // Full algorithm
    if (options.get_str("ENERGY_TYPE") == "LMRCIS") {
        H.reset(new Matrix("Hamiltonian Matrix", dim_ref_sd_dets, dim_ref_sd_dets));

        ForteTimer t_h_build2;
#pragma omp parallel for schedule(dynamic)
        for (size_t I = 0; I < dim_ref_sd_dets; ++I) {
            const StringDeterminant& detI = ref_sd_dets[I];
            for (size_t J = I; J < dim_ref_sd_dets; ++J) {
                const StringDeterminant& detJ = ref_sd_dets[J];
                double HIJ = detI.slater_rules(detJ);
                if (I == J)
                    HIJ += nuclear_repulsion_energy_;
                H->set(I, J, HIJ);
                H->set(J, I, HIJ);
            }
        }
        outfile->Printf("\n  Time spent building H               = %f s", t_h_build2.elapsed());

        // 4) Diagonalize the Hamiltonian
        ForteTimer t_hdiag_large2;
        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
            outfile->Printf("\n  Using the Davidson-Liu algorithm.");
            davidson_liu(H, evals, evecs, nroot);
        } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
            outfile->Printf("\n  Performing full diagonalization.");
            H->diagonalize(evecs, evals);
        }

        outfile->Printf("\n  Time spent diagonalizing H          = %f s", t_hdiag_large2.elapsed());

    }
    // Sparse algorithm
    else {
        ForteTimer t_h_build2;
        std::vector<std::vector<std::pair<int, double>>> H_sparse;

        size_t num_nonzero = 0;
        // Form the Hamiltonian matrix
        for (size_t I = 0; I < dim_ref_sd_dets; ++I) {
            std::vector<std::pair<int, double>> H_row;
            const StringDeterminant& detI = ref_sd_dets[I];
            double HII = detI.slater_rules(detI) + nuclear_repulsion_energy_;
            H_row.push_back(std::make_pair(int(I), HII));
            for (size_t J = 0; J < dim_ref_sd_dets; ++J) {
                if (I != J) {
                    const StringDeterminant& detJ = ref_sd_dets[J];
                    double HIJ = detI.slater_rules(detJ);
                    if (std::fabs(HIJ) >= 1.0e-12) {
                        H_row.push_back(std::make_pair(int(J), HIJ));
                        num_nonzero += 1;
                    }
                }
            }
            H_sparse.push_back(H_row);
        }

        outfile->Printf("\n  %ld nonzero elements out of %ld (%e)", num_nonzero,
                        size_t(dim_ref_sd_dets * dim_ref_sd_dets),
                        double(num_nonzero) / double(dim_ref_sd_dets * dim_ref_sd_dets));
        outfile->Printf("\n  Time spent building H               = %f s", t_h_build2.elapsed());

        // 4) Diagonalize the Hamiltonian
        ForteTimer t_hdiag_large2;
        outfile->Printf("\n  Using the Davidson-Liu algorithm.");
        davidson_liu_sparse(H_sparse, evals, evecs, nroot);
        outfile->Printf("\n  Time spent diagonalizing H          = %f s", t_hdiag_large2.elapsed());
    }
    outfile->Printf("\n  Finished building H");

    outfile->Printf("\n\n  => Lambda+S-CI <=\n");
    // 5) Print the energy
    for (int i = 0; i < nroot; ++i) {
        outfile->Printf("\n  Adaptive CI Energy Root %3d        = %20.12f Eh = %8.4f eV", i + 1,
                        evals->get(i), pc_hartree2ev * (evals->get(i) - evals->get(0)));
        outfile->Printf("\n  Adaptive CI Energy + EPT2 Root %3d = %20.12f Eh = %8.4f eV", i + 1,
                        evals->get(i) + multistate_pt2_energy_correction_[i],
                        pc_hartree2ev *
                            (evals->get(i) - evals->get(0) + multistate_pt2_energy_correction_[i] -
                             multistate_pt2_energy_correction_[0]));
    }

    // Set some environment variables
    Process::environment.globals["LAMBDA+S-CI ENERGY"] = evals->get(options_.get_int("ROOT"));

    print_results_lambda_sd_ci(ref_sd_dets, evecs, evals, nroot);
}
}
} // EndNamespaces
