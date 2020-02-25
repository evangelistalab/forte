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

#include "adaptive-ci.h"

#include <algorithm>
#include <boost/unordered_map.hpp>
#include <cmath>
#include <functional>

#include "mini-boost/boost/format.hpp"
#include "mini-boost/boost/timer.hpp"

#include <libqt/qt.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <libciomr/libciomr.h>
//#include <libqt/qt.h>

#include "cartographer.h"
#include "dynamic_bitset_determinant.h"
#include "lambda-ci.h"
#include "string_determinant.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

/**
 * Diagonalize the
 */
void LambdaCI::iterative_adaptive_mrcisd(std::shared_ptr<ForteOptions> options) {
    ForteTimer t_iamrcisd;

    outfile->Printf("\n\n  Iterative Adaptive MRCISD");

    int nroot = options.get_int("NROOT");

    double tau_p = options.get_double("TAUP");
    double tau_q = options.get_double("TAUQ");

    outfile->Printf("\n\n  TAU_P = %f Eh", tau_p);
    outfile->Printf("\n  TAU_Q = %.12f Eh\n", tau_q);

    double ia_mrcisd_threshold = 1.0e-9;

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

    std::vector<StringDeterminant> ref_space;
    std::map<StringDeterminant, int> ref_space_map;
    ref_space.push_back(reference_determinant_);
    ref_space_map[reference_determinant_] = 1;

    double old_energy = reference_determinant_.energy() + nuclear_repulsion_energy_;
    double new_energy = 0.0;

    int maxcycle = 20;
    for (int cycle = 0; cycle < maxcycle; ++cycle) {
        // Build the Hamiltonian in the P space

        size_t dim_ref_space = ref_space.size();

        outfile->Printf("\n\n  Cycle %3d. The model space contains %zu determinants", cycle,
                        dim_ref_space);

        int num_ref_roots = (cycle == 0 ? std::max(nroot, 4) : nroot);

        H.reset(new Matrix("Hamiltonian Matrix", dim_ref_space, dim_ref_space));
        evecs.reset(new Matrix("U", dim_ref_space, num_ref_roots));
        evals.reset(new Vector("e", num_ref_roots));

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

        // Diagonalize the Hamiltonian
        ForteTimer t_hdiag_large;
        if (cycle == 0) {
            outfile->Printf("\n  Performing full diagonalization.");
            H->diagonalize(evecs, evals);
        } else {
            if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
                outfile->Printf("\n  Using the Davidson-Liu algorithm.");
                davidson_liu(H, evals, evecs, num_ref_roots);
            } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
                outfile->Printf("\n  Performing full diagonalization.");
                H->diagonalize(evecs, evals);
            }
        }

        outfile->Printf("\n  Time spent diagonalizing H          = %f s", t_hdiag_large.elapsed());

        // Print the energy
        for (int i = 0; i < num_ref_roots; ++i) {
            outfile->Printf("\n  P-space CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                            evals->get(i) + nuclear_repulsion_energy_,
                            pc_hartree2ev * (evals->get(i) - evals->get(0)));
        }

        int nmo = reference_determinant_.nmo();
        //        size_t nfrzc = frzc_.size();
        //        size_t nfrzv = frzv_.size();

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
                    //                    if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                    aocc[i] = p;
                    i++;
                    //                    }
                } else {
                    //                    if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                    avir[a] = p;
                    a++;
                    //                    }
                }
            }
            for (int p = 0, i = 0, a = 0; p < ncmo_; ++p) {
                if (det.get_beta_bit(p)) {
                    //                    if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                    bocc[i] = p;
                    i++;
                    //                    }
                } else {
                    //                    if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                    bvir[a] = p;
                    a++;
                    //                    }
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
                            if ((mo_symmetry_[ii] ^
                                 (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) ==
                                wavefunction_symmetry_) {
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
                            if ((mo_symmetry_[ii] ^
                                 (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) ==
                                wavefunction_symmetry_) {
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
                            if ((mo_symmetry_[ii] ^
                                 (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) ==
                                wavefunction_symmetry_) {
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
        outfile->Printf("\n  Time spent building the model space = %f s", t_ms_build.elapsed());

        // Remove the duplicate determinants
        ForteTimer t_ms_unique;
        sort(sd_dets_vec.begin(), sd_dets_vec.end());
        sd_dets_vec.erase(unique(sd_dets_vec.begin(), sd_dets_vec.end()), sd_dets_vec.end());
        outfile->Printf("\n  The SD excitation space has dimension: %zu (unique)",
                        sd_dets_vec.size());
        outfile->Printf("\n  Time spent to eliminate duplicate   = %f s", t_ms_unique.elapsed());

        ForteTimer t_ms_screen;
        // This will contain all the determinants
        std::vector<StringDeterminant> ref_sd_dets;

        // Add  the P-space determinants
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
            //#pragma omp parallel for schedule(dynamic)
            for (size_t J = 0, max_J = ref_space.size(); J < max_J; ++J) {
                double HIJ = sd_dets_vec[I].slater_rules(ref_space[J]);
                //#pragma omp critical
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

            std::pair<double, double> max_C1 = *std::max_element(C1.begin(), C1.end());
            std::pair<double, double> max_E2 = *std::max_element(E2.begin(), E2.end());

            double select_value = energy_select ? max_E2.first : max_C1.first;

            // Do not select now, just store the determinant index and the selection criterion
            if (aimed_selection) {
                if (energy_select) {
                    new_dets_importance_vec.push_back(std::make_pair(select_value, I));
                    aimed_selection_sum += select_value;
                } else {
                    new_dets_importance_vec.push_back(
                        std::make_pair(select_value * select_value, I));
                    aimed_selection_sum += select_value * select_value;
                }
            } else {
                if (std::fabs(select_value) > tau_q) {
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

        outfile->Printf("\n  After screening the ia-MRCISD space contains %zu determinants",
                        dim_ref_sd_dets);
        outfile->Printf("\n  Time spent screening the model space = %f s", t_ms_screen.elapsed());

        evecs.reset(new Matrix("U", dim_ref_sd_dets, nroot));
        evals.reset(new Vector("e", nroot));
        // Full algorithm
        if (options.get_str("ENERGY_TYPE") == "IMRCISD") {
            H.reset(new Matrix("Hamiltonian Matrix", dim_ref_sd_dets, dim_ref_sd_dets));

            ForteTimer t_h_build2;
#pragma omp parallel for schedule(dynamic)
            for (size_t I = 0; I < dim_ref_sd_dets; ++I) {
                const StringDeterminant& detI = ref_sd_dets[I];
                for (size_t J = I; J < dim_ref_sd_dets; ++J) {
                    const StringDeterminant& detJ = ref_sd_dets[J];
                    double HIJ = detI.slater_rules(detJ);
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

            outfile->Printf("\n  Time spent diagonalizing H          = %f s",
                            t_hdiag_large2.elapsed());

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
                double HII = detI.slater_rules(detI);
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
            outfile->Printf("\n  Time spent diagonalizing H          = %f s",
                            t_hdiag_large2.elapsed());
        }

        //
        for (int i = 0; i < nroot; ++i) {
            outfile->Printf("\n  Adaptive CI Energy Root %3d        = %.12f Eh = %8.4f eV", i + 1,
                            evals->get(i) + nuclear_repulsion_energy_,
                            pc_hartree2ev * (evals->get(i) - evals->get(0)));
            outfile->Printf("\n  Adaptive CI Energy + EPT2 Root %3d = %.12f Eh = %8.4f eV", i + 1,
                            evals->get(i) + nuclear_repulsion_energy_ +
                                multistate_pt2_energy_correction_[i],
                            pc_hartree2ev * (evals->get(i) - evals->get(0) +
                                             multistate_pt2_energy_correction_[i] -
                                             multistate_pt2_energy_correction_[0]));
        }

        // Select the new reference space
        ref_space.clear();
        ref_space_map.clear();

        new_energy = evals->get(0) + nuclear_repulsion_energy_;

        if (std::fabs(new_energy - old_energy) < ia_mrcisd_threshold) {
            break;
        }
        old_energy = new_energy;

        std::vector<std::pair<double, size_t>> dm_det_list;

        for (size_t I = 0; I < dim_ref_sd_dets; ++I) {
            double max_dm = 0.0;
            for (int n = 0; n < nroot; ++n) {
                max_dm = std::max(max_dm, std::fabs(evecs->get(I, n)));
            }
            dm_det_list.push_back(std::make_pair(max_dm, I));
        }

        std::sort(dm_det_list.begin(), dm_det_list.end());
        std::reverse(dm_det_list.begin(), dm_det_list.end());

        // Decide which will go in ref_space
        for (size_t I = 0; I < dim_ref_sd_dets; ++I) {
            if (dm_det_list[I].first > tau_p) {
                ref_space.push_back(ref_sd_dets[dm_det_list[I].second]);
                ref_space_map[ref_sd_dets[dm_det_list[I].second]] = 1;
            }
        }
        //        unordered_map<std::vector<bool>,int> a_str_hash;
        //        unordered_map<std::vector<bool>,int> b_str_hash;

        //        ForteTimer t_stringify;
        //        for (size_t I = 0; I < dim_ref_sd_dets; ++I){
        //            const StringDeterminant& detI = ref_sd_dets[I];
        //            const std::vector<bool> a_str = detI.get_alfa_bits_vector_bool();
        //            const std::vector<bool> b_str = detI.get_beta_bits_vector_bool();
        //            a_str_hash[a_str] = 1;
        //            b_str_hash[b_str] = 1;
        //        }
        //        outfile->Printf("\n  Size of the @MRCISD space: %zu",dim_ref_sd_dets);
        //        outfile->Printf("\n  Size of the alpha strings: %zu",a_str_hash.size());
        //        outfile->Printf("\n  Size of the beta  strings: %zu",b_str_hash.size());

        //        outfile->Printf("\n\n  Time to stringify: %f s",t_stringify.elapsed());
    }

    for (int i = 0; i < nroot; ++i) {
        outfile->Printf("\n  * IA-MRCISD total energy (%3d)        = %.12f Eh = %8.4f eV", i + 1,
                        evals->get(i) + nuclear_repulsion_energy_,
                        pc_hartree2ev * (evals->get(i) - evals->get(0)));
        outfile->Printf(
            "\n  * IA-MRCISD total energy (%3d) + EPT2 = %.12f Eh = %8.4f eV", i + 1,
            evals->get(i) + nuclear_repulsion_energy_ + multistate_pt2_energy_correction_[i],
            pc_hartree2ev * (evals->get(i) - evals->get(0) + multistate_pt2_energy_correction_[i] -
                             multistate_pt2_energy_correction_[0]));
    }

    outfile->Printf("\n\n  iterative_adaptive_mrcisd        ran in %f s", t_iamrcisd.elapsed());
}

/**
 * Diagonalize the
 */
void LambdaCI::iterative_adaptive_mrcisd_bitset(std::shared_ptr<ForteOptions> options) {
    ForteTimer t_iamrcisd;
    outfile->Printf("\n\n  Iterative Adaptive MRCISD");

    int nroot = options.get_int("NROOT");

    double tau_p = options.get_double("TAUP");
    double tau_q = options.get_double("TAUQ");

    outfile->Printf("\n\n  TAU_P = %f Eh", tau_p);
    outfile->Printf("\n  TAU_Q = %.12f Eh\n", tau_q);

    double ia_mrcisd_threshold = 1.0e-9;

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

    std::vector<DynamicBitsetDeterminant> ref_space;
    std::map<DynamicBitsetDeterminant, int> ref_space_map;

    // Copy the determinants from the previous Lambda-CI
    for (size_t I = 0, maxI = determinants_.size(); I < maxI; ++I) {
        boost::tuple<double, int, int, int, int>& determinantI = determinants_[I];
        const int I_class_a = determinantI.get<1>(); // std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();       // std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); // std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();       // std::get<2>(determinantI);
        StringDeterminant det(vec_astr_symm_[I_class_a][Isa].get<2>(),
                              vec_bstr_symm_[I_class_b][Isb].get<2>());

        std::vector<bool> alfa_bits = det.get_alfa_bits_vector_bool();
        std::vector<bool> beta_bits = det.get_beta_bits_vector_bool();
        DynamicBitsetDeterminant bs_det(alfa_bits, beta_bits);
        ref_space.push_back(bs_det);
        ref_space_map[bs_det] = 1;
    }

    size_t dim_ref_space = ref_space.size();

    outfile->Printf("\n  The model space contains %zu determinants", dim_ref_space);

    //    H.reset(new Matrix("Hamiltonian Matrix",dim_ref_space,dim_ref_space));
    //    evecs.reset(new Matrix("U",dim_ref_space,nroot));
    //    evals.reset(new Vector("e",nroot));

    //    ForteTimer t_h_build;
    //#pragma omp parallel for schedule(dynamic)
    //    for (size_t I = 0; I < dim_ref_space; ++I){
    //        const StringDeterminant& detI = ref_space[I];
    //        for (size_t J = I; J < dim_ref_space; ++J){
    //            const StringDeterminant& detJ = ref_space[J];
    //            double HIJ = detI.slater_rules(detJ);
    //            H->set(I,J,HIJ);
    //            H->set(J,I,HIJ);
    //        }
    //    }
    //    outfile->Printf("\n  Time spent building H               = %f s",t_h_build.elapsed());
    //

    //    // 4) Diagonalize the Hamiltonian
    //    ForteTimer t_hdiag_large;
    //    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
    //        outfile->Printf("\n  Using the Davidson-Liu algorithm.");
    //        davidson_liu(H,evals,evecs,nroot);
    //    }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
    //        outfile->Printf("\n  Performing full diagonalization.");
    //        H->diagonalize(evecs,evals);
    //    }

    //    outfile->Printf("\n  Time spent diagonalizing H          = %f s",t_hdiag_large.elapsed());
    //

    //    std::vector<bool> ref_abits = reference_determinant_.get_alfa_bits_vector_bool();
    //    std::vector<bool> ref_bbits = reference_determinant_.get_beta_bits_vector_bool();
    //    DynamicBitsetDeterminant bs_ref_(ref_abits,ref_bbits);
    //    ref_space.push_back(bs_ref_);
    //    ref_space_map[bs_ref_] = 1;

    std::vector<std::vector<double>> energy_history;

    double old_energy = reference_determinant_.energy() + nuclear_repulsion_energy_;
    double new_energy = 0.0;

    int maxcycle = 20;
    for (int cycle = 0; cycle < maxcycle; ++cycle) {
        // Build the Hamiltonian in the P space

        dim_ref_space = ref_space.size();

        outfile->Printf("\n\n  Cycle %3d. The model space contains %zu determinants", cycle,
                        dim_ref_space);
        //        outfile->Printf("\n  Solving for %d roots",num_ref_roots);

        H.reset(new Matrix("Hamiltonian Matrix", dim_ref_space, dim_ref_space));

        ForteTimer t_h_build;
#pragma omp parallel for schedule(dynamic)
        for (size_t I = 0; I < dim_ref_space; ++I) {
            const DynamicBitsetDeterminant& detI = ref_space[I];
            for (size_t J = I; J < dim_ref_space; ++J) {
                const DynamicBitsetDeterminant& detJ = ref_space[J];
                double HIJ = detI.slater_rules(detJ);
                H->set(I, J, HIJ);
                H->set(J, I, HIJ);
            }
        }
        outfile->Printf("\n  Time spent building H               = %f s", t_h_build.elapsed());

        // Be careful, we might not have as many reference dets as roots (just in the first cycle)
        int num_ref_roots = std::min(nroot, int(dim_ref_space));

        // Diagonalize the Hamiltonian
        evecs.reset(new Matrix("U", dim_ref_space, num_ref_roots));
        evals.reset(new Vector("e", num_ref_roots));
        outfile->Printf("\n  Using the Davidson-Liu algorithm.");
        ForteTimer t_hdiag_large;
        davidson_liu(H, evals, evecs, num_ref_roots);
        outfile->Printf("\n  Time spent diagonalizing H          = %f s", t_hdiag_large.elapsed());

        // Print the energy
        for (int i = 0; i < num_ref_roots; ++i) {
            outfile->Printf("\n  P-space CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                            evals->get(i) + nuclear_repulsion_energy_,
                            pc_hartree2ev * (evals->get(i) - evals->get(0)));
        }

        int nmo = reference_determinant_.nmo();
        //        size_t nfrzc = frzc_.size();
        //        size_t nfrzv = frzv_.size();

        std::vector<int> aocc(nalpha_);
        std::vector<int> bocc(nbeta_);
        std::vector<int> avir(ncmo_ - nalpha_);
        std::vector<int> bvir(ncmo_ - nbeta_);

        int noalpha = nalpha_;
        int nobeta = nbeta_;
        int nvalpha = ncmo_ - nalpha_;
        int nvbeta = ncmo_ - nbeta_;

        // Find the SD space out of the reference
        std::vector<DynamicBitsetDeterminant> sd_dets_vec;
        std::map<DynamicBitsetDeterminant, int> new_dets_map;

        ForteTimer t_ms_build;

        // This hash saves the determinant coupling to the model space eigenfunction
        std::map<DynamicBitsetDeterminant, std::vector<double>> V_hash;

        for (size_t I = 0, max_I = ref_space_map.size(); I < max_I; ++I) {
            const DynamicBitsetDeterminant& det = ref_space[I];
            for (int p = 0, i = 0, a = 0; p < ncmo_; ++p) {
                if (det.get_alfa_bit(p)) {
                    //                    if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                    aocc[i] = p;
                    i++;
                    //                    }
                } else {
                    //                    if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                    avir[a] = p;
                    a++;
                    //                    }
                }
            }
            for (int p = 0, i = 0, a = 0; p < ncmo_; ++p) {
                if (det.get_beta_bit(p)) {
                    //                    if (std::count (frzc_.begin(),frzc_.end(),p) == 0){
                    bocc[i] = p;
                    i++;
                    //                    }
                } else {
                    //                    if (std::count (frzv_.begin(),frzv_.end(),p) == 0){
                    bvir[a] = p;
                    a++;
                    //                    }
                }
            }

            // Generate aa excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_) {
                        DynamicBitsetDeterminant new_det(det);
                        new_det.set_alfa_bit(ii, false);
                        new_det.set_alfa_bit(aa, true);
                        if (ref_space_map.find(new_det) == ref_space_map.end()) {
                            double HIJ = det.slater_rules(new_det);
                            if (V_hash.count(new_det) == 0) {
                                V_hash[new_det] = std::vector<double>(num_ref_roots);
                            }
                            for (int n = 0; n < num_ref_roots; ++n) {
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
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_) {
                        DynamicBitsetDeterminant new_det(det);
                        new_det.set_beta_bit(ii, false);
                        new_det.set_beta_bit(aa, true);
                        if (ref_space_map.find(new_det) == ref_space_map.end()) {
                            double HIJ = det.slater_rules(new_det);
                            if (V_hash.count(new_det) == 0) {
                                V_hash[new_det] = std::vector<double>(num_ref_roots);
                            }
                            for (int n = 0; n < num_ref_roots; ++n) {
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
                                 mo_symmetry_[bb]) == wavefunction_symmetry_) {
                                DynamicBitsetDeterminant new_det(det);
                                new_det.set_alfa_bit(ii, false);
                                new_det.set_alfa_bit(jj, false);
                                new_det.set_alfa_bit(aa, true);
                                new_det.set_alfa_bit(bb, true);
                                if (ref_space_map.find(new_det) == ref_space_map.end()) {
                                    double HIJ = det.slater_rules(new_det);
                                    if (V_hash.count(new_det) == 0) {
                                        V_hash[new_det] = std::vector<double>(num_ref_roots);
                                    }
                                    for (int n = 0; n < num_ref_roots; ++n) {
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
                                 mo_symmetry_[bb]) == wavefunction_symmetry_) {
                                DynamicBitsetDeterminant new_det(det);
                                new_det.set_alfa_bit(ii, false);
                                new_det.set_beta_bit(jj, false);
                                new_det.set_alfa_bit(aa, true);
                                new_det.set_beta_bit(bb, true);
                                if (ref_space_map.find(new_det) == ref_space_map.end()) {
                                    double HIJ = det.slater_rules(new_det);
                                    if (V_hash.count(new_det) == 0) {
                                        V_hash[new_det] = std::vector<double>(num_ref_roots);
                                    }
                                    for (int n = 0; n < num_ref_roots; ++n) {
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
                                 (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) ==
                                wavefunction_symmetry_) {
                                DynamicBitsetDeterminant new_det(det);
                                new_det.set_beta_bit(ii, false);
                                new_det.set_beta_bit(jj, false);
                                new_det.set_beta_bit(aa, true);
                                new_det.set_beta_bit(bb, true);
                                if (ref_space_map.find(new_det) == ref_space_map.end()) {
                                    double HIJ = det.slater_rules(new_det);
                                    if (V_hash.count(new_det) == 0) {
                                        V_hash[new_det] = std::vector<double>(num_ref_roots);
                                    }
                                    for (int n = 0; n < num_ref_roots; ++n) {
                                        V_hash[new_det][n] += HIJ * evecs->get(I, n);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        outfile->Printf("\n  The SD excitation space has dimension: %zu (unique)", V_hash.size());
        outfile->Printf("\n  Time spent building the model space = %f s", t_ms_build.elapsed());

        // This will contain all the determinants
        std::vector<DynamicBitsetDeterminant> ref_sd_dets;

        // Add  the P-space determinants
        for (size_t J = 0, max_J = ref_space.size(); J < max_J; ++J) {
            ref_sd_dets.push_back(ref_space[J]);
        }

        ForteTimer t_ms_screen;

        typedef std::map<DynamicBitsetDeterminant, std::vector<double>>::iterator bsmap_it;
        std::vector<std::pair<double, double>> C1(nroot, std::make_pair(0.0, 0.0));
        std::vector<std::pair<double, double>> E2(nroot, std::make_pair(0.0, 0.0));
        std::vector<double> ept2(nroot, 0.0);

        // Check the coupling between the reference and the SD space
        for (bsmap_it it = V_hash.begin(), endit = V_hash.end(); it != endit; ++it) {
            double EI = it->first.energy();
            for (int n = 0; n < num_ref_roots; ++n) {
                double V = it->second[n];
                double C1_I = -V / (EI - evals->get(n));
                double E2_I = -V * V / (EI - evals->get(n));

                C1[n] = std::make_pair(std::fabs(C1_I), C1_I);
                E2[n] = std::make_pair(std::fabs(E2_I), E2_I);
            }

            std::pair<double, double> max_C1 = *std::max_element(C1.begin(), C1.end());
            std::pair<double, double> max_E2 = *std::max_element(E2.begin(), E2.end());

            double select_value = energy_select ? max_E2.first : max_C1.first;

            if (std::fabs(select_value) > tau_q) {
                ref_sd_dets.push_back(it->first);
            } else {
                for (int n = 0; n < num_ref_roots; ++n) {
                    ept2[n] += E2[n].second;
                }
            }
        }

        multistate_pt2_energy_correction_ = ept2;

        size_t dim_ref_sd_dets = ref_sd_dets.size();

        outfile->Printf("\n  After screening the ia-MRCISD space contains %zu determinants",
                        dim_ref_sd_dets);
        outfile->Printf("\n  Time spent screening the model space = %f s", t_ms_screen.elapsed());

        evecs.reset(new Matrix("U", dim_ref_sd_dets, nroot));
        evals.reset(new Vector("e", nroot));
        // Full algorithm
        if (options.get_str("ENERGY_TYPE") == "IMRCISD") {
            H.reset(new Matrix("Hamiltonian Matrix", dim_ref_sd_dets, dim_ref_sd_dets));

            ForteTimer t_h_build2;
#pragma omp parallel for schedule(dynamic)
            for (size_t I = 0; I < dim_ref_sd_dets; ++I) {
                const DynamicBitsetDeterminant& detI = ref_sd_dets[I];
                for (size_t J = I; J < dim_ref_sd_dets; ++J) {
                    const DynamicBitsetDeterminant& detJ = ref_sd_dets[J];
                    double HIJ = detI.slater_rules(detJ);
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

            outfile->Printf("\n  Time spent diagonalizing H          = %f s",
                            t_hdiag_large2.elapsed());

        }
        // Sparse algorithm
        else {
            ForteTimer t_h_build2;
            std::vector<std::vector<std::pair<int, double>>> H_sparse;

            size_t num_nonzero = 0;
            // Form the Hamiltonian matrix
            for (size_t I = 0; I < dim_ref_sd_dets; ++I) {
                std::vector<std::pair<int, double>> H_row;
                const DynamicBitsetDeterminant& detI = ref_sd_dets[I];
                double HII = detI.slater_rules(detI);
                H_row.push_back(std::make_pair(int(I), HII));
                for (size_t J = 0; J < dim_ref_sd_dets; ++J) {
                    if (I != J) {
                        const DynamicBitsetDeterminant& detJ = ref_sd_dets[J];
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
            outfile->Printf("\n  Time spent diagonalizing H          = %f s",
                            t_hdiag_large2.elapsed());
        }

        //
        for (int i = 0; i < nroot; ++i) {
            outfile->Printf("\n  Adaptive CI Energy Root %3d        = %.12f Eh = %8.4f eV", i + 1,
                            evals->get(i) + nuclear_repulsion_energy_,
                            pc_hartree2ev * (evals->get(i) - evals->get(0)));
            outfile->Printf("\n  Adaptive CI Energy + EPT2 Root %3d = %.12f Eh = %8.4f eV", i + 1,
                            evals->get(i) + nuclear_repulsion_energy_ +
                                multistate_pt2_energy_correction_[i],
                            pc_hartree2ev * (evals->get(i) - evals->get(0) +
                                             multistate_pt2_energy_correction_[i] -
                                             multistate_pt2_energy_correction_[0]));
        }

        new_energy = 0.0;
        std::vector<double> energies;
        for (int n = 0; n < nroot; ++n) {
            double state_n_energy = evals->get(n) + nuclear_repulsion_energy_;
            energies.push_back(state_n_energy);
            new_energy += state_n_energy;
        }
        new_energy /= static_cast<double>(nroot);

        // Check for convergence
        if (std::fabs(new_energy - old_energy) < ia_mrcisd_threshold and cycle > 1) {
            break;
        }
        //        // Check the history of energies to avoid cycling in a loop
        //        if(cycle > 3){
        //            bool stuck = true;
        //            for(int cycle_test = cycle - 2; cycle_test < cycle; ++cycle_test){
        //                for (int n = 0; n < nroot; ++n){
        //                    if(std::fabs(energy_history[cycle_test][n] - energies[n]) < 1.0e-12){
        //                        stuck = true;
        //                    }
        //                }
        //            }
        //            if(stuck) break; // exit the cycle
        //        }

        old_energy = new_energy;
        energy_history.push_back(energies);

        std::vector<std::pair<double, size_t>> dm_det_list;

        for (size_t I = 0; I < dim_ref_sd_dets; ++I) {
            double max_dm = 0.0;
            for (int n = 0; n < nroot; ++n) {
                max_dm = std::max(max_dm, std::fabs(evecs->get(I, n)));
            }
            dm_det_list.push_back(std::make_pair(max_dm, I));
        }

        std::sort(dm_det_list.begin(), dm_det_list.end());
        std::reverse(dm_det_list.begin(), dm_det_list.end());

        // Select the new reference space
        ref_space.clear();
        ref_space_map.clear();

        // Decide which will go in ref_space
        for (size_t I = 0; I < dim_ref_sd_dets; ++I) {
            if (dm_det_list[I].first > tau_p) {
                ref_space.push_back(ref_sd_dets[dm_det_list[I].second]);
                ref_space_map[ref_sd_dets[dm_det_list[I].second]] = 1;
            }
        }

        for (int n = 0; n < nroot; ++n) {
            outfile->Printf("\n\n  Root %d", n);

            std::vector<std::pair<double, size_t>> det_weight;
            for (size_t I = 0; I < dim_ref_sd_dets; ++I) {
                det_weight.push_back(std::make_pair(std::fabs(evecs->get(I, n)), I));
            }
            std::sort(det_weight.begin(), det_weight.end());
            std::reverse(det_weight.begin(), det_weight.end());
            for (size_t I = 0; I < 10; ++I) {
                outfile->Printf("\n  %3zu  %9.6f %.9f  %10zu %s", I,
                                evecs->get(det_weight[I].second, n),
                                det_weight[I].first * det_weight[I].first, det_weight[I].second,
                                ref_sd_dets[det_weight[I].second].str().c_str());
            }
        }
    }

    for (int i = 0; i < nroot; ++i) {
        outfile->Printf("\n  * IA-MRCISD total energy (%3d)        = %.12f Eh = %8.4f eV", i + 1,
                        evals->get(i) + nuclear_repulsion_energy_,
                        pc_hartree2ev * (evals->get(i) - evals->get(0)));
        outfile->Printf(
            "\n  * IA-MRCISD total energy (%3d) + EPT2 = %.12f Eh = %8.4f eV", i + 1,
            evals->get(i) + nuclear_repulsion_energy_ + multistate_pt2_energy_correction_[i],
            pc_hartree2ev * (evals->get(i) - evals->get(0) + multistate_pt2_energy_correction_[i] -
                             multistate_pt2_energy_correction_[0]));
    }
    outfile->Printf("\n\n  iterative_adaptive_mrcisd_bitset ran in %f s", t_iamrcisd.elapsed());
}
}
} // EndNamespaces
