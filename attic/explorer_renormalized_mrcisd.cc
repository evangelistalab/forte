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
void LambdaCI::renormalized_mrcisd(std::shared_ptr<ForteOptions> options) {
    outfile->Printf("\n\n  Renormalized MRCISD");

    int nroot = options.get_int("NROOT");
    size_t imrcisd_size = options.get_int("IMRCISD_SIZE");
    size_t imrcisd_test_size = options.get_int("IMRCISD_TEST_SIZE");

    double selection_threshold = t2_threshold_;
    double rmrci_threshold = 1.0e-9;
    int maxcycle = 50;

    bool energy_select = (options.get_str("SELECT_TYPE") == "ENERGY");

    outfile->Printf("\n\n  Diagonalizing the Hamiltonian in the model space");
    outfile->Printf("\n  using an iterative procedure keeping %zu determinants\n", imrcisd_size);
    if (imrcisd_test_size == 0) {
        if (energy_select) {
            outfile->Printf("\n  and testing those with a second-order energy contribution "
                            "greather than %f mEh\n",
                            1000.0 * selection_threshold);
        } else {
            outfile->Printf(
                "\n  and testing those with a first-order wfn coefficient greather than %f\n",
                selection_threshold);
        }
    } else {
        if (energy_select) {
            outfile->Printf("\n  and testing the first %zu ordered according to the second-order "
                            "energy contribution\n",
                            imrcisd_test_size);
        } else {
            outfile->Printf("\n  and testing the first %zu ordered according to the first-order "
                            "wfn coefficient\n",
                            imrcisd_test_size);
        }
    }

    int nmo = reference_determinant_.nmo();
    std::vector<int> aocc(nalpha_);
    std::vector<int> bocc(nbeta_);
    std::vector<int> avir(ncmo_ - nalpha_);
    std::vector<int> bvir(ncmo_ - nbeta_);

    SharedMatrix H;
    SharedMatrix evecs;
    SharedVector evals;

    int nvalpha = ncmo_ - nalpha_;
    int nvbeta = ncmo_ - nbeta_;

    std::vector<StringDeterminant> old_dets_vec;
    std::map<StringDeterminant, int> old_dets_map;
    std::vector<double> coefficient;
    std::vector<double> energy;

    StringDeterminant D0(reference_determinant_);
    //    D0.print();
    srand((unsigned)time(NULL));
    {
        for (int p = 0, i = 0, a = 0; p < ncmo_; ++p) {
            if (D0.get_alfa_bit(p)) {
                aocc[i] = p;
                i++;
            } else {
                avir[a] = p;
                a++;
            }
        }
        //        int ii = aocc[std::rand() % nalpha_];
        //        int aa = avir[std::rand() % nvalpha];
        //        D0.set_alfa_bit(ii,false);
        //        D0.set_beta_bit(ii,false);
        //        D0.set_alfa_bit(aa,true);
        //        D0.set_beta_bit(aa,true);
        //        outfile->Printf("\n\n  The reference determinant was excited (%d) -> (%d)",ii,aa);
    }
    D0.print();

    old_dets_vec.push_back(D0);
    old_dets_map[D0] = 1;
    coefficient.push_back(1.0);
    energy.push_back(D0.energy());
    double old_energy = D0.energy() + nuclear_repulsion_energy_;
    double new_energy = 0.0;

    for (int cycle = 0; cycle < maxcycle; ++cycle) {

        outfile->Printf("\n\n  Cycle %3d: %zu determinants in the RMRCISD wave function", cycle,
                        old_dets_vec.size());

        // Find the SD space out of the reference
        std::vector<StringDeterminant> sd_dets_vec;
        std::map<StringDeterminant, int> new_dets_map;
        ForteTimer t_ms_build;
        for (size_t I = 0, max_I = old_dets_vec.size(); I < max_I; ++I) {
            const StringDeterminant& det = old_dets_vec[I];
            for (int p = 0, i = 0, a = 0; p < ncmo_; ++p) {
                if (det.get_alfa_bit(p)) {
                    aocc[i] = p;
                    i++;
                } else {
                    avir[a] = p;
                    a++;
                }
            }
            for (int p = 0, i = 0, a = 0; p < ncmo_; ++p) {
                if (det.get_beta_bit(p)) {
                    bocc[i] = p;
                    i++;
                } else {
                    bvir[a] = p;
                    a++;
                }
            }

            // Generate aa excitations
            for (int i = 0; i < nalpha_; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_) {
                        StringDeterminant new_det(det);
                        new_det.set_alfa_bit(ii, false);
                        new_det.set_alfa_bit(aa, true);
                        if (old_dets_map.find(new_det) == old_dets_map.end()) {
                            sd_dets_vec.push_back(new_det);
                            //                            new_dets_map[new_det] = 1;
                        }
                    }
                }
            }

            for (int i = 0; i < nbeta_; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_) {
                        StringDeterminant new_det(det);
                        new_det.set_beta_bit(ii, false);
                        new_det.set_beta_bit(aa, true);
                        if (old_dets_map.find(new_det) == old_dets_map.end()) {
                            sd_dets_vec.push_back(new_det);
                            //                            new_dets_map[new_det] = 1;
                        }
                    }
                }
            }

            // Generate aa excitations
            for (int i = 0; i < nalpha_; ++i) {
                int ii = aocc[i];
                for (int j = i + 1; j < nalpha_; ++j) {
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
                                if (old_dets_map.find(new_det) == old_dets_map.end()) {
                                    sd_dets_vec.push_back(new_det);
                                    //                                    new_dets_map[new_det] = 2;
                                }
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < nalpha_; ++i) {
                int ii = aocc[i];
                for (int j = 0; j < nbeta_; ++j) {
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
                                if (old_dets_map.find(new_det) == old_dets_map.end()) {
                                    sd_dets_vec.push_back(new_det);
                                    //                                    new_dets_map[new_det] = 2;
                                }
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < nbeta_; ++i) {
                int ii = bocc[i];
                for (int j = i + 1; j < nbeta_; ++j) {
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
                                if (old_dets_map.find(new_det) == old_dets_map.end()) {
                                    sd_dets_vec.push_back(new_det);
                                    //                                    new_dets_map[new_det] = 2;
                                }
                            }
                        }
                    }
                }
            }
        }

        //        for (auto it = new_dets_map.begin(), endit = new_dets_map.end(); it != endit;
        //        ++it){
        //            sd_dets_vec.push_back(it->first);
        //        }

        outfile->Printf("\n  The SD excitation space has dimension: %zu", sd_dets_vec.size());
        outfile->Printf("\n  Time spent building the model space = %f s", t_ms_build.elapsed());

        ForteTimer t_ms_screen;

        sort(sd_dets_vec.begin(), sd_dets_vec.end());
        sd_dets_vec.erase(unique(sd_dets_vec.begin(), sd_dets_vec.end()), sd_dets_vec.end());
        std::vector<StringDeterminant> refsd_dets_vec;

        for (size_t J = 0, max_J = old_dets_vec.size(); J < max_J; ++J) {
            refsd_dets_vec.push_back(old_dets_vec[J]);
        }

        {
            // Check the coupling between the reference and the SD space
            std::vector<std::pair<double, size_t>> new_dets_importance_vec;

            for (size_t I = 0, max_I = sd_dets_vec.size(); I < max_I; ++I) {
                double V = 0.0;
                double ERef = 0.0;
                double EI = sd_dets_vec[I].energy();
                for (size_t J = 0, max_J = old_dets_vec.size(); J < max_J; ++J) {
                    V += sd_dets_vec[I].slater_rules(old_dets_vec[J]) * coefficient[J]; // HJI * C_I
                }
                double C1 = std::fabs(V / (EI - energy[0]));
                double E2 = std::fabs(V * V / (EI - energy[0]));

                double select_value = (energy_select ? E2 : C1);

                // Save the importance metrics
                new_dets_importance_vec.push_back(std::make_pair(select_value, I));
            }

            std::sort(new_dets_importance_vec.begin(), new_dets_importance_vec.end());
            std::reverse(new_dets_importance_vec.begin(), new_dets_importance_vec.end());

            // Select only those determinants above the threshold
            if (imrcisd_test_size == 0) {
                outfile->Printf("\n  Adding all SD determinants above %f", selection_threshold);

                for (size_t I = 0, maxI = new_dets_importance_vec.size(); I < maxI; ++I) {
                    if (new_dets_importance_vec[I].first > selection_threshold) {
                        refsd_dets_vec.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
                    }
                }
            } else {
                size_t maxI = std::min(imrcisd_test_size, new_dets_importance_vec.size());
                outfile->Printf("\n  Adding the most important %zu SD determinants", maxI);
                for (size_t I = 0; I < maxI; ++I) {
                    refsd_dets_vec.push_back(sd_dets_vec[new_dets_importance_vec[I].second]);
                }
            }
        }

        size_t num_mrcisd_dets = refsd_dets_vec.size();

        outfile->Printf("\n  The i-MRCISD full space contains %zu determinants", num_mrcisd_dets);
        outfile->Printf("\n  Time spent screening the model space = %f s", t_ms_screen.elapsed());

        evecs.reset(new Matrix("U", num_mrcisd_dets, nroot));
        evals.reset(new Vector("e", nroot));

        if (options.get_str("ENERGY_TYPE") != "IMRCISD_SPARSE") {
            H.reset(new Matrix("Hamiltonian Matrix", num_mrcisd_dets, num_mrcisd_dets));
            ForteTimer t_h_build;
#pragma omp parallel for schedule(dynamic)
            for (size_t I = 0; I < num_mrcisd_dets; ++I) {
                const StringDeterminant& detI = refsd_dets_vec[I];
                for (size_t J = I; J < num_mrcisd_dets; ++J) {
                    const StringDeterminant& detJ = refsd_dets_vec[J];
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

            outfile->Printf("\n  Time spent diagonalizing H          = %f s",
                            t_hdiag_large.elapsed());

        } else
        // Sparse algorithm
        {
            ForteTimer t_h_build2;
            std::vector<std::vector<std::pair<int, double>>> H_sparse;

            size_t num_nonzero = 0;
            // Form the Hamiltonian matrix
            for (size_t I = 0; I < num_mrcisd_dets; ++I) {
                std::vector<std::pair<int, double>> H_row;
                const StringDeterminant& detI = refsd_dets_vec[I];
                double HII = detI.slater_rules(detI);
                H_row.push_back(std::make_pair(int(I), HII));
                for (size_t J = 0; J < num_mrcisd_dets; ++J) {
                    if (I != J) {
                        const StringDeterminant& detJ = refsd_dets_vec[J];
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
                            size_t(num_mrcisd_dets * num_mrcisd_dets),
                            double(num_nonzero) / double(num_mrcisd_dets * num_mrcisd_dets));
            outfile->Printf("\n  Time spent building H               = %f s", t_h_build2.elapsed());

            // 4) Diagonalize the Hamiltonian
            ForteTimer t_hdiag_large2;
            outfile->Printf("\n  Using the Davidson-Liu algorithm.");
            davidson_liu_sparse(H_sparse, evals, evecs, nroot);
            outfile->Printf("\n  Time spent diagonalizing H          = %f s",
                            t_hdiag_large2.elapsed());

            outfile->Printf("\n  Time spent diagonalizing H          = %f s",
                            t_hdiag_large2.elapsed());
        }

        // 5) Print the energy
        for (int i = 0; i < nroot; ++i) {
            outfile->Printf("\n  Ren. step CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                            evals->get(i) + nuclear_repulsion_energy_,
                            pc_hartree2ev * (evals->get(i) - evals->get(0)));
            //        outfile->Printf("\n  Ren. step CI Energy + EPT2 Root %3d = %.12f = %.12f +
            //        %.12f",i + 1,evals->get(i) + multistate_pt2_energy_correction_[i],
            //                evals->get(i),multistate_pt2_energy_correction_[i]);
        }
        outfile->Printf("\n  Finished building H");

        new_energy = evals->get(0) + nuclear_repulsion_energy_;

        std::vector<std::pair<double, size_t>> dm_det_list;

        for (size_t I = 0; I < num_mrcisd_dets; ++I) {
            double max_dm = 0.0;
            for (int n = 0; n < nroot; ++n) {
                max_dm = std::max(max_dm, std::fabs(evecs->get(I, n)));
            }
            dm_det_list.push_back(std::make_pair(max_dm, I));
        }

        std::sort(dm_det_list.begin(), dm_det_list.end());
        std::reverse(dm_det_list.begin(), dm_det_list.end());

        old_dets_vec.clear();
        old_dets_map.clear();
        coefficient.clear();
        size_t size_small_ci = std::min(dm_det_list.size(), imrcisd_size);

        for (size_t I = 0; I < size_small_ci; ++I) {
            old_dets_vec.push_back(refsd_dets_vec[dm_det_list[I].second]);
            old_dets_map[refsd_dets_vec[dm_det_list[I].second]] = 1;
        }

        evecs.reset(new Matrix("U", size_small_ci, nroot));
        evals.reset(new Vector("e", nroot));

        if (options.get_str("ENERGY_TYPE") == "IMRCISD_SPARSE") {
            H.reset(new Matrix("Hamiltonian Matrix", size_small_ci, size_small_ci));

            ForteTimer t_h_small_build;
#pragma omp parallel for schedule(dynamic)
            for (size_t I = 0; I < size_small_ci; ++I) {
                for (size_t J = I; J < size_small_ci; ++J) {
                    double HIJ = old_dets_vec[I].slater_rules(old_dets_vec[J]);
                    H->set(I, J, HIJ);
                    H->set(J, I, HIJ);
                }
            }
            outfile->Printf("\n  Time spent building H               = %f s",
                            t_h_small_build.elapsed());

            // 4) Diagonalize the Hamiltonian
            if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
                outfile->Printf("\n  Using the Davidson-Liu algorithm.");
                davidson_liu(H, evals, evecs, nroot);
            } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
                outfile->Printf("\n  Performing full diagonalization.");
                H->diagonalize(evecs, evals);
            }
        } else
        // Sparse algorithm
        {
            ForteTimer t_h_build2;
            std::vector<std::vector<std::pair<int, double>>> H_sparse;

            size_t num_nonzero = 0;
            // Form the Hamiltonian matrix
            for (size_t I = 0; I < size_small_ci; ++I) {
                std::vector<std::pair<int, double>> H_row;
                const StringDeterminant& detI = old_dets_vec[I];
                double HII = detI.slater_rules(detI);
                H_row.push_back(std::make_pair(int(I), HII));
                for (size_t J = 0; J < size_small_ci; ++J) {
                    if (I != J) {
                        const StringDeterminant& detJ = old_dets_vec[J];
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
                            size_t(size_small_ci * size_small_ci),
                            double(num_nonzero) / double(size_small_ci * size_small_ci));
            outfile->Printf("\n  Time spent building H               = %f s", t_h_build2.elapsed());

            // 4) Diagonalize the Hamiltonian
            ForteTimer t_hdiag_large2;
            outfile->Printf("\n  Using the Davidson-Liu algorithm.");
            davidson_liu_sparse(H_sparse, evals, evecs, nroot);
            outfile->Printf("\n  Time spent diagonalizing H          = %f s",
                            t_hdiag_large2.elapsed());
        }

        for (size_t I = 0; I < size_small_ci; ++I) {
            coefficient.push_back(evecs->get(I, 0));
        }
        energy[0] = evals->get(0);

        // 5) Print the energy
        for (int i = 0; i < nroot; ++i) {
            outfile->Printf("\n  Ren. step (small) CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                            evals->get(i) + nuclear_repulsion_energy_,
                            pc_hartree2ev * (evals->get(i) - evals->get(0)));
        }

        new_energy = evals->get(0) + nuclear_repulsion_energy_;
        outfile->Printf("\n   @IMRCISD %2d  %24.16f  %24.16f", cycle, new_energy,
                        new_energy - old_energy);

        if (std::fabs(new_energy - old_energy) < rmrci_threshold) {
            break;
        }
        old_energy = new_energy;

        outfile->Printf("\n  After diagonalization there are %zu determinants",
                        old_dets_vec.size());
    }

    outfile->Printf("\n\n   * IMRCISD total energy =  %24.16f\n", new_energy);
}

/**
 * Diagonalize the
 */
void LambdaCI::renormalized_mrcisd_simple(std::shared_ptr<ForteOptions> options) {
    outfile->Printf("\n\n  Renormalized MRCISD");

    int nroot = options.get_int("NROOT");
    size_t ren_ndets = options.get_int("IMRCISD_SIZE");
    double selection_threshold = t2_threshold_;
    double rmrci_threshold = 1.0e-9;
    int maxcycle = 50;

    outfile->Printf("\n\n  Diagonalizing the Hamiltonian in the model space");
    outfile->Printf("\n  using a renormalization procedure keeping %zu determinants\n", ren_ndets);
    outfile->Printf("\n  and exciting those with a first-order coefficient greather than %f\n",
                    selection_threshold);

    int nmo = reference_determinant_.nmo();
    std::vector<int> aocc(nalpha_);
    std::vector<int> bocc(nbeta_);
    std::vector<int> avir(ncmo_ - nalpha_);
    std::vector<int> bvir(ncmo_ - nbeta_);

    SharedMatrix H;
    SharedMatrix evecs;
    SharedVector evals;

    int nvalpha = ncmo_ - nalpha_;
    int nvbeta = ncmo_ - nbeta_;

    std::vector<StringDeterminant> old_dets_vec;
    std::vector<double> coefficient;
    old_dets_vec.push_back(reference_determinant_);
    coefficient.push_back(1.0);
    double old_energy = reference_determinant_.energy() + nuclear_repulsion_energy_;
    double new_energy = 0.0;

    for (int cycle = 0; cycle < maxcycle; ++cycle) {

        outfile->Printf("\n  Cycle %3d: %zu determinants in the RMRCISD wave function", cycle,
                        old_dets_vec.size());

        // Find the SD space out of the reference
        std::map<StringDeterminant, int> new_dets_map;
        ForteTimer t_ms_build;
        for (size_t I = 0, max_I = old_dets_vec.size(); I < max_I; ++I) {
            const StringDeterminant& det = old_dets_vec[I];
            double Em = det.energy();

            new_dets_map[det] = 1;
            for (int p = 0, i = 0, a = 0; p < ncmo_; ++p) {
                if (det.get_alfa_bit(p)) {
                    aocc[i] = p;
                    i++;
                } else {
                    avir[a] = p;
                    a++;
                }
            }
            for (int p = 0, i = 0, a = 0; p < ncmo_; ++p) {
                if (det.get_beta_bit(p)) {
                    bocc[i] = p;
                    i++;
                } else {
                    bvir[a] = p;
                    a++;
                }
            }

            // Generate aa excitations
            for (int i = 0; i < nalpha_; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_) {
                        StringDeterminant new_det(det);
                        new_det.set_alfa_bit(ii, false);
                        new_det.set_alfa_bit(aa, true);
                        new_dets_map[new_det] = 1;
                    }
                }
            }

            for (int i = 0; i < nbeta_; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_) {
                        StringDeterminant new_det(det);
                        new_det.set_beta_bit(ii, false);
                        new_det.set_beta_bit(aa, true);
                        new_dets_map[new_det] = 1;
                    }
                }
            }

            // Generate aa excitations
            for (int i = 0; i < nalpha_; ++i) {
                int ii = aocc[i];
                for (int j = i + 1; j < nalpha_; ++j) {
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

                                double Ex = new_det.energy();
                                double V = det.slater_rules(new_det);
                                double c1 = std::fabs(V / (Em - Ex) * coefficient[I]);
                                if (c1 > selection_threshold) {
                                    new_dets_map[new_det] = 1;
                                }
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < nalpha_; ++i) {
                int ii = aocc[i];
                for (int j = 0; j < nbeta_; ++j) {
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

                                double Ex = new_det.energy();
                                double V = det.slater_rules(new_det);
                                double c1 = std::fabs(V / (Em - Ex) * coefficient[I]);
                                if (c1 > selection_threshold) {
                                    new_dets_map[new_det] = 1;
                                }
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < nbeta_; ++i) {
                int ii = bocc[i];
                for (int j = i + 1; j < nbeta_; ++j) {
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

                                double Ex = new_det.energy();
                                double V = det.slater_rules(new_det);
                                double c1 = std::fabs(V / (Em - Ex) * coefficient[I]);
                                if (c1 > selection_threshold) {
                                    new_dets_map[new_det] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        std::vector<StringDeterminant> new_dets_vec;
        for (std::map<StringDeterminant, int>::iterator it = new_dets_map.begin(),
                                                        endit = new_dets_map.end();
             it != endit; ++it) {
            new_dets_vec.push_back(it->first);
        }

        size_t num_mrcisd_dets = new_dets_vec.size();

        outfile->Printf("\n  The model space contains %zu determinants", num_mrcisd_dets);
        outfile->Printf("\n  Time spent building the model space = %f s", t_ms_build.elapsed());

        H.reset(new Matrix("Hamiltonian Matrix", num_mrcisd_dets, num_mrcisd_dets));
        evecs.reset(new Matrix("U", num_mrcisd_dets, nroot));
        evals.reset(new Vector("e", nroot));

        ForteTimer t_h_build;
        for (size_t I = 0; I < num_mrcisd_dets; ++I) {
            for (size_t J = I; J < num_mrcisd_dets; ++J) {
                double HIJ = new_dets_vec[I].slater_rules(new_dets_vec[J]);
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
        outfile->Printf("\n  Finished building H");

        new_energy = evals->get(0) + nuclear_repulsion_energy_;

        std::vector<std::pair<double, size_t>> dm_det_list;

        for (size_t I = 0; I < num_mrcisd_dets; ++I) {
            double max_dm = 0.0;
            for (int n = 0; n < nroot; ++n) {
                max_dm = std::max(max_dm, std::fabs(evecs->get(I, n)));
            }
            dm_det_list.push_back(std::make_pair(max_dm, I));
        }

        std::sort(dm_det_list.begin(), dm_det_list.end());
        std::reverse(dm_det_list.begin(), dm_det_list.end());

        old_dets_vec.clear();
        coefficient.clear();
        for (size_t I = 0, max_I = std::min(dm_det_list.size(), ren_ndets); I < max_I; ++I) {
            old_dets_vec.push_back(new_dets_vec[dm_det_list[I].second]);
        }

        size_t size_small_ci = std::min(dm_det_list.size(), ren_ndets);
        H.reset(new Matrix("Hamiltonian Matrix", size_small_ci, size_small_ci));
        evecs.reset(new Matrix("U", size_small_ci, nroot));
        evals.reset(new Vector("e", nroot));

        ForteTimer t_h_small_build;
        for (size_t I = 0; I < size_small_ci; ++I) {
            for (size_t J = I; J < size_small_ci; ++J) {
                double HIJ = old_dets_vec[I].slater_rules(old_dets_vec[J]);
                H->set(I, J, HIJ);
                H->set(J, I, HIJ);
            }
        }
        outfile->Printf("\n  Time spent building H               = %f s",
                        t_h_small_build.elapsed());

        // 4) Diagonalize the Hamiltonian
        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON") {
            outfile->Printf("\n  Using the Davidson-Liu algorithm.");
            davidson_liu(H, evals, evecs, nroot);
        } else if (options.get_str("DIAG_ALGORITHM") == "FULL") {
            outfile->Printf("\n  Performing full diagonalization.");
            H->diagonalize(evecs, evals);
        }

        for (size_t I = 0; I < size_small_ci; ++I) {
            coefficient.push_back(evecs->get(I, 0));
        }

        // 5) Print the energy
        for (int i = 0; i < nroot; ++i) {
            outfile->Printf("\n  Ren. step (small) CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                            evals->get(i) + nuclear_repulsion_energy_,
                            pc_hartree2ev * (evals->get(i) - evals->get(0)));
        }

        new_energy = evals->get(0) + nuclear_repulsion_energy_;
        outfile->Printf("\n ->  %2d  %24.16f", cycle, new_energy);

        if (std::fabs(new_energy - old_energy) < rmrci_threshold) {
            break;
        }
        old_energy = new_energy;

        outfile->Printf("\n  After diagonalization there are %zu determinants",
                        old_dets_vec.size());
    }
}

// void Explorer::renormalized_mrcisd_simple(std::shared_ptr<ForteOptions> options)
//{
//    typedef boost::shared_ptr<StringDeterminant> shared_det;
//    outfile->Printf("\n\n  Renormalized MRCISD");

//    int nroot = options.get_int("NROOT");
//    size_t ren_ndets = options.get_int("IMRCISD_SIZE");
//    double selection_threshold = t2_threshold_;
//    double rmrci_threshold = 1.0e-9;
//    int maxcycle = 50;

//    outfile->Printf("\n\n  Diagonalizing the Hamiltonian in the model space");
//    outfile->Printf("\n  using a renormalization procedure keeping %zu determinants\n",ren_ndets);
//    outfile->Printf("\n  and exciting those with a first-order coefficient greather than
//    %f\n",selection_threshold);
//

//    int nmo = reference_determinant_.nmo();
//    std::vector<int> aocc(nalpha_);
//    std::vector<int> bocc(nbeta_);
//    std::vector<int> avir(nmo_ - nalpha_);
//    std::vector<int> bvir(nmo_ - nbeta_);

//    SharedMatrix H;
//    SharedMatrix evecs;
//    SharedVector evals;

//    int nvalpha = nmo_ - nalpha_;
//    int nvbeta = nmo_ - nbeta_;

//    std::vector<shared_det> old_dets_vec;
//    old_dets_vec.push_back(shared_det(new StringDeterminant(reference_determinant_)));
//    double old_energy = reference_determinant_.energy() + nuclear_repulsion_energy_;
//    double new_energy = 0.0;

//    size_t size_small_ci = std::min(old_dets_vec.size(),ren_ndets);

////    H.reset(new Matrix("Hamiltonian Matrix",size_small_ci,size_small_ci));
////    evecs.reset(new Matrix("U",size_small_ci,size_small_ci));
////    evals.reset(new Vector("e",size_small_ci));

//    SharedMatrix H_small(new Matrix("Hamiltonian Matrix",ren_ndets,ren_ndets));
//    SharedMatrix evecs_small(new Matrix("U",ren_ndets,nroot));
//    SharedVector evals_small(new Vector("e",nroot));
//    evecs_small->set(0,0,1.0);

////    H->set(0,0,reference_determinant_.energy());
////    evecs->set(0,0,1.0);
////    evals->set(0,reference_determinant_.energy());

//    for (int cycle = 0; cycle < maxcycle; ++cycle){

//        outfile->Printf("\n  Cycle %3d: %zu determinants in the RMRCISD wave
//        function",cycle,old_dets_vec.size());

//        std::map<shared_det,int> new_dets_map;
//        ForteTimer t_ms_build;
////        outfile->Printf("\n  Processing determinants:\n");
//        for (size_t I = 0, max_I = old_dets_vec.size(); I < max_I; ++I){
//            const shared_det& det = old_dets_vec[I];

//            double Em = det->energy();

//            // Add all the determinants, then scr
//            new_dets_map[det] = 1;
//            for (int p = 0, i = 0, a = 0; p < nmo_; ++p){
//                if (det->get_alfa_bit(p)){
//                    aocc[i] = p;
//                    i++;
//                }else{
//                    avir[a] = p;
//                    a++;
//                }
//            }
//            for (int p = 0, i = 0, a = 0; p < nmo_; ++p){
//                if (det->get_beta_bit(p)){
//                    bocc[i] = p;
//                    i++;
//                }else{
//                    bvir[a] = p;
//                    a++;
//                }
//            }

//            // Generate aa excitations
//            for (int i = 0; i < nalpha_; ++i){
//                int ii = aocc[i];
//                for (int a = 0; a < nvalpha; ++a){
//                    int aa = avir[a];
//                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_){
//                        shared_det new_det(new StringDeterminant(*det));
//                        new_det->set_alfa_bit(ii,false);
//                        new_det->set_alfa_bit(aa,true);
//                        new_dets_map[new_det] = 1;
//                    }
//                }
//            }

//            for (int i = 0; i < nbeta_; ++i){
//                int ii = bocc[i];
//                for (int a = 0; a < nvbeta; ++a){
//                    int aa = bvir[a];
//                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == wavefunction_symmetry_){
//                        shared_det new_det(new StringDeterminant(*det));
//                        new_det->set_beta_bit(ii,false);
//                        new_det->set_beta_bit(aa,true);
//                        new_dets_map[new_det] = 1;
//                    }
//                }
//            }

//            // Generate aa excitations
//            for (int i = 0; i < nalpha_; ++i){
//                int ii = aocc[i];
//                for (int j = i + 1; j < nalpha_; ++j){
//                    int jj = aocc[j];
//                    for (int a = 0; a < nvalpha; ++a){
//                        int aa = avir[a];
//                        for (int b = a + 1; b < nvalpha; ++b){
//                            int bb = avir[b];
//                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
//                            mo_symmetry_[bb]) == wavefunction_symmetry_){
//                                shared_det new_det(new StringDeterminant(*det));
//                                new_det->set_alfa_bit(ii,false);
//                                new_det->set_alfa_bit(jj,false);
//                                new_det->set_alfa_bit(aa,true);
//                                new_det->set_alfa_bit(bb,true);

//                                double Ex = new_det->energy();
//                                double V = det->slater_rules(*new_det);
//                                double c1 = std::fabs(evecs_small->get(I,0) * V / (Em - Ex));
//                                if (c1 > selection_threshold){
//                                    new_dets_map[new_det] = 1;
//                                }
//                            }
//                        }
//                    }
//                }
//            }

//            for (int i = 0; i < nalpha_; ++i){
//                int ii = aocc[i];
//                for (int j = 0; j < nbeta_; ++j){
//                    int jj = bocc[j];
//                    for (int a = 0; a < nvalpha; ++a){
//                        int aa = avir[a];
//                        for (int b = 0; b < nvbeta; ++b){
//                            int bb = bvir[b];
//                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
//                            mo_symmetry_[bb]) == wavefunction_symmetry_){
//                                shared_det new_det(new StringDeterminant(*det));
//                                new_det->set_alfa_bit(ii,false);
//                                new_det->set_beta_bit(jj,false);
//                                new_det->set_alfa_bit(aa,true);
//                                new_det->set_beta_bit(bb,true);

//                                double Ex = new_det->energy();
//                                double V = det->slater_rules(*new_det);
//                                double c1 = std::fabs(evecs_small->get(I,0) * V / (Em - Ex));
//                                if (c1 > selection_threshold){
//                                    new_dets_map[new_det] = 1;
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//            for (int i = 0; i < nbeta_; ++i){
//                int ii = bocc[i];
//                for (int j = i + 1; j < nbeta_; ++j){
//                    int jj = bocc[j];
//                    for (int a = 0; a < nvbeta; ++a){
//                        int aa = bvir[a];
//                        for (int b = a + 1; b < nvbeta; ++b){
//                            int bb = bvir[b];
//                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
//                            mo_symmetry_[bb]) == wavefunction_symmetry_){
//                                shared_det new_det(new StringDeterminant(*det));

//                                new_det->set_beta_bit(ii,false);
//                                new_det->set_beta_bit(jj,false);
//                                new_det->set_beta_bit(aa,true);
//                                new_det->set_beta_bit(bb,true);

//                                double Ex = new_det->energy();
//                                double V = det->slater_rules(*new_det);
//                                double c1 = std::fabs(evecs_small->get(I,0) * V / (Em - Ex));
//                                if (c1 > selection_threshold){
//                                    new_dets_map[new_det] = 1;
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }

//        std::vector<shared_det> new_dets_vec;
//        for (auto it = new_dets_map.begin(), endit = new_dets_map.end(); it != endit; ++it){
//            new_dets_vec.push_back(it->first);
//        }

//        size_t num_mrcisd_dets = new_dets_vec.size();

//        outfile->Printf("\n  The model space contains %zu determinants",num_mrcisd_dets);
//        outfile->Printf("\n  Time spent building the model space = %f s",t_ms_build.elapsed());
//

//        H.reset(new Matrix("Hamiltonian Matrix",num_mrcisd_dets,num_mrcisd_dets));
//        evecs.reset(new Matrix("U",num_mrcisd_dets,nroot));
//        evals.reset(new Vector("e",nroot));

//        ForteTimer t_h_build;
//        for (size_t I = 0; I < num_mrcisd_dets; ++I){
////            new_dets_vec[I]->print();
//            for (size_t J = I; J < num_mrcisd_dets; ++J){
//                double HIJ = new_dets_vec[I]->slater_rules(*new_dets_vec[J]);
//                H->set(I,J,HIJ);
//                H->set(J,I,HIJ);
//                if(std::fabs(HIJ) > 150.0)
//                    outfile->Printf("\n  H(%5zu,%5zu) = %26.16f",I,J,H->get(I,J));
//            }

//        }
//        outfile->Printf("\n  Time spent building H               = %f s",t_h_build.elapsed());
//

//        // 4) Diagonalize the Hamiltonian
//        ForteTimer t_hdiag_large;
//        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
//            outfile->Printf("\n  Using the Davidson-Liu algorithm.");
//            davidson_liu(H,evals,evecs,nroot);
//        }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
//            outfile->Printf("\n  Performing full diagonalization.");
//            H->diagonalize(evecs,evals);
//        }

//        outfile->Printf("\n  Time spent diagonalizing H          = %f s",t_hdiag_large.elapsed());
//

//        // 5) Print the energy
//        for (int i = 0; i < nroot; ++ i){
//            outfile->Printf("\n  Ren. step CI Energy Root %3d = %.12f Eh = %8.4f eV",i +
//            1,evals->get(i) + nuclear_repulsion_energy_,pc_hartree2ev * (evals->get(i) -
//            evals->get(0)));
//            //        outfile->Printf("\n  Ren. step CI Energy + EPT2 Root %3d = %.12f = %.12f +
//            %.12f",i + 1,evals->get(i) + multistate_pt2_energy_correction_[i],
//            //                evals->get(i),multistate_pt2_energy_correction_[i]);
//        }
//        outfile->Printf("\n  Finished building H");
//

//        new_energy = evals->get(0) + nuclear_repulsion_energy_;

//        std::vector<std::pair<double,shared_det> > dm_det_list;

//        for (size_t I = 0; I < num_mrcisd_dets; ++I){
//            double max_dm = 0.0;
//            for (int n = 0; n < nroot; ++n){
//                max_dm = std::max(max_dm,std::fabs(evecs->get(I,n)));
//            }
//            dm_det_list.push_back(std::make_pair(max_dm,new_dets_vec[I]));
//        }

//        std::sort(dm_det_list.begin(),dm_det_list.end());
//        std::reverse(dm_det_list.begin(),dm_det_list.end());

//        old_dets_vec.clear();
//        for (size_t I = 0, max_I = std::min(dm_det_list.size(),ren_ndets); I < max_I; ++I){
//            old_dets_vec.push_back(dm_det_list[I].second);
//        }

//        size_small_ci = std::min(old_dets_vec.size(),ren_ndets);

//        outfile->Printf("\n  Solving a %zu x %zu eigensystem",size_small_ci,size_small_ci);
//        H.reset(new Matrix("Hamiltonian Matrix",size_small_ci,size_small_ci));
//        evecs.reset(new Matrix("U",size_small_ci,nroot));
//        evals.reset(new Vector("e",nroot));

//        ForteTimer t_h_small_build;
//        for (size_t I = 0; I < size_small_ci; ++I){
//            for (size_t J = I; J < size_small_ci; ++J){
//                double HIJ = old_dets_vec[I]->slater_rules(*old_dets_vec[J]);
//                H_small->set(I,J,HIJ);
//                H_small->set(J,I,HIJ);

//                outfile->Printf("\n  H(%5zu,%5zu) = %26.16f (%26.16f)",I,J,H->get(I,J),HIJ);
//                new_dets_vec[I]->print();
//                new_dets_vec[J]->print();

//            }
//        }
//        outfile->Printf("\n  Time spent building H               = %f
//        s",t_h_small_build.elapsed());
//        H_small->print();
//

//        // 4) Diagonalize the Hamiltonian
//        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
//            outfile->Printf("\n  Using the Davidson-Liu algorithm.");
//            davidson_liu(H_small,evals_small,evecs_small,nroot);
////            H->diagonalize(evecs,evals);
//        }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
//            outfile->Printf("\n  Performing full diagonalization.");
//        }

//        // 5) Print the energy
//        for (int i = 0; i < nroot; ++ i){
//            outfile->Printf("\n  Ren. step (small) CI Energy Root %3d = %.12f Eh = %8.4f eV",i +
//            1,evals_small->get(i) + nuclear_repulsion_energy_,pc_hartree2ev * (evals->get(i) -
//            evals->get(0)));
//        }
//

//        new_energy = evals_small->get(0) + nuclear_repulsion_energy_;
//        outfile->Printf("\n ->  %2d  %24.16f",cycle,new_energy);

//
//        if (std::fabs(new_energy - old_energy) < rmrci_threshold){
//            break;
//        }
//        old_energy = new_energy;

//        outfile->Printf("\n  After diagonalization there are %zu
//        determinants",old_dets_vec.size());
//
//    }

//
//}
}
} // EndNamespaces
