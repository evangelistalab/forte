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

#include "ga-ci.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>

#include "mini-boost/boost/format.hpp"
#include "mini-boost/boost/timer.hpp"
#include <boost/unordered_map.hpp>

#include <libciomr/libciomr.h>
#include <libmints/molecule.h>
#include <libpsio/psio.h>
#include <libpsio/psio.hpp>
#include <libqt/qt.h>

#include "cartographer.h"
#include "dynamic_bitset_determinant.h"
#include "sparse_ci_solver.h"
#include "string_determinant.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

GeneticAlgorithmCI::GeneticAlgorithmCI(boost::shared_ptr<Wavefunction> wfn, Options& options,
                                       ForteIntegrals* ints)
    : Wavefunction(options, _default_psio_lib_), options_(options), ints_(ints) {
    // Copy the wavefunction information
    copy(wfn);

    startup();
    print_info();
}

void GeneticAlgorithmCI::startup() {
    // Connect the integrals to the determinant class
    StringDeterminant::set_ints(ints_);
    DynamicBitsetDeterminant::set_ints(ints_);

    // The number of correlated molecular orbitals
    ncmo_ = ints_->ncmo();
    ncmopi_ = ints_->ncmopi();

    // Overwrite the frozen orbitals arrays
    frzcpi_ = ints_->frzcpi();
    frzvpi_ = ints_->frzvpi();

    ncalpha_ = nalpha_ - frzcpi_.sum();
    ncbeta_ = nbeta_ - frzcpi_.sum();

    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();

    // Create the array with mo symmetry
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < ncmopi_[h]; ++p) {
            mo_symmetry_.push_back(h);
        }
    }

    wavefunction_symmetry_ = 0;
    if (options_["ROOT_SYM"].has_changed()) {
        wavefunction_symmetry_ = options_.get_int("ROOT_SYM");
    }

    // Build the reference determinant and compute its energy
    std::vector<int> occupation(2 * ncmo_, 0);
    int cumidx = 0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < doccpi_[h] - frzcpi_[h]; ++i) {
            occupation[i + cumidx] = 1;
            occupation[ncmo_ + i + cumidx] = 1;
        }
        for (int i = 0; i < soccpi_[h]; ++i) {
            occupation[i + cumidx] = 1;
        }
        cumidx += ncmopi_[h];
    }
    reference_determinant_ =
        SharedDynamicBitsetDeterminant(new DynamicBitsetDeterminant(occupation));

    outfile->Printf("\n  The reference determinant is:\n");
    reference_determinant_->print();

    // Read options
    nroot_ = options_.get_int("NROOT");

    root_ = options_.get_int("ROOT");

    npop_ = options_.get_int("NPOP");
}

GeneticAlgorithmCI::~GeneticAlgorithmCI() {}

void GeneticAlgorithmCI::print_info() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{
        {"Symmetry", wavefunction_symmetry_},
        {"Number of roots", nroot_},
        {"Root used for properties", root_},
        {"Size of the population", npop_},
        {"Number of electrons", nalpha_ + nbeta_},
        {"Number of correlated alpha electrons", ncalpha_},
        {"Number of correlated beta electrons", ncbeta_},
    };

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Convergence threshold", options_.get_double("E_CONVERGENCE")}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{};

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

double GeneticAlgorithmCI::compute_energy() {
    ForteTimer t_iamrcisd;
    outfile->Printf("\n\n  Genetic Algorithm CI");

    SparseCISolver sparse_solver;

    // These hold the determinants and the value of the fitness function
    std::vector<SharedDynamicBitsetDeterminant> population;
    std::vector<double> fitness;
    std::unordered_map<std::vector<bool>, bool> unique_list;

    // Step 1.  Generate an initial population
    generate_initial_pop(population, npop_, unique_list);
    population.push_back(reference_determinant_);
    std::vector<bool> occupation_ab(2 * ncmo_, false);
    for (int i = 0; i < ncmo_; ++i) {
        occupation_ab[i] = reference_determinant_->get_alfa_bit(i);
        occupation_ab[i + ncmo_] = reference_determinant_->get_beta_bit(i);
    }
    unique_list[occupation_ab] = true;

    for (int cycle = 0; cycle < 20; ++cycle) {
        int total_pop = population.size();
        SharedMatrix evecs(new Matrix("Eigenvectors", total_pop, nroot_));
        SharedVector evals(new Vector("Eigenvalues", nroot_));
        sparse_solver.diagonalize_hamiltonian(population, evals, evecs, nroot_, Full);

        // Print the energy
        outfile->Printf("\n");
        for (int i = 0; i < nroot_; ++i) {
            double abs_energy = evals->get(i) + nuclear_repulsion_energy_;
            double exc_energy = pc_hartree2ev * (evals->get(i) - evals->get(0));
            outfile->Printf("\n    Population CI Energy Root %3d        = %.12f Eh = %8.4f eV (%d)",
                            i + 1, abs_energy, exc_energy, total_pop);
        }
        outfile->Printf("\n");
        
        fitness.clear();

        for (int I = 0; I < total_pop; ++I) {
            fitness.push_back(evecs->get(I, root_) * evecs->get(I, root_));
        }

        weed_out(population, fitness, unique_list);

        crossover(population, fitness, unique_list);
    }

    //    int maxcycle = 20;
    //    for (int cycle = 0; cycle < maxcycle; ++cycle){
    //        // Step 1. Diagonalize the Hamiltonian in the P space
    //        int num_ref_roots = std::min(nroot_,int(P_space_.size()));

    //        outfile->Printf("\n\n  Cycle %3d",cycle);
    //        outfile->Printf("\n  %s: %zu determinants","Dimension of the P
    //        space",P_space_.size());
    //        

    SharedMatrix H;
    SharedMatrix P_evecs;
    SharedMatrix PQ_evecs;
    SharedVector P_evals;
    SharedVector PQ_evals;

    //    // Use the reference determinant as a starting point
    //    std::vector<bool> alfa_bits = reference_determinant_.get_alfa_bits_vector_bool();
    //    std::vector<bool> beta_bits = reference_determinant_.get_beta_bits_vector_bool();
    //    SharedDynamicBitsetDeterminant bs_det(alfa_bits,beta_bits);
    //    P_space_.push_back(bs_det);
    //    P_space_map_[bs_det] = 1;

    //    outfile->Printf("\n  The model space contains %zu determinants",P_space_.size());
    //    

    //    double old_avg_energy = reference_determinant_.energy() + nuclear_repulsion_energy_;
    //    double new_avg_energy = 0.0;

    //    std::vector<std::vector<double> > energy_history;
    //    SparseCISolver sparse_solver;
    //    int maxcycle = 20;
    //    for (int cycle = 0; cycle < maxcycle; ++cycle){
    //        // Step 1. Diagonalize the Hamiltonian in the P space
    //        int num_ref_roots = std::min(nroot_,int(P_space_.size()));

    //        outfile->Printf("\n\n  Cycle %3d",cycle);
    //        outfile->Printf("\n  %s: %zu determinants","Dimension of the P
    //        space",P_space_.size());
    //        

    //        sparse_solver.diagonalize_hamiltonian(P_space_,P_evals,P_evecs,nroot_,DavidsonLiuSparse);

    //        // Step 2. Find determinants in the Q space
    //        find_q_space(num_ref_roots,P_evals,P_evecs);

    //        // Step 3. Diagonalize the Hamiltonian in the P + Q space
    //        sparse_solver.diagonalize_hamiltonian(PQ_space_,PQ_evals,PQ_evecs,nroot_,DavidsonLiuSparse);

    //        // Print the energy
    //        outfile->Printf("\n");
    //        for (int i = 0; i < nroot_; ++ i){
    //            double abs_energy = PQ_evals->get(i) + nuclear_repulsion_energy_;
    //            double exc_energy = pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
    //            outfile->Printf("\n    PQ-space CI Energy Root %3d        = %.12f Eh = %8.4f eV",i
    //            + 1,abs_energy,exc_energy);
    //            outfile->Printf("\n    PQ-space CI Energy + EPT2 Root %3d = %.12f Eh = %8.4f eV",i
    //            + 1,abs_energy + multistate_pt2_energy_correction_[i],
    //                            exc_energy + pc_hartree2ev * (multistate_pt2_energy_correction_[i]
    //                            - multistate_pt2_energy_correction_[0]));
    //        }
    //        outfile->Printf("\n");
    //        

    //        // Step 4. Check convergence and break if needed
    //        bool converged = check_convergence(energy_history,PQ_evals);
    //        if (converged) break;

    //        // Step 5. Prune the P + Q space to get an update P space
    //        prune_q_space(PQ_space_,P_space_,P_space_map_,PQ_evecs,nroot_);

    //        // Print information about the wave function
    //        print_wfn(PQ_space_,PQ_evecs,nroot_);
    //    }

    //    outfile->Printf("\n\n  ==> Post-Iterations <==\n");
    //    for (int i = 0; i < nroot_; ++ i){
    //        double abs_energy = PQ_evals->get(i) + nuclear_repulsion_energy_;
    //        double exc_energy = pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));
    //        outfile->Printf("\n  * Adaptive-CI Energy Root %3d        = %.12f Eh = %8.4f eV",i +
    //        1,abs_energy,exc_energy);
    //        outfile->Printf("\n  * Adaptive-CI Energy Root %3d + EPT2 = %.12f Eh = %8.4f eV",i +
    //        1,abs_energy + multistate_pt2_energy_correction_[i],
    //                exc_energy + pc_hartree2ev * (multistate_pt2_energy_correction_[i] -
    //                multistate_pt2_energy_correction_[0]));
    //    }
    //    outfile->Printf("\n\n  %s: %f s","Adaptive-CI (bitset) ran in ",t_iamrcisd.elapsed());
    //    outfile->Printf("\n\n  %s: %d","Saving information for root",options_.get_int("ROOT") +
    //    1);
    //    

    //    return PQ_evals->get(options_.get_int("ROOT")) + nuclear_repulsion_energy_;
    return 0.0;
}

void GeneticAlgorithmCI::generate_initial_pop(
    std::vector<SharedDynamicBitsetDeterminant>& population, int size,
    std::unordered_map<std::vector<bool>, bool>& unique_list) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, ncmo_ - 1);
    auto rand_int = std::bind(distribution, generator);

    std::vector<bool> occupation_ab(2 * ncmo_, false);

    for (int I = 0; I < size; ++I) {
        while (true) {
            std::vector<bool> occupation_a(ncmo_, false);
            std::vector<bool> occupation_b(ncmo_, false);

            int symmetry = 0;

            for (int na = 0; na < ncalpha_;) {
                int i = rand_int();
                if (not occupation_a[i]) {
                    occupation_a[i] = true;
                    symmetry = symmetry ^ mo_symmetry_[i];
                    na++;
                }
            }

            for (int nb = 0; nb < ncbeta_;) {
                int i = rand_int();
                if (not occupation_b[i]) {
                    occupation_b[i] = true;
                    symmetry = symmetry ^ mo_symmetry_[i];
                    nb++;
                }
            }
            if (symmetry == wavefunction_symmetry_) {
                for (int i = 0; i < ncmo_; ++i) {
                    occupation_ab[i] = occupation_a[i];
                    occupation_ab[i + ncmo_] = occupation_b[i];
                }
                if (unique_list.count(occupation_ab) == 0) {
                    unique_list[occupation_ab] = true;
                    SharedDynamicBitsetDeterminant det(
                        new DynamicBitsetDeterminant(occupation_a, occupation_b));
                    population.push_back(det);
                    break;
                }
            }
        }
    }
}

void GeneticAlgorithmCI::weed_out(std::vector<SharedDynamicBitsetDeterminant>& population,
                                  std::vector<double> fitness,
                                  std::unordered_map<std::vector<bool>, bool>& unique_list) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, population.size() - 1);
    auto rand_int = std::bind(distribution, generator);

    int ndel = double(population.size()) * 0.2;

    for (int K = 0; K < ndel; ++K) {
        int I = rand_int() % population.size();
        if (fitness[I] < 0.0001) {
            SharedDynamicBitsetDeterminant detI = population[I];
            std::vector<bool> occupation_ab(2 * ncmo_, false);
            for (int i = 0; i < ncmo_; ++i) {
                occupation_ab[i] = detI->get_alfa_bit(i);
                occupation_ab[i + ncmo_] = detI->get_beta_bit(i);
            }
            population.erase(population.begin() + I);
            fitness.erase(fitness.begin() + I);
            unique_list.erase(occupation_ab);
        }
    }
}

void GeneticAlgorithmCI::crossover(std::vector<SharedDynamicBitsetDeterminant>& population,
                                   std::vector<double> fitness,
                                   std::unordered_map<std::vector<bool>, bool>& unique_list) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, population.size() - 1);
    auto rand_int = std::bind(distribution, generator);

    std::uniform_int_distribution<int> distribution_cmo(0, ncmo_ - 1);
    auto rand_cmo = std::bind(distribution_cmo, generator);

    std::uniform_real_distribution<double> distribution_real(-0.001, 0.001);
    auto rand_real = std::bind(distribution_real, generator);

    std::vector<bool> occupation_ab(2 * ncmo_, false);

    for (int K = 0; K < npop_; ++K) {
        while (true) {
            int I = rand_int();
            int J = rand_int();
            double importance = fitness[I] + fitness[J];
            if (importance > rand_real()) {
                SharedDynamicBitsetDeterminant detI = population[I];
                SharedDynamicBitsetDeterminant detJ = population[J];
                std::vector<bool> occupation_a(ncmo_, false);
                std::vector<bool> occupation_b(ncmo_, false);
                std::vector<bool> unoccupied_a(ncmo_, false);
                std::vector<bool> unoccupied_b(ncmo_, false);

                int symmetry = 0;

                int nafix = 0;
                int nbfix = 0;
                for (int i = 0; i < ncmo_; ++i) {
                    if (detI->get_alfa_bit(i) and detJ->get_alfa_bit(i)) {
                        occupation_a[i] = true;
                        symmetry = symmetry ^ mo_symmetry_[i];
                        nafix++;
                    }
                    if (not detI->get_alfa_bit(i) and not detJ->get_alfa_bit(i)) {
                        unoccupied_a[i] = true;
                    }
                    if (detI->get_beta_bit(i) and detJ->get_beta_bit(i)) {
                        occupation_b[i] = true;
                        symmetry = symmetry ^ mo_symmetry_[i];
                        nbfix++;
                    }
                    if (not detI->get_beta_bit(i) and not detJ->get_beta_bit(i)) {
                        unoccupied_b[i] = true;
                    }
                }

                for (int na = 0; na < ncalpha_ - nafix;) {
                    int i = rand_cmo();
                    if ((not occupation_a[i]) and (not unoccupied_a[i])) {
                        occupation_a[i] = true;
                        symmetry = symmetry ^ mo_symmetry_[i];
                        na++;
                    }
                }

                for (int nb = 0; nb < ncbeta_ - nbfix;) {
                    int i = rand_cmo();
                    if ((not occupation_b[i]) and (not unoccupied_b[i])) {
                        occupation_b[i] = true;
                        symmetry = symmetry ^ mo_symmetry_[i];
                        nb++;
                    }
                }

                if (symmetry == wavefunction_symmetry_) {
                    for (int i = 0; i < ncmo_; ++i) {
                        occupation_ab[i] = occupation_a[i];
                        occupation_ab[i + ncmo_] = occupation_b[i];
                    }
                    if (unique_list.count(occupation_ab) == 0) {
                        unique_list[occupation_ab] = true;
                        SharedDynamicBitsetDeterminant det(
                            new DynamicBitsetDeterminant(occupation_a, occupation_b));
                        population.push_back(det);
                        break;
                    }
                }
            }
        }
    }

    for (int I = 0; I < population.size(); ++I) {
        for (int J = I + 1; J < population.size(); ++J) {
            if (*population[I] == *population[J]) {
                outfile->Printf("\n  Determinants %d and %d are identical", I, J);
            }
        }
    }
}
}
} // EndNamespaces
