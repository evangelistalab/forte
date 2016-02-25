/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#ifndef _ga_ci_h_
#define _ga_ci_h_

#include <fstream>
#include <unordered_map>

#include <libmints/wavefunction.h>
#include <liboptions/liboptions.h>
#include <physconst.h>

#include "integrals.h"
#include "string_determinant.h"
#include "dynamic_bitset_determinant.h"

namespace psi{ namespace forte{

/**
 * @brief The Genetic Algorithm CI class
 * This class implements a genetic CI algorithm
 */
class GeneticAlgorithmCI : public Wavefunction
{
public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     */
    GeneticAlgorithmCI(boost::shared_ptr<Wavefunction> wfn, Options &options, ForteIntegrals* ints);

    /// Destructor
    ~GeneticAlgorithmCI();

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

private:

    // ==> Class data <==

    /// A reference to the options object
    Options& options_;
    /// The molecular integrals required by Explorer
    ForteIntegrals* ints_;
    /// The wave function symmetry
    int wavefunction_symmetry_;
    /// The symmetry of each orbital in Pitzer ordering
    std::vector<int> mo_symmetry_;
    /// The number of correlated molecular orbitals
    int ncmo_;
    /// The number of correlated molecular orbitals per irrep
    Dimension ncmopi_;
    /// The number of correlated alpha electrons
    int ncalpha_;
    /// The number of correlated beta electrons
    int ncbeta_;
    /// The nuclear repulsion energy
    double nuclear_repulsion_energy_;

    /// The size of the population
    int npop_;

    /// The reference determinant
    SharedDynamicBitsetDeterminant reference_determinant_;
//    /// The PT2 energy correction
//    std::vector<double> multistate_pt2_energy_correction_;

//    /// The threshold applied to the primary space
//    double tau_p_;
//    /// The threshold applied to the secondary space
//    double tau_q_;
    /// The number of roots computed
    int nroot_;
    /// The root used to compute properties and select the determinants
    int root_;

//    /// Enable aimed selection
//    bool aimed_selection_;
//    /// If true select by energy, if false use first-order coefficient
//    bool energy_selection_;

//    /// A vector of determinants in the P space
//    std::vector<SharedDynamicBitsetDeterminant> P_space_;
//    /// A vector of determinants in the P + Q space
//    std::vector<SharedDynamicBitsetDeterminant> PQ_space_;
//    /// A map of determinants in the P space
//    std::map<SharedDynamicBitsetDeterminant,int> P_space_map_;


    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();

    /// Print information about this calculation
    void print_info();

    /// Generate an initial random population of a given size
    void generate_initial_pop(std::vector<SharedDynamicBitsetDeterminant>& population,int size,std::unordered_map<std::vector<bool>,bool>& unique_list);

    /// Weed out the population
    void weed_out(std::vector<SharedDynamicBitsetDeterminant>& population,std::vector<double> fitness,std::unordered_map<std::vector<bool>,bool>& unique_list);

    /// Crossover the population
    void crossover(std::vector<SharedDynamicBitsetDeterminant>& population,std::vector<double> fitness,std::unordered_map<std::vector<bool>,bool>& unique_list);


//    /// Print a wave function
//    void print_wfn(std::vector<SharedDynamicBitsetDeterminant> space, SharedMatrix evecs, int nroot);

//    /// Diagonalize the Hamiltonian in a space of determinants
//    void diagonalize_hamiltonian(const std::vector<SharedDynamicBitsetDeterminant>& space, SharedVector &evals, SharedMatrix &evecs, int nroot);

//    /// Diagonalize the Hamiltonian in a space of determinants
//    void diagonalize_hamiltonian2(const std::vector<SharedDynamicBitsetDeterminant>& space, SharedVector &evals, SharedMatrix &evecs, int nroot);

//    /// Find all the relevant SD excitations out of the P space.
//    void find_q_space(int nroot, SharedVector evals, SharedMatrix evecs);

//    /// Prune the space of determinants
//    void prune_q_space(std::vector<SharedDynamicBitsetDeterminant>& large_space,std::vector<SharedDynamicBitsetDeterminant>& pruned_space,
//                                   std::map<SharedDynamicBitsetDeterminant,int>& pruned_space_map,SharedMatrix evecs,int nroot);

//    /// Check if the procedure has converged
//    bool check_convergence(std::vector<std::vector<double>>& energy_history,SharedVector new_energies);

};

}} // End Namespaces

#endif // _ga_ci_h_
