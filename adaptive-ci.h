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

#ifndef _adaptive_ci_h_
#define _adaptive_ci_h_

#include <fstream>

#include <libmints/wavefunction.h>
#include <liboptions/liboptions.h>
#include <physconst.h>

#include "integrals.h"
#include "string_determinant.h"
#include "bitset_determinant.h"

namespace psi{ namespace libadaptive{

/**
 * @brief The AdaptiveCI class
 * This class implements an adaptive CI algorithm
 */
class AdaptiveCI : public Wavefunction
{
public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param wfn The main wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     */
    AdaptiveCI(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints);

    /// Destructor
    ~AdaptiveCI();

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

private:

    // ==> Class data <==

    /// A reference to the options object
    Options& options_;
    /// The molecular integrals required by Explorer
    ExplorerIntegrals* ints_;
    /// The wave function symmetry
    int wavefunction_symmetry_;
    /// The symmetry of each orbital in Pitzer ordering
    std::vector<int> mo_symmetry_;
    /// The number of correlated molecular orbitals
    int ncmo_;
    /// The number of correlated molecular orbitals per irrep
    Dimension ncmopi_;
    /// The nuclear repulsion energy
    double nuclear_repulsion_energy_;
    /// The reference determinant
    StringDeterminant reference_determinant_;
    /// The PT2 energy correction
    std::vector<double> multistate_pt2_energy_correction_;

    /// The threshold applied to the primary space
    double tau_p_;
    /// The threshold applied to the secondary space
    double tau_q_;
    /// The number of roots computed
    int nroot_;
    /// Enable aimed selection
    bool aimed_selection_;
    /// If true select by energy, if false use first-order coefficient
    bool energy_selection_;

    /// A vector of determinants in the P space
    std::vector<BitsetDeterminant> P_space_;
    /// A vector of determinants in the P + Q space
    std::vector<BitsetDeterminant> PQ_space_;
    /// A map of determinants in the P space
    std::map<BitsetDeterminant,int> P_space_map_;


    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();

    /// Print information about this calculation
    void print_info();

    /// Print a wave function
    void print_wfn(std::vector<BitsetDeterminant> space, SharedMatrix evecs, int nroot);

    /// Diagonalize the Hamiltonian in a space of determinants
    void diagonalize_hamiltonian(const std::vector<BitsetDeterminant>& space, SharedVector &evals, SharedMatrix &evecs, int nroot);

    /// Diagonalize the Hamiltonian in a space of determinants
    void diagonalize_hamiltonian2(const std::vector<BitsetDeterminant>& space, SharedVector &evals, SharedMatrix &evecs, int nroot);

    /// Find all the relevant SD excitations out of the P space.
    void find_q_space(int nroot, SharedVector evals, SharedMatrix evecs);

    /// Prune the space of determinants
    void prune_q_space(std::vector<BitsetDeterminant>& large_space,std::vector<BitsetDeterminant>& pruned_space,
                                   std::map<BitsetDeterminant,int>& pruned_space_map,SharedMatrix evecs,int nroot);

    /// Check if the procedure has converged
    bool check_convergence(std::vector<std::vector<double>>& energy_history,SharedVector new_energies);

    int david2(double **A, int N, int M, double *eps, double **v,double cutoff, int print);
    /// Perform a Davidson-Liu diagonalization
    void davidson_liu(SharedMatrix H,SharedVector Eigenvalues,SharedMatrix Eigenvectors,int nroots);

    /// Perform a Davidson-Liu diagonalization on a sparse matrix
    bool davidson_liu_sparse(std::vector<std::vector<std::pair<int,double> > > H_sparse,SharedVector Eigenvalues,SharedMatrix Eigenvectors,int nroots);
};

}} // End Namespaces

#endif // _adaptive_ci_h_
