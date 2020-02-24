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

#ifndef _fcimc_h_
#define _fcimc_h_

#include <fstream>


#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include <random>

#include "integrals/integrals.h"
#include "sparse_ci/determinant.h"
#include "fci/fci_vector.h"

namespace psi {
namespace forte {

enum SpawnType { random, all, ground_and_random };

typedef std::map<Determinant, double> walker_map;

struct ObtCount {
    std::vector<int> naocc, nbocc, navir, nbvir;
};

class MOSpaceInfo;

class FCIQMC : public Wavefunction {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     */
    FCIQMC(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
           std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~FCIQMC();

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

  private:
    //    /// The wave function symmetry
    //    int wavefunction_symmetry_;
    //    /// The symmetry of each orbital in Pitzer ordering
    //    std::vector<int> mo_symmetry_;
    //    /// The symmetry of each orbital in the qt ordering
    //    std::vector<int> mo_symmetry_qt_;
    //    /// A vector that contains all the frozen core
    //    std::vector<int> frzc_;
    //    /// A vector that contains all the frozen virtual
    //    std::vector<int> frzv_;
    //    /// The nuclear repulsion energy
    //    double nuclear_repulsion_energy_;

    //    int compute_pgen(Determinant& detI);
    /// The reference determinant
    Determinant reference_;
    /// The maximum number of threads
    int num_threads_;
    /// Do we have OpenMP?
    static bool have_omp_;
    /// The molecular integrals required by fcimc
    std::shared_ptr<ForteIntegrals> ints_;
    /// The information of mo space
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// Store all the integrals locally
    static std::shared_ptr<FCIIntegrals> fci_ints_;

    /// The wave function symmetry
    int wavefunction_symmetry_;
    /// The symmetry of each orbital in Pitzer ordering
    std::vector<int> mo_symmetry_;
    /// The number of correlated molecular orbitals
    int ncmo_;
    /// The number of correlated molecular orbitals per irrep
    Dimension ncmopi_;
    /// The cumulative number of correlated molecular orbitals per irrep
    std::vector<int> cume_ncmopi_;
    /// cume number of irrep combination catagories per alpha beta combination:
    /// ab->ab, aiai->ajaj, aiaj->akal, bibi->bjbj, bibj->bkbl, a->a, b->b
    size_t cume_excit_irrep_[7];
    /// Nuclear repulsion energy
    double nuclear_repulsion_energy_;
    /// number of excitations by category
    size_t nsa_, nsb_, ndaa_, ndab_, ndbb_;
    size_t sumgen_;
    size_t cume_sumgen_[5];

    // * Calculation info
    /// spawn type
    SpawnType spawn_type_;
    /// The size of the time step (TAU)
    double time_step_;
    /// The maximum number of FCIQMC steps
    size_t maxiter_;
    /// HartreeForkEnergy
    double Ehf_;
    /// Projectional energy
    double Eproj_, AvgEproj_, ErrEproj_;
    /// Variational energy
    double Evar_;
    /// Start Number of walkers
    double start_num_walkers_;
    /// The shift of energy
    double shift_, AvgShift_, ErrShift_;
    /// Number of walkers
    double nWalkers_;
    /// Number of determinants
    double nDets_;
    /// Shift the Hamiltonian?
    bool do_shift_;
    double shift_num_walkers_;
    int shift_freq_;
    double shift_damp_;
    /// Clone/Death only parents?
    bool death_parent_only_;
    /// Initiator
    bool use_initiator_;
    double initiator_na_;
    /// The frequency of approximate variational estimation of the energy
    int energy_estimate_freq_;
    /// The frequency of print information
    int print_freq_;
    /// number of iterations
    size_t iter_;

    void startup();
    void print_info();

    // adjust shift
    void adjust_shift(double pre_nWalker, size_t pre_iter);

    // Spawning step
    void spawn(walker_map& walkers, walker_map& new_walkers);
    void spawn_generative(walker_map& walkers, walker_map& new_walkers);
    void singleWalkerSpawn(Determinant& new_det, const Determinant& det,
                           std::tuple<size_t, size_t, size_t, size_t, size_t> pgen, size_t sumgen);
    // Death/Clone step
    void death_clone(walker_map& walkers, double shift);
    void detClone(walker_map& walkers, const Determinant& det, double coef, double pDeathClone);
    void detDeath(walker_map& walkers, const Determinant& det, double coef, double pDeathClone);
    // Merge step
    void merge(walker_map& walkers, walker_map& new_walkers);
    // Annihilation step
    void annihilate(walker_map& walkers, walker_map& new_walkers);

    // Count the number of allowed single and double excitations
    std::tuple<size_t, size_t, size_t, size_t, size_t> compute_pgen(const Determinant& det);
    std::tuple<size_t, size_t, size_t, size_t, size_t> compute_pgen_C1(const Determinant& det);
    void
    compute_excitations(const Determinant& det,
                        std::vector<std::tuple<size_t, size_t>>& singleExcitations,
                        std::vector<std::tuple<size_t, size_t, size_t, size_t>>& doubleExcitations);
    void compute_single_excitations(const Determinant& det,
                                    std::vector<std::tuple<size_t, size_t>>& singleExcitations);
    void compute_double_excitations(
        const Determinant& det,
        std::vector<std::tuple<size_t, size_t, size_t, size_t>>& doubleExcitations);
    size_t compute_irrep_divided_excitations(
        const Determinant& det, std::vector<size_t>& excitationDivides,
        std::vector<std::tuple<int, int, int, int>>& excitationType, ObtCount& obtCount);
    bool detSingleRandomExcitation(Determinant& new_det, const std::vector<int>& occ,
                                   const std::vector<int>& vir, bool isAlpha);
    void detSingleExcitation(Determinant& new_det, std::tuple<size_t, size_t>& rand_ext);
    void detDoubleExcitation(Determinant& new_det,
                             std::tuple<size_t, size_t, size_t, size_t>& rand_ext);
    bool detDoubleSoloSpinRandomExcitation(Determinant& new_det, const std::vector<int>& occ,
                                           const std::vector<int>& vir, bool isAlpha);
    bool detDoubleMixSpinRandomExcitation(Determinant& new_det, const std::vector<int>& aocc,
                                          const std::vector<int>& bocc,
                                          const std::vector<int>& avir,
                                          const std::vector<int>& bvir);
    void detExcitation(Determinant& new_det, size_t rand_ext,
                       std::vector<size_t>& excitationDivides,
                       std::vector<std::tuple<int, int, int, int>>& excitationType,
                       ObtCount& obtCount);
    double count_walkers(walker_map& walkers);
    double compute_proj_energy(Determinant& ref, walker_map& walkers);
    double compute_var_energy(walker_map& walkers);
    void print_iter_info(size_t iter);
    void compute_avg_Eproj(std::vector<double> Eprojs);
    void compute_err_Eproj(std::vector<double> Eprojs);
    void compute_avg_shift(std::vector<double> shifts);
    void compute_err_shift(std::vector<double> shifts);
};
}
} // End Namespaces

#endif // _fcimc_h_
