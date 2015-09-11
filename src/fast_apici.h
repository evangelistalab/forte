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

#ifndef _fast_adaptive_pifci_h_
#define _fast_adaptive_pifci_h_

#include <fstream>

#include <libmints/wavefunction.h>
#include <liboptions/liboptions.h>
#include <libpsio/psio.h>
#include <libpsio/psio.hpp>
#include <physconst.h>

#include "integrals.h"
#include "string_determinant.h"
#include "fast_determinant.h"

namespace psi{ namespace forte{


/**
 * @brief The SparsePathIntegralCI class
 * This class implements an a sparse path-integral FCI algorithm
 */
class FastAdaptivePathIntegralCI : public Wavefunction
{
    enum PropagatorType {LinearPropagator,
                         QuadraticPropagator,
                         CubicPropagator,
                         QuarticPropagator,
                         PowerPropagator,
                         TrotterLinearPropagator,
                         OlsenPropagator,
                         DavidsonLiuPropagator};
public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param wfn The main wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     */
    FastAdaptivePathIntegralCI(boost::shared_ptr<Wavefunction> wfn, Options &options, std::shared_ptr<ForteIntegrals>  ints, std::shared_ptr<psi::forte::MOSpaceInfo> mo_space_info);

    /// Destructor
    ~FastAdaptivePathIntegralCI();

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

    double compute_energy_parallel();

private:

    // ==> Class data <==

    // * Calculation data
    /// A reference to the options object
    Options& options_;
    /// The molecular integrals required by Explorer
    std::shared_ptr<ForteIntegrals>  ints_;
    /// The maximum number of threads
    int num_threads_;
    /// The type of propagator used
    PropagatorType propagator_;
    /// A string that describes the propagator type
    std::string propagator_description_;
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
    std::vector<std::map<FastDeterminant,double>> solutions_;
    /// The information of mo space
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// Store all the integrals locally
    FCIIntegrals fciInts_;


    // * Calculation info
    /// The threshold applied to the primary space
    double spawning_threshold_;
    /// The threshold applied during the initial guess
    double initial_guess_spawning_threshold_;
    /// The size of the time step (TAU)
    double time_step_;
    /// Use an adaptive time step?
    bool adaptive_beta_;
    /// Shift the Hamiltonian?
    bool do_shift_;
    /// Use intermediate normalization?
    bool use_inter_norm_;
    /// The number of roots computed
    int nroot_;
    /// The energy convergence criterium
    double e_convergence_;
    /// The maximum number of iterations
    int maxiter_;
    /// The current iteration
    int iter_;
    /// The current root
    int current_root_;
    /// Diagonalize the Hamiltonian in the APIFCI basis after running a ground state calculation?
    bool post_diagonalization_;

    // * Simple Prescreening
    /// Prescreen spawning using general integral upper bounds
    bool do_simple_prescreening_;
    /// Maximum value of the one-electron coupling
    double new_max_one_HJI_;
    double old_max_one_HJI_;
    /// Maximum value of the two-electron coupling
    double new_max_two_HJI_;
    double old_max_two_HJI_;
    /// The tollerance factor applied when prescreening singles
    double prescreening_tollerance_factor_;


    // * Dynamics Prescreening
    /// Prescreen spawning using a dynamic integral upper bounds
    bool do_dynamic_prescreening_;
    /// A map used to store the largest absolute value of the couplings of a
    /// determinant to all of its singly and doubly excited states.
    /// Bounds are stored as a pair (f_max,v_max) where f_max and v_max are
    /// the couplings to the singles and doubles, respectively.
    std::map<FastDeterminant,std::pair<double,double>> dets_max_couplings_;

    // * Energy estimation
    /// Estimate the variational energy via a fast procedure?
    bool fast_variational_estimate_;
    /// The frequency of approximate variational estimation of the energy
    int energy_estimate_freq_;
    /// The threshold with which we estimate the energy during the iterations
    double energy_estimate_threshold_;


    // * Calculation statistics
    /// Number of determinants visited during a time step
    size_t ndet_visited_;
    /// Number of determinants accepted during a time step
    size_t ndet_accepted_;
    /// Number of determinants spawned during a time step
    size_t nspawned_;
    /// Number of determinants that don't spawn
    size_t nzerospawn_;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();

    /// Print information about this calculation
    void print_info();

    /// Print a wave function
    void print_wfn(std::vector<FastDeterminant> &space, std::vector<double> &C);

    /// Save a wave function
    void save_wfn(std::vector<FastDeterminant> &space, std::vector<double> &C,std::vector<std::map<FastDeterminant,double>>& solutions);

    /// Orthogonalize the wave function to previous solutions
    void orthogonalize(std::vector<FastDeterminant>& space,std::vector<double>& C,std::vector<std::map<FastDeterminant,double>>& solutions);

    /// Initial wave function guess
    double initial_guess(std::vector<FastDeterminant>& dets,std::vector<double>& C);

    /**
    * Propagate the wave function by a step of length tau
    * @param propagator The type of propagator used
    * @param dets The set of determinants that form the wave function at time n
    * @param C The wave function coefficients at time n
    * @param tau The time step in a.u.
    * @param spawning_threshold The threshold used to accept or reject spawning events
    * @param S An energy shift subtracted from the Hamiltonian
    */
    void propagate(PropagatorType propagator,std::vector<FastDeterminant>& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);

    /// A first-order propagator
    void propagate_first_order(std::vector<FastDeterminant>& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);

    /// An experimental second-order propagator
    void propagate_second_order(std::vector<FastDeterminant>& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);

    /// An experimental arbitrary-order Taylor series propagator
    void propagate_Taylor(int order,std::vector<FastDeterminant>& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);

    /// An experimental arbitrary-order Chebyshev series propagator
    void propagate_Chebyshev(int order,std::vector<FastDeterminant>& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);

    /// The power propagator
    void propagate_power(std::vector<FastDeterminant>& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);

    /// The power propagator
    void propagate_power_quadratic_extrapolation(std::vector<FastDeterminant>& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);

    /// The Trotter propagator
    void propagate_Trotter(std::vector<FastDeterminant>& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);

    /// The Olsen propagator
    void propagate_Olsen(std::vector<FastDeterminant>& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);

    /// The Davidson-Liu propagator
    void propagate_DavidsonLiu(std::vector<FastDeterminant>& dets, std::vector<double>& C, double tau, double spawning_threshold);

    /// Estimates the energy give a wave function
    std::map<std::string, double> estimate_energy(std::vector<FastDeterminant>& dets,std::vector<double>& C);

    /// Estimates the projective energy
    double estimate_proj_energy(std::vector<FastDeterminant>& dets,std::vector<double>& C);

    /// Estimates the variational energy
    /// @param dets The set of determinants that form the wave function
    /// @param C The wave function coefficients
    /// @param tollerance The accuracy of the estimate.  Used to impose |C_I C_J| < tollerance
    double estimate_var_energy(std::vector<FastDeterminant>& dets, std::vector<double>& C, double tollerance = 1.0e-14);

    /// Estimates the variational energy using a sparse algorithm
    /// @param dets The set of determinants that form the wave function
    /// @param C The wave function coefficients
    /// @param tollerance The accuracy of the estimate.  Used to impose |C_I C_J| < tollerance
    double estimate_var_energy_sparse(std::vector<FastDeterminant>& dets, std::vector<double>& C, double tollerance = 1.0e-14);

    /// Perform a time step
    double time_step_optimized(double spawning_threshold,FastDeterminant& detI, double CI, std::map<FastDeterminant,double>& new_space_C, double E0);

    /// Apply tau H to a determinant using dynamic screening
    size_t apply_tau_H(double tau, double spawning_threshold, std::vector<FastDeterminant> &dets, const std::vector<double>& C, std::map<FastDeterminant,double>& dets_C_map, double S);
//    size_t apply_tau_H(double tau,double spawning_threshold,std::map<FastDeterminant,double>& det_C_old, std::map<FastDeterminant,double>& dets_C_map, double S);

    /// Apply tau H to a determinant
    size_t apply_tau_H_det(double tau,double spawning_threshold,const FastDeterminant& detI, double CI, std::map<FastDeterminant,double>& new_space_C, double E0);


    /// Apply tau H to a determinant using dynamic screening
    size_t apply_tau_H_det_dynamic(double tau,double spawning_threshold,const FastDeterminant& detI, double CI, std::map<FastDeterminant,double>& new_space_C, double E0,std::pair<double,double>& max_coupling);

    /// Form the product H c
    double form_H_C(double tau,double spawning_threshold,FastDeterminant& detI, double CI, std::map<FastDeterminant,double>& det_C,std::pair<double,double>& max_coupling);

    static void scale(std::vector<double>& A,double alpha);
    static double normalize(std::vector<double>& C);
    static double normalize(std::map<FastDeterminant,double>& dets_C);
    static bool have_omp;
};

}} // End Namespaces

#endif // _fast_adaptive_pifci_h_
