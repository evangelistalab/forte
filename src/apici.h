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

#ifndef _adaptive_pifci_h_
#define _adaptive_pifci_h_

#include <fstream>

#include <libmints/wavefunction.h>
#include <liboptions/liboptions.h>
#include <physconst.h>
#include <boost/unordered_map.hpp>

#include "forte-def.h"
#include "integrals.h"
#include "stl_bitset_determinant.h"
#include "dynamic_bitset_determinant.h"
#include "helpers.h"
#include "fci_vector.h"

namespace psi{ namespace forte{

enum PropagatorType {LinearPropagator,
                     QuadraticPropagator,
                     CubicPropagator,
                     QuarticPropagator,
                     PowerPropagator,
                     OlsenPropagator,
                     DavidsonLiuPropagator};

/**
 * @brief The SparsePathIntegralCI class
 * This class implements an a sparse path-integral FCI algorithm
 */
class AdaptivePathIntegralCI : public Wavefunction
{
public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param wfn The main wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     */
    AdaptivePathIntegralCI(boost::shared_ptr<Wavefunction> wfn, Options &options, std::shared_ptr<ForteIntegrals>  ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

private:

    // ==> Class data <==

    // * Calculation data
    /// A reference to the options object
    Options& options_;
    /// The molecular integrals required by Explorer
    std::shared_ptr<ForteIntegrals>  ints_;
    /// Store all the integrals locally
    static std::shared_ptr<FCIIntegrals> fci_ints_;
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
    /// The multiplicity of the wave function
    int wavefunction_multiplicity_;
    /// The nuclear repulsion energy
    double nuclear_repulsion_energy_;
    /// The reference determinant
    Determinant reference_determinant_;
    std::vector<det_hash<>> solutions_;
    /// The information of mo space
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// (pq|pq) matrix for prescreening
    double *pqpq_aa_, *pqpq_ab_, *pqpq_bb_;
    /// maximum element in (pq|pq) matrix
    double pqpq_max_aa_, pqpq_max_ab_, pqpq_max_bb_;
    /// maximum element in (pq|pq) matrix
    std::vector<double> pqpq_row_max_;
    /// 2loop total count
    size_t schwarz_total_;
    /// 2loop schwarz succeed count
    size_t schwarz_succ_;

    // * Calculation info
    /// The threshold applied to the primary space
    double spawning_threshold_;
    /// The maximum size of the guess wave function
    size_t max_guess_size_;
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
    /// Prescreen spawning using schwarz inequality
    bool do_schwarz_prescreening_;
    /// Prescreen spawning using initiator approximation
    bool do_initiator_approx_;
    /// Initiator approximation factor
    double initiator_approx_factor_;
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
    std::unordered_map<Determinant,std::pair<double,double>,Determinant::Hash> dets_max_couplings_;

    // * Energy estimation
    /// Estimate the variational energy via a fast procedure?
    bool fast_variational_estimate_;
    /// The frequency of approximate variational estimation of the energy
    int energy_estimate_freq_;
    /// The threshold with which we estimate the energy during the iterations
    double energy_estimate_threshold_;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();

    /// Print information about this calculation
    void print_info();

    /// Print a wave function
    void print_wfn(det_vec &space, std::vector<double> &C);

    /// Save a wave function
    void save_wfn(det_vec &space, std::vector<double> &C,std::vector<det_hash<>>& solutions);

    /// Orthogonalize the wave function to previous solutions
    void orthogonalize(det_vec& space,std::vector<double>& C,std::vector<det_hash<>>& solutions);

    /// Initial wave function guess
    double initial_guess(det_vec& dets,std::vector<double>& C);

    /**
    * Propagate the wave function by a step of length tau
    * @param propagator The type of propagator used
    * @param dets The set of determinants that form the wave function at time n
    * @param C The wave function coefficients at time n
    * @param tau The time step in a.u.
    * @param spawning_threshold The threshold used to accept or reject spawning events
    * @param S An energy shift subtracted from the Hamiltonian
    */
    void propagate(PropagatorType propagator,det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// A first-order propagator
    void propagate_first_order(det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// An experimental second-order propagator
    void propagate_second_order(det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// An experimental arbitrary-order Taylor series propagator
    void propagate_Taylor(int order,det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// The power propagator
    void propagate_power(det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// The power propagator
    void propagate_power_quadratic_extrapolation(det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// The Olsen propagator
    void propagate_Olsen(det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// The Davidson-Liu propagator
    void propagate_DavidsonLiu(det_vec& dets, std::vector<double>& C, double tau, double spawning_threshold);

    /// Apply tau H to a set of determinants
    void apply_tau_H(double tau, double spawning_threshold, det_vec &dets, const std::vector<double>& C, det_hash<>& dets_C_map, double S);
    /// Apply tau H to a determinant using screening based on the maxim couplings
    std::pair<double, double> apply_tau_H_det_prescreening(double tau, double spawning_threshold, Determinant& detI, double CI, std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0);
    /// Apply tau H to a determinant using dynamic screening
    void apply_tau_H_det_dynamic(double tau,double spawning_threshold,const Determinant& detI, double CI, std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0,std::pair<double,double>& max_coupling);
    /// Apply tau H to a determinant using Schwarz screening
    void apply_tau_H_det_schwarz(double tau, double spawning_threshold, const Determinant &detI, double CI, std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0);

    /// Estimates the energy give a wave function
    std::map<std::string, double> estimate_energy(det_vec& dets,std::vector<double>& C);
    /// Estimates the projective energy
    double estimate_proj_energy(det_vec& dets,std::vector<double>& C);
    /// Estimates the variational energy
    /// @param dets The set of determinants that form the wave function
    /// @param C The wave function coefficients
    /// @param tollerance The accuracy of the estimate.  Used to impose |C_I C_J| < tollerance
    double estimate_var_energy(det_vec& dets, std::vector<double>& C, double tollerance = 1.0e-14);
    /// Estimates the variational energy using a sparse algorithm
    /// @param dets The set of determinants that form the wave function
    /// @param C The wave function coefficients
    /// @param tollerance The accuracy of the estimate.  Used to impose |C_I C_J| < tollerance
    double estimate_var_energy_sparse(det_vec& dets, std::vector<double>& C, double tollerance = 1.0e-14);

    /// Form the product H c
    double form_H_C(double tau,double spawning_threshold,Determinant& detI, double CI, det_hash<>& det_C,std::pair<double,double>& max_coupling);
    /// Do we have OpenMP?
    static bool have_omp_;
};

}} // End Namespaces

#endif // _adaptive_pifci_h_
