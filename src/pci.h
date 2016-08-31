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

#ifndef _pci_h_
#define _pci_h_

#include <fstream>

#include <libmints/wavefunction.h>
#include <liboptions/liboptions.h>
#include <physconst.h>

#include "forte-def.h"
#include "integrals.h"
#include "stl_bitset_determinant.h"
#include "dynamic_bitset_determinant.h"
#include "sparse_ci_solver.h"
#include "helpers.h"
#include "fci_vector.h"

namespace psi{ namespace forte{

enum GeneratorType {LinearGenerator,
                     TrotterLinear,
                     QuadraticGenerator,
                     CubicGenerator,
                     QuarticGenerator,
                     PowerGenerator,
                     OlsenGenerator,
                     DavidsonLiuGenerator,
                     ExpChebyshevGenerator,
                     DeltaChebyshevGenerator,
                     ChebyshevGenerator,
                     DeltaGenerator,
                     LanczosGenerator};

/**
 * @brief The SparsePathIntegralCI class
 * This class implements an a sparse path-integral FCI algorithm
 */
class ProjectorCI : public Wavefunction
{
public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     */
    ProjectorCI(SharedWavefunction ref_wfn, Options &options, std::shared_ptr<ForteIntegrals>  ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

private:

    // ==> Class data <==

    // * Calculation data
    /// The molecular integrals required by Explorer
    std::shared_ptr<ForteIntegrals>  ints_;
    /// Store all the integrals locally
    static std::shared_ptr<FCIIntegrals> fci_ints_;
    /// The maximum number of threads
    int num_threads_;
    /// The type of Generator used
    GeneratorType generator_;
    /// A string that describes the Generator type
    std::string generator_description_;
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
    /// The threshold applied for initial guess
    double initial_guess_spawning_threshold_;
    /// The maximum size of the guess wave function
    size_t max_guess_size_;
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
    /// The eigensolver type
    DiagonalizationMethod diag_method_;
    /// Print full wavefunction in the APIFCI basis after running a ground state calculation?
    bool print_full_wavefunction_;

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
    /// Estimate the variational energy?
    bool variational_estimate_;
    /// Estimate the variational energy via a fast procedure?
    bool fast_variational_estimate_;
    /// The frequency of approximate variational estimation of the energy
    int energy_estimate_freq_;
    /// The threshold with which we estimate the energy during the iterations
    double energy_estimate_threshold_;
    /// Flag for conducting CHC energy estimation
    bool approx_E_flag_;
    double approx_energy_, old_approx_energy_;
    double approx_E_tau_, approx_E_S_;


    // * Energy extrapolation
    /// Estimated variational energy at each step
    std::vector<std::pair<double, double>> iter_Evar_steps_;
//    std::tuple<double, double, double> fit_exp(std::vector<std::pair<double, double>> data);
//    std::tuple<double, double, double, double> fit_Aetx_c_opt(std::vector<std::pair<double, double>> data, double threshold);
//    std::pair<double, double> fit_Aetx_c_opt(std::vector<std::pair<double, double>> data, double threshold);

    // * Chebyshev Generator
    /// Range of Hamiltonian
    double range_;
    /// Order of Chebyshev truncate
    int chebyshev_order_;
    /// Order of Krylov subspace truncate
    int krylov_order_;
    /// Threshold for norm of orthogonal basis to be colinear.
    double colinear_threshold_;

    // * Convergence analysis
    /// Shift of Hamiltonian
    double shift_;
    /// lowest e-value in initial guess
    double lambda_1_;
    /// Second lowest e-value in initial guess
//    double lambda_2_;
    /// Highest possible e-value
    double lambda_h_;
    /// Characteristic function coefficients
    std::vector<double> cha_func_coefs_;
    /// Do result perturbation analysis
    bool do_perturb_analysis_;
    /// Use symmetric approximated hamiltonian
    bool symm_approx_H_;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();

    /// Print information about this calculation
    void print_info();

    /// Print a wave function
    void print_wfn(det_vec &space, std::vector<double> &C, size_t max_output = 10);

    /// Save a wave function
    void save_wfn(det_vec &space, std::vector<double> &C,std::vector<det_hash<>>& solutions);

    /// Orthogonalize the wave function to previous solutions
    void orthogonalize(det_vec& space,std::vector<double>& C,std::vector<det_hash<>>& solutions);

    /// Initial wave function guess
    double initial_guess(det_vec& dets,std::vector<double>& C);

    /**
    * Propagate the wave function by a step of length tau
    * @param Generator The type of Generator used
    * @param dets The set of determinants that form the wave function at time n
    * @param C The wave function coefficients at time n
    * @param tau The time step in a.u.
    * @param spawning_threshold The threshold used to accept or reject spawning events
    * @param S An energy shift subtracted from the Hamiltonian
    */
    void propagate(GeneratorType generator,det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// A Delta projector fitted by 10th order chebyshev polynomial
    void propagate_delta(det_vec& dets,std::vector<double>& C,double spawning_threshold,double S);
    /// A first-order Generator
    void propagate_Linear(det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// An Trotter-decomposed Generator (H = H^d + H^od)
    void propagate_Trotter_linear(det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// An experimental second-order Generator
    void propagate_second_order(det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// An experimental arbitrary-order Taylor series Generator
    void propagate_Taylor(int order,det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// The power Generator
    void propagate_power(det_vec& dets, std::vector<double>& C, double spawning_threshold, double S);
    /// The power Generator
    void propagate_power_quadratic_extrapolation(det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// The Olsen Generator
    void propagate_Olsen(det_vec& dets, std::vector<double>& C, double spawning_threshold, double S);
    /// The Davidson-Liu Generator
    void propagate_DavidsonLiu(det_vec& dets, std::vector<double>& C, double spawning_threshold);
    /// The Chebyshev Generator
    void propagate_Chebyshev(det_vec& dets, std::vector<double>& C, double spawning_threshold);
//    void propagate_Chebyshev(det_vec& dets,std::vector<double>& C,double tau,double spawning_threshold,double S);
    /// The Polynomial Generator
    void propagate_Polynomial(det_vec& dets,std::vector<double>& C, std::vector<double>& coef,double spawning_threshold);
    /// The Lanczos Generator
    void propagate_Lanczos(det_vec& dets, std::vector<double>& C, double spawning_threshold, double S);

    /// Apply tau H to a set of determinants
    void apply_tau_H(double tau, double spawning_threshold, det_vec &dets, const std::vector<double>& C, det_hash<>& dets_C_map, double S);
    /// Apply symmetric approx tau H to a set of determinants
    void apply_tau_H_symm(double tau,double spawning_threshold,det_vec& dets,const std::vector<double>& C, det_hash<>& dets_C_hash, double S);
    /// Apply symmetric approx tau H to a determinant using dynamic screening
    void apply_tau_H_symm_det_dynamic(double tau, double spawning_threshold, det_hash<> &pre_dets_C_hash, const Determinant &detI, double CI, std::vector<std::pair<Determinant, double> > &new_space_C_vec, double E0, std::pair<double,double>& max_coupling);
    /// Apply tau H to a subset of determinants
    void apply_tau_H_subset(double tau, double spawning_threshold, det_vec &dets, const std::vector<double>& C, det_hash<> &dets_sum_map, det_hash<>& dets_C_hash, double S);
    /// Apply tau H to a determinant using screening based on the maxim couplings
    std::pair<double, double> apply_tau_H_det_prescreening(double tau, double spawning_threshold, Determinant& detI, double CI, std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0);
    /// Apply tau H to a determinant using dynamic screening
    void apply_tau_H_det_dynamic(double tau,double spawning_threshold,const Determinant& detI, double CI, std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0,std::pair<double,double>& max_coupling);
    /// Apply tau H to a determinant using Schwarz screening
    void apply_tau_H_det_schwarz(double tau, double spawning_threshold, const Determinant &detI, double CI, std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0);
    /// Apply tau H to a determinant within subset
    void apply_tau_H_det_subset(double tau, Determinant& detI, double CI, det_hash<>& dets_sum_map, std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0);
    /// Apply tau H to a determinant by selection within subset
    void apply_tau_H_det_subset_prescreening(double tau, double spawning_threshold, Determinant& detI, double CI, det_hash<>& dets_sum_map, std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0);
    /// Apply symmetric approx tau H to a set of determinants with selection according to reference coefficients
    void apply_tau_H_ref_C_symm(double tau,double spawning_threshold,det_vec& dets,const std::vector<double>& C, const std::vector<double>& ref_C, det_hash<>& dets_C_hash, double S);
    /// Apply symmetric approx tau H to a determinant using dynamic screening with selection according to a reference coefficient
    void apply_tau_H_ref_C_symm_det_dynamic(double tau, double spawning_threshold, det_hash<> &pre_dets_C_hash, det_hash<> &ref_dets_C_hash, const Determinant &detI, double CI, double ref_CI, std::vector<std::pair<Determinant, double> > &new_space_C_vec, double E0, std::pair<double,double>& max_coupling);


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
    /// Estimate the pertubation energy for the result
    std::tuple<double, double> estimate_perturbation(det_vec& dets, std::vector<double>& C, double spawning_threshold);
    /// Estimate the 1st order pertubation energy for the result.
    double estimate_1st_order_perturbation(det_vec& dets, std::vector<double>& C, double spawning_threshold);
    /// Estimate the 2nd order pertubation energy for the result within subspace
    double estimate_2nd_order_perturbation_sub(det_vec& dets, std::vector<double>& C, double spawning_threshold);
    /// Estimate the path-filtering error
    double estimate_path_filtering_error(det_vec& dets, std::vector<double>& C, double spawning_threshold);

    /// Form the product H c
    double form_H_C(double tau,double spawning_threshold,Determinant& detI, double CI, det_hash<>& det_C,std::pair<double,double>& max_coupling);
    /// Do we have OpenMP?
    static bool have_omp_;

    /// Estimate the highest possible energy
    double estimate_high_energy();
    /// Convergence estimation
    void convergence_analysis();
    /// Compute the characteristic function for projector
    void compute_characteristic_function();
    /// Print the characteristic function
    void print_characteristic_function();
};

}} // End Namespaces

#endif // _pci_h_
