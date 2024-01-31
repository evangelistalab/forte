/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#pragma once

#include <vector>
#include <unordered_set>

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#include "ambit/tensor.h"

namespace ambit {
class BlockedTensor;
}

#include "base_classes/rdms.h"
#include "base_classes/state_info.h"

#include "helpers/printing.h"
#include "integrals/one_body_integrals.h"
#include "sparse_ci/determinant.h"
#include "sparse_ci/determinant_hashvector.h"

namespace forte {

class ActiveSpaceIntegrals;
class ForteIntegrals;
class ForteOptions;
class MOSpaceInfo;
class SCFInfo;

/**
 * @class ActiveSpaceMethod
 *
 * @brief Base class for methods that solve the Schrodinger equation in an active space
 *
 * This class is the base class for methods that solve for the wavefunction in a
 * small subset of the full orbital space (<30-40 orbitals).
 * The molecular orbital used by the active space methods are stored in a ActiveSpaceIntegrals
 * and must be passed at the time of creation.
 *
 * All methods that derive from this class offer the following interface
 *
 * - Compute the energy
 *    double compute_energy();
 *
 * - Compute a RDMs object
 *    RDMs rdms();
 *
 * - Set the options for the derived methods
 *    set_options(std::shared_ptr<ForteOptions> options);
 *
 * All methods must also use the following base class variables
 *
 * - Information about the state
 *    StateInfo state_;
 *
 * - Number of states computed
 *    size_t nroot_;
 *
 * - Store final energies in the vector (including nuclear repulsion):
 *    std::vector<double> energies_;
 *
 * @note This class is not aware of which orbitals are considered active. This information
 * is contained in the ActiveSpaceIntegrals object. Orbitals that are double occupied
 * are not correlated and are trated via effective scalar and one-body interactions.
 */
class ActiveSpaceMethod {
  public:
    // ==> Class Constructor and Destructor <==
    /**
     * @brief ActiveSpaceMethod Constructor for a single state computation
     * @param state the electronic state to compute
     * @param nroot the number of roots
     * @param mo_space_info a MOSpaceInfo object that defines the orbital spaces
     * @param as_ints molecular integrals defined only for the active space orbitals
     */
    ActiveSpaceMethod(StateInfo state, size_t nroot, std::shared_ptr<MOSpaceInfo> mo_space_info,
                      std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    /// Default constructor
    ActiveSpaceMethod() = default;

    /// Virtual destructor to enable deletion of a Derived* through a Base*
    virtual ~ActiveSpaceMethod() = default;

    // ==> Class Interface <==

    /// Compute the energy and return it
    virtual double compute_energy() = 0;

    /**
     * @brief Compute the reduced density matrices up to a given particle rank (max_rdm_level)
     *
     *        This function can be used to compute transition density matrices between
     *        states of the same symmetry,
     *
     *        D^{p}_{q} = <I, this| a+_p1 ... a_qn |J, this>
     *
     *        where |I> is the I-th state of this object
     *              |J> is the J-th state of this object
     *
     * @param root_list     a list of pairs of roots to compute [(I_1, J_1), (I_2, J_2), ...]
     * @param max_rdm_level the maximum RDM rank
     * @return
     */
    virtual std::vector<std::shared_ptr<RDMs>>
    rdms(const std::vector<std::pair<size_t, size_t>>& root_list, int max_rdm_level,
         RDMsType type) = 0;

    /// @brief Compute the transition density matrices between two states of different symmetry
    ///        This function can be used to compute transition density matrices between
    ///        states of different symmetry,
    ///
    ///        D^{p}_{q} = <I, this, symmetry_l| a+_p1 ... a_qn |J, method2, symmetry_r>
    ///
    ///        where |I, this, symmetry_l> is the I-th state of this object
    ///              |J, method2, symmetry_r> is the J-th state of the method2 object
    ///
    /// @param root_list     a list of pairs of roots to compute [(I_1, J_1), (I_2, J_2), ...]
    /// @param method2       the method that computes the right wave function
    /// @param max_rdm_level the maximum RDM rank
    /// @return a vector of RDMs objects
    virtual std::vector<std::shared_ptr<RDMs>>
    transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                    std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level,
                    RDMsType type) = 0;

    /// Dump transition RDMs to disk
    void save_transition_rdms(const std::vector<std::shared_ptr<RDMs>>& rdms,
                              const std::vector<std::pair<size_t, size_t>>& root_list,
                              std::shared_ptr<ActiveSpaceMethod> method2);

    /// Compute the overlap of two wave functions acted by complementary operators
    /// Return a map from state to roots of values
    /// Computes the overlap of \sum_{p} \sum_{σ} <Ψ| h^+_{pσ} (v) h_{pσ} (t) |Ψ>, where
    /// h_{pσ} (t) = \sum_{uvw} t^{uv}_{pw} \sum_{τ} w^+_{τ} v_{τ} u_{σ}
    /// Useful to get the 3-RDM contribution of fully contracted term of two 2-body operators:
    /// \sum_{puvwxyzστθ} v_{pwxy} t_{uvpz} <Ψ| xσ^+ yτ^+ wτ zθ^+ vθ uσ |Ψ>
    virtual std::vector<double>
    compute_complementary_H2caa_overlap(const std::vector<size_t>& /*roots*/,
                                        ambit::Tensor /*Tbra*/, ambit::Tensor /*Tket*/) {
        throw std::runtime_error(
            "ActiveSpaceMethod::compute_complementary_H2caa_overlap: Not yet implemented!");
    }

    /// Compute generalized RDM
    ///     Gamma' = C_I <Phi_I| H |Phi_J> X_J where H is the active space Hamiltonian (fci_ints)
    /// @param x: the X vector to be contracted with H_IJ
    virtual void generalized_rdms([[maybe_unused]] size_t root,
                                  [[maybe_unused]] const std::vector<double>& X,
                                  [[maybe_unused]] ambit::BlockedTensor& result,
                                  [[maybe_unused]] bool c_right, [[maybe_unused]] int rdm_level,
                                  [[maybe_unused]] std::vector<std::string> spin) {
        throw std::runtime_error(
            "The function generalized_rdms is not implemented for this ActiveSpaceMethod type!");
    }

    /// Add k-body contributions to the sigma vector
    ///    σ_I += h_{p1,p2,...}^{q1,q2,...} <Phi_I| a^+_p1 a^+_p2 .. a_q2 a_q1 |Phi_J> C_J
    /// @param root: the root number of the state
    /// @param h: the antisymmetrized k-body integrals
    /// @param block_label_to_factor: map from the block labels of integrals to its factors
    /// @param sigma: the sigma vector to be added
    virtual void
    add_sigma_kbody([[maybe_unused]] size_t root, [[maybe_unused]] ambit::BlockedTensor& h,
                    [[maybe_unused]] const std::map<std::string, double>& block_label_to_factor,
                    [[maybe_unused]] std::vector<double>& sigma) {
        throw std::runtime_error(
            "The function add_sigma_kbody is not implemented for this ActiveSpaceMethod type!");
    }

    /// Compute generalized sigma vector
    ///     σ_I = <Phi_I| H |Phi_J> X_J where H is the active space Hamiltonian (fci_ints)
    /// @param x: the X vector to be contracted with H_IJ
    /// @param sigma: the sigma vector (will be zeroed first)
    virtual void generalized_sigma([[maybe_unused]] std::shared_ptr<psi::Vector> x,
                                   [[maybe_unused]] std::shared_ptr<psi::Vector> sigma) {
        throw std::runtime_error(
            "The function generalized_sigma is not implemented for this ActiveSpaceMethod type!");
    }

    /// Return the space size
    virtual size_t space_size() {
        throw std::runtime_error(
            "The function space_size is not implemented for this ActiveSpaceMethod type!");
    }

    /// Set options from an option object
    /// @param options the options passed in
    virtual void set_options(std::shared_ptr<ForteOptions> options) = 0;

    /// Return the eigen vectors
    virtual std::vector<ambit::Tensor> eigenvectors() {
        throw std::runtime_error(
            "ActiveSpaceMethod::eigenvectors(): Not Implemented for this class!");
    }
    /// Compute permanent dipole moments
    std::vector<std::vector<double>>
    compute_permanent_dipole(const std::vector<std::pair<size_t, size_t>>& root_list,
                             const ambit::Tensor& Ua, const ambit::Tensor& Ub);

    /// Compute permanent dipole moments (electronic + nuclear)
    std::vector<std::shared_ptr<psi::Vector>>
    compute_permanent_dipole(std::shared_ptr<ActiveMultipoleIntegrals> ampints,
                             std::vector<std::pair<size_t, size_t>>& root_list);

    /// Compute permanent quadrupole moments (electronic + nuclear)
    std::vector<std::shared_ptr<psi::Vector>>
    compute_permanent_quadrupole(std::shared_ptr<ActiveMultipoleIntegrals> ampints,
                                 const std::vector<std::pair<size_t, size_t>>& root_list);

    /// Compute transition dipole moments assuming same orbitals
    std::vector<std::shared_ptr<psi::Vector>>
    compute_transition_dipole_same_orbs(std::shared_ptr<ActiveMultipoleIntegrals> ampints,
                                        const std::vector<std::pair<size_t, size_t>>& root_list,
                                        std::shared_ptr<ActiveSpaceMethod> method2);

    /// Compute oscillator strength assuming same orbitals
    std::vector<double>
    compute_oscillator_strength_same_orbs(std::shared_ptr<ActiveMultipoleIntegrals> ampints,
                                          const std::vector<std::pair<size_t, size_t>>& root_list,
                                          std::shared_ptr<ActiveSpaceMethod> method2);

    /// Store spectra results from calculation
    void add_spectra_results(const std::vector<std::string>& data);

    /// Dump the wave function to file
    /// @param file name
    virtual void dump_wave_function(const std::string&) {
        throw std::runtime_error("ActiveSpaceMethod::dump_wave_function: Not yet implemented!");
    }

    /// Read the wave function from file
    /// @param file name
    /// @return the number of active orbitals, the set of determinants, CI coefficients
    virtual std::tuple<size_t, std::vector<Determinant>, std::shared_ptr<psi::Matrix>>
    read_wave_function(const std::string&) {
        throw std::runtime_error("ActiveSpaceMethod::read_wave_function: Not yet implemented!");
    }

    /// @return the CI wave functions for the current StateInfo (deterministic determinant space)
    virtual std::shared_ptr<psi::Matrix> ci_wave_functions() {
        throw std::runtime_error("ActiveSpaceMethod::ci_wave_functions: Not yet implemented!");
    }

    // ==> Base Class Functionality (inherited by derived classes) <==

    /// Pass a set of ActiveSpaceIntegrals to the solver (e.g. an effective Hamiltonian)
    /// @param as_ints the integrals passed in
    void set_active_space_integrals(std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    /// Return the eigenvalues
    std::shared_ptr<psi::Vector> evals();

    /// Return a vector with the energies of all the states
    const std::vector<double>& energies() const;

    /// Return a vector with the average value of S^2 of all the states
    const std::vector<double>& spin2() const;

    /// Return the number of roots computed
    size_t nroot() const { return nroot_; }

    /// Return the state info
    const StateInfo& state() const { return state_; }

    /// Return the wave function file name
    std::string wfn_filename() const { return wfn_filename_; }

    /// Return if we dump wave function to disk
    bool dump_wfn() const { return dump_wfn_; }

    /// Return if we read wave function guess from disk
    bool read_wfn_guess() const { return read_wfn_guess_; }

    // ==> Base Class Handles Set Functions <==

    /// Set the energy convergence criterion
    /// @param value the convergence criterion in a.u.
    void set_e_convergence(double value);

    /// Set the residual 2-norm convergence criterion
    /// @param value the convergence criterion in a.u.
    void set_r_convergence(double value);

    /// Set the maximum number of iterations
    /// @param value the maximum number of iterations
    void set_maxiter(size_t value);

    /// Set if we dump the wave function to disk
    void set_read_wfn_guess(bool read);

    /// Set if we dump the wave function to disk
    void set_dump_wfn(bool dump);

    /// Set if we dump the transition dipole moment to disk
    void set_dump_trdm(bool dump);

    /// Set the file name for storing wave function on disk
    /// @param name the wave function file name
    void set_wfn_filename(const std::string& name);

    /// Set the root that will be used to compute the properties
    /// @param the root (root = 0, 1, 2, ...)
    void set_root(int value);

    /// Set the print level
    /// @param level the print level (0 = no printing, 1 default)
    void set_print(PrintLevel level);

    /// Quiet mode (no printing, for use with CASSCF)
    void set_quiet_mode();

    // Get map with spectra data
    std::map<std::string, std::vector<std::string>> get_spectra_results();

    /// Get the model space
    DeterminantHashVec get_PQ_space();

    /// Get model space coefficients
    std::shared_ptr<psi::Matrix> get_PQ_evecs();

  protected:
    /// The list of active orbitals (absolute ordering)
    std::vector<size_t> active_mo_;

    /// The list of doubly occupied orbitals (absolute ordering)
    std::vector<size_t> core_mo_;

    /// The state to calculate
    StateInfo state_;

    /// The number of roots (default = 1)
    size_t nroot_ = 1;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The molecular integrals for the active space
    /// This object holds only the integrals for the orbital contained in the active_mo_ vector.
    /// The one-electron integrals and scalar energy contains contributions from the
    /// doubly occupied orbitals specified by the core_mo_ vector.
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;

    DeterminantHashVec final_wfn_;

    std::shared_ptr<psi::Matrix> evecs_;

    // ==> Base Class Handles [can be changed before running compute_energy()]  <==

    /// The energy convergence criterion
    double e_convergence_ = 1.0e-12;

    /// The residual 2-norm convergence criterion
    double r_convergence_ = 1.0e-6;

    /// The maximum number of iterations
    size_t maxiter_ = 100;

    /// The root used to compute properties (zero based, default = 0)
    int root_ = 0;

    /// A variable to control printing information
    PrintLevel print_ = PrintLevel::Default;

    /// Eigenvalues
    std::shared_ptr<psi::Vector> evals_;

    /// The energies (including nuclear repulsion) of all the states
    std::vector<double> energies_;

    /// The average value of S^2 of all the states. If empty this quantity will not be checked
    std::vector<double> spin2_;

    /// Map of state_i->state_f and the spectroscopic results
    std::map<std::string, std::vector<std::string>> spectra_results_;

    /// Read wave function from disk as initial guess?
    bool read_wfn_guess_ = false;
    /// Dump transition density matrix to disk?
    bool dump_trdm_ = false;
    /// Dump wave function to disk?
    bool dump_wfn_ = false;
    /// The file name for storing wave function (determinants, CI coefficients)
    std::string wfn_filename_;
};

/**
 * @brief make_active_space_method Make an active space method object
 * @param type a string that specifies the type (e.g. "FCI", "ACI", ...)
 * @param state information about the elecronic state
 * @param scf_info information about a previous SCF computation
 * @param mo_space_info orbital space information
 * @param as_ints an active space integral object
 * @param options user-provided options
 * @return a shared pointer for the base class ActiveSpaceMethod
 */
std::shared_ptr<ActiveSpaceMethod> make_active_space_method(
    const std::string& type, StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
    std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ActiveSpaceIntegrals> as_ints,
    std::shared_ptr<ForteOptions> options);

// std::vector<std::shared_ptr<RDMs>> transition_rdms(std::shared_ptr<ActiveSpaceMethod> m1,
//                                                    std::shared_ptr<ActiveSpaceMethod> m2,
//                                                    std::vector<std::pair<size_t, size_t>>,
//                                                    int max_rdm_level);

} // namespace forte
