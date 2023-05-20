#ifndef _detci_h_
#define _detci_h_

#include "ci_rdm/ci_rdms.h"
#include "base_classes/mo_space_info.h"
#include "helpers/helpers.h"
#include "integrals/integrals.h"
#include "base_classes/active_space_method.h"
#include "base_classes/rdms.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "sparse_ci/ci_reference.h"
#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant.h"
#include "sparse_ci/sigma_vector.h"

namespace forte {
class DETCI : public ActiveSpaceMethod {
  public:
    /**
     * @brief DETCI Constructor
     * @param state The state info (symmetry, multiplicity, na, nb, etc.)
     * @param nroot Number of roots of interests
     * @param scf_info SCF information
     * @param options Forte options
     * @param mo_space_info MOSpaceInfo
     * @param as_ints Active space integrals
     */
    DETCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
          std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
          std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    ~DETCI();

    /// Compute the energy
    double compute_energy() override;

    /// RDMs override
    std::vector<std::shared_ptr<RDMs>> rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                            int max_rdm_level, RDMsType rdm_type) override;

    /// Transition RDMs override
    std::vector<std::shared_ptr<RDMs>>
    transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                    std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level,
                    RDMsType rdm_type) override;

    /// Return the CI wave functions for current state symmetry
    std::shared_ptr<psi::Matrix> ci_wave_functions() override { return evecs_; }

    /// Set options override
    void set_options(std::shared_ptr<ForteOptions> options) override;

    /// Set projected roots
    void project_roots(std::vector<std::vector<std::pair<size_t, double>>>& projected) {
        projected_roots_ = projected;
    }

    /// Set initial guess
    void set_initial_guess(std::vector<std::vector<std::pair<size_t, double>>>& guess) {
        initial_guess_ = guess;
    }

    /// Dump wave function to disk
    void dump_wave_function(const std::string& filename) override;

    /// Read wave function from disk
    /// Return the number of active orbitals, set of determinants, CI coefficients
    std::tuple<size_t, std::vector<Determinant>, std::shared_ptr<psi::Matrix>>
    read_wave_function(const std::string& filename) override;

    /// Return the number of determinants
    size_t space_size() override { return p_space_.size(); }

    /// Return the eigen vector in ambit Tensor format
    std::vector<ambit::Tensor> eigenvectors() override;

  private:
    /// SCFInfo object
    std::shared_ptr<SCFInfo> scf_info_;

    /// ForteOptions
    std::shared_ptr<ForteOptions> options_;

    /// Start up function
    void startup();

    /// Number of active orbitals
    size_t nactv_;
    /// Number of active orbitals per irrep
    psi::Dimension actv_dim_;

    /// Active space type (CAS, GAS, DOCI, CIS/CID/CISD)
    std::string actv_space_type_;
    /// Exclude HF determinant in CID/CISD
    bool exclude_hf_in_cid_;

    /// The determinant space
    DeterminantHashVec p_space_;
    /// Build determinant space
    void build_determinant_space();

    /// State label
    std::string state_label_;
    /// Multiplicity
    int multiplicity_;
    /// Twice Ms
    int twice_ms_;
    /// Wave function symmetry
    int wfn_irrep_;
    /// Number of irreps
    int nirrep_;

    /// Max iteration of Davidson-Liu
    int maxiter_;

    /// Number of guess basis for Davidson-Liu
    int dl_guess_size_;
    /// Initial guess vector
    std::vector<std::vector<std::pair<size_t, double>>> initial_guess_;

    /// Roots to be projected out in the diagonalization
    std::vector<std::vector<std::pair<size_t, double>>> projected_roots_;

    /// Number of trial vector to keep after collapsing of Davidson-Liu
    int ncollapse_per_root_;
    /// Number of trial vectors per root for Davidson-Liu
    int nsubspace_per_root_;

    /// Diagonalize the Hamiltonian
    void diagonalize_hamiltonian();
    /// Prepare Davidson-Liu solver
    std::shared_ptr<SparseCISolver> prepare_ci_solver();

    std::shared_ptr<SigmaVector> sigma_vector_;
    /// Algorithm to build sigma vector
    SigmaVectorType sigma_vector_type_;
    /// Max memory can be used for sigma build
    size_t sigma_max_memory_;

    /// Eigen vectors
    std::shared_ptr<psi::Matrix> evecs_;
    /// Print important CI vectors
    void print_ci_wfn();
    /// Threshold to print CI coefficients
    double ci_print_threshold_;

    /// Compute 1RDMs
    void compute_1rdms();
    /// 1RDMs alpha spin
    std::vector<std::shared_ptr<psi::Matrix>> opdm_a_;
    /// 1RDMs beta spin
    std::vector<std::shared_ptr<psi::Matrix>> opdm_b_;

    /// Compute the (transition) 1RDMs, same orbital, same set of determinants
    std::vector<ambit::Tensor> compute_trans_1rdms_sosd(int root1, int root2);
    /// Compute the (transition) 2RDMs, same orbital, same set of determinants
    std::vector<ambit::Tensor> compute_trans_2rdms_sosd(int root1, int root2);
    /// Compute the (transition) 3RDMs, same orbital, same set of determinants
    std::vector<ambit::Tensor> compute_trans_3rdms_sosd(int root1, int root2);

    /// Printing for CI_RDMs
    bool print_ci_rdms_ = true;

    /// Read wave function from disk as initial guess
    bool read_initial_guess(const std::string& filename);

    // Functions for gradients

    /// Compute the generalized reduced density matrix of a given level
    void generalized_rdms(size_t root, const std::vector<double>& X, ambit::BlockedTensor& grdms,
                          bool c_right, int rdm_level, std::vector<std::string> spin) override;

    /// Add k-body contributions to the sigma vector
    ///    σ_I += h_{p1,p2,...}^{q1,q2,...} <Phi_I| a^+_p1 a^+_p2 .. a_q2 a_q1 |Phi_J> C_J
    /// @param root: the root number of the state
    /// @param h: the antisymmetrized k-body integrals
    /// @param block_label_to_factor: map from the block labels of integrals to its factors
    /// @param sigma: the sigma vector to be added
    void add_sigma_kbody(size_t root, ambit::BlockedTensor& h,
                         const std::map<std::string, double>& block_label_to_factor,
                         std::vector<double>& sigma) override;

    /// Compute generalized sigma vector
    ///     σ_I = <Phi_I| H |Phi_J> X_J where H is the active space Hamiltonian (fci_ints)
    /// @param x: the X vector to be contracted with H_IJ
    /// @param sigma: the sigma vector (will be zeroed first)
    void generalized_sigma(psi::std::shared_ptr<psi::Vector> x,
                           psi::std::shared_ptr<psi::Vector> sigma) override;
};
} // namespace forte

#endif // _detci_h_
