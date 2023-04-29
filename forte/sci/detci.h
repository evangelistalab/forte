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

class DeterminantSubstitutionLists;

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
    psi::SharedMatrix ci_wave_functions() override { return evecs_; }

    /// Return the determinants
    DeterminantHashVec determinants() const { return p_space_; }
    /// Return the number of active orbitals
    psi::Dimension actv_dim() const { return actv_dim_; }

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
    std::tuple<size_t, std::vector<Determinant>, psi::SharedMatrix>
    read_wave_function(const std::string& filename) override;

  private:
    /// SCFInfo object
    std::shared_ptr<SCFInfo> scf_info_;

    /// ForteOptions
    std::shared_ptr<ForteOptions> options_;

    /// Start up function
    void startup();

    /// Number of active orbitals
    int nactv_;
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
    /// Substitution lists
    std::shared_ptr<DeterminantSubstitutionLists> sub_lists_;

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

    /// Sparse CI solver
    std::unique_ptr<SparseCISolver> sparse_ci_solver_;
    /// Set up sparse CI solver
    void set_sparse_ci_solver();

    /// Diagonalize the Hamiltonian
    void diagonalize_hamiltonian();
    /// Algorithm to build sigma vector
    SigmaVectorType sigma_vector_type_;
    /// Max memory can be used for sigma build
    size_t sigma_max_memory_;
    /// Sigma vector builder
    std::shared_ptr<SigmaVector> sigma_vector_;

    /// Eigen vectors
    psi::SharedMatrix evecs_;
    /// Print important CI vectors
    void print_ci_wfn();
    /// Threshold to print CI coefficients
    double ci_print_threshold_;

    /// Compute 1RDMs
    void compute_1rdms();
    /// 1RDMs alpha spin
    std::vector<psi::SharedMatrix> opdm_a_;
    /// 1RDMs beta spin
    std::vector<psi::SharedMatrix> opdm_b_;

    /// Compute the (transition) 1RDMs, same orbital, same set of determinants
    std::vector<ambit::Tensor> compute_trans_1rdms_sosd(int root1, int root2);
    /// Compute the (transition) 2RDMs, same orbital, same set of determinants
    std::vector<ambit::Tensor> compute_trans_2rdms_sosd(int root1, int root2);
    /// Compute the (transition) 3RDMs, same orbital, same set of determinants
    std::vector<ambit::Tensor> compute_trans_3rdms_sosd(int root1, int root2);
    /// Compute the (transition) RDMs using dynamic algorithm
    /// same orbital, same set of determinants
    std::shared_ptr<RDMs> compute_trans_rdms_sosd_dynamic(CI_RDMS& ci_rdms, int max_rdm_level,
                                                          RDMsType rdm_type);

    /// Printing for CI_RDMs
    bool print_ci_rdms_ = true;

    /// Read wave function from disk as initial guess
    bool read_initial_guess(const std::string& filename);
};
} // namespace forte

#endif // _detci_h_
