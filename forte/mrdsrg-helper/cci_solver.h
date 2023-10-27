#ifndef _cci_solver_h_
#define _cci_solver_h_

#include <vector>
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

namespace forte {

class ActiveSpaceSolver;
class ActiveSpaceIntegrals;

/**
 * @class ContractedCISolver
 * @brief Contracted Configuration Interaction Solver for multi-state DSRG
 */
class ContractedCISolver {
  public:
    /**
     * @brief ContractedCISolver constructor
     *
     * @param as_solver active space solver
     * @param as_ints active space integrals
     */
    ContractedCISolver(std::shared_ptr<ActiveSpaceSolver> as_solver,
                       std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                       int max_rdm_level, int max_body);

    /// TODO: the as_ints should handle 3-body integrals, as_ints may not be Hermitian

    /// build the effective Hamiltonian <A|H|B> and diagonalize it
    void compute_Heff();

    /// compute the new densities and return a new RDMs??

    /// get the eigen values
    std::vector<std::vector<double>> get_energies() { return evals_; }

    /// get the eigen vectors
    std::vector<psi::Matrix> get_evecs() { return evecs_; }

  private:
    /// active space solver
    std::shared_ptr<ActiveSpaceSolver> as_solver_;

    /// active space integrals
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;

    /// max many-body level available for as_ints
    int max_body_;

    /// max rdm level
    int max_rdm_level_;

    /// eigen values
    std::vector<std::vector<double>> evals_;

    /// eigen vectors
    std::vector<psi::Matrix> evecs_;
};
}
#endif // _cci_solver_h_
