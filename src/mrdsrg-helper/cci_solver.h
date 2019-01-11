#ifndef _cci_solver_h_
#define _cci_solver_h_

#include <vector>
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

class ActiveSpaceSolver;
class ActiveSpaceIntegrals;

/**
 * @class ContractedCISolver
 * @brief Contracted Configuration Interaction Solver for multi-state DSRG
 */
class ContractedCISolver
{
public:
    /**
     * @brief ContractedCISolver constructor
     *
     * @param as_solver active space solver
     * @param as_ints active space integrals
     */
    ContractedCISolver(std::shared_ptr<ActiveSpaceSolver> as_solver, std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    /// build the effective Hamiltonian <A|H|B> and diagonalize it
    void compute_Heff();

    /// compute the new densities and return a new Reference??

    /// get the eigen values
    std::vector<double> get_cCI_energies() {return evals_;}

    /// get the eigen vectors
    std::vector<SharedVector> get_cCI_evecs() {return evecs_;}

private:
    /// active space solver
    std::shared_ptr<ActiveSpaceSolver> as_solver_;

    /// active space integrals
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;

    /// effective Hamiltonian
    SharedMatrix Heff_;

    /// eigen values
    std::vector<double> evals_;

    /// eigen vectors
    std::vector<SharedVector> evecs_;

    /// build the effective Hamiltonian
    void build_Heff();
};

#endif // _cci_solver_h_
