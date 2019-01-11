#include "cci_solver.h"

#include "base_classes/active_space_solver.h"
#include "integrals/active_space_integrals.h"

ContractedCISolver::ContractedCISolver(std::shared_ptr<ActiveSpaceSolver> as_solver,
                       std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : as_solver_(as_solver), as_ints_(as_ints) {}

void ContractedCISolver::compute_Heff(){}

