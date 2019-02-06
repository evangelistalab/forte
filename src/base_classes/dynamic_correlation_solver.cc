#include "base_classes/dynamic_correlation_solver.h"

namespace forte {
DynamicCorrelationSolver::DynamicCorrelationSolver(Reference reference,
                                                   std::shared_ptr<SCFInfo> scf_info,
                                                   std::shared_ptr<ForteOptions> options,
                                                   std::shared_ptr<ForteIntegrals> ints,
                                                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ints_(ints), mo_space_info_(mo_space_info), reference_(reference), scf_info_(scf_info),
      foptions_(options) {}

std::shared_ptr<DynamicCorrelationSolver>
make_dynamic_correlation_solver(const std::string& type, std::shared_ptr<ForteOptions> options,
                                std::shared_ptr<ForteIntegrals> ints,
                                std::shared_ptr<MOSpaceInfo> mo_space_info) {
    // TODO fill and return objects!
    //    return DynamicCorrelationSolver();
}

} // namespace forte
