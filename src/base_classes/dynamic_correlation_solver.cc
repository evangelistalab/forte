#include "base_classes/dynamic_correlation_solver.h"

namespace forte {
DynamicCorrelationSolver::DynamicCorrelationSolver(RDMs rdms,
                                                   std::shared_ptr<SCFInfo> scf_info,
                                                   std::shared_ptr<ForteOptions> options,
                                                   std::shared_ptr<ForteIntegrals> ints,
                                                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ints_(ints), mo_space_info_(mo_space_info), rdms_(rdms), scf_info_(scf_info),
      foptions_(options) {
    Enuc_ = ints_->nuclear_repulsion_energy();
    Efrzc_ = ints_->frozen_core_energy();
}

std::shared_ptr<DynamicCorrelationSolver>
make_dynamic_correlation_solver(const std::string& /*type*/, std::shared_ptr<ForteOptions> /*options*/,
                                std::shared_ptr<ForteIntegrals> /*ints*/,
                                std::shared_ptr<MOSpaceInfo> /*mo_space_info*/) {
    // TODO fill and return objects!
    //    return DynamicCorrelationSolver();
    return std::shared_ptr<DynamicCorrelationSolver>();
}

} // namespace forte
