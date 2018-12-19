#include "base_classes/dynamic_correlation_solver.h"

namespace forte {
DynamicCorrelationSolver::DynamicCorrelationSolver(Reference reference,
                                                   psi::SharedWavefunction ref_wfn,
                                                   psi::Options& options,
                                                   std::shared_ptr<ForteIntegrals> ints,
                                                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info), reference_(reference) {
    shallow_copy(ref_wfn);
}

std::shared_ptr<DynamicCorrelationSolver>
make_dynamic_correlation_solver(const std::string& type, std::shared_ptr<ForteOptions> options,
                                std::shared_ptr<ForteIntegrals> ints,
                                std::shared_ptr<MOSpaceInfo> mo_space_info) {
    //    return DynamicCorrelationSolver();
}

} // namespace forte
