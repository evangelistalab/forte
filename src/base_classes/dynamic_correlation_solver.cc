#include "base_classes/dynamic_correlation_solver.h"


namespace forte {
DynamicCorrelationSolver::DynamicCorrelationSolver(Reference reference, psi::SharedWavefunction ref_wfn,
                                                   Options& options,
                                                   std::shared_ptr<ForteIntegrals> ints,
                                                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(psi::Options), ints_(ints), mo_space_info_(mo_space_info), reference_(reference) {
    shallow_copy(ref_wfn);
}
} // namespace forte

