#include "active_space_solver.h"

namespace psi {
namespace forte {

ActiveSpaceSolver::ActiveSpaceSolver(SharedWavefunction ref_wfn, Options& options,
                                     std::shared_ptr<ForteIntegrals> ints,
                                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info) {}
}
}
