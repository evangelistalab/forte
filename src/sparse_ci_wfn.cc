#include "sparse_ci_wfn.h"

namespace psi{ namespace forte{

SparseCIWavefunction::SparseCIWavefunction() {}

SparseCIWavefunction::SparseCIWavefunction(wfn_hash wfn) : wfn_(wfn) {}

wfn_hash& SparseCIWavefunction::wfn()
{
    return wfn_;
}

}}
