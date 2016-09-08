//[forte-public]
#include <cmath>
#include <numeric>

#include <libmints/basisset.h>
#include <libthce/thce.h>
#include <libthce/lreri.h>
#include <libqt/qt.h>

#include "blockedtensorfactory.h"

namespace psi{ namespace forte{

OwnIntegrals::OwnIntegrals(psi::Options &options, SharedWavefunction ref_wfn,  IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core,
std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ForteIntegrals(options, ref_wfn, restricted, resort_frozen_core, mo_space_info){
    integral_type_ = Own;
    // If code calls constructor print things
    // But if someone calls retransform integrals do not print it

    wfn_ = ref_wfn;

    outfile->Printf("\n Avoiding Generation of Integrals");
    if (ncmo_ < nmo_){
        freeze_core_orbitals();
        // Set the new value of the number of orbitals to be used in indexing routines
        aptei_idx_ = ncmo_;
    }
}
OwnIntegrals::~OwnIntegrals()
{
}
}}
