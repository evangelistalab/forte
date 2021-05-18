/*
 *@BEGIN LICENSE
 *
 * v2RDM-CASSCF, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Copyright (c) 2014, The Florida State University. All rights reserved.
 *
 *@END LICENSE
 *
 */

#include "backtransform_tpdm.h"
#include <psi4/libtrans/integraltransform.h>
#include <psi4/libpsio/psio.hpp>
#include <psi4/libciomr/libciomr.h>
#include <psi4/libqt/qt.h>
#include <psi4/libiwl/iwl.hpp>
#include <psi4/libtrans/integraltransform_functors.h>
#include <psi4/psifiles.h>
#include <psi4/libtrans/mospace.h>
#define EXTERN

using namespace psi;

TPDMBackTransform::TPDMBackTransform(SharedWavefunction wfn, SpaceVec spaces,
                                     TransformationType transformationType, OutputType outputType,
                                     MOOrdering moOrdering, FrozenOrbitals frozenOrbitals,
                                     bool init)
    : IntegralTransform(wfn, spaces, transformationType, outputType, moOrdering, frozenOrbitals,
                        init) {}

void TPDMBackTransform::backtransform_density() {
    outfile->Printf("\n\n  ==> Back-transforming MO-basis TPDMs <==\n");

    check_initialized();

    // This limitation can be remedied by accounting for the fact that Pitzer orbital numbering is
    // not dense, so certain quantities must be alloc'd for the full MO space.  It's no limitation,
    // though
    if (frozenOrbitals_ != FrozenOrbitals::None)
        throw SanityCheckError("No orbitals can be frozen in density matrix transformations\n",
                               __FILE__, __LINE__);

    // The full MO space must be in the list of spaces used, let's check
    bool allFound = false;
    for (size_t i = 0; i < spacesUsed_.size(); ++i)
        if (spacesUsed_[i] == MOSPACE_ALL)
            allFound = true;
    if (!allFound)
        throw PSIEXCEPTION(
            "MOSpace::all must be amongst the spaces passed to the integral object's constructor");

    if (transformationType_ == TransformationType::Restricted) {
        backtransform_tpdm_restricted();
    } else {
        backtransform_tpdm_unrestricted();
    }
}
