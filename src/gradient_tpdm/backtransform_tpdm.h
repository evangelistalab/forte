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

#ifndef BACKTRANSFORM_TPDM_H
#define BACKTRANSFORM_TPDM_H

#include <map>
#include <vector>
#include <string>
#include <psi4/libmints/dimension.h>
#include <psi4/libmints/typedefs.h>
#include <psi4/libtrans/mospace.h>

#include <psi4/libtrans/integraltransform.h>

namespace psi {

struct dpdfile4;
struct dpdbuf4;
class Matrix;
class Dimension;
class Wavefunction;

typedef std::vector<std::shared_ptr<MOSpace>> SpaceVec;

class TPDMBackTransform : public IntegralTransform {

  public:
    /**
     * Set up a transformation involving four MO spaces
     *
     * @param wfn                A (shared pointer to a) wavefunction object with the orbital info
     * @param spaces             A vector containing smart pointers to the unique space(s) involved
     *                           in any transformations that this object will perform
     * @param transformationType The type of transformation, described by the
     *                           enum TransformationType
     * @param moOrdering         The ordering convention of the resulting integrals, see
     *                           enum MOOrdering.  This only affects IWL output.
     * @param outputType         The storage format of the transformed integrals, see
     *                           enum OutputType
     * @param frozenOrbitals     Which orbitals are to be excluded from the transformation, see
     *                           enum FrozenOrbitals
     * @param initialize         Whether to initialize during construction or not.  Useful if some
     *                           options need to be tweaked before initialization.
     */
    TPDMBackTransform(
        SharedWavefunction wfn, SpaceVec spaces,
        TransformationType transformationType = IntegralTransform::TransformationType::Restricted,
        OutputType outputType = IntegralTransform::OutputType::DPDOnly,
        MOOrdering moOrdering = IntegralTransform::MOOrdering::QTOrder,
        FrozenOrbitals frozenOrbitals = IntegralTransform::FrozenOrbitals::OccAndVir,
        bool initialize = true);
    ~TPDMBackTransform();

    void backtransform_density();

  protected:
    void backtransform_tpdm_unrestricted();
    void presort_mo_tpdm_unrestricted();
    void sort_so_tpdm(const dpdbuf4* B, int irrep, size_t first_row, size_t num_rows,
                      bool first_run);
    void setup_tpdm_buffer(const dpdbuf4* D);
};

} // namespace psi

#endif
