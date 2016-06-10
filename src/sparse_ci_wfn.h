/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
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
 *@END LICENSE
 */

#ifndef _sparse_ci_wfn_h_
#define _sparse_ci_wfn_h_

#include "stl_bitset_determinant.h"

namespace psi{ namespace forte{

/**
 * @brief A class to store sparse configuration interaction wave functions
 * Stores a hash of determinants.
 */

using wfn_hash = det_hash<double>;

class SparseCIWavefunction
{
public:

    /// Default constructor
    SparseCIWavefunction();
    /// Copy constructor
    SparseCIWavefunction(const wfn_hash& wfn_);

    /// @return The hash
    wfn_hash& wfn();

protected:
    /// A hash of (determinants,coefficients)
    wfn_hash wfn_;
};

}}

#endif // _sparse_ci_wfn_h_
