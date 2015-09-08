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

#ifndef _mp2_nos_h_
#define _mp2_nos_h_

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>

#include "helpers.h"
#include "integrals.h"
#include "reference.h"

namespace psi{

namespace forte{

/**
 * @brief The MP2_NOS class
 * Computes MP2 natural orbitals
 */
class MP2_NOS
{
public:
    // => Constructor <= //
    MP2_NOS(boost::shared_ptr<Wavefunction> wfn, Options &options, ForteIntegrals* ints, std::shared_ptr<MOSpaceInfo> mo_space_info);
    //  => Destructor <= //
};


/**
 * @brief The SemiCanonical class
 * Computes semi-canonical orbitals
 */
class SemiCanonical
{
public:
    // => Constructor <= //
    SemiCanonical(boost::shared_ptr<Wavefunction> wfn, Options &options, ForteIntegrals* ints, std::shared_ptr<MOSpaceInfo> mo_space_info, Reference& reference);
};

}} // End Namespaces

#endif // _mp2_nos_h_
