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

#include "integrals.h"

namespace psi{

namespace libadaptive{

/**
 * @brief The MP2_NOS class
 * Computes MP2 natural orbitals
 */
class MP2_NOS
{
    // => Constructor <= //
    MP2_NOS(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints);
};

}} // End Namespaces

#endif // _mp2_nos_h_

//protected:

//    // => Class data <= //

//    /// The molecular integrals required by MethodBase
//    ExplorerIntegrals* ints_;

//    /// A pointer to the Wavefunction object
//    boost::shared_ptr<Wavefunction> wfn_;

//    // => Class initialization and termination <= //

//    /// Called in the constructor
//    void startup();
//    /// Called in the destructor
//    void cleanup();
