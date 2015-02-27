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

#ifndef _methodbase_h_
#define _methodbase_h_

#include <fstream>

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>

#include "integrals.h"
//#include <ambit/blocked_tensor.h>
#include "tensor_basic.h"
#include "tensor_labeled.h"
#include "tensor_product.h"
#include "tensor_blocked.h"

namespace psi{

//class PSIO;
//class Chkpt;

namespace libadaptive{

/**
 * @brief The MethodBase class
 * This class provides basic functions to write electronic structure
 * pilot codes using the Tensor classes
 */
class MethodBase : public Wavefunction
{
protected:

    // => Class data <= //

    /// The molecular integrals required by MethodBase
    ExplorerIntegrals* ints_;

    /// List of alpha occupied MOs
    std::vector<size_t> a_occ_mos;
    /// List of beta occupied MOs
    std::vector<size_t> b_occ_mos;
    /// List of alpha virtual MOs
    std::vector<size_t> a_vir_mos;
    /// List of beta virtual MOs
    std::vector<size_t> b_vir_mos;

    /// Map from all the MOs to the alpha occupied
    std::map<size_t,size_t> mos_to_aocc;
    /// Map from all the MOs to the beta occupied
    std::map<size_t,size_t> mos_to_bocc;
    /// Map from all the MOs to the alpha virtual
    std::map<size_t,size_t> mos_to_avir;
    /// Map from all the MOs to the beta virtual
    std::map<size_t,size_t> mos_to_bvir;

    ::BlockedTensor H;
    ::BlockedTensor F;
    ::BlockedTensor V;
    ::BlockedTensor CG1;
    ::BlockedTensor G1;
    ::BlockedTensor D1;
    ::BlockedTensor D2;

    ::BlockedTensor Ha;
    ::BlockedTensor Hb;
    ::BlockedTensor Fa;
    ::BlockedTensor Fb;
    ::BlockedTensor G1a;
    ::BlockedTensor G1b;
    ::BlockedTensor Vaa;
    ::BlockedTensor Vab;
    ::BlockedTensor Vbb;
    ::BlockedTensor D2aa;
    ::BlockedTensor D2ab;
    ::BlockedTensor D2bb;

    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();
public:

    // => Constructors <= //

    MethodBase(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints);
    ~MethodBase();

    /// The energy of the reference
    double E0_;

    /// The frozen-core energy
    double frozen_core_energy_;
};

}} // End Namespaces

#endif // _methodbase_h_
