/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef _methodbase_h_
#define _methodbase_h_

#include <fstream>


#include "psi4/libmints/wavefunction.h"

#include "integrals/integrals.h"
#include "ambit/blocked_tensor.h"

namespace psi {

namespace forte {

/**
 * @brief The MethodBase class
 * This class provides basic functions to write electronic structure
 * pilot codes using the Tensor classes
 */
class MethodBase : public Wavefunction {
  protected:
    // => Class data <= //

    /// The molecular integrals required by MethodBase
    std::shared_ptr<ForteIntegrals> ints_;

    /// List of alpha occupied MOs
    std::vector<size_t> a_occ_mos;
    /// List of beta occupied MOs
    std::vector<size_t> b_occ_mos;
    /// List of alpha virtual MOs
    std::vector<size_t> a_vir_mos;
    /// List of beta virtual MOs
    std::vector<size_t> b_vir_mos;

    /// Map from all the MOs to the alpha occupied
    std::map<size_t, size_t> mos_to_aocc;
    /// Map from all the MOs to the beta occupied
    std::map<size_t, size_t> mos_to_bocc;
    /// Map from all the MOs to the alpha virtual
    std::map<size_t, size_t> mos_to_avir;
    /// Map from all the MOs to the beta virtual
    std::map<size_t, size_t> mos_to_bvir;

    ambit::TensorType tensor_type_;

    ambit::BlockedTensor H;
    ambit::BlockedTensor F;
    ambit::BlockedTensor V;
    ambit::BlockedTensor CG1;
    ambit::BlockedTensor G1;
    ambit::BlockedTensor InvD1;
    ambit::BlockedTensor InvD2;

    ambit::BlockedTensor Ha;
    ambit::BlockedTensor Hb;
    ambit::BlockedTensor Fa;
    ambit::BlockedTensor Fb;
    ambit::BlockedTensor G1a;
    ambit::BlockedTensor G1b;
    ambit::BlockedTensor Vaa;
    ambit::BlockedTensor Vab;
    ambit::BlockedTensor Vbb;
    ambit::BlockedTensor D2aa;
    ambit::BlockedTensor D2ab;
    ambit::BlockedTensor D2bb;

    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();
    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    SharedWavefunction ref_wfn_;

  public:
    // => Constructors <= //

    MethodBase(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
               std::shared_ptr<MOSpaceInfo> mo_space_info);
    ~MethodBase();

    /// The energy of the reference
    double E0_;

    /// The frozen-core energy
    double frozen_core_energy_;
};
}
} // End Namespaces

#endif // _methodbase_h_
