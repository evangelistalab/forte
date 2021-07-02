/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _v2rdm_h_
#define _v2rdm_h_

#include <map>
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include "integrals/integrals.h"
#include "base_classes/rdms.h"

#define PSIF_V2RDM_D2AA 270
#define PSIF_V2RDM_D2AB 271
#define PSIF_V2RDM_D2BB 272
#define PSIF_V2RDM_D3AAA 273
#define PSIF_V2RDM_D3AAB 274
#define PSIF_V2RDM_D3BBA 275
#define PSIF_V2RDM_D3BBB 276

using namespace ambit;

namespace forte {

class V2RDM : public psi::Wavefunction {
  public:
    /**
     * V2RDM Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    V2RDM(psi::SharedWavefunction ref_wfn, psi::Options& options, std::shared_ptr<ForteIntegrals> ints,
          std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~V2RDM();

    /// Returns the reference object of forte
    RDMs reference();

  protected:
    /// Start-up function called in the constructor
    void startup();

    /// The molecular integrals
    std::shared_ptr<ForteIntegrals> ints_;

    /// The frozen-core energy
    double frozen_core_energy_;

    /// MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Number of irrep
    size_t nirrep_;
    /// MO per irrep;
    psi::Dimension nmopi_;
    /// Frozen docc per irrep
    psi::Dimension fdoccpi_;
    /// Restricted docc per irrep
    psi::Dimension rdoccpi_;
    /// Active per irrep
    psi::Dimension active_;
    /// Map active absolute index to relative index
    std::map<size_t, size_t> abs_to_rel_;

    /// List of core MOs
    std::vector<size_t> core_mos_;
    /// List of active MOs
    std::vector<size_t> actv_mos_;

    /// Read two particle density
    void read_2pdm();
    /// Build one particle density
    void build_opdm();
    /// Read three particle density
    void read_3pdm();

    /// Compute the reference energy
    double compute_ref_energy();

    /// One particle density matrix (active only)
    ambit::Tensor D1a_;
    ambit::Tensor D1b_;

    /// Two particle density matrix (active only)
    std::vector<ambit::Tensor> D2_; // D2aa, D2ab, D2bb

    /// Three particle density matrix (active only)
    std::vector<ambit::Tensor> D3_; // D3aaa, D3aab, D3abb, D3bbb

    /// Write densities (or cumulants) to files
    /// 1PDM: file_opdm_a, file_opdm_b
    /// 2PDM: file_2pdm_aa, file_2pdm_ab, file_2pdm_bb
    /// 3PDM: file_3pdm_aaa, file_3pdm_aab, file_3pdm_abb, file_3pdm_bbb
    void write_density_to_file();
};
}

#endif // V2RDM_H
