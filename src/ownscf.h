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

#ifndef _ownscf_h_
#define _ownscf_h_

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"

#include "ci_rdm/ci_rdms.h"
#include "helpers.h"
#include "reference.h"
#include "sparse_ci/determinant.h"
#include "integrals/integrals.h"
#include "fci/fci_integrals.h"
#include "determinant_hashvector.h"
#include "operator.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "forte_options.h"

namespace psi {

namespace forte {

	void set_SCF_options(ForteOptions& foptions);

/**
 * @brief The OwnSCF class
 * Computes embedding excitation with iao
 */
class OwnSCF : public Wavefunction {
  public:
    // => Constructor <= //
    OwnSCF(SharedWavefunction ref_wfn, Options& options, 
                  std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~OwnSCF();

    double compute_energy();

  private:
    void startup();

    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    SharedWavefunction ref_wfn_;

	double E_conv_;

	double D_conv_;

	int Maxcyc_;
};
}
} // End Namespaces

#endif // _ownscf_h_
