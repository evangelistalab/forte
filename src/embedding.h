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

#ifndef _embedding_h_
#define _embedding_h_

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

	void set_EMBEDDING_options(ForteOptions& foptions);

/**
 * @brief The Embedding class
 * Computes embedding excitation with iao
 */
class Embedding : public Wavefunction {
  public:
    // => Constructor <= //
    Embedding(SharedWavefunction ref_wfn, Options& options, 
              std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~Embedding();

    double compute_energy();

  private:
    void startup();

    double do_env(SharedWavefunction wfn, SharedMatrix h, Options& options);

    double do_sys(SharedWavefunction wfn, SharedMatrix h, Options& options);

    std::map<std::string, SharedMatrix> localize(SharedWavefunction wfn, Options& options);

    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    SharedWavefunction ref_wfn_;
};
} // namespace forte
} // namespace psi

#endif // _embedding_h_
