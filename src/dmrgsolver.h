/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef DMRGSOLVER_H
#define DMRGSOLVER_H

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libfock/jk.h"
#include "reference.h"
#include "integrals/integrals.h"
#include "helpers.h"

#include "chemps2/Irreps.h"
#include "chemps2/Problem.h"
#include "chemps2/CASSCF.h"
#include "chemps2/Initialize.h"
#include "chemps2/EdmistonRuedenberg.h"

namespace psi {
namespace forte {

class DMRGSolver {
  public:
    DMRGSolver(SharedWavefunction ref_wfn, Options& options,
               std::shared_ptr<MOSpaceInfo> mo_space_info,
               std::shared_ptr<ForteIntegrals> ints);
    DMRGSolver(SharedWavefunction ref_wfn, Options& options,
               std::shared_ptr<MOSpaceInfo> mo_space_info);
    void compute_energy();

    Reference reference() { return dmrg_ref_; }
    void set_max_rdm(int max_rdm) { max_rdm_ = max_rdm; }
    void spin_free_rdm(bool spin_free) { spin_free_rdm_ = spin_free; }
    void disk_3_rdm(bool use_disk_for_3rdm) { disk_3_rdm_ = use_disk_for_3rdm; }
    void set_up_integrals(const ambit::Tensor& active_integrals,
                          const std::vector<double>& one_body) {
        active_integrals_ = active_integrals;
        one_body_integrals_ = one_body;
        use_user_integrals_ = true;
    }
    void set_scalar(double energy) { scalar_energy_ = energy; }

  private:
    Reference dmrg_ref_;
    SharedWavefunction wfn_;
    Options& options_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    std::shared_ptr<ForteIntegrals> ints_;
    bool disk_3_rdm_ = false;
    /// Form CAS-CI Hamiltonian stuff

    void compute_reference(double* one_rdm, double* two_rdm, double* three_rdm,
                           CheMPS2::DMRGSCFindices* iHandler);
    /// Ported over codes from DMRGSCF plugin
    void startup();
    /// By default, compute the second rdm.  If you are doing MRPT2, may need to
    /// change this.
    int max_rdm_ = 3;
    bool spin_free_rdm_ = false;
    int chemps2_groupnumber(const string SymmLabel);
    ambit::Tensor active_integrals_;
    std::vector<double> one_body_integrals_;
    double scalar_energy_ = 0.0;
    std::vector<double> one_body_operator();
    bool use_user_integrals_ = false;
    void print_natural_orbitals(double* one_rdm);
};
}
}
#endif // DMRG_H
