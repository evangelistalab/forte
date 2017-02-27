
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

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"

#include "mrci.h"

namespace psi {
namespace forte {

MRCI::MRCI( SharedWavefunction ref_wfn, Options& options,
            std::shared_ptr<ForteIntegrals> ints, 
            std::shared_ptr<MOSpaceInfo> mo_space_info,
            DeterminantMap& reference)
    : Wavefunction(options), ints_(ints), 
      mo_space_info_(mo_space_info), 
      reference_(reference)
{
    shallow_copy(ref_wfn);
    print_method_banner(
        {"Multireference CISD", "Jeff Schriber"});
   startup();
}

MRCI::~MRCI() {}

void MRCI::startup()
{
    // Define the correlated space
    auto correlated_mo = mo_space_info_->get_corr_abs_mo("GENERALIZED PARTICLE");

    fci_ints_ = std::make_shared<FCIIntegrals>(
        ints_,mo_space_info_->get_corr_abs_mo("GENERALIZED PARTICLE"), 
        mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));

    // Set the integrals
    ambit::Tensor tei_active_aa = ints_->aptei_aa_block(correlated_mo, correlated_mo, correlated_mo, correlated_mo); 
    ambit::Tensor tei_active_ab = ints_->aptei_ab_block(correlated_mo, correlated_mo, correlated_mo, correlated_mo); 
    ambit::Tensor tei_active_bb = ints_->aptei_bb_block(correlated_mo, correlated_mo, correlated_mo, correlated_mo); 

    fci_ints_->compute_restricted_one_body_operator();

    STLBitsetDeterminant::set_ints(fci_ints_);

    nroot_ = options_.get_int("NROOT");
    multiplicity_ = options_.get_int("MULTIPLICITY");

    diag_method_ = DLSolver;

}

double MRCI::compute_energy()
{

    std::vector<int> mo_symmetry = mo_space_info_->symmetry("GENERALIZED PARTICLE");

    WFNOperator op(mo_symmetry);

    outfile->Printf("\n  Adding single and double excitations ...");
    Timer add;
    op.add_singles(reference_);    
    op.add_doubles(reference_);    
    outfile->Printf("\n  Excitations took %1.5f s", add.get());
    outfile->Printf("\n  Dimension of model space: %zu", reference_.size());

    op.op_lists(reference_);
    op.tp_lists(reference_);

    // Diagonalize MR-CISD Hamiltonian
    SharedMatrix evecs;
    SharedVector evals;    

    SparseCISolver sparse_solver;
    sparse_solver.diagonalize_hamiltonian_map(reference_,op,evals,evecs,nroot_,multiplicity_,diag_method_ ); 

    std::vector<double> energy(nroot_);
    double scalar = fci_ints_->scalar_energy() + molecule_->nuclear_repulsion_energy();

    for( int n = 0; n < nroot_; ++n ){
        energy[n] = scalar + evals->get(n);
        outfile->Printf("\n  MR-CISD energy root %d: %1.13f eh", n, energy[n]);
    }

    return energy[0];
}

}}
