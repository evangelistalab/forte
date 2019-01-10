/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <map>
#include <numeric>
#include <regex>
#include <vector>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/element_to_Z.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/petitelist.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/masses.h"

#include "boost/format.hpp"

#include "fragmentprojector.h"

using namespace psi;

namespace forte {

psi::SharedMatrix create_fragment_projector(psi::SharedWavefunction wfn, psi::Options& options) {
    SharedMatrix Pf;

    // Run this code only if user specified fragments
	std::shared_ptr<Molecule> molecule = wfn->molecule();
	int nfrag = mol->nfragments();
	if (nfrag == 1) {
		throw PSIEXCEPTION("A input molecule with fragments (-- in atom list) is required "
			"for embedding!");
	}
	outfile->Printf(
		"\n The input molecule have %d fragments, assigning the first fragment as system! \n",
		nfrag);

	std::shared_ptr<BasisSet> prime_basis = wfn->basisset();
	//std::shared_ptr<BasisSet> minao_basis = wfn->get_basisset("MINAO_BASIS");

        // Create a fragmentprojector object
		FragmentProjector FP(molecule, prime_basis);

		// Create a fragmentprojector with the second constructor if we want to project to minAO or use IAO procedure
		// FragmentProjector FP(molecule, prime_basis, minao_basis);

        // Compute and return the projector matrix
        Pf = FP.build_f_projector(molecule, prime_basis);
    }
    return Pf;
}

FragmentProjector::FragmentProjector(std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basis)
    : molecule_(molecule), basis_(basis) {
    startup();
}

void FragmentProjector::startup() {

	std::vector<int> none_list = {};
	std::vector<int> sys_list = { 0 };
	std::vector<int> env_list = { 1 };

	std::shared_ptr<Molecule> mol_sys = molecule_->extract_subsets(sys_list, none_list);
	std::shared_ptr<Molecule> mol_env = molecule_->extract_subsets(env_list, none_list);
	outfile->Printf("\n System Fragment \n");
	mol_sys->print();
	//outfile->Printf("\n Environment Fragment(s) \n");
	//mol_env->print();

	nbf_ = basis_->nbf();
	outfile->Printf("\n number of basis on all atoms: %d", nbf);

	natom_A_ = mol_sys->natom();
	int count_basis = 0;
	for (int mu = 0; mu < nbf_; mu++) {
		int A = basis_->function_to_center(mu);
		// outfile->Printf("\n  Function %d is on atom %d", mu, A);
		if (A < natom_A_) {
			count_basis += 1;
		}
	}
	outfile->Printf("\n number of basis in \"system\": %d", count_basis);
	nbf_A_ = count_basis;

	//Create fragment slice (0 -> nbf_A, AA block)
	A_begin_[0] = 0;
	A_end_[0] = nbf_A_;
}

psi::SharedMatrix FragmentProjector::build_f_projector(std::shared_ptr<psi::Molecule> molecule, 
	std::shared_ptr<psi::BasisSet> basis) {

	std::shared_ptr<IntegralFactory> integral_pp(
		new IntegralFactory(basis, basis, basis, basis));

	std::shared_ptr<OneBodyAOInt> S_int(integral_pp->ao_overlap());
	SharedMatrix S_nn = std::make_shared<psi::Matrix>("S_nn", nbf_, nbf_);
	S_int->compute(S_nn);

	S_nn->print();

	Slice fragA(A_begin_, A_end_);

	// Construct S_A
	SharedMatrix S_A = S_nn->get_block(fragA, fragA);

	// Construct S_A^-1 in n*n size
	S_A->general_invert();
	SharedMatrix S_A_nn(new Matrix("S system in fullsize", nbf_, nbf_));
	S_A_nn->set_block(fragA, fragA, S_A);

	// Evaluate AO basis projector
	S_A_nn->transform(S_nn);

    return S_A_nn;
}

} // namespace forte
