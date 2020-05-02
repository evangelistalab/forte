/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/dimension.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"

#include "base_classes/forte_options.h"

#include "fragment_projector.h"

using namespace psi;

namespace forte {

std::pair<psi::SharedMatrix, int> make_fragment_projector(SharedWavefunction wfn,
                                                          std::shared_ptr<ForteOptions> options) {
    // Run this code only if user specified fragments
    std::shared_ptr<Molecule> molecule = wfn->molecule();
    int nfrag = molecule->nfragments();
    if (nfrag == 1) {
        throw PSIEXCEPTION("A input molecule with fragments (-- in atom list) is required "
                           "for embedding!");
    }
    outfile->Printf(
        "\n  The input molecule has %d fragments, treating the first fragment as the system.\n",
        nfrag);

    std::shared_ptr<BasisSet> prime_basis = wfn->basisset();

    // Create a fragmentprojector object
    FragmentProjector FP(molecule, prime_basis);

    // Create a fragmentprojector with the second constructor if we want to project to minAO or use
    // IAO procedure FragmentProjector FP(molecule, prime_basis, minao_basis);

    // Compute and return the projector matrix
    psi::SharedMatrix Pf = FP.build_f_projector(prime_basis);
    int nbfA = FP.get_nbf_A();
    std::pair<psi::SharedMatrix, int> Projector = std::make_pair(Pf, nbfA);

    return Projector;
}

FragmentProjector::FragmentProjector(std::shared_ptr<Molecule> molecule,
                                     std::shared_ptr<BasisSet> basis)
    : molecule_(molecule), basis_(basis) {
    startup();
}

void FragmentProjector::startup() {

    std::vector<int> sys_list = {0};  // the first fragment in the input is the system
    std::vector<int> ghost_list = {}; // leave empty to include no fragments with ghost atoms

    // extract the sys molecule objects
    std::shared_ptr<Molecule> mol_sys = molecule_->extract_subsets(sys_list, ghost_list);

    outfile->Printf("\n  System Fragment \n");
    mol_sys->print();

    nbf_ = basis_->nbf();
    outfile->Printf("\n  Number of basis on all atoms: %d", nbf_);

    natom_A_ = mol_sys->natom();
    int count_basis = 0;
    for (int mu = 0; mu < nbf_; mu++) {
        int A = basis_->function_to_center(mu);
        // outfile->Printf("\n  Function %d is on atom %d", mu, A);
        if (A < natom_A_) {
            count_basis += 1;
        }
    }
    outfile->Printf("\n  Number of basis in the system fragment: %d", count_basis);
    nbf_A_ = count_basis;
}

SharedMatrix FragmentProjector::build_f_projector(std::shared_ptr<psi::BasisSet> basis) {

    std::vector<int> zeropi(1, 0);
    Dimension A_begin(zeropi);
    Dimension A_end(zeropi);
    A_begin[0] = 0;
    A_end[0] = nbf_A_;

    std::shared_ptr<IntegralFactory> integral_pp(new IntegralFactory(basis, basis, basis, basis));
    std::shared_ptr<OneBodyAOInt> S_int(integral_pp->ao_overlap());
    SharedMatrix S_nn = std::make_shared<psi::Matrix>("S_nn", nbf_, nbf_);
    S_int->compute(S_nn);

    Slice fragA(A_begin, A_end);

    // Construct the system portion of S (S_A)
    SharedMatrix S_A = S_nn->get_block(fragA, fragA);

    // Construct S_A^-1 and store it in a matrix of size nbf x nbf
    S_A->general_invert();
    SharedMatrix S_A_nn(new Matrix("S system in fullsize", nbf_, nbf_));
    S_A_nn->set_block(fragA, fragA, S_A);

    // Evaluate AO basis projector  P = S^T (S_A)^{-1} S
    S_A_nn->transform(S_nn);

    return S_A_nn;
}

} // namespace forte
