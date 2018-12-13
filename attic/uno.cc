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

#include "uno.h"
#include "helpers/mo_space_info.h"
#include <cmath>
#include <cstdio>
#include <libmints/basisset.h>
#include <libmints/integralparameters.h>
#include <libmints/matrix.h>
#include <libmints/mintshelper.h>
#include <libmints/multipolesymmetry.h>
#include <libmints/petitelist.h>
#include <libmints/typedefs.h>
#include <libmints/vector.h>
#include <libmints/wavefunction.h>
#include <libqt/qt.h>
#include <map>
#include <vector>

namespace psi {
namespace forte {

UNO::UNO(Options& options) {

    print_method_banner({"Unrestricted Natural Orbitals (UNO)", "Chenyang (York) Li"});

    // wavefunction from psi
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    // number of irrep
    int nirrep = wfn->nirrep();
    Dimension nsopi = wfn->nsopi();

    // total density
    SharedMatrix Dt((wfn->Da())->clone());
    Dt->add(wfn->Db());

    // size of basis function
    size_t nao = (wfn->basisset())->nao();

    // AO overlap
    SharedMatrix overlap = wfn->S();
    SharedMatrix Svector(new Matrix("Overlap Eigen Vectors", nsopi, nsopi));
    SharedVector Svalues(new Vector("Overlap Eigen Values", nsopi));
    overlap->diagonalize(Svector, Svalues);
    //    Svector->eivprint(Svalues);

    // AO overlap one half and minus one half
    SharedMatrix S_onehalf(new Matrix("Overlap One Half", nsopi, nsopi));
    SharedMatrix S_minus_onehalf(new Matrix("Overlap Minus One Half", nsopi, nsopi));
    for (int h = 0; h != nirrep; ++h) {
        size_t m = nsopi[h];
        for (size_t i = 0; i != m; ++i) {
            S_onehalf->set(h, i, i, sqrt(Svalues->get(h, i)));
            S_minus_onehalf->set(h, i, i, 1 / sqrt(Svalues->get(h, i)));
        }
    }
    S_onehalf->back_transform(Svector);
    S_minus_onehalf->back_transform(Svector);
    //    S_onehalf->print();

    // diagonalize S^(1/2) * Dt * S^(1/2)
    //    (wfn->Da())->print();
    //    (wfn->Db())->print();
    //    Dt->print();
    Dt->transform(S_onehalf); // because S is symmetric
                              //    Dt->print();
    SharedMatrix Xvector(new Matrix("S*Dt Eigen Vectors", nsopi, nsopi));
    SharedVector occ(new Vector("Occupation Numbers", nsopi));
    Dt->diagonalize(Xvector, occ, descending);
    Xvector->eivprint(occ);

    // UNO coefficients
    SharedMatrix Ca = wfn->Ca();
    SharedMatrix Cb = wfn->Cb();
    Ca->print();
    SharedMatrix Cnew(new Matrix("NO Coefficients", nsopi, nsopi));
    Cnew->gemm(false, false, 1.0, S_minus_onehalf, Xvector, 0.0);
    Ca->copy(Cnew);
    Cb->copy(Cnew);
    Cnew->print();

    // form new density
    SharedMatrix Ca_scale(Ca->clone());
    for (int h = 0; h != nirrep; ++h) {
        for (int i = 0; i != nsopi[h]; ++i) {
            Ca_scale->scale_column(h, i, occ->get(h, i));
        }
    }
    SharedMatrix Dnew(new Matrix("New Density (SO basis)", nsopi, nsopi));
    Dnew->gemm(false, true, 1.0, Ca_scale, Ca, 0.0);
    Dnew->scale(0.5);

    // transform Dnew to AO
    boost::shared_ptr<BasisSet> basisset = wfn->basisset();
    PetiteList petite(basisset, wfn->integral(), true);
    SharedMatrix aotoso = petite.aotoso();
    SharedMatrix sotoao = petite.sotoao();
    SharedMatrix Dao(new Matrix("D (AO basis)", nao, nao));
    Dao->remove_symmetry(Dnew, sotoao);
    //    Dnew->print();
    //    Dao->print();

    // get ao eri
    MintsHelper helper;
    SharedMatrix eri = helper.ao_eri();
    outfile->Printf("\n I am here");
    outfile->Printf("\n  size of ao: %zu", nao);
    outfile->Printf("\n  size of so: %zu", wfn->nso());
    outfile->Printf("\n  size of row of eri: %zu", eri->nrow());
    outfile->Printf("\n  size of col of eri: %zu", eri->ncol());

    // form G
    SharedMatrix Gao(new Matrix("G (AO basis)", nao, nao));
    SharedMatrix G(new Matrix("G (SO basis)", nsopi, nsopi));
    for (size_t m = 0; m < nao; ++m) {
        for (size_t n = 0; n < nao; ++n) {
            double temp = 0.0;
            for (size_t r = 0; r < nao; ++r) {
                for (size_t s = 0; s < nao; ++s) {
                    double J = eri->get(m * nao + n, r * nao + s);
                    double K = eri->get(m * nao + r, n * nao + s);
                    temp += Dao->get(r, s) * (2 * J - K);
                }
            }
            Gao->set(m, n, temp);
        }
    }
    G->apply_symmetry(Gao, aotoso);

    // form Fock matrix
    SharedMatrix F((wfn->H())->clone());
    F->add(G);
    F->transform(Ca);
    SharedVector F_diag(new Vector("Fock Diagonal Elements", nsopi));
    for (int h = 0; h != nirrep; ++h) {
        for (int i = 0; i != nsopi[h]; ++i) {
            F_diag->set(h, i, F->get(h, i, i));
        }
    }
    F_diag->print();
    F->print();

    // print occupation number
    std::vector<size_t> closed, active;
    double unomin = options.get_double("UNOMIN");
    double unomax = options.get_double("UNOMAX");
    outfile->Printf("\n  UNO Orbital Spaces for CASSCF/CASCI (Min. Occ.: %.3f, Max. Occ.: %.3f)",
                    unomin, unomax);
    outfile->Printf("\n");
    for (int h = 0; h != nirrep; ++h) {
        size_t closedpi = 0, activepi = 0;
        for (int i = 0; i != nsopi[h]; ++i) {
            double occ_num = occ->get(h, i);
            if (occ_num < unomin) {
                continue;
            } else if (occ_num >= unomax) {
                ++closedpi;
            } else {
                ++activepi;
            }
        }
        closed.push_back(closedpi);
        active.push_back(activepi);
    }
    outfile->Printf("\n  %-25s ", "CLOSED:");
    for (auto& rocc : closed) {
        outfile->Printf("%5zu", rocc);
    }
    outfile->Printf("\n  %-25s ", "ACTIVE:");
    for (auto& aocc : active) {
        outfile->Printf("%5zu", aocc);
    }
    outfile->Printf("\n");

    // print UNO details
    if (options.get_bool("UNO_PRINT")) {
        outfile->Printf("\n  UNO Occupation Number:\n");
        occ->print();
        outfile->Printf("\n  UNO Coefficients:\n");
        Ca->print();
    }

    // write molden
    if (options.get_bool("MOLDEN_WRITE")) {
        boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(wfn));
        std::string filename = get_writer_file_prefix() + ".molden";

        SharedVector occ_a(new Vector("Occ. Alpha", nsopi));
        SharedVector occ_b(new Vector("Occ. Beta", nsopi));
        occ_a->copy(*occ);
        occ_b->copy(*occ);
        occ_a->scale(0.5);
        occ_b->scale(0.5);

        if (remove(filename.c_str()) == 0) {
            outfile->Printf("\n  Remove previous molden file named %s.", filename.c_str());
        }
        outfile->Printf("\n  Write molden file to %s.", filename.c_str());
        molden->write(filename, Ca, Ca, F_diag, F_diag, occ_a, occ_b);
    }
}

UNO::~UNO() {}
}
}
