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

#include <cmath>

#include "multidimensional_arrays.h"
#include <libmints/molecule.h>
#include <libmints/wavefunction.h>
#include <libpsio/psio.hpp>

#include "sobase.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

SOBase::SOBase(Options& options, ForteIntegrals* ints, TwoIndex G1)
    : ints_(ints), options_(options) {
    startup(G1);
}

SOBase::~SOBase() { release(); }

void SOBase::startup(TwoIndex G1) {
    // Extract data from the reference (SCF) wavefunction
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    nirrep_ = wfn->nirrep();
    nmo_ = wfn->nmo();
    nso_ = 2 * nmo_;
    nmopi_ = wfn->nmopi();

    //    frzcpi_ = Dimension(nirrep_);
    //    frzvpi_ = Dimension(nirrep_);

    allocate();
    sort_integrals();

    loop_p {
        No_[p] = G1[p][p];
        Nv_[p] = 1.0 - G1[p][p];
    }
    loop_p loop_q {
        G1_[p][q] = G1[p][q];
        E1_[p][q] = (p == q ? 1.0 : 0.0) - G1_[p][q];
    }

    build_fock();
}

void SOBase::allocate() {
    No_ = new double[nso_];
    Nv_ = new double[nso_];
    allocate(G1_);
    allocate(E1_);
    allocate(G2_);
    allocate(L2_);
    allocate(H1_);
    allocate(F_);
    allocate(V_);
}

void SOBase::release() {
    delete[] No_;
    delete[] Nv_;
    release(G1_);
    release(E1_);
    release(G2_);
    release(L2_);
    release(H1_);
    release(F_);
    release(V_);
}

void SOBase::allocate(TwoIndex& two_index) { init_matrix<double>(two_index, nso_, nso_); }

void SOBase::allocate(FourIndex& four_index) {
    init_matrix<double>(four_index, nso_, nso_, nso_, nso_);
}

void SOBase::release(TwoIndex& two_index) { free_matrix<double>(two_index, nso_, nso_); }

void SOBase::release(FourIndex& four_index) {
    free_matrix<double>(four_index, nso_, nso_, nso_, nso_);
}

void SOBase::sort_integrals() {
    loop_p {
        so_mo.push_back(p % nmo_);
        bool spin = p >= nmo_;
        so_spin.push_back(spin);
        outfile->Printf("\n  so %3d = mo %3d  spin %s", p, so_mo[p], not so_spin[p] ? "a" : "b");
    }
    loop_p loop_q {
        H1_[p][q] = 0.0;
        if (so_spin[p] == true and so_spin[q] == true)
            H1_[p][q] = ints_->oei_a(so_mo[p], so_mo[q]);
        if (so_spin[p] == false and so_spin[q] == false)
            H1_[p][q] = ints_->oei_b(so_mo[p], so_mo[q]);
    }
    loop_p loop_q loop_r loop_s {
        V_[p][q][r][s] = 0.0;
        //        if((so_spin[p] == so_spin[r]) and (so_spin[q] == so_spin[s]))
        //            V_[p][q][r][s] += ints_->rtei(so_mo[p],so_mo[r],so_mo[q],so_mo[s]);
        //        if((so_spin[p] == so_spin[s]) and (so_spin[q] == so_spin[r]))
        //            V_[p][q][r][s] -= ints_->rtei(so_mo[p],so_mo[s],so_mo[q],so_mo[r]);
    }
    outfile->Printf("\n\n  WARNING: I had to temporarily disable the SO code! :(");
    
    exit(1);
}

void SOBase::build_fock() {
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    boost::shared_ptr<Molecule> molecule_ = wfn->molecule();
    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();

    // Compute the reference energy
    E0_ = nuclear_repulsion_energy_;
    loop_p loop_q { E0_ += H1_[p][q] * G1_[q][p]; }
    loop_p loop_q loop_r loop_s {
        E0_ += 0.25 * V_[p][q][r][s] * (G1_[p][r] * G1_[q][s] - G1_[p][s] * G1_[q][r]);
    }
    loop_p loop_q {
        F_[p][q] = H1_[p][q];
        loop_r loop_s { F_[p][q] += V_[p][r][q][s] * G1_[s][r]; }
    }

    //    loop_p loop_q{
    //        outfile->Printf("\nF[%2d][%2d] = %20.12f (ON = %.6f)",p,q,F_[p][q],G1_[p][q]);
    //    }
    outfile->Printf("\n  The energy of the reference is: %20.12f Eh", E0_);
    outfile->Printf("\n  Diagonal elements of the Fock matrix:");
    outfile->Printf("\n  SO            Epsilon         ON");
    loop_p { outfile->Printf("\n  %2d  %20.12f   %8.6f", p, F_[p][p], G1_[p][p]); }
}

void SOBase::add(double fA, TwoIndex& A, double fB, TwoIndex& B) {
    loop_p loop_q { B[p][q] = fA * A[p][q] + fB * B[p][q]; }
}

void SOBase::add(double fA, FourIndex& A, double fB, FourIndex& B) {
    loop_p loop_q loop_r loop_s { B[p][q][r][s] = fA * A[p][q][r][s] + fB * B[p][q][r][s]; }
}

double SOBase::norm(TwoIndex& A) {
    double norm = 0.0;
    loop_p loop_q { norm += std::pow(A[p][q], 2.0); }
    return std::sqrt(norm);
}

double SOBase::norm(FourIndex& A) {
    double norm = 0.0;
    loop_p loop_q loop_r loop_s { norm += std::pow(A[p][q][r][s], 2.0); }
    return std::sqrt(0.25 * norm);
}

void SOBase::zero(TwoIndex& A) {
    loop_p loop_q { A[p][q] = 0.0; }
}

void SOBase::zero(FourIndex& A) {
    loop_p loop_q loop_r loop_s { A[p][q][r][s] = 0.0; }
}
}
} // EndNamespaces
