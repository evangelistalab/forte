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

#include <libmoinfo/libmoinfo.h>

#include "string_determinant.h"

using namespace std;
using namespace psi;

#include <libmints/matrix.h>
#include <psi4-dec.h>

namespace psi {
namespace forte {

double SlaterSign(bool* I, int n);
double SlaterSign(const std::vector<bool>& I, int n);

std::shared_ptr<ForteIntegrals> StringDeterminant::ints_ = 0;
double StringDeterminant::ahole_[20];
double StringDeterminant::bhole_[20];
double StringDeterminant::apart_[20];
double StringDeterminant::bpart_[20];

StringDeterminant::StringDeterminant() : nmo_(0) {}

StringDeterminant::StringDeterminant(const std::vector<int>& occupation, bool print_det)
    : nmo_(occupation.size() / 2) {
    allocate();
    for (int p = 0; p < nmo_; ++p) {
        alfa_bits_[p] = occupation[p];
        beta_bits_[p] = occupation[p + nmo_];
    }
    if (print_det)
        print();
}

StringDeterminant::StringDeterminant(const std::vector<bool>& occupation, bool print_det)
    : nmo_(occupation.size() / 2) {
    allocate();
    for (int p = 0; p < nmo_; ++p) {
        alfa_bits_[p] = occupation[p];
        beta_bits_[p] = occupation[p + nmo_];
    }
    if (print_det)
        print();
}

StringDeterminant::StringDeterminant(const std::vector<bool>& occupation_a,
                                     const std::vector<bool>& occupation_b, bool print_det)
    : nmo_(occupation_a.size()) {
    allocate();
    for (int p = 0; p < nmo_; ++p) {
        alfa_bits_[p] = occupation_a[p];
        beta_bits_[p] = occupation_b[p];
    }
    if (print_det)
        print();
}

StringDeterminant::StringDeterminant(const StringDeterminant& det) : nmo_(det.nmo_) {
    allocate();
    for (int n = 0; n < 2 * nmo_; ++n) {
        alfa_bits_[n] = det.alfa_bits_[n];
    }
}

StringDeterminant::StringDeterminant(StringDeterminant& det) : nmo_(det.nmo_) {
    allocate();
    for (int n = 0; n < 2 * nmo_; ++n) {
        alfa_bits_[n] = det.alfa_bits_[n];
    }
}

StringDeterminant::StringDeterminant(const StringDeterminant& ref, const ExcitationDeterminant& ex)
    : nmo_(ref.nmo_) {
    allocate();
    for (int n = 0; n < 2 * nmo_; ++n) {
        alfa_bits_[n] = ref.alfa_bits_[n];
    }
    for (int aex = 0; aex < ex.naex_; ++aex) {
        alfa_bits_[ex.aann(aex)] = false;
        alfa_bits_[ex.acre(aex)] = true;
    }
    for (int bex = 0; bex < ex.nbex_; ++bex) {
        beta_bits_[ex.bann(bex)] = false;
        beta_bits_[ex.bcre(bex)] = true;
    }
}

StringDeterminant& StringDeterminant::operator=(const StringDeterminant& rhs) {
    if (nmo_ != rhs.nmo_) {
        deallocate();
        nmo_ = rhs.nmo_;
        allocate();
    }
    for (int n = 0; n < 2 * nmo_; ++n) {
        alfa_bits_[n] = rhs.alfa_bits_[n];
    }
    return *this;
}

StringDeterminant::~StringDeterminant() { deallocate(); }

void StringDeterminant::allocate() {
    if (nmo_ > 0) {
        alfa_bits_ = new bool[2 * nmo_];
        beta_bits_ = &(alfa_bits_[nmo_]);
    }
}

void StringDeterminant::deallocate() {
    if (nmo_ > 0) {
        delete[] alfa_bits_;
    }
}

void StringDeterminant::set_bits(bool*& alfa_bits, bool*& beta_bits) {
    std::copy(alfa_bits, alfa_bits + nmo_, alfa_bits_);
    std::copy(beta_bits, beta_bits + nmo_, beta_bits_);
}

void StringDeterminant::set_bits(std::vector<bool>& alfa_bits, std::vector<bool>& beta_bits) {
    std::copy(alfa_bits.begin(), alfa_bits.end(), alfa_bits_);
    std::copy(beta_bits.begin(), beta_bits.end(), beta_bits_);
}

/**
 * Print the determinant
 */
void StringDeterminant::print() const {
    outfile->Printf("\n  |");
    for (int p = 0; p < nmo_; ++p) {
        outfile->Printf("%d", alfa_bits_[p] ? 1 : 0);
    }
    outfile->Printf("|");
    for (int p = 0; p < nmo_; ++p) {
        outfile->Printf("%d", beta_bits_[p] ? 1 : 0);
    }
    outfile->Printf(">");
}

/**
 * Compute the energy of this determinant
 * @return the electronic energy (does not include the nuclear repulsion energy)
 */
double StringDeterminant::energy() const {
    double matrix_element = 0.0;
    matrix_element = ints_->frozen_core_energy() + ints_->scalar();
    //    for(int p = 0; p < nmo_; ++p){
    //        if(alfa_bits_[p]) matrix_element += ints_->diag_roei(p);
    //        if(beta_bits_[p]) matrix_element += ints_->diag_roei(p);
    //        if(alfa_bits_[p]) outfile->Printf("\n  One-electron terms: %20.12f + %20.12f
    //        (string)",ints_->diag_roei(p),ints_->diag_roei(p));
    //        for(int q = 0; q < nmo_; ++q){
    //            if(alfa_bits_[p] and alfa_bits_[q]){
    //                matrix_element +=   0.5 * ints_->diag_ce_rtei(p,q);
    //                outfile->Printf("\n  One-electron terms (%da,%da): 0.5 * %20.12f
    //                (string)",p,q,ints_->diag_ce_rtei(p,q));
    //            }
    //            if(alfa_bits_[p] and beta_bits_[q]){
    //                matrix_element += ints_->diag_c_rtei(p,q);
    //                outfile->Printf("\n  One-electron terms (%da,%db): 1.0 * %20.12f
    //                (string)",p,q,ints_->diag_c_rtei(p,q));
    //            }
    //            if(beta_bits_[p] and beta_bits_[q]){
    //                matrix_element +=   0.5 * ints_->diag_ce_rtei(p,q);
    //                outfile->Printf("\n  One-electron terms (%db,%db): 0.5 * %20.12f
    //                (string)",p,q,ints_->diag_ce_rtei(p,q));
    //            }
    //        }
    //    }
    for (int p = 0; p < nmo_; ++p) {
        if (alfa_bits_[p])
            matrix_element += ints_->oei_a(p, p);
        if (beta_bits_[p])
            matrix_element += ints_->oei_b(p, p);
        for (int q = 0; q < nmo_; ++q) {
            if (alfa_bits_[p] and alfa_bits_[q]) {
                matrix_element += 0.5 * ints_->diag_aptei_aa(p, q);
            }
            if (alfa_bits_[p] and beta_bits_[q]) {
                matrix_element += ints_->diag_aptei_ab(p, q);
            }
            if (beta_bits_[p] and beta_bits_[q]) {
                matrix_element += 0.5 * ints_->diag_aptei_bb(p, q);
            }
        }
    }
    return (matrix_element);
}

/**
 * Compute the kinetic energy of this determinant
 * @return the kinetic energy
 */
double StringDeterminant::one_electron_energy() {
    double matrix_element = 0.0;
    for (int p = 0; p < nmo_; ++p) {
        if (alfa_bits_[p])
            matrix_element += ints_->oei_a(p, p);
        if (beta_bits_[p])
            matrix_element += ints_->oei_b(p, p);
    }
    return (matrix_element);
}

/**
 * Compute the energy of this determinant with respect to a reference determinant
 * @return the electronic energy (does not include the nuclear repulsion energy)
 */
double StringDeterminant::excitation_energy(const StringDeterminant& reference) {
    double matrix_element = 0.0;
    // Find the difference in orbital occupation
    int naexp = 0;
    int nbexp = 0;
    int naexh = 0;
    int nbexh = 0;
    for (int p = 0; p < nmo_; ++p) {
        if (reference.alfa_bits_[p] and not alfa_bits_[p]) {
            ahole_[naexh] = p;
            naexh++;
        }
        if (alfa_bits_[p] and not reference.alfa_bits_[p]) {
            apart_[naexp] = p;
            naexp++;
        }
        if (reference.beta_bits_[p] and not beta_bits_[p]) {
            bhole_[nbexh] = p;
            nbexh++;
        }
        if (beta_bits_[p] and not reference.beta_bits_[p]) {
            bpart_[nbexp] = p;
            nbexp++;
        }
    }
    for (int i = 0; i < naexh; ++i) {
        matrix_element -= ints_->diag_fock_a(ahole_[i]);
        matrix_element += ints_->diag_fock_a(apart_[i]);
    }
    for (int i = 0; i < nbexh; ++i) {
        matrix_element -= ints_->diag_fock_b(bhole_[i]);
        matrix_element += ints_->diag_fock_b(bpart_[i]);
    }

    for (int i = 0; i < naexh; ++i) {
        for (int j = i + 1; j < naexh; ++j) {
            //            matrix_element += ints_->diag_ce_rtei(ahole_[i],ahole_[j]);
            matrix_element += ints_->diag_aptei_aa(ahole_[i], ahole_[j]);
        }
    }
    for (int i = 0; i < nbexh; ++i) {
        for (int j = i + 1; j < nbexh; ++j) {
            //            matrix_element += ints_->diag_ce_rtei(bhole_[i],bhole_[j]);
            matrix_element += ints_->diag_aptei_bb(bhole_[i], bhole_[j]);
        }
    }
    for (int i = 0; i < naexh; ++i) {
        for (int j = 0; j < nbexh; ++j) {
            //            matrix_element += ints_->diag_c_rtei(ahole_[i],bhole_[j]);
            matrix_element += ints_->diag_aptei_ab(ahole_[i], bhole_[j]);
        }
    }

    for (int i = 0; i < naexh; ++i) {
        for (int a = 0; a < naexp; ++a) {
            //            matrix_element -= ints_->diag_ce_rtei(ahole_[i],apart_[a]);
            matrix_element -= ints_->diag_aptei_aa(ahole_[i], apart_[a]);
        }
    }
    for (int i = 0; i < nbexh; ++i) {
        for (int a = 0; a < nbexp; ++a) {
            //            matrix_element -= ints_->diag_ce_rtei(bhole_[i],bpart_[a]);
            matrix_element -= ints_->diag_aptei_bb(bhole_[i], bpart_[a]);
        }
    }
    for (int i = 0; i < naexh; ++i) {
        for (int a = 0; a < nbexp; ++a) {
            //            matrix_element -= ints_->diag_c_rtei(ahole_[i],bpart_[a]);
            matrix_element -= ints_->diag_aptei_ab(ahole_[i], bpart_[a]);
        }
    }
    for (int i = 0; i < nbexh; ++i) {
        for (int a = 0; a < naexp; ++a) {
            //            matrix_element -= ints_->diag_c_rtei(bhole_[i],apart_[a]);
            matrix_element -= ints_->diag_aptei_ab(apart_[a], bhole_[i]);
        }
    }

    for (int a = 0; a < naexp; ++a) {
        for (int b = a + 1; b < naexp; ++b) {
            //            matrix_element += ints_->diag_ce_rtei(apart_[a],apart_[b]);
            matrix_element += ints_->diag_aptei_aa(apart_[a], apart_[b]);
        }
    }
    for (int a = 0; a < nbexp; ++a) {
        for (int b = a + 1; b < nbexp; ++b) {
            //            matrix_element += ints_->diag_ce_rtei(bpart_[a],bpart_[b]);
            matrix_element += ints_->diag_aptei_bb(bpart_[a], bpart_[b]);
        }
    }
    for (int a = 0; a < naexp; ++a) {
        for (int b = 0; b < nbexp; ++b) {
            //            matrix_element += ints_->diag_c_rtei(apart_[a],bpart_[b]);
            matrix_element += ints_->diag_aptei_ab(apart_[a], bpart_[b]);
        }
    }
    return (matrix_element);
}

/**
 * Compute the energy of this determinant with respect to a reference determinant
 * @return the electronic energy (does not include the nuclear repulsion energy)
 */
double StringDeterminant::excitation_ab_energy(const StringDeterminant& reference) {
    double matrix_element = 0.0;
    // Find the difference in orbital occupation
    int naexp = 0;
    int nbexp = 0;
    int naexh = 0;
    int nbexh = 0;
    for (int p = 0; p < nmo_; ++p) {
        if (reference.alfa_bits_[p] and not alfa_bits_[p]) {
            ahole_[naexh] = p;
            naexh++;
        }
        if (alfa_bits_[p] and not reference.alfa_bits_[p]) {
            apart_[naexp] = p;
            naexp++;
        }
        if (reference.beta_bits_[p] and not beta_bits_[p]) {
            bhole_[nbexh] = p;
            nbexh++;
        }
        if (beta_bits_[p] and not reference.beta_bits_[p]) {
            bpart_[nbexp] = p;
            nbexp++;
        }
    }

    for (int i = 0; i < naexh; ++i) {
        for (int j = 0; j < nbexh; ++j) {
            //            matrix_element += ints_->diag_c_rtei(ahole_[i],bhole_[j]);
            matrix_element += ints_->diag_aptei_ab(ahole_[i], bhole_[j]);
        }
    }
    for (int i = 0; i < naexh; ++i) {
        for (int a = 0; a < nbexp; ++a) {
            //            matrix_element -= ints_->diag_c_rtei(ahole_[i],bpart_[a]);
            matrix_element -= ints_->diag_aptei_ab(ahole_[i], bpart_[a]);
        }
    }
    for (int i = 0; i < nbexh; ++i) {
        for (int a = 0; a < naexp; ++a) {
            //            matrix_element -= ints_->diag_c_rtei(bhole_[i],apart_[a]);
            matrix_element -= ints_->diag_aptei_ab(apart_[a], bhole_[i]);
        }
    }
    for (int a = 0; a < naexp; ++a) {
        for (int b = 0; b < nbexp; ++b) {
            //            matrix_element += ints_->diag_c_rtei(apart_[a],bpart_[b]);
            matrix_element += ints_->diag_aptei_ab(apart_[a], bpart_[b]);
        }
    }
    return (matrix_element);
}

/**
 * Compute the relative excitation of two determinants
 * @return the relative excitation level
 */
int StringDeterminant::excitation_level(const StringDeterminant& reference) {
    int nex = 0;
    for (int p = 0; p < nmo_; ++p) {
        if (reference.alfa_bits_[p] and (not alfa_bits_[p]))
            nex += 1;
        if (reference.beta_bits_[p] and (not beta_bits_[p]))
            nex += 1;
    }
    return (nex);
}

/**
 * Compute the relative excitation of two determinants
 * @return the relative excitation level
 */
int StringDeterminant::excitation_level(const bool* Ia, const bool* Ib) {
    int nex = 0;
    for (int p = 0; p < nmo_; ++p) {
        if (alfa_bits_[p] and (not Ia[p]))
            nex += 1;
        if (beta_bits_[p] and (not Ib[p]))
            nex += 1;
    }
    return (nex);
}

/**
 * Compute the matrix element of the Hamiltonian between this determinant and a given one
 * @param rhs
 * @return
 */
double StringDeterminant::slater_rules(const StringDeterminant& rhs) const {
    bool* Ia = alfa_bits_;
    bool* Ib = beta_bits_;
    bool* Ja = rhs.alfa_bits_;
    bool* Jb = rhs.beta_bits_;

    int nadiff = 0;
    int nbdiff = 0;

    // Count how many differences in mos are there
    for (int n = 0; n < nmo_; ++n) {
        if (Ia[n] != Ja[n])
            nadiff++;
        if (Ib[n] != Jb[n])
            nbdiff++;
        if (nadiff + nbdiff > 4)
            return 0.0; // Get out of this as soon as possible
    }
    nadiff /= 2;
    nbdiff /= 2;

    double matrix_element = 0.0;
    // Slater rule 1 PhiI = PhiJ
    if ((nadiff == 0) and (nbdiff == 0)) {
        matrix_element = ints_->frozen_core_energy() + ints_->scalar();
        for (int p = 0; p < nmo_; ++p) {
            if (alfa_bits_[p])
                matrix_element += ints_->oei_a(p, p);
            if (beta_bits_[p])
                matrix_element += ints_->oei_b(p, p);
            for (int q = 0; q < nmo_; ++q) {
                if (alfa_bits_[p] and alfa_bits_[q])
                    matrix_element += 0.5 * ints_->diag_aptei_aa(p, q);
                if (beta_bits_[p] and beta_bits_[q])
                    matrix_element += 0.5 * ints_->diag_aptei_bb(p, q);
                if (alfa_bits_[p] and beta_bits_[q])
                    matrix_element += ints_->diag_aptei_ab(p, q);
            }
        }
    }

    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 0)) {
        // Diagonal contribution
        int i = 0;
        int j = 0;
        for (int p = 0; p < nmo_; ++p) {
            if ((Ia[p] != Ja[p]) and Ia[p])
                i = p;
            if ((Ia[p] != Ja[p]) and Ja[p])
                j = p;
        }
        double sign = SlaterSign(Ia, i) * SlaterSign(Ja, j);
        matrix_element = sign * ints_->oei_a(i, j);
        for (int p = 0; p < nmo_; ++p) {
            if (Ia[p] and Ja[p]) {
                matrix_element += sign * ints_->aptei_aa(i, p, j, p);
            }
            if (Ib[p] and Jb[p]) {
                matrix_element += sign * ints_->aptei_ab(i, p, j, p);
            }
        }
    }
    // Slater rule 2 PhiI = j_b^+ i_b PhiJ
    if ((nadiff == 0) and (nbdiff == 1)) {
        // Diagonal contribution
        int i = 0;
        int j = 0;
        for (int p = 0; p < nmo_; ++p) {
            if ((Ib[p] != Jb[p]) and Ib[p])
                i = p;
            if ((Ib[p] != Jb[p]) and Jb[p])
                j = p;
        }
        double sign = SlaterSign(Ib, i) * SlaterSign(Jb, j);
        matrix_element = sign * ints_->oei_b(i, j);
        for (int p = 0; p < nmo_; ++p) {
            if (Ia[p] and Ja[p]) {
                matrix_element += sign * ints_->aptei_ab(p, i, p, j);
            }
            if (Ib[p] and Jb[p]) {
                matrix_element += sign * ints_->aptei_bb(i, p, j, p);
            }
        }
    }

    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
    if ((nadiff == 2) and (nbdiff == 0)) {
        // Diagonal contribution
        int i = -1;
        int j = 0;
        int k = -1;
        int l = 0;
        for (int p = 0; p < nmo_; ++p) {
            if ((Ia[p] != Ja[p]) and Ia[p]) {
                if (i == -1) {
                    i = p;
                } else {
                    j = p;
                }
            }
            if ((Ia[p] != Ja[p]) and Ja[p]) {
                if (k == -1) {
                    k = p;
                } else {
                    l = p;
                }
            }
        }
        double sign = SlaterSign(Ia, i) * SlaterSign(Ia, j) * SlaterSign(Ja, k) * SlaterSign(Ja, l);
        matrix_element = sign * ints_->aptei_aa(i, j, k, l);
    }

    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
    if ((nadiff == 0) and (nbdiff == 2)) {
        // Diagonal contribution
        int i, j, k, l;
        i = -1;
        j = -1;
        k = -1;
        l = -1;
        for (int p = 0; p < nmo_; ++p) {
            if ((Ib[p] != Jb[p]) and Ib[p]) {
                if (i == -1) {
                    i = p;
                } else {
                    j = p;
                }
            }
            if ((Ib[p] != Jb[p]) and Jb[p]) {
                if (k == -1) {
                    k = p;
                } else {
                    l = p;
                }
            }
        }
        double sign = SlaterSign(Ib, i) * SlaterSign(Ib, j) * SlaterSign(Jb, k) * SlaterSign(Jb, l);
        matrix_element = sign * ints_->aptei_bb(i, j, k, l);
    }

    // Slater rule 3 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Diagonal contribution
        int i, j, k, l;
        i = j = k = l = -1;
        for (int p = 0; p < nmo_; ++p) {
            if ((Ia[p] != Ja[p]) and Ia[p])
                i = p;
            if ((Ib[p] != Jb[p]) and Ib[p])
                j = p;
            if ((Ia[p] != Ja[p]) and Ja[p])
                k = p;
            if ((Ib[p] != Jb[p]) and Jb[p])
                l = p;
        }
        double sign = SlaterSign(Ia, i) * SlaterSign(Ib, j) * SlaterSign(Ja, k) * SlaterSign(Jb, l);
        matrix_element = sign * ints_->aptei_ab(i, j, k, l);
    }
    return (matrix_element);
}

/**
 * Compute the S^2 matrix element of the Hamiltonian between two determinants specified by the
 * strings (Ia,Ib) and (Ja,Jb)
 * @return S^2
 */
double StringDeterminant::spin2(const StringDeterminant& rhs) const {
    bool* Ia = alfa_bits_;
    bool* Ib = beta_bits_;
    bool* Ja = rhs.alfa_bits_;
    bool* Jb = rhs.beta_bits_;

    // Compute the matrix elements of the operator S^2
    // S^2 = S- S+ + Sz (Sz + 1)
    //     = Sz (Sz + 1) + Nbeta + Npairs - sum_pq' a+(qa) a+(pb) a-(qb) a-(pa)
    double matrix_element = 0.0;

    int nadiff = 0;
    int nbdiff = 0;
    int na = 0;
    int nb = 0;
    int npair = 0;
    int nmo = nmo_;
    // Count how many differences in mos are there and the number of alpha/beta electrons
    for (int n = 0; n < nmo; ++n) {
        if (Ia[n] != Ja[n])
            nadiff++;
        if (Ib[n] != Jb[n])
            nbdiff++;
        if (Ia[n])
            na++;
        if (Ib[n])
            nb++;
        if ((Ia[n] and Ib[n]))
            npair += 1;
    }
    nadiff /= 2;
    nbdiff /= 2;

    double Ms = 0.5 * static_cast<double>(na - nb);

    // PhiI = PhiJ -> S^2 = Sz (Sz + 1) + Nbeta + Npairs
    if ((nadiff == 0) and (nbdiff == 0)) {
        matrix_element += Ms * (Ms + 1.0) + double(nb) - double(npair);
    }

    // PhiI = a+(qa) a+(pb) a-(qb) a-(pa) PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Find a pair of spin coupled electrons
        int i = -1;
        int j = -1;
        // The logic here is a bit complex
        for (int p = 0; p < nmo; ++p) {
            if (Ja[p] and Ib[p] and (not Jb[p]) and (not Ia[p]))
                i = p; //(p)
            if (Jb[p] and Ia[p] and (not Ja[p]) and (not Ib[p]))
                j = p; //(q)
        }
        if (i != j and i >= 0 and j >= 0) {
            double sign =
                SlaterSign(Ja, i) * SlaterSign(Jb, j) * SlaterSign(Ib, i) * SlaterSign(Ia, j);
            matrix_element -= sign;
        }
    }
    return (matrix_element);
}

/**
 * @brief Compute the cotribution of a determinant to the diagonal density matrix (number operator)
 * @param Da Diagonal alpha density matrix
 * @param Db Diagonal beta density matrix
 * @param w Weight of this contribution
 */
void StringDeterminant::diag_opdm(std::vector<double>& Da, std::vector<double>& Db, double w) {
    bool* Ia = alfa_bits_;
    bool* Ib = beta_bits_;
    for (int n = 0; n < nmo_; ++n) {
        if (Ia[n])
            Da[n] += w;
        if (Ib[n])
            Db[n] += w;
    }
}

/**
 * Compute the matrix element of the Hamiltonian between this determinant and a given one
 * @param rhs
 * @return
 */
double StringDeterminant::SlaterRules(const std::vector<bool>& Ia, const std::vector<bool>& Ib,
                                      const std::vector<bool>& Ja, const std::vector<bool>& Jb) {
    double matrix_element = 0.0;

    int nadiff = 0;
    int nbdiff = 0;

    int nmo = static_cast<int>(Ia.size());
    // Count how many differences in mos are there
    for (int n = 0; n < nmo; ++n) {
        if (Ia[n] != Ja[n])
            nadiff++;
        if (Ib[n] != Jb[n])
            nbdiff++;
        if (nadiff + nbdiff > 4)
            return 0.0; // Get our of this as soon as possible
    }
    nadiff /= 2;
    nbdiff /= 2;

    // Slater rule 1 PhiI = PhiJ
    if ((nadiff == 0) and (nbdiff == 0)) {
        matrix_element = ints_->frozen_core_energy() + ints_->scalar();
        for (int p = 0; p < nmo; ++p) {
            if (Ia[p])
                matrix_element += ints_->oei_a(p, p);
            if (Ib[p])
                matrix_element += ints_->oei_b(p, p);
            for (int q = 0; q < nmo; ++q) {
                if (Ia[p] and Ia[q]) {
                    matrix_element += 0.5 * ints_->diag_aptei_aa(p, q);
                }
                if (Ib[p] and Ib[q]) {
                    matrix_element += 0.5 * ints_->diag_aptei_bb(p, q);
                }
                if (Ia[p] and Ib[q]) {
                    matrix_element += ints_->diag_aptei_ab(p, q);
                }
            }
        }
    }

    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 0)) {
        // Diagonal contribution
        int i = 0;
        int j = 0;
        for (int p = 0; p < nmo; ++p) {
            if ((Ia[p] != Ja[p]) and Ia[p])
                i = p;
            if ((Ia[p] != Ja[p]) and Ja[p])
                j = p;
        }
        double sign = SlaterSign(Ia, i) * SlaterSign(Ja, j);
        matrix_element = sign * ints_->oei_a(i, j);
        for (int p = 0; p < nmo; ++p) {
            if (Ia[p] and Ja[p]) {
                matrix_element += sign * ints_->aptei_aa(i, p, j, p);
            }
            if (Ib[p] and Jb[p]) {
                matrix_element += sign * ints_->aptei_ab(i, p, j, p);
            }
        }
    }
    // Slater rule 2 PhiI = j_b^+ i_b PhiJ
    if ((nadiff == 0) and (nbdiff == 1)) {
        // Diagonal contribution
        int i = 0;
        int j = 0;
        for (int p = 0; p < nmo; ++p) {
            if ((Ib[p] != Jb[p]) and Ib[p])
                i = p;
            if ((Ib[p] != Jb[p]) and Jb[p])
                j = p;
        }
        double sign = SlaterSign(Ib, i) * SlaterSign(Jb, j);
        matrix_element = sign * ints_->oei_b(i, j);
        for (int p = 0; p < nmo; ++p) {
            if (Ia[p] and Ja[p]) {
                matrix_element += sign * ints_->aptei_ab(p, i, p, j);
            }
            if (Ib[p] and Jb[p]) {
                matrix_element += sign * ints_->aptei_bb(i, p, j, p);
            }
        }
    }

    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
    if ((nadiff == 2) and (nbdiff == 0)) {
        // Diagonal contribution
        int i = -1;
        int j = 0;
        int k = -1;
        int l = 0;
        for (int p = 0; p < nmo; ++p) {
            if ((Ia[p] != Ja[p]) and Ia[p]) {
                if (i == -1) {
                    i = p;
                } else {
                    j = p;
                }
            }
            if ((Ia[p] != Ja[p]) and Ja[p]) {
                if (k == -1) {
                    k = p;
                } else {
                    l = p;
                }
            }
        }
        double sign = SlaterSign(Ia, i) * SlaterSign(Ia, j) * SlaterSign(Ja, k) * SlaterSign(Ja, l);
        //        matrix_element = sign * (ints_->rtei(i,k,j,l) - ints_->rtei(i,l,j,k));
        matrix_element = sign * ints_->aptei_aa(i, j, k, l);
    }

    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
    if ((nadiff == 0) and (nbdiff == 2)) {
        // Diagonal contribution
        int i, j, k, l;
        i = -1;
        j = -1;
        k = -1;
        l = -1;
        for (int p = 0; p < nmo; ++p) {
            if ((Ib[p] != Jb[p]) and Ib[p]) {
                if (i == -1) {
                    i = p;
                } else {
                    j = p;
                }
            }
            if ((Ib[p] != Jb[p]) and Jb[p]) {
                if (k == -1) {
                    k = p;
                } else {
                    l = p;
                }
            }
        }
        double sign = SlaterSign(Ib, i) * SlaterSign(Ib, j) * SlaterSign(Jb, k) * SlaterSign(Jb, l);
        matrix_element = sign * ints_->aptei_bb(i, j, k, l);
    }

    // Slater rule 3 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Diagonal contribution
        int i, j, k, l;
        i = j = k = l = -1;
        for (int p = 0; p < nmo; ++p) {
            if ((Ia[p] != Ja[p]) and Ia[p])
                i = p;
            if ((Ib[p] != Jb[p]) and Ib[p])
                j = p;
            if ((Ia[p] != Ja[p]) and Ja[p])
                k = p;
            if ((Ib[p] != Jb[p]) and Jb[p])
                l = p;
        }
        double sign = SlaterSign(Ia, i) * SlaterSign(Ib, j) * SlaterSign(Ja, k) * SlaterSign(Jb, l);
        matrix_element = sign * ints_->aptei_ab(i, j, k, l);
    }
    return (matrix_element);
}

/**
 * Compute the S^2 matrix element of the Hamiltonian between two determinants specified by the
 * strings (Ia,Ib) and (Ja,Jb)
 * @return S^2
 */
double StringDeterminant::Spin2(const std::vector<bool>& Ia, const std::vector<bool>& Ib,
                                const std::vector<bool>& Ja, const std::vector<bool>& Jb) {
    // Compute the matrix elements of the operator S^2
    // S^2 = S- S+ + Sz (Sz + 1)
    //     = Sz (Sz + 1) + Nbeta + Npairs - sum_pq' a+(qa) a+(pb) a-(qb) a-(pa)
    double matrix_element = 0.0;

    int nadiff = 0;
    int nbdiff = 0;
    int na = 0;
    int nb = 0;
    int npair = 0;
    int nmo = static_cast<int>(Ia.size());
    // Count how many differences in mos are there and the number of alpha/beta electrons
    for (int n = 0; n < nmo; ++n) {
        if (Ia[n] != Ja[n])
            nadiff++;
        if (Ib[n] != Jb[n])
            nbdiff++;
        if (Ia[n])
            na++;
        if (Ib[n])
            nb++;
        if ((Ia[n] and Ib[n]))
            npair += 1;
    }
    nadiff /= 2;
    nbdiff /= 2;

    double Ms = 0.5 * static_cast<double>(na - nb);

    // PhiI = PhiJ -> S^2 = Sz (Sz + 1) + Nbeta + Npairs
    if ((nadiff == 0) and (nbdiff == 0)) {
        matrix_element += Ms * (Ms + 1.0) + double(nb) - double(npair);
    }

    // PhiI = a+(qa) a+(pb) a-(qb) a-(pa) PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Find a pair of spin coupled electrons
        int i = -1;
        int j = -1;
        // The logic here is a bit complex
        for (int p = 0; p < nmo; ++p) {
            if (Ja[p] and Ib[p] and (not Jb[p]) and (not Ia[p]))
                i = p; //(p)
            if (Jb[p] and Ia[p] and (not Ja[p]) and (not Ib[p]))
                j = p; //(q)
        }
        if (i != j and i >= 0 and j >= 0) {
            double sign =
                SlaterSign(Ja, i) * SlaterSign(Jb, j) * SlaterSign(Ib, i) * SlaterSign(Ia, j);
            matrix_element -= sign;
        }
    }
    return (matrix_element);
}

void StringDeterminant::SlaterOPDM(const std::vector<bool>& Ia, const std::vector<bool>& Ib,
                                   const std::vector<bool>& Ja, const std::vector<bool>& Jb,
                                   SharedMatrix Da, SharedMatrix Db, double w) {
    int nmo = static_cast<int>(Ia.size());

    int nadiff = 0;
    int nbdiff = 0;
    // Count how many differences in mos are there and the number of alpha/beta electrons
    for (int n = 0; n < nmo; ++n) {
        if (Ia[n] != Ja[n])
            nadiff++;
        if (Ib[n] != Jb[n])
            nbdiff++;
    }

    nadiff /= 2;
    nbdiff /= 2;

    // Case 1: PhiI = PhiJ
    if ((nadiff == 0) and (nbdiff == 0)) {
        for (int n = 0; n < nmo; ++n) {
            if (Ia[n])
                Da->add(n, n, w);
            if (Ib[n])
                Db->add(n, n, w);
        }
    }
    // Case 2: PhiI = a+ i PhiJ
    int i, a;
    if ((nadiff == 1) and (nbdiff == 0)) {
        for (int n = 0; n < nmo; ++n) {
            if ((Ia[n] != Ja[n]) and Ja[n])
                i = n;
            if ((Ia[n] != Ja[n]) and Ia[n])
                a = n;
        }
        Da->add(i, a, w * SlaterSign(Ia, a) * SlaterSign(Ja, i));
    }
    if ((nadiff == 0) and (nbdiff == 1)) {
        for (int n = 0; n < nmo; ++n) {
            if ((Ib[n] != Jb[n]) and Jb[n])
                i = n;
            if ((Ib[n] != Jb[n]) and Ib[n])
                a = n;
        }
        Db->add(i, a, w * SlaterSign(Ib, a) * SlaterSign(Jb, i));
    }
}

void StringDeterminant::SlaterdiagOPDM(const std::vector<bool>& Ia, const std::vector<bool>& Ib,
                                       std::vector<double>& Da, std::vector<double>& Db, double w) {
    // Case 1: PhiI = PhiJ
    int nmo = static_cast<int>(Ia.size());
    for (int n = 0; n < nmo; ++n) {
        if (Ia[n])
            Da[n] += w;
        if (Ib[n])
            Db[n] += w;
    }
}

// double SlaterRules(StringDeterminant& PhiI, StringDeterminant& PhiJ,boost::shared_ptr<Integrals>&
// ints)
//{
//    double matrix_element = 0.0;
//    bool* Ia = PhiI.get_alfa_bits();
//    bool* Ib = PhiI.get_beta_bits();
//    bool* Ja = PhiJ.get_alfa_bits();
//    bool* Jb = PhiJ.get_beta_bits();

//    int nmo = PhiI.nmo();

//    int nadiff = 0;
//    int nbdiff = 0;

//    // Count how many differences in mos are there
//    for (int n = 0; n < nmo; ++n) {
//        if (Ia[n] != Ja[n]) nadiff++;
//        if (Ib[n] != Jb[n]) nbdiff++;
//    }
//    nadiff /= 2;
//    nbdiff /= 2;

//    // Slater rule 1 PhiI = PhiJ
//    if ((nadiff == 0) and (nbdiff == 0)) {
//        // Diagonal contribution
//        matrix_element = ints->frozen_core_energy();
//        for(int p = 0; p < nmo; ++p){
//            if(Ia[p]) matrix_element += ints->get_oei_aa_all(p,p);
//            if(Ib[p]) matrix_element += ints->get_oei_bb_all(p,p);
//            for(int q = 0; q < nmo; ++q){
//                if(Ia[p] and Ia[q])
//                    matrix_element +=   0.5 * ints->get_tei_aaaa_all(p,p,q,q)
//                            - 0.5 * ints->get_tei_aaaa_all(p,q,p,q);
//                if(Ib[p] and Ib[q])
//                    matrix_element +=   0.5 * ints->get_tei_bbbb_all(p,p,q,q)
//                            - 0.5 * ints->get_tei_bbbb_all(p,q,p,q);
//                if(Ia[p] and Ib[q])
//                    matrix_element += ints->get_tei_aabb_all(p,p,q,q);
//            }
//        }
//    }

//    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
//    if ((nadiff == 1) and (nbdiff == 0)) {
//        // Diagonal contribution
//        int i = 0;
//        int j = 0;
//        for(int p = 0; p < nmo; ++p){
//            if((Ia[p] != Ja[p]) and Ia[p]) i = p;
//            if((Ia[p] != Ja[p]) and Ja[p]) j = p;
//        }
//        double sign = SlaterSign(Ia,i) * SlaterSign(Ja,j);
//        matrix_element = sign * ints->get_oei_aa(i,j);
//        for(int p = 0; p < nmo; ++p){
//            if(Ia[p] and Ja[p]){
//                matrix_element += sign * (ints->get_tei_aaaa_all(i,j,p,p) -
//                ints->get_tei_aaaa_all(i,p,p,j));
//            }
//            if(Ib[p] and Jb[p]){
//                matrix_element += sign * ints->get_tei_aabb_all(i,j,p,p);
//            }
//        }
//    }
//    // Slater rule 2 PhiI = j_b^+ i_b PhiJ
//    if ((nadiff == 0) and (nbdiff == 1)) {
//        // Diagonal contribution
//        int i = 0;
//        int j = 0;
//        for(int p = 0; p < nmo; ++p){
//            if((Ib[p] != Jb[p]) and Ib[p]) i = p;
//            if((Ib[p] != Jb[p]) and Jb[p]) j = p;
//        }
//        double sign = SlaterSign(Ib,i) * SlaterSign(Jb,j);
//        matrix_element = sign * ints->get_oei_bb_all(i,j);
//        for(int p = 0; p < nmo; ++p){
//            if(Ia[p] and Ja[p]){
//                matrix_element += sign * ints->get_tei_aabb_all(p,p,i,j);
//            }
//            if(Ib[p] and Jb[p]){
//                matrix_element += sign * (ints->get_tei_bbbb_all(i,j,p,p) -
//                ints->get_tei_bbbb_all(i,p,p,j));
//            }
//        }
//    }

//    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
//    if ((nadiff == 2) and (nbdiff == 0)) {
//        // Diagonal contribution
//        int i = -1;
//        int j =  0;
//        int k = -1;
//        int l =  0;
//        for(int p = 0; p < nmo; ++p){
//            if((Ia[p] != Ja[p]) and Ia[p]){
//                if (i == -1) { i = p; } else { j = p; }
//            }
//            if((Ia[p] != Ja[p]) and Ja[p]){
//                if (k == -1) { k = p; } else { l = p; }
//            }
//        }
//        double sign = SlaterSign(Ia,i) * SlaterSign(Ia,j) * SlaterSign(Ja,k) * SlaterSign(Ja,l);
//        matrix_element = sign * (ints->get_tei_aaaa_all(i,k,j,l) -
//        ints->get_tei_aaaa_all(i,l,j,k));
//    }

//    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
//    if ((nadiff == 0) and (nbdiff == 2)) {
//        // Diagonal contribution
//        int i,j,k,l;
//        i = -1;
//        j = -1;
//        k = -1;
//        l = -1;
//        for(int p = 0; p < nmo; ++p){
//            if((Ib[p] != Jb[p]) and Ib[p]){
//                if (i == -1) { i = p; } else { j = p; }
//            }
//            if((Ib[p] != Jb[p]) and Jb[p]){
//                if (k == -1) { k = p; } else { l = p; }
//            }
//        }
//        double sign = SlaterSign(Ib,i) * SlaterSign(Ib,j) * SlaterSign(Jb,k) * SlaterSign(Jb,l);
//        matrix_element = sign * (ints->get_tei_bbbb_all(i,k,j,l) -
//        ints->get_tei_bbbb_all(i,l,j,k));
//    }

//    // Slater rule 3 PhiI = j_a^+ i_a PhiJ
//    if ((nadiff == 1) and (nbdiff == 1)) {
//        // Diagonal contribution
//        int i,j,k,l;
//        i = j = k = l = -1;
//        for(int p = 0; p < nmo; ++p){
//            if((Ia[p] != Ja[p]) and Ia[p]) i = p;
//            if((Ib[p] != Jb[p]) and Ib[p]) j = p;
//            if((Ia[p] != Ja[p]) and Ja[p]) k = p;
//            if((Ib[p] != Jb[p]) and Jb[p]) l = p;
//        }
//        double sign = SlaterSign(Ia,i) * SlaterSign(Ib,j) * SlaterSign(Ja,k) * SlaterSign(Jb,l);
//        matrix_element = sign * ints->get_tei_aabb_all(i,k,j,l);
//    }
//    return(matrix_element);
//}

double SlaterSign(bool* I, int n) {
    double sign = 1.0;
    for (int i = 0; i < n; ++i) { // This runs up to the operator before n
        if (I[i])
            sign *= -1.0;
    }
    return (sign);
}

double SlaterSign(const std::vector<bool>& I, int n) {
    double sign = 1.0;
    for (int i = 0; i < n; ++i) { // This runs up to the operator before n
        if (I[i])
            sign *= -1.0;
    }
    return (sign);
}

// double SlaterAnnihilate(bool* I,int n)
//{
//    if (I[n]) {
//        I[n] = false;
//        double sign = 1.0;
//        for(int i = 0; i < n; ++i){  // This runs up to the operator before n
//            if(I[i]) sign *= -1.0;
//        }
//        return(sign);
//    }
//    return 0.0;
//}

// double SlaterOPDM(StringDeterminant& PhiI, StringDeterminant& PhiJ,int p,bool spin_p,int q,bool
// spin_q)
//{
//    double sign = 1.0;
//    bool* Ia = PhiI.get_alfa_bits();
//    bool* Ib = PhiI.get_beta_bits();
//    bool* Ja = PhiJ.get_alfa_bits();
//    bool* Jb = PhiJ.get_beta_bits();

//    int n = PhiI.nmo();

//    if(spin_p){
//        sign *= SlaterAnnihilate(Ia,p);
//    }else{
//        sign *= SlaterSign(Ia,n); // Count the alpha part as well
//        sign *= SlaterAnnihilate(Ib,p);
//    }

//    if(spin_q){
//        sign *= SlaterAnnihilate(Ja,q);
//    }else{
//        sign *= SlaterSign(Ja,n); // Count the alpha part as well
//        sign *= SlaterAnnihilate(Jb,q);
//    }

//    if((Ia == Ja) and (Ib == Jb))
//        return(sign);
//    else
//        return 0.0;
//}

// double SlaterTPDM(StringDeterminant& PhiI, StringDeterminant& PhiJ,int p,bool spin_p,int q,bool
// spin_q,int r,bool spin_r,int s,bool spin_s)
//{
//    double sign = 1.0;
//    bool* Ia = PhiI.get_alfa_bits();
//    bool* Ib = PhiI.get_beta_bits();
//    bool* Ja = PhiJ.get_alfa_bits();
//    bool* Jb = PhiJ.get_beta_bits();

//    int n = PhiI.nmo();

//    if(spin_p){
//        sign *= SlaterAnnihilate(Ia,p);
//    }else{
//        sign *= SlaterSign(Ia,n); // Count the alpha part as well
//        sign *= SlaterAnnihilate(Ib,p);
//    }

//    if(spin_q){
//        sign *= SlaterAnnihilate(Ia,q);
//    }else{
//        sign *= SlaterSign(Ia,n); // Count the alpha part as well
//        sign *= SlaterAnnihilate(Ib,q);
//    }

//    if(spin_r){
//        sign *= SlaterAnnihilate(Ja,r);
//    }else{
//        sign *= SlaterSign(Ja,n); // Count the alpha part as well
//        sign *= SlaterAnnihilate(Jb,r);
//    }

//    if(spin_s){
//        sign *= SlaterAnnihilate(Ja,s);
//    }else{
//        sign *= SlaterSign(Ja,n); // Count the alpha part as well
//        sign *= SlaterAnnihilate(Jb,s);
//    }

//    if((Ia == Ja) and (Ib == Jb))
//        return(sign);
//    else
//        return 0.0;
//}

// double Slater3PDM(StringDeterminant& PhiI, StringDeterminant& PhiJ,int p,bool spin_p,int q,bool
// spin_q,int r,bool spin_r,int s,bool spin_s,int t,bool spin_t,int u,bool spin_u)
//{
//    double sign = 1.0;
//    bool* Ia = PhiI.get_alfa_bits();
//    bool* Ib = PhiI.get_beta_bits();
//    bool* Ja = PhiJ.get_alfa_bits();
//    bool* Jb = PhiJ.get_beta_bits();

//    int n = PhiI.nmo();

//    if(spin_p){
//        sign *= SlaterAnnihilate(Ia,p);
//    }else{
//        sign *= SlaterSign(Ia,n); // Count the alpha part as well
//        sign *= SlaterAnnihilate(Ib,p);
//    }

//    if(spin_q){
//        sign *= SlaterAnnihilate(Ia,q);
//    }else{
//        sign *= SlaterSign(Ia,n); // Count the alpha part as well
//        sign *= SlaterAnnihilate(Ib,q);
//    }

//    if(spin_r){
//        sign *= SlaterAnnihilate(Ia,r);
//    }else{
//        sign *= SlaterSign(Ia,n); // Count the alpha part as well
//        sign *= SlaterAnnihilate(Ib,r);
//    }

//    if(spin_s){
//        sign *= SlaterAnnihilate(Ja,s);
//    }else{
//        sign *= SlaterSign(Ja,n); // Count the alpha part as well
//        sign *= SlaterAnnihilate(Jb,s);
//    }

//    if(spin_t){
//        sign *= SlaterAnnihilate(Ja,t);
//    }else{
//        sign *= SlaterSign(Ja,n); // Count the alpha part as well
//        sign *= SlaterAnnihilate(Jb,t);
//    }

//    if(spin_u){
//        sign *= SlaterAnnihilate(Ja,u);
//    }else{
//        sign *= SlaterSign(Ja,n); // Count the alpha part as well
//        sign *= SlaterAnnihilate(Jb,u);
//    }

//    if((Ia == Ja) and (Ib == Jb))
//        return(sign);
//    else
//        return 0.0;
//}
}
} // End Namespaces
