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

#include "psi4/libmints/matrix.h"
#include "psi4/psi4-dec.h"

#include "dynamic_bitset_determinant.h"
#include "fci/fci_vector.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

std::size_t hash_value(const DynamicBitsetDeterminant& input) {
    return (input.alfa_bits_.to_ulong() % 100000 + input.beta_bits_.to_ulong() % 100000);
}

// Static members
std::vector<DynamicBitsetDeterminant::bit_t> DynamicBitsetDeterminant::bit_mask_;
std::shared_ptr<FCIIntegrals> DynamicBitsetDeterminant::fci_ints_;

void DynamicBitsetDeterminant::set_ints(std::shared_ptr<FCIIntegrals> ints) {
    fci_ints_ = ints;

    // Initialize the bit masks
    int n = ints->nmo();

    bit_mask_.clear();
    for (int i = 0; i < n; ++i) {
        bit_t b(n);
        for (int j = 0; j < i; ++j) {
            b[j] = 1;
        }
        bit_mask_.push_back(b);
    }
}

void DynamicBitsetDeterminant::reset_ints() { fci_ints_ = nullptr; }

DynamicBitsetDeterminant::DynamicBitsetDeterminant() : nmo_(0) {}

DynamicBitsetDeterminant::DynamicBitsetDeterminant(int nmo)
    : nmo_(nmo), alfa_bits_(nmo_), beta_bits_(nmo_) {}

/// Construct the determinant from an occupation vector that
/// specifies the alpha and beta strings.  occupation = [Ia,Ib]
DynamicBitsetDeterminant::DynamicBitsetDeterminant(const std::vector<int>& occupation,
                                                   bool print_det)
    : nmo_(occupation.size() / 2), alfa_bits_(nmo_), beta_bits_(nmo_) {
    for (int p = 0; p < nmo_; ++p) {
        alfa_bits_[p] = occupation[p];
        beta_bits_[p] = occupation[p + nmo_];
    }
    if (print_det)
        print();
}

/// Construct the determinant from an occupation vector that
/// specifies the alpha and beta strings.  occupation = [Ia,Ib]
DynamicBitsetDeterminant::DynamicBitsetDeterminant(const std::vector<bool>& occupation,
                                                   bool print_det)
    : nmo_(occupation.size() / 2), alfa_bits_(nmo_), beta_bits_(nmo_) {
    for (int p = 0; p < nmo_; ++p) {
        alfa_bits_[p] = occupation[p];
        beta_bits_[p] = occupation[p + nmo_];
    }
    if (print_det)
        print();
}

DynamicBitsetDeterminant::DynamicBitsetDeterminant(const std::vector<bool>& occupation_a,
                                                   const std::vector<bool>& occupation_b,
                                                   bool print_det)
    : nmo_(occupation_a.size()), alfa_bits_(nmo_), beta_bits_(nmo_) {
    for (int p = 0; p < nmo_; ++p) {
        alfa_bits_[p] = occupation_a[p];
        beta_bits_[p] = occupation_b[p];
    }
    if (print_det)
        print();
}

std::vector<int> DynamicBitsetDeterminant::get_alfa_occ() {
    std::vector<int> occ(alfa_bits_.count());
    size_t index = alfa_bits_.find_first();
    int i = 0;
    while (index != boost::dynamic_bitset<>::npos) {
        occ[i] = index;
        index = alfa_bits_.find_next(index);
        i++;
    }
    return occ;
}

std::vector<int> DynamicBitsetDeterminant::get_beta_occ() {
    std::vector<int> occ(beta_bits_.count());
    size_t index = beta_bits_.find_first();
    int i = 0;
    while (index != boost::dynamic_bitset<>::npos) {
        occ[i] = index;
        index = beta_bits_.find_next(index);
        i++;
    }
    return occ;
}

std::vector<int> DynamicBitsetDeterminant::get_alfa_vir() {
    alfa_bits_.flip();
    std::vector<int> vir(alfa_bits_.count());
    size_t index = alfa_bits_.find_first();
    int i = 0;
    while (index != boost::dynamic_bitset<>::npos) {
        vir[i] = index;
        index = alfa_bits_.find_next(index);
        i++;
    }
    alfa_bits_.flip();
    return vir;
}

std::vector<int> DynamicBitsetDeterminant::get_beta_vir() {
    beta_bits_.flip();
    std::vector<int> vir(beta_bits_.count());
    size_t index = beta_bits_.find_first();
    int i = 0;
    while (index != boost::dynamic_bitset<>::npos) {
        vir[i] = index;
        index = beta_bits_.find_next(index);
        i++;
    }
    beta_bits_.flip();
    return vir;
}

std::vector<int> DynamicBitsetDeterminant::get_alfa_occ() const {
    std::vector<int> occ(alfa_bits_.count());
    size_t index = alfa_bits_.find_first();
    int i = 0;
    while (index != boost::dynamic_bitset<>::npos) {
        occ[i] = index;
        index = alfa_bits_.find_next(index);
        i++;
    }
    return occ;
}

std::vector<int> DynamicBitsetDeterminant::get_beta_occ() const {
    std::vector<int> occ(beta_bits_.count());
    size_t index = beta_bits_.find_first();
    int i = 0;
    while (index != boost::dynamic_bitset<>::npos) {
        occ[i] = index;
        index = beta_bits_.find_next(index);
        i++;
    }
    return occ;
}

std::vector<int> DynamicBitsetDeterminant::get_alfa_vir() const {
    boost::dynamic_bitset<> alfa_bits(alfa_bits_);
    alfa_bits.flip();
    std::vector<int> vir(alfa_bits.count());
    size_t index = alfa_bits.find_first();
    int i = 0;
    while (index != boost::dynamic_bitset<>::npos) {
        vir[i] = index;
        index = alfa_bits.find_next(index);
        i++;
    }
    return vir;
}

std::vector<int> DynamicBitsetDeterminant::get_beta_vir() const {
    boost::dynamic_bitset<> beta_bits(beta_bits_);
    beta_bits.flip();
    std::vector<int> vir(beta_bits.count());
    size_t index = beta_bits.find_first();
    int i = 0;
    while (index != boost::dynamic_bitset<>::npos) {
        vir[i] = index;
        index = beta_bits.find_next(index);
        i++;
    }
    return vir;
}

double DynamicBitsetDeterminant::create_alfa_bit(int n) {
    if (alfa_bits_[n])
        return 0.0;
    alfa_bits_[n] = true;
    return SlaterSign(alfa_bits_, n);
}

double DynamicBitsetDeterminant::create_beta_bit(int n) {
    if (beta_bits_[n])
        return 0.0;
    beta_bits_[n] = true;
    return SlaterSign(beta_bits_, n);
}

double DynamicBitsetDeterminant::destroy_alfa_bit(int n) {
    if (not alfa_bits_[n])
        return 0.0;
    alfa_bits_[n] = false;
    return SlaterSign(alfa_bits_, n);
}

/// Set the value of a beta bit
double DynamicBitsetDeterminant::destroy_beta_bit(int n) {
    if (not beta_bits_[n])
        return 0.0;
    beta_bits_[n] = false;
    return SlaterSign(beta_bits_, n);
}

/// Switch alfa and beta bits
void DynamicBitsetDeterminant::spin_flip() { std::swap(alfa_bits_, beta_bits_); }

/**
 * Print the determinant
 */
void DynamicBitsetDeterminant::print() const {
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
 * Print the determinant
 */
std::string DynamicBitsetDeterminant::str() const {
    std::string s;
    s += "|";
    for (int p = 0; p < nmo_; ++p) {
        s += alfa_bits_[p] ? "1" : "0";
    }
    s += "|";
    for (int p = 0; p < nmo_; ++p) {
        s += beta_bits_[p] ? "1" : "0";
    }
    s += ">";
    return s;
}

/**
 * Compute the energy of this determinant
 * @return the electronic energy (does not include the nuclear repulsion energy)
 */
double DynamicBitsetDeterminant::energy() const {
    double matrix_element = fci_ints_->frozen_core_energy();
    for (int p = 0; p < nmo_; ++p) {
        if (alfa_bits_[p]) {
            matrix_element += fci_ints_->oei_a(p, p);
            for (int q = p + 1; q < nmo_; ++q) {
                if (alfa_bits_[q]) {
                    matrix_element += fci_ints_->diag_tei_aa(p, q);
                }
            }
            for (int q = 0; q < nmo_; ++q) {
                if (beta_bits_[q]) {
                    matrix_element += fci_ints_->diag_tei_ab(p, q);
                }
            }
        }
        if (beta_bits_[p]) {
            matrix_element += fci_ints_->oei_b(p, p);
            for (int q = p + 1; q < nmo_; ++q) {
                if (beta_bits_[q]) {
                    matrix_element += fci_ints_->diag_tei_bb(p, q);
                }
            }
        }
    }
    return (matrix_element);
}

/**
 * Compute the matrix element of the Hamiltonian between this determinant and a
 * given one
 * @param rhs
 * @return
 */
double DynamicBitsetDeterminant::slater_rules(const DynamicBitsetDeterminant& rhs) const {
    const boost::dynamic_bitset<>& Ia = alfa_bits_;
    const boost::dynamic_bitset<>& Ib = beta_bits_;
    const boost::dynamic_bitset<>& Ja = rhs.alfa_bits_;
    const boost::dynamic_bitset<>& Jb = rhs.beta_bits_;

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
        matrix_element = fci_ints_->frozen_core_energy();
        for (int p = 0; p < nmo_; ++p) {
            if (alfa_bits_[p])
                matrix_element += fci_ints_->oei_a(p, p);
            if (beta_bits_[p])
                matrix_element += fci_ints_->oei_b(p, p);
            for (int q = 0; q < nmo_; ++q) {
                if (alfa_bits_[p] and alfa_bits_[q])
                    matrix_element += 0.5 * fci_ints_->diag_tei_aa(p, q);
                //                    matrix_element +=   0.5 *
                //                    ints_->diag_ce_rtei(p,q);
                if (beta_bits_[p] and beta_bits_[q])
                    matrix_element += 0.5 * fci_ints_->diag_tei_bb(p, q);
                //                    matrix_element +=   0.5 *
                //                    ints_->diag_ce_rtei(p,q);
                if (alfa_bits_[p] and beta_bits_[q])
                    matrix_element += fci_ints_->diag_tei_ab(p, q);
                //                    matrix_element +=
                //                    fci_ints_->diag_c_rtei(p,q);
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
        matrix_element = sign * fci_ints_->oei_a(i, j);
        for (int p = 0; p < nmo_; ++p) {
            if (Ia[p] and Ja[p]) {
                matrix_element += sign * fci_ints_->tei_aa(i, p, j, p);
            }
            if (Ib[p] and Jb[p]) {
                matrix_element += sign * fci_ints_->tei_ab(i, p, j, p);
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
        matrix_element = sign * fci_ints_->oei_b(i, j);
        for (int p = 0; p < nmo_; ++p) {
            if (Ia[p] and Ja[p]) {
                matrix_element += sign * fci_ints_->tei_ab(p, i, p, j);
            }
            if (Ib[p] and Jb[p]) {
                matrix_element += sign * fci_ints_->tei_bb(i, p, j, p);
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
        matrix_element = sign * fci_ints_->tei_aa(i, j, k, l);
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
        matrix_element = sign * fci_ints_->tei_bb(i, j, k, l);
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
        matrix_element = sign * fci_ints_->tei_ab(i, j, k, l);
    }
    return (matrix_element);
}

double DynamicBitsetDeterminant::slater_rules_single_alpha(int i, int a) const {
    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    double sign = SlaterSign(alfa_bits_, i) * SlaterSign(alfa_bits_, a) * (a > i ? -1.0 : 1.0);
    double matrix_element = fci_ints_->oei_a(i, a);
    for (int p = 0; p < nmo_; ++p) {
        if (alfa_bits_[p]) {
            matrix_element += fci_ints_->tei_aa(i, p, a, p);
        }
        if (beta_bits_[p]) {
            matrix_element += fci_ints_->tei_ab(i, p, a, p);
        }
    }
    return sign * matrix_element;
}

double DynamicBitsetDeterminant::slater_rules_single_beta(int i, int a) const {
    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    double sign = SlaterSign(beta_bits_, i) * SlaterSign(beta_bits_, a) * (a > i ? -1.0 : 1.0);
    double matrix_element = fci_ints_->oei_b(i, a);
    for (int p = 0; p < nmo_; ++p) {
        if (alfa_bits_[p]) {
            matrix_element += fci_ints_->tei_ab(p, i, p, a);
        }
        if (beta_bits_[p]) {
            matrix_element += fci_ints_->tei_bb(i, p, a, p);
        }
    }
    return sign * matrix_element;
}

/**
 * Compute the S^2 matrix element of the Hamiltonian between two determinants
 * specified by the strings (Ia,Ib) and (Ja,Jb)
 * @return S^2
 */
double DynamicBitsetDeterminant::spin2(const DynamicBitsetDeterminant& rhs) const {
    const boost::dynamic_bitset<>& Ia = alfa_bits_;
    const boost::dynamic_bitset<>& Ib = beta_bits_;
    const boost::dynamic_bitset<>& Ja = rhs.alfa_bits_;
    const boost::dynamic_bitset<>& Jb = rhs.beta_bits_;

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
    // Count how many differences in mos are there and the number of alpha/beta
    // electrons
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

    // PhiI = PhiJ -> S^2 = Sz (Sz + 1) + Nbeta - Npairs
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

std::vector<std::pair<DynamicBitsetDeterminant, double>>
DynamicBitsetDeterminant::spin_plus() const {
    std::vector<std::pair<DynamicBitsetDeterminant, double>> res;
    for (int i = 0; i < nmo_; ++i) {
        if ((not alfa_bits_[i]) and beta_bits_[i]) {
            double sign = SlaterSign(alfa_bits_, i) * SlaterSign(beta_bits_, i);
            DynamicBitsetDeterminant new_det(*this);
            new_det.set_alfa_bit(i, true);
            new_det.set_beta_bit(i, false);
            res.push_back(std::make_pair(new_det, sign));
        }
    }
    return res;
}

std::vector<std::pair<DynamicBitsetDeterminant, double>>
DynamicBitsetDeterminant::spin_minus() const {
    std::vector<std::pair<DynamicBitsetDeterminant, double>> res;
    for (int i = 0; i < nmo_; ++i) {
        if (alfa_bits_[i] and (not beta_bits_[i])) {
            double sign = SlaterSign(alfa_bits_, i) * SlaterSign(beta_bits_, i);
            DynamicBitsetDeterminant new_det(*this);
            new_det.set_alfa_bit(i, false);
            new_det.set_beta_bit(i, true);
            res.push_back(std::make_pair(new_det, sign));
        }
    }
    return res;
}

double DynamicBitsetDeterminant::spin_z() const {
    return 0.5 * static_cast<double>(alfa_bits_.count() - beta_bits_.count());
}

double DynamicBitsetDeterminant::spin2_slow(const DynamicBitsetDeterminant& rhs) const {
    double s2 = 0.0;
    if (rhs == *this) {
        double sz = spin_z();
        s2 += sz * (sz + 1.0);
    }
    return s2;
}

double DynamicBitsetDeterminant::slater_sign_alpha(int n) const {
    double sign = 1.0;
    for (int i = 0; i < n; ++i) { // This runs up to the operator before n
        if (alfa_bits_[i])
            sign *= -1.0;
    }
    return (sign);
}

double DynamicBitsetDeterminant::slater_sign_beta(int n) const {
    double sign = 1.0;
    for (int i = 0; i < n; ++i) { // This runs up to the operator before n
        if (beta_bits_[i])
            sign *= -1.0;
    }
    return (sign);
}

double DynamicBitsetDeterminant::double_excitation_aa(int i, int j, int a, int b) {
    double sign = 1.0;
    sign *= slater_sign_alpha(i);
    sign *= slater_sign_alpha(j);
    alfa_bits_[i] = false;
    alfa_bits_[j] = false;
    alfa_bits_[a] = true;
    alfa_bits_[b] = true;
    sign *= slater_sign_alpha(a);
    sign *= slater_sign_alpha(b);
    return sign;
}

double DynamicBitsetDeterminant::double_excitation_ab(int i, int j, int a, int b) {
    double sign = 1.0;
    sign *= slater_sign_alpha(i);
    sign *= slater_sign_beta(j);
    alfa_bits_[i] = false;
    beta_bits_[j] = false;
    alfa_bits_[a] = true;
    beta_bits_[b] = true;
    sign *= slater_sign_alpha(a);
    sign *= slater_sign_beta(b);
    return sign;
}

double DynamicBitsetDeterminant::double_excitation_bb(int i, int j, int a, int b) {
    double sign = 1.0;
    sign *= slater_sign_beta(i);
    sign *= slater_sign_beta(j);
    beta_bits_[i] = false;
    beta_bits_[j] = false;
    beta_bits_[a] = true;
    beta_bits_[b] = true;
    sign *= slater_sign_beta(a);
    sign *= slater_sign_beta(b);
    return sign;
}

double DynamicBitsetDeterminant::SlaterSign(const boost::dynamic_bitset<>& I, int n) {
    double sign = 1.0;
    for (int i = 0; i < n; ++i) { // This runs up to the operator before n
        if (I[i])
            sign *= -1.0;
    }
    return (sign);
}

double DynamicBitsetDeterminant::FastSlaterSign(const boost::dynamic_bitset<>& I, int n) {
    return ((bit_mask_[n] & I).count() % 2) ? -1.0 : 1.0;
}

void DynamicBitsetDeterminant::check_uniqueness(
    const std::vector<DynamicBitsetDeterminant> det_space) {
    size_t duplicates = 0;
    size_t dim = det_space.size();
    std::unordered_map<DynamicBitsetDeterminant, size_t, function<decltype(hash_value)>> det_map(
        dim, hash_value);

    for (const auto& i : det_space) {
        ++det_map[i];
    }
    for (const auto& d : det_map) {
        if (d.second > 1) {
            outfile->Printf("\n  Duplicate determinant! ==> %s", d.first.str().c_str());
            duplicates += d.second;
        }
    }
    outfile->Printf("\n  Number of duplicate determinants:  %zu  ", duplicates);
}
}
} // end namespace
