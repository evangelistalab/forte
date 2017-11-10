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

#include <algorithm>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "stl_bitset_determinant.h"

using namespace psi;

#define ALFA(n) bits_[n]
#define BETA(n) bits_[num_str_bits + n]

namespace psi {
namespace forte {

const STLBitsetDeterminant::bit_t STLBitsetDeterminant::alfa_mask =
    bit_t(0xFFFFFFFFFFFFFFFF) |
    (bit_t(0xFFFFFFFFFFFFFFFF) << STLBitsetDeterminant::num_str_bits / 2);
const STLBitsetDeterminant::bit_t STLBitsetDeterminant::beta_mask =
    alfa_mask << STLBitsetDeterminant::num_str_bits;

STLBitsetDeterminant::STLBitsetDeterminant() {}

STLBitsetDeterminant::STLBitsetDeterminant(const bit_t& bits) { bits_ = bits; }

STLBitsetDeterminant::STLBitsetDeterminant(const std::vector<bool>& occupation) {
    int size = occupation.size() / 2;
    for (int p = 0; p < size; ++p)
        ALFA(p) = occupation[p];
    for (int p = 0; p < size; ++p)
        BETA(p) = occupation[size + p];
}

STLBitsetDeterminant::STLBitsetDeterminant(const std::vector<bool>& occupation_a,
                                           const std::vector<bool>& occupation_b) {
    int size = occupation_a.size();
    for (int p = 0; p < size; ++p) {
        ALFA(p) = occupation_a[p];
        BETA(p) = occupation_b[p];
    }
}

const STLBitsetDeterminant::bit_t& STLBitsetDeterminant::bits() const { return bits_; }

bool STLBitsetDeterminant::less_than(const STLBitsetDeterminant& rhs,
                                     const STLBitsetDeterminant& lhs) {
    // check beta first
    for (int i = num_det_bits - 1; i >= num_str_bits; i--) {
        if (rhs.bits_[i] ^ lhs.bits_[i])
            return lhs.bits_[i];
    }
    for (int i = num_str_bits - 1; i >= 0; i--) {
        if (rhs.bits_[i] ^ lhs.bits_[i])
            return lhs.bits_[i];
    }
    return false;
}

bool STLBitsetDeterminant::reverse_less_then(const STLBitsetDeterminant& rhs,
                                             const STLBitsetDeterminant& lhs) {
    // check alpha first
    for (int i = num_str_bits - 1; i >= 0; i--) {
        if (rhs.bits_[i] ^ lhs.bits_[i])
            return lhs.bits_[i];
    }
    for (int i = num_det_bits - 1; i >= num_str_bits; i--) {
        if (rhs.bits_[i] ^ lhs.bits_[i])
            return lhs.bits_[i];
    }
    return false;
}

bool STLBitsetDeterminant::operator==(const STLBitsetDeterminant& lhs) const {
    return (bits_ == lhs.bits_);
}

bool STLBitsetDeterminant::operator<(const STLBitsetDeterminant& lhs) const {
    // check beta first
    for (int i = num_det_bits - 1; i >= num_str_bits; i--) {
        if (bits_[i] ^ lhs.bits_[i])
            return lhs.bits_[i];
    }
    for (int i = num_str_bits - 1; i >= 0; i--) {
        if (bits_[i] ^ lhs.bits_[i])
            return lhs.bits_[i];
    }
    return false;
}

STLBitsetDeterminant& STLBitsetDeterminant::flip() {
    bits_.flip();
    return *this;
}

int STLBitsetDeterminant::count_alfa() const { return (bits_ & alfa_mask).count(); }

int STLBitsetDeterminant::count_beta() const { return (bits_ & beta_mask).count(); }

bool STLBitsetDeterminant::get_alfa_bit(int n) const { return ALFA(n); }

bool STLBitsetDeterminant::get_beta_bit(int n) const { return BETA(n); }

void STLBitsetDeterminant::set_alfa_bit(int n, bool value) { ALFA(n) = value; }

void STLBitsetDeterminant::set_beta_bit(int n, bool value) { BETA(n) = value; }

// std::vector<int> STLBitsetDeterminant::get_alfa_occ() {
//    std::vector<int> occ;
//    for (int p = 0; p < num_str_bits; ++p) {
//        if (ALFA(p))
//            occ.push_back(p);
//    }
//    return occ;
//}

// std::vector<int> STLBitsetDeterminant::get_beta_occ() {
//    std::vector<int> occ;
//    for (int p = 0; p < num_str_bits; ++p) {
//        if (BETA(p))
//            occ.push_back(p);
//    }
//    return occ;
//}

// std::vector<int> STLBitsetDeterminant::get_alfa_vir() {
//    std::vector<int> vir;
//    for (int p = 0; p < num_str_bits; ++p) {
//        if (not ALFA(p))
//            vir.push_back(p);
//    }
//    return vir;
//}

// std::vector<int> STLBitsetDeterminant::get_beta_vir() {
//    std::vector<int> vir;
//    for (int p = 0; p < num_str_bits; ++p) {
//        if (not BETA(p))
//            vir.push_back(p);
//    }
//    return vir;
//}

std::vector<int> STLBitsetDeterminant::get_alfa_occ(int norb) const {
    std::vector<int> orbs;
    for (int p = 0; p < norb; ++p) {
        if (ALFA(p)) {
            orbs.push_back(p);
        }
    }
    return orbs;
}

std::vector<int> STLBitsetDeterminant::get_beta_occ(int norb) const {
    std::vector<int> orbs;
    for (int p = 0; p < norb; ++p) {
        if (BETA(p)) {
            orbs.push_back(p);
        }
    }
    return orbs;
}

std::vector<int> STLBitsetDeterminant::get_alfa_vir(int norb) const {
    std::vector<int> orbs;
    for (int p = 0; p < norb; ++p) {
        if (not ALFA(p)) {
            orbs.push_back(p);
        }
    }
    return orbs;
}

std::vector<int> STLBitsetDeterminant::get_beta_vir(int norb) const {
    std::vector<int> orbs;
    for (int p = 0; p < norb; ++p) {
        if (not BETA(p)) {
            orbs.push_back(p);
        }
    }
    return orbs;
}

double STLBitsetDeterminant::create_alfa_bit(int n) {
    if (ALFA(n))
        return 0.0;
    ALFA(n) = true;
    return slater_sign_a(n);
}

double STLBitsetDeterminant::create_beta_bit(int n) {
    if (BETA(n))
        return 0.0;
    BETA(n) = true;
    return slater_sign_b(n);
}

double STLBitsetDeterminant::destroy_alfa_bit(int n) {
    if (not ALFA(n))
        return 0.0;
    ALFA(n) = false;
    return slater_sign_a(n);
}

/// Set the value of a beta bit
double STLBitsetDeterminant::destroy_beta_bit(int n) {
    if (not BETA(n))
        return 0.0;
    BETA(n) = false;
    return slater_sign_b(n);
}

/// Return determinant with one spin annihilated, 0 == alpha
void STLBitsetDeterminant::zero_spin(DetSpinType spin_type) {
    if (spin_type == DetSpinType::Alpha) {
        bits_ &= beta_mask;
        //        for (int p = 0; p < size_; ++p) {
        //            ALFA(p) = false;
        //        }
    } else {
        bits_ &= alfa_mask;
        //        for (int p = 0; p < size_; ++p) {
        //            BETA(p) = false;
        //        }
    }
}

std::string STLBitsetDeterminant::str(int n) const {
    std::string s;
    s += "|";
    for (int p = 0; p < n; ++p) {
        if (ALFA(p) and BETA(p)) {
            s += "2";
        } else if (ALFA(p) and not BETA(p)) {
            s += "+";
        } else if (not ALFA(p) and BETA(p)) {
            s += "-";
        } else {
            s += "0";
        }
    }
    s += ">";
    return s;
}

double STLBitsetDeterminant::slater_sign_a(int n) const {
    double sign = 1.0;
    for (int i = 0; i < n; ++i) { // This runs up to the operator before n
        if (ALFA(i))
            sign *= -1.0;
    }
    return (sign);
}

double STLBitsetDeterminant::slater_sign_aa(int n, int m) const {
    double sign = 1.0;
    for (int i = m + 1; i < n; ++i) { // This runs up to the operator before n
        if (ALFA(i))
            sign *= -1.0;
    }
    for (int i = n + 1; i < m; ++i) {
        if (ALFA(i)) {
            sign *= -1.0;
        }
    }
    return (sign);
}

double STLBitsetDeterminant::slater_sign_b(int n) const {
    double sign = 1.0;
    for (int i = 0; i < n; ++i) { // This runs up to the operator before n
        if (BETA(i))
            sign *= -1.0;
    }
    return (sign);
}

double STLBitsetDeterminant::slater_sign_bb(int n, int m) const {
    double sign = 1.0;
    for (int i = m + 1; i < n; ++i) { // This runs up to the operator before n
        if (BETA(i))
            sign *= -1.0;
    }
    for (int i = n + 1; i < m; ++i) {
        if (BETA(i)) {
            sign *= -1.0;
        }
    }
    return (sign);
}

double STLBitsetDeterminant::slater_sign_aaaa(int i, int j, int a, int b) const {
    if ((((i < a) && (j < a) && (i < b) && (j < b)) == true) ||
        (((i < a) || (j < a) || (i < b) || (j < b)) == false)) {
        if ((i < j) ^ (a < b)) {
            return (-1.0 * slater_sign_aa(i, j) * slater_sign_aa(a, b));
        } else {
            return (slater_sign_aa(i, j) * slater_sign_aa(a, b));
        }
    } else {
        if ((i < j) ^ (a < b)) {
            return (-1.0 * slater_sign_aa(i, b) * slater_sign_aa(j, a));
        } else {
            return (slater_sign_aa(i, a) * slater_sign_aa(j, b));
        }
    }
}

double STLBitsetDeterminant::slater_sign_bbbb(int i, int j, int a, int b) const {
    if ((((i < a) && (j < a) && (i < b) && (j < b)) == true) ||
        (((i < a) || (j < a) || (i < b) || (j < b)) == false)) {
        if ((i < j) ^ (a < b)) {
            return (-1.0 * slater_sign_bb(i, j) * slater_sign_bb(a, b));
        } else {
            return (slater_sign_bb(i, j) * slater_sign_bb(a, b));
        }
    } else {
        if ((i < j) ^ (a < b)) {
            return (-1.0 * slater_sign_bb(i, b) * slater_sign_bb(j, a));
        } else {
            return (slater_sign_bb(i, a) * slater_sign_bb(j, b));
        }
    }
}

double STLBitsetDeterminant::single_excitation_a(int i, int a) {
    ALFA(i) = false;
    ALFA(a) = true;
    return slater_sign_aa(i, a);
}

double STLBitsetDeterminant::single_excitation_b(int i, int a) {
    BETA(i) = false;
    BETA(a) = true;
    return slater_sign_bb(i, a);
}

double STLBitsetDeterminant::double_excitation_aa(int i, int j, int a, int b) {
    ALFA(i) = false;
    ALFA(j) = false;
    ALFA(b) = true;
    ALFA(a) = true;
    return slater_sign_aaaa(i, j, a, b);
}

double STLBitsetDeterminant::double_excitation_ab(int i, int j, int a, int b) {
    ALFA(i) = false;
    BETA(j) = false;
    BETA(b) = true;
    ALFA(a) = true;
    return slater_sign_aa(i, a) * slater_sign_bb(j, b);
}

double STLBitsetDeterminant::double_excitation_bb(int i, int j, int a, int b) {
    BETA(i) = false;
    BETA(j) = false;
    BETA(b) = true;
    BETA(a) = true;
    return slater_sign_bbbb(i, j, a, b);
}

std::vector<std::pair<STLBitsetDeterminant, double>> STLBitsetDeterminant::spin_plus() const {
    std::vector<std::pair<STLBitsetDeterminant, double>> res;
    for (int i = 0; i < num_str_bits; ++i) {
        if ((not ALFA(i)) and BETA(i)) {
            double sign = slater_sign_a(i) * slater_sign_b(i);
            STLBitsetDeterminant new_det(*this);
            new_det.set_alfa_bit(i, true);
            new_det.set_beta_bit(i, false);
            res.push_back(std::make_pair(new_det, sign));
        }
    }
    return res;
}

std::vector<std::pair<STLBitsetDeterminant, double>> STLBitsetDeterminant::spin_minus() const {
    std::vector<std::pair<STLBitsetDeterminant, double>> res;
    for (int i = 0; i < num_str_bits; ++i) {
        if (ALFA(i) and (not BETA(i))) {
            double sign = slater_sign_a(i) * slater_sign_b(i);
            STLBitsetDeterminant new_det(*this);
            new_det.set_alfa_bit(i, false);
            new_det.set_beta_bit(i, true);
            res.push_back(std::make_pair(new_det, sign));
        }
    }
    return res;
}

double STLBitsetDeterminant::spin_z() const {
    return 0.5 * static_cast<double>(count_alfa() - count_beta());
}

int STLBitsetDeterminant::npair() {
    int npair = 0;
    for (int n = 0; n < num_str_bits; ++n) {
        if (ALFA(n) and BETA(n)) {
            npair++;
        }
    }
    return npair;
}

STLBitsetDeterminant common_occupation(const STLBitsetDeterminant& lhs,
                                       const STLBitsetDeterminant& rhs) {
    STLBitsetDeterminant::bit_t bits = rhs.bits() & lhs.bits();
    return STLBitsetDeterminant(bits);
}

STLBitsetDeterminant different_occupation(const STLBitsetDeterminant& lhs,
                                          const STLBitsetDeterminant& rhs) {
    STLBitsetDeterminant::bit_t bits = rhs.bits() ^ lhs.bits();
    return STLBitsetDeterminant(bits);
}

STLBitsetDeterminant union_occupation(const STLBitsetDeterminant& lhs,
                                      const STLBitsetDeterminant& rhs) {
    STLBitsetDeterminant::bit_t bits = rhs.bits() | lhs.bits();
    return STLBitsetDeterminant(bits);
}

double spin2(const STLBitsetDeterminant& lhs, const STLBitsetDeterminant& rhs) {
    int num_str_bits = STLBitsetDeterminant::num_str_bits;
    int size = num_str_bits;
    const STLBitsetDeterminant::bit_t& I = lhs.bits();
    const STLBitsetDeterminant::bit_t& J = rhs.bits();

    // Compute the matrix elements of the operator S^2
    // S^2 = S- S+ + Sz (Sz + 1)
    //     = Sz (Sz + 1) + Nbeta + Npairs - sum_pq' a+(qa) a+(pb) a-(qb) a-(pa)
    double matrix_element = 0.0;

    int nadiff = 0;
    int nbdiff = 0;
    int na = 0;
    int nb = 0;
    int npair = 0;
    // Count how many differences in mos are there and the number of alpha/beta
    // electrons
    for (int n = 0; n < size; ++n) {
        if (I[n] != J[n])
            nadiff++;
        if (I[num_str_bits + n] != J[num_str_bits + n])
            nbdiff++;
        if (I[n])
            na++;
        if (I[num_str_bits + n])
            nb++;
        if ((I[n] and I[num_str_bits + n]))
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
        for (int p = 0; p < size; ++p) {
            if (J[p] and I[num_str_bits + p] and (not J[num_str_bits + p]) and (not I[p]))
                i = p;
            if (J[num_str_bits + p] and I[p] and (not J[p]) and (not I[num_str_bits + p]))
                j = p;
        }
        if (i != j and i >= 0 and j >= 0) {
            double sign = rhs.slater_sign_a(i) * rhs.slater_sign_b(j) * lhs.slater_sign_a(j) *
                          lhs.slater_sign_b(i);
            matrix_element -= sign;
        }
    }
    return (matrix_element);
}

void enforce_spin_completeness(std::vector<STLBitsetDeterminant>& det_space, int nmo) {
    std::unordered_map<STLBitsetDeterminant, bool, STLBitsetDeterminant::Hash> det_map;
    // Add all determinants to the map, assume set is mostly spin complete
    for (auto& I : det_space) {
        det_map[I] = true;
    }
    // Loop over determinants
    size_t ndet_added = 0;
    std::vector<size_t> closed(nmo, 0);
    std::vector<size_t> open(nmo, 0);
    std::vector<size_t> open_bits(nmo, 0);
    for (size_t I = 0, det_size = det_space.size(); I < det_size; ++I) {
        const STLBitsetDeterminant& det = det_space[I];
        // outfile->Printf("\n  Original determinant: %s", det.str().c_str());
        for (int i = 0; i < nmo; ++i) {
            closed[i] = open[i] = 0;
            open_bits[i] = false;
        }
        int naopen = 0;
        int nbopen = 0;
        int nclosed = 0;
        for (int i = 0; i < nmo; ++i) {
            if (det.get_alfa_bit(i) and (not det.get_beta_bit(i))) {
                open[naopen + nbopen] = i;
                naopen += 1;
            } else if ((not det.get_alfa_bit(i)) and det.get_beta_bit(i)) {
                open[naopen + nbopen] = i;
                nbopen += 1;
            } else if (det.get_alfa_bit(i) and det.get_beta_bit(i)) {
                closed[nclosed] = i;
                nclosed += 1;
            }
        }

        if (naopen + nbopen == 0)
            continue;

        // Generate the strings 1111100000
        //                      {nao}{nbo}
        for (int i = 0; i < nbopen; ++i)
            open_bits[i] = false; // 0
        for (int i = nbopen; i < naopen + nbopen; ++i)
            open_bits[i] = true; // 1
        do {
            STLBitsetDeterminant new_det;
            for (int c = 0; c < nclosed; ++c) {
                new_det.set_alfa_bit(closed[c], true);
                new_det.set_beta_bit(closed[c], true);
            }
            for (int o = 0; o < naopen + nbopen; ++o) {
                if (open_bits[o]) { //? not
                    new_det.set_alfa_bit(open[o], true);
                } else {
                    new_det.set_beta_bit(open[o], true);
                }
            }
            if (det_map.count(new_det) == 0) {
                det_space.push_back(new_det);
                det_map[new_det] = true;
                // outfile->Printf("\n  added determinant:    %s", new_det.str().c_str());
                ndet_added++;
            }
        } while (std::next_permutation(open_bits.begin(), open_bits.begin() + naopen + nbopen));
    }
    // if( ndet_added > 0 ){
    //    outfile->Printf("\n\n  Determinant space is spin incomplete!");
    //    outfile->Printf("\n  %zu more determinants were needed.", ndet_added);
    //}else{
    //    outfile->Printf("\n\n  Determinant space is spin complete.");
    //}
}
}
} // end namespace
