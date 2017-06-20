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

#include "mini-boost/boost/lexical_cast.hpp"

#include "fast_determinant.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

inline bool test_bit01(bit_t a, bit_t i)
// Return whether bit[i] is set.
{
    return (0 != (a & (bit_t(1) << i)));
}

inline bit_t bit_count(bit_t x)
// Return number of bits set
{
    x = (0x5555555555555555UL & x) + (0x5555555555555555UL & (x >> 1));
    x = (0x3333333333333333UL & x) + (0x3333333333333333UL & (x >> 2));
    x = (0x0f0f0f0f0f0f0f0fUL & x) + (0x0f0f0f0f0f0f0f0fUL & (x >> 4));
    x = (0x00ff00ff00ff00ffUL & x) + (0x00ff00ff00ff00ffUL & (x >> 8));
    x = (0x0000ffff0000ffffUL & x) + (0x0000ffff0000ffffUL & (x >> 16));
    x = (0x00000000ffffffffUL & x) + (0x00000000ffffffffUL & (x >> 32));
    return x;
    //    x = ((x>>1) & 0x5555555555555555UL) + (x & 0x5555555555555555UL);
    //    x = ((x>>2) & 0x3333333333333333UL) + (x & 0x3333333333333333UL);
    //    x = ((x>>4) + x) & 0x0f0f0f0f0f0f0f0fUL;
    //    x+=x>>8;
    //    x += x>>16;
    //    x += x>>32;
    //    return x & 0xff;

    //    x -= (x>>1) & 0x5555555555555555UL;
    //    x = ((x>>2) & 0x3333333333333333UL) + (x & 0x3333333333333333UL); // 0-4 in 4 bits
    //    x = ((x>>4) + x) & 0x0f0f0f0f0f0f0f0fUL; // 0-8 in 8 bits
    //    x *= 0x0101010101010101UL;
    //    return x>>56;
}

inline bit_t bit_count_sparse(bit_t x) {
    bit_t n = 0;
    while (x) {
        ++n;
        x &= (x - 1);
    }
    return n;
}

inline bit_t lowest_one_idx(bit_t x) {
    if (1 >= x)
        return x - 1; // 0 if 1, ~0 if 0
    bit_t r = 0;
    x &= -x; // isolate lowest bit
    if (x & 0xffffffff00000000UL)
        r += 32;
    if (x & 0xffff0000ffff0000UL)
        r += 16;
    if (x & 0xff00ff00ff00ff00UL)
        r += 8;
    if (x & 0xf0f0f0f0f0f0f0f0UL)
        r += 4;
    if (x & 0xccccccccccccccccUL)
        r += 2;
    if (x & 0xaaaaaaaaaaaaaaaaUL)
        r += 1;
    return r;
}

inline bit_t clear_lowest_one(bit_t x)
// Return word where the lowest bit set in x is cleared
// Return 0 for input == 0
{
    return x & (x - 1);
}

inline std::vector<int> get_set(bit_t x, bit_t range) {
    uint64_t mask = (bit_t(1) << range) - bit_t(1);
    x = x & mask;
    std::vector<int> r(bit_count(x));
    bit_t index = lowest_one_idx(x);
    int i = 0;
    while (index != -1) {
        r[i] = index;
        x = clear_lowest_one(x);
        index = lowest_one_idx(x);
        i++;
    }
    return r;
}

std::shared_ptr<ForteIntegrals> FastDeterminant::ints_ = 0;

/// Construct the determinant from an occupation vector that
/// specifies the alpha and beta strings.  occupation = [Ia,Ib]
FastDeterminant::FastDeterminant(const std::vector<int>& occupation, bool print_det)
    : nmo_(occupation.size() / 2), alfa_bits_(0UL), beta_bits_(0UL) {
    for (int p = 0; p < nmo_; ++p) {
        set_alfa_bit(p, occupation[p]);
        set_beta_bit(p, occupation[p + nmo_]);
    }
    if (print_det)
        print();
}

/// Construct the determinant from an occupation vector that
/// specifies the alpha and beta strings.  occupation = [Ia,Ib]
FastDeterminant::FastDeterminant(const std::vector<bool>& occupation, bool print_det)
    : nmo_(occupation.size() / 2), alfa_bits_(0UL), beta_bits_(0UL) {
    for (int p = 0; p < nmo_; ++p) {
        set_alfa_bit(p, occupation[p]);
        set_beta_bit(p, occupation[p + nmo_]);
    }
    if (print_det)
        print();
}

FastDeterminant::FastDeterminant(const std::vector<bool>& occupation_a,
                                 const std::vector<bool>& occupation_b, bool print_det)
    : nmo_(occupation_a.size()), alfa_bits_(0UL), beta_bits_(0UL) {
    for (int p = 0; p < nmo_; ++p) {
        set_alfa_bit(p, occupation_a[p]);
        set_beta_bit(p, occupation_b[p]);
    }
    if (print_det)
        print();
}

std::vector<int> FastDeterminant::get_alfa_occ() { return get_set(alfa_bits_, nmo_); }

std::vector<int> FastDeterminant::get_beta_occ() { return get_set(beta_bits_, nmo_); }

std::vector<int> FastDeterminant::get_alfa_vir() { return get_set(~alfa_bits_, nmo_); }

std::vector<int> FastDeterminant::get_beta_vir() { return get_set(~beta_bits_, nmo_); }

std::vector<int> FastDeterminant::get_alfa_occ() const { return get_set(alfa_bits_, nmo_); }

std::vector<int> FastDeterminant::get_beta_occ() const { return get_set(beta_bits_, nmo_); }

std::vector<int> FastDeterminant::get_alfa_vir() const { return get_set(~alfa_bits_, nmo_); }

std::vector<int> FastDeterminant::get_beta_vir() const { return get_set(~beta_bits_, nmo_); }

/**
 * Print the determinant
 */
void FastDeterminant::print() const {
    outfile->Printf("\n  |");
    for (int p = 0; p < nmo_; ++p) {
        outfile->Printf("%d", get_alfa_bit(p) ? 1 : 0);
    }
    outfile->Printf("|");
    for (int p = 0; p < nmo_; ++p) {
        outfile->Printf("%d", get_beta_bit(p) ? 1 : 0);
    }
    outfile->Printf(">");
    
}

/**
 * Print the determinant
 */
std::string FastDeterminant::str() const {
    std::string s;
    s += "|";
    for (int p = 0; p < nmo_; ++p) {
        s += boost::lexical_cast<std::string>(get_alfa_bit(p));
    }
    s += "|";
    for (int p = 0; p < nmo_; ++p) {
        s += boost::lexical_cast<std::string>(get_beta_bit(p));
    }
    s += ">";
    return s;
}

/**
 * Compute the energy of this determinant
 * @return the electronic energy (does not include the nuclear repulsion energy)
 */
double FastDeterminant::energy() const {
    double matrix_element = 0.0;
    matrix_element = ints_->frozen_core_energy();
    //    for(int p = 0; p < nmo_; ++p){
    //        if(get_alfa_bit(p)) matrix_element += ints_->diag_roei(p);
    //        if(get_beta_bit(p)) matrix_element += ints_->diag_roei(p);
    //        if(get_alfa_bit(p)) outfile->Printf("\n  One-electron terms: %20.12f + %20.12f
    //        (string)",ints_->diag_roei(p),ints_->diag_roei(p));
    //        for(int q = 0; q < nmo_; ++q){
    //            if(get_alfa_bit(p) and get_alfa_bit(q)){
    //                matrix_element +=   0.5 * ints_->diag_ce_rtei(p,q);
    //                outfile->Printf("\n  One-electron terms (%da,%da): 0.5 * %20.12f
    //                (string)",p,q,ints_->diag_ce_rtei(p,q));
    //            }
    //            if(get_alfa_bit(p) and get_beta_bit(q)){
    //                matrix_element += ints_->diag_c_rtei(p,q);
    //                outfile->Printf("\n  One-electron terms (%da,%db): 1.0 * %20.12f
    //                (string)",p,q,ints_->diag_c_rtei(p,q));
    //            }
    //            if(get_beta_bit(p) and get_beta_bit(q)){
    //                matrix_element +=   0.5 * ints_->diag_ce_rtei(p,q);
    //                outfile->Printf("\n  One-electron terms (%db,%db): 0.5 * %20.12f
    //                (string)",p,q,ints_->diag_ce_rtei(p,q));
    //            }
    //        }
    //    }
    for (int p = 0; p < nmo_; ++p) {
        if (get_alfa_bit(p))
            matrix_element += ints_->oei_a(p, p);
        if (get_beta_bit(p))
            matrix_element += ints_->oei_b(p, p);
        //        if(get_alfa_bit(p)) outfile->Printf("\n  One-electron terms: %20.12f + %20.12f
        //        (string)",ints_->diag_roei(p),ints_->diag_roei(p));
        for (int q = 0; q < nmo_; ++q) {
            if (get_alfa_bit(p) and get_alfa_bit(q)) {
                matrix_element += 0.5 * ints_->diag_aptei_aa(p, q);
                //                outfile->Printf("\n  One-electron terms (%da,%da): 0.5 * %20.12f
                //                (string)",p,q,ints_->diag_aptei_aa(p,q));
            }
            if (get_alfa_bit(p) and get_beta_bit(q)) {
                matrix_element += ints_->diag_aptei_ab(p, q);
                //                outfile->Printf("\n  One-electron terms (%da,%db): 1.0 * %20.12f
                //                (string)",p,q,ints_->diag_aptei_ab(p,q));
            }
            if (get_beta_bit(p) and get_beta_bit(q)) {
                matrix_element += 0.5 * ints_->diag_aptei_bb(p, q);
                //                outfile->Printf("\n  One-electron terms (%db,%db): 0.5 * %20.12f
                //                (string)",p,q,ints_->diag_aptei_bb(p,q));
            }
        }
    }
    return (matrix_element);
}

/**
 * Compute the matrix element of the Hamiltonian between this determinant and a given one
 * @param rhs
 * @return
 */
double FastDeterminant::slater_rules(const FastDeterminant& rhs) const {
    const bit_t& Ia = alfa_bits_;
    const bit_t& Ib = beta_bits_;
    const bit_t& Ja = rhs.alfa_bits_;
    const bit_t& Jb = rhs.beta_bits_;

    int nadiff = 0;
    int nbdiff = 0;
    // Count how many differences in mos are there
    for (int n = 0; n < nmo_; ++n) {
        if (test_bit01(Ia, n) != test_bit01(Ja, n))
            nadiff++;
        if (test_bit01(Ib, n) != test_bit01(Jb, n))
            nbdiff++;
        if (nadiff + nbdiff > 4)
            return 0.0; // Get out of this as soon as possible
    }
    nadiff /= 2;
    nbdiff /= 2;

    //    this->print();
    //    rhs.print();
    //    outfile->Printf("\n  nadiff = %d  nbdiff = %d",nadiff,nbdiff);

    double matrix_element = 0.0;
    // Slater rule 1 PhiI = PhiJ
    if ((nadiff == 0) and (nbdiff == 0)) {
        matrix_element = ints_->frozen_core_energy();
        for (int p = 0; p < nmo_; ++p) {
            if (get_alfa_bit(p))
                matrix_element += ints_->oei_a(p, p);
            if (get_beta_bit(p))
                matrix_element += ints_->oei_b(p, p);
            for (int q = 0; q < nmo_; ++q) {
                if (get_alfa_bit(p) and get_alfa_bit(q))
                    matrix_element += 0.5 * ints_->diag_aptei_aa(p, q);
                //                    matrix_element +=   0.5 * ints_->diag_ce_rtei(p,q);
                if (get_beta_bit(p) and get_beta_bit(q))
                    matrix_element += 0.5 * ints_->diag_aptei_bb(p, q);
                //                    matrix_element +=   0.5 * ints_->diag_ce_rtei(p,q);
                if (get_alfa_bit(p) and get_beta_bit(q))
                    matrix_element += ints_->diag_aptei_ab(p, q);
                //                    matrix_element += ints_->diag_c_rtei(p,q);
            }
        }
    }

    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 0)) {
        // Diagonal contribution
        int i = 0;
        int j = 0;
        for (int p = 0; p < nmo_; ++p) {
            if ((test_bit01(Ia, p) != test_bit01(Ja, p)) and test_bit01(Ia, p))
                i = p;
            if ((test_bit01(Ia, p) != test_bit01(Ja, p)) and test_bit01(Ja, p))
                j = p;
        }
        double sign = SlaterSign(Ia, i) * SlaterSign(Ja, j);
        matrix_element = sign * ints_->oei_a(i, j);
        for (int p = 0; p < nmo_; ++p) {
            if (test_bit01(Ia, p) and test_bit01(Ja, p)) {
                matrix_element += sign * ints_->aptei_aa(i, p, j, p);
            }
            if (test_bit01(Ib, p) and test_bit01(Jb, p)) {
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
            if ((test_bit01(Ib, p) != test_bit01(Jb, p)) and test_bit01(Ib, p))
                i = p;
            if ((test_bit01(Ib, p) != test_bit01(Jb, p)) and test_bit01(Jb, p))
                j = p;
        }
        double sign = SlaterSign(Ib, i) * SlaterSign(Jb, j);
        matrix_element = sign * ints_->oei_b(i, j);
        for (int p = 0; p < nmo_; ++p) {
            if (test_bit01(Ia, p) and test_bit01(Ja, p)) {
                matrix_element += sign * ints_->aptei_ab(p, i, p, j);
            }
            if (test_bit01(Ib, p) and test_bit01(Jb, p)) {
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
            if ((test_bit01(Ia, p) != test_bit01(Ja, p)) and test_bit01(Ia, p)) {
                if (i == -1) {
                    i = p;
                } else {
                    j = p;
                }
            }
            if ((test_bit01(Ia, p) != test_bit01(Ja, p)) and test_bit01(Ja, p)) {
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
            if ((test_bit01(Ib, p) != test_bit01(Jb, p)) and test_bit01(Ib, p)) {
                if (i == -1) {
                    i = p;
                } else {
                    j = p;
                }
            }
            if ((test_bit01(Ib, p) != test_bit01(Jb, p)) and test_bit01(Jb, p)) {
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
            if ((test_bit01(Ia, p) != test_bit01(Ja, p)) and test_bit01(Ia, p))
                i = p;
            if ((test_bit01(Ib, p) != test_bit01(Jb, p)) and test_bit01(Ib, p))
                j = p;
            if ((test_bit01(Ia, p) != test_bit01(Ja, p)) and test_bit01(Ja, p))
                k = p;
            if ((test_bit01(Ib, p) != test_bit01(Jb, p)) and test_bit01(Jb, p))
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
double FastDeterminant::spin2(const FastDeterminant& rhs) const {
    const bit_t& Ia = alfa_bits_;
    const bit_t& Ib = beta_bits_;
    const bit_t& Ja = rhs.alfa_bits_;
    const bit_t& Jb = rhs.beta_bits_;

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
        if (test_bit01(Ia, n) != test_bit01(Ja, n))
            nadiff++;
        if (test_bit01(Ib, n) != test_bit01(Jb, n))
            nbdiff++;
        if (test_bit01(Ia, n))
            na++;
        if (test_bit01(Ib, n))
            nb++;
        if ((test_bit01(Ia, n) and test_bit01(Ib, n)))
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
            if (test_bit01(Ja, p) and test_bit01(Ib, p) and (not test_bit01(Jb, p)) and
                (not test_bit01(Ia, p)))
                i = p; //(p)
            if (test_bit01(Jb, p) and test_bit01(Ia, p) and (not test_bit01(Ja, p)) and
                (not test_bit01(Ib, p)))
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

double FastDeterminant::SlaterSign(const bit_t I, int n) {
    double sign = 1.0;
    for (int i = 0; i < n; ++i) { // This runs up to the operator before n
        if (test_bit01(I, i))
            sign *= -1.0;
    }
    return (sign);
}
}
} // end namespace
