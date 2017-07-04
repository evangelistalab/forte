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

#include "stl_bitset_string.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

// Static members
int STLBitsetString::nmo_;

void STLBitsetString::set_nmo(int nmo) { nmo_ = nmo; }

STLBitsetString::STLBitsetString() {}

STLBitsetString::STLBitsetString(const std::vector<int>& occupation) {
    for (int p = 0; p < nmo_; ++p)
        bits_[p] = occupation[p];
}

STLBitsetString::STLBitsetString(const std::vector<bool>& occupation) {
    for (int p = 0; p < nmo_; ++p) {
        if (occupation[p]) {
            bits_[p] = 1;
        } else {
            bits_[p] = 0;
        }
    }
}

STLBitsetString::STLBitsetString(const std::bitset<128>& bits) { bits_ = bits; }

bool STLBitsetString::operator==(const STLBitsetString& lhs) const { return (bits_ == lhs.bits_); }

bool STLBitsetString::operator<(const STLBitsetString& lhs) const {
    for (int p = nmo_ - 1; p >= 0; --p) {
        if ((bits_[p] == false) and (lhs.bits_[p] == true))
            return true;
        if ((bits_[p] == true) and (lhs.bits_[p] == false))
            return false;
    }
    return false;
}

STLBitsetString STLBitsetString::operator^(const STLBitsetString& lhs) const {
    STLBitsetString ndet(bits_ ^ lhs.bits());
    return ndet;
}

const std::bitset<128>& STLBitsetString::bits() const { return bits_; }

bool STLBitsetString::get_bit(int n) const { return bits_[n]; }

void STLBitsetString::set_bit(int n, bool value) { bits_[n] = value; }

std::vector<bool> STLBitsetString::get_bits_vector_bool() {
    std::vector<bool> result(nmo_);
    for (int n = 0; n < nmo_; ++n) {
        result[n] = bits_[n];
    }
    return result;
}

std::vector<int> STLBitsetString::get_occ() {
    std::vector<int> occ;
    for (int p = 0; p < nmo_; ++p) {
        if (bits_[p])
            occ.push_back(p);
    }
    return occ;
}

std::vector<int> STLBitsetString::get_vir() {
    std::vector<int> vir;
    for (int p = 0; p < nmo_; ++p) {
        if (not bits_[p])
            vir.push_back(p);
    }
    return vir;
}

void STLBitsetString::print() const {
    outfile->Printf("\n  |");
    for (int p = 0; p < nmo_; ++p) {
        if (bits_[p]) {
            outfile->Printf("1");
        } else {
            outfile->Printf("0");
        }
    }
    outfile->Printf(">");
}

std::string STLBitsetString::str() const {
    std::string s;
    s += "|";
    for (int p = 0; p < nmo_; ++p) {
        if (bits_[p]) {
            s += "1";
        } else {
            s += "0";
        }
    }
    s += ">";
    return s;
}

double STLBitsetString::get_nocc() {
    int nocc = 0;
    for (int p = 0; p < nmo_; ++p) {
        if (bits_[p])
            ++nocc;
    }
    return nocc;
}

double STLBitsetString::SlaterSign(int n) {
    double sign = 1.0;
    for (int i = 0; i < n; ++i) { // This runs up to the operator before n
        if (bits_[i])
            sign *= -1.0;
    }
    return (sign);
}
}
} // end namespace
