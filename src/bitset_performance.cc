///*
// * @BEGIN LICENSE
// *
// * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
// * that implements a variety of quantum chemistry methods for strongly
// * correlated electrons.
// *
// * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
// *
// * The copyrights for code used from other parties are included in
// * the corresponding files.
// *
// * This program is free software: you can redistribute it and/or modify
// * it under the terms of the GNU Lesser General Public License as published by
// * the Free Software Foundation, either version 3 of the License, or
// * (at your option) any later version.
// *
// * This program is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// * GNU Lesser General Public License for more details.
// *
// * You should have received a copy of the GNU Lesser General Public License
// * along with this program.  If not, see http://www.gnu.org/licenses/.
// *
// * @END LICENSE
// */

//#include <chrono>

//#include "psi4/psi4-dec.h"
//#include "psi4/libmints/matrix.h"

//#include "mini-boost/boost/lexical_cast.hpp"

//#include "sparse_ci/determinant.h"
//#include "fci/fci_vector.h"

// using namespace psi;

// namespace psi {
// namespace forte {

// template <size_t sz> struct bitset_comparer {
//    bool operator()(const bitset<sz>& b1, const bitset<sz>& b2) const {
//        return b1.to_ulong() < b2.to_ulong();
//    }
//};

// DynamicBitsetDeterminant random_bitset_determinant(int nmo, int nset) {
//    DynamicBitsetDeterminant I(nmo);
//    for (int k = 0; k < nset; ++k) {
//        int a = rand() % nmo;
//        int b = rand() % nmo;
//        I.set_alfa_bit(a, true);
//        I.set_beta_bit(b, true);
//    }
//    return I;
//}

// Determinant random_bitset_determinant2(int nmo, int nset) {
//    Determinant I;
//    for (int k = 0; k < nset; ++k) {
//        I.set_alfa_bit(rand() % nmo, true);
//        I.set_beta_bit(rand() % nmo, true);
//    }
//    return I;
//}

// boost::dynamic_bitset<> random_dynamic_bitset(int nmo, int nset) {
//    boost::dynamic_bitset<> I(2 * nmo);
//    for (int k = 0; k < nset; ++k) {
//        I[rand() % nmo] = true;
//        I[nmo + rand() % nmo] = true;
//    }
//    return I;
//}

// std::bitset<64> random_bitset_64(int nmo, int nset) {
//    std::bitset<64> I(2 * nmo);
//    for (int k = 0; k < nset; ++k) {
//        I[rand() % nmo] = true;
//        I[nmo + rand() % nmo] = true;
//    }
//    return I;
//}

// std::bitset<128> random_bitset_128(int nmo, int nset) {
//    std::bitset<128> I(2 * nmo);
//    for (int k = 0; k < nset; ++k) {
//        I[rand() % nmo] = true;
//        I[nmo + rand() % nmo] = true;
//    }
//    return I;
//}

// bit_t random_bitset_256(int nmo, int nset) {
//    bit_t I(2 * nmo);
//    for (int k = 0; k < nset; ++k) {
//        I[rand() % nmo] = true;
//        I[nmo + rand() % nmo] = true;
//    }
//    return I;
//}

// double test_bitset_determinant_copy_constructor(int nmo, int nset, int repeat) {
//    DynamicBitsetDeterminant I = random_bitset_determinant(nmo, nset);
//    I.print();
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        DynamicBitsetDeterminant J(I);
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_determinant2_copy_constructor(int nmo, int nset,
//                                                 int repeat) {
//    Determinant I = random_bitset_determinant2(nmo, nset);
//    I.print();
//    auto start = chrono::steady_clock::now();
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_dynamic_bitset_copy_constructor(int nmo, int nset, int repeat) {
//    boost::dynamic_bitset<> I = random_dynamic_bitset(nmo, nset);
//    // Converting dynamic_bitset to a string
//    string buffer;
//    to_string(I, buffer);
//    outfile->Printf("\n dynamic_bitset: %s", buffer.c_str());
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        boost::dynamic_bitset<> J(I);
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_64_copy_constructor(int nmo, int nset, int repeat) {
//    std::bitset<64> I = random_bitset_64(nmo, nset);
//    // Converting dynamic_bitset to a string
//    //    string buffer;
//    //    to_string(I, buffer);
//    //    outfile->Printf("\n dynamic_bitset: %s",buffer.c_str());
//    auto start = chrono::steady_clock::now();
//    size_t sum = 0;
//    for (int n = 0; n < repeat; ++n) {
//        std::bitset<64> J(I);
//        sum += J[0];
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_determinant_place_in_map(int nmo, int nset, int repeat) {
//    std::vector<DynamicBitsetDeterminant> det_vec;
//    std::map<DynamicBitsetDeterminant, double> det_map;
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(random_bitset_determinant(nmo, nset));
//    }
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_map[det_vec[n]] = 1.1;
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_determinant2_place_in_map(int nmo, int nset, int repeat) {
//    //    std::vector<Determinant> det_vec;
//    //    std::map<Determinant,double> det_map;
//    //    for (int n = 0; n < repeat; ++n){
//    //        det_vec.push_back(random_bitset_determinant2(nmo,nset));
//    //    }
//    auto start = chrono::steady_clock::now();
//    //    for (int n = 0; n < repeat; ++n){
//    //        det_map[det_vec[n]] = 1.1;
//    //    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_dynamic_bitset_place_in_map(int nmo, int nset, int repeat) {
//    std::vector<boost::dynamic_bitset<>> det_vec;
//    std::map<boost::dynamic_bitset<>, double> det_map;
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(random_dynamic_bitset(nmo, nset));
//    }
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_map[det_vec[n]] = 1.1;
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_64_place_in_map(int nmo, int nset, int repeat) {
//    std::vector<std::bitset<64>> det_vec;
//    std::map<std::bitset<64>, double, bitset_comparer<64>> det_map;
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(random_bitset_64(nmo, nset));
//    }
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_map[det_vec[n]] = 1.1;
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_128_place_in_map(int nmo, int nset, int repeat) {
//    std::vector<std::bitset<128>> det_vec;
//    std::map<std::bitset<128>, double, bitset_comparer<128>> det_map;
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(random_bitset_128(nmo, nset));
//    }
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_map[det_vec[n]] = 1.1;
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_size_t_place_in_map(int nmo, int nset, int repeat) {
//    std::vector<size_t> det_vec;
//    std::map<size_t, double> det_map;
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(rand());
//    }
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_map[det_vec[n]] = 1.1;
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test___int128_place_in_map(int nmo, int nset, int repeat) {
//    std::vector<__int128> det_vec;
//    std::map<__int128, double> det_map;
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(rand());
//    }
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_map[det_vec[n]] = 1.1;
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_determinant_place_in_hash(int nmo, int nset, int repeat) {
//    std::vector<DynamicBitsetDeterminant> det_vec;
//    std::unordered_map<DynamicBitsetDeterminant, double,
//                       DynamicBitsetDeterminant::Hash> det_hash;
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(random_bitset_determinant(nmo, nset));
//    }
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_hash[det_vec[n]] = 1.1;
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_determinant2_place_in_hash(int nmo, int nset, int repeat) {
//    std::vector<Determinant> det_vec;
//    std::unordered_map<Determinant, double, Determinant::Hash>
//        det_hash;
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(random_bitset_determinant2(nmo, nset));
//    }
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_hash[det_vec[n]] = 1.1;
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_64_place_in_hash(int nmo, int nset, int repeat) {
//    std::vector<std::bitset<64>> det_vec;
//    std::unordered_map<std::bitset<64>, double> det_hash;
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(random_bitset_64(nmo, nset));
//    }
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_hash[det_vec[n]] = 1.1;
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_128_place_in_hash(int nmo, int nset, int repeat) {
//    std::vector<std::bitset<128>> det_vec;
//    std::unordered_map<std::bitset<128>, double> det_hash;
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(random_bitset_128(nmo, nset));
//    }
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_hash[det_vec[n]] = 1.1;
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_256_place_in_hash(int nmo, int nset, int repeat) {
//    std::vector<bit_t> det_vec;
//    std::unordered_map<bit_t, double> det_hash;
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(random_bitset_256(nmo, nset));
//    }
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_hash[det_vec[n]] = 1.1;
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_determinant_place_in_vector(int nmo, int nset, int repeat) {
//    std::vector<DynamicBitsetDeterminant> det_vec;
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(random_bitset_determinant(nmo, nset));
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_determinant2_place_in_vector(int nmo, int nset, int repeat) {
//    std::vector<Determinant> det_vec;
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(random_bitset_determinant2(nmo, nset));
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_256_place_in_vector(int nmo, int nset, int repeat) {
//    std::vector<bit_t> det_vec;
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        det_vec.push_back(random_bitset_256(nmo, nset));
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_determinant_generate(int nmo, int nset, int repeat) {
//    std::vector<DynamicBitsetDeterminant> det_vec;
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        random_bitset_determinant(nmo, nset);
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// double test_bitset_determinant2_generate(int nmo, int nset, int repeat) {
//    std::vector<Determinant> det_vec;
//    auto start = chrono::steady_clock::now();
//    for (int n = 0; n < repeat; ++n) {
//        random_bitset_determinant2(nmo, nset);
//    }
//    auto end = chrono::steady_clock::now();
//    auto diff = end - start;
//    return chrono::duration<double, nano>(diff).count();
//}

// std::vector<std::pair<std::string, double>> run_tests(int nmo, int nset,
//                                                      int nrepeat) {
//    std::vector<std::pair<std::string, double>> test_results;

//    outfile->Printf("\n\n\n  ==> nmo: %d, nset: %d, nrepeat: %d <==\n", nmo,
//                    nset, nrepeat);

//    // Test generation
//    test_results.push_back(
//        std::make_pair("DynamicBitsetDeterminant generate",
//                       test_bitset_determinant_generate(nmo, nset, nrepeat)));
//    test_results.push_back(
//        std::make_pair("Determinant generate",
//                       test_bitset_determinant2_generate(nmo, nset, nrepeat)));

//    test_results.push_back(std::make_pair(
//        "DynamicBitsetDeterminant place in map",
//        test_bitset_determinant_place_in_map(nmo, nset, nrepeat)));

//    // Test placing in hash
//    test_results.push_back(std::make_pair(
//        "DynamicBitsetDeterminant place in hash",
//        test_bitset_determinant_place_in_hash(nmo, nset, nrepeat)));
//    if (nmo <= 256) {
//        test_results.push_back(std::make_pair(
//            "Determinant place in hash",
//            test_bitset_determinant2_place_in_hash(nmo, nset, nrepeat)));
//    }
//    if (nmo <= 64) {
//        test_results.push_back(
//            std::make_pair("bitset_128 place in hash",
//                           test_bitset_128_place_in_hash(nmo, nset, nrepeat)));
//    }
//    if (nmo <= 128) {
//        test_results.push_back(
//            std::make_pair("bitset_256 place in hash",
//                           test_bitset_256_place_in_hash(nmo, nset, nrepeat)));
//    }

//    // Test placing in vector
//    test_results.push_back(std::make_pair(
//        "DynamicBitsetDeterminant place in vector",
//        test_bitset_determinant_place_in_vector(nmo, nset, nrepeat)));
//    test_results.push_back(std::make_pair(
//        "Determinant place in vector",
//        test_bitset_determinant2_place_in_vector(nmo, nset, nrepeat)));
//    test_results.push_back(
//        std::make_pair("bitset_256 place in vector",
//                       test_bitset_256_place_in_vector(nmo, nset, nrepeat)));

//    for (auto kv : test_results) {
//        double value = kv.second / nrepeat;
//        int digits = std::floor(std::log10(value));
//        digits = std::max(digits, 0);
//        std::string filler = string(100 - kv.first.size() - digits, '.');
//        outfile->Printf("\n%-s%s%.3f ns/operation", kv.first.c_str(),
//                        filler.c_str(), value);
//    }

//    return test_results;
//}

// void test_bitset_performance() {
//    outfile->Printf("\nTesting performance of bitset");

//    int nrepeat = 1000000;

//    for (int nmo : {8, 16, 32, 64}) {
//        for (double pop : {0.1, 0.25, 0.5}) {
//            int nset = std::ceil(double(nmo) * pop);
//            run_tests(nmo, nset, nrepeat);
//        }
//    }
//}
//}
//} // end namespace
