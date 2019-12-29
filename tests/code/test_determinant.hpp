#ifndef _test_determinant_
#define _test_determinant_

#include <cstddef>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <functional>

using namespace std;
using namespace forte;

template <size_t N> std::pair<DeterminantImpl<N>, vector<int>> generate_random_determinant() {
    std::random_device rd;
    std::mt19937 mersenne_engine(rd());            // Generates random integers
    std::uniform_int_distribution<int> dist{0, 1}; // Possible choices are [0,1]

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    vector<int> vec(N);
    generate(begin(vec), end(vec), gen);

    DeterminantImpl<N> d;
    for (size_t i = 0; i < N; i++) {
        d.set_bit(i, vec[i]);
    }
    return std::make_pair(d, vec);
}

template <size_t N> std::vector<DeterminantImpl<N>> generate_test_determinants() {
    std::vector<DeterminantImpl<N>> dets;
    size_t nbits = DeterminantImpl<N>::nbits;
    // the "111...1" determinant
    DeterminantImpl<N> d_all_one;
    for (size_t i = 0; i < nbits; i++)
        d_all_one.set_bit(i, 1);
    dets.push_back(d_all_one);

    // the "000...0" determinant
    DeterminantImpl<N> d_all_zero;
    for (size_t i = 0; i < nbits; i++)
        d_all_zero.set_bit(i, 0);
    dets.push_back(d_all_zero);

    // the "1010..10" determinant
    DeterminantImpl<N> d_p1;
    for (size_t i = 0; i < nbits; i++)
        d_p1.set_bit(i, i % 2 + 1);
    dets.push_back(d_p1);

    // the "0101..01" determinant
    DeterminantImpl<N> d_p2;
    for (size_t i = 0; i < nbits; i++)
        d_p1.set_bit(i, i % 2);
    dets.push_back(d_p2);

    // a random determinant
    DeterminantImpl<N> d_rand = generate_random_determinant<N>().first;
    dets.push_back(d_rand);

    return dets;
}

// Test that a BitArray object is initialized to zero
template <size_t N> void test_bitarray_init() {
    auto ba = forte::BitArray<N>();
    for (size_t i = 0; i < N; i++) {
        REQUIRE(ba.get_bit(i) == 0);
    }
    REQUIRE(ba.count() == 0);
    REQUIRE(ba.find_first_one() == ~(unsigned long int)(0));
}

template <size_t N> void test_determinantimpl_init() {
    size_t Nhalf = N / 2;
    forte::DeterminantImpl<N> d;
    for (size_t i = 0; i < N; i++) {
        REQUIRE(d.get_bit(i) == 0);
    }
    REQUIRE(d.count_alfa() == 0);
    REQUIRE(d.count_beta() == 0);
    REQUIRE(d.npair() == 0);

    std::vector<int> aocc = d.get_alfa_occ(Nhalf);
    std::vector<int> bocc = d.get_beta_occ(Nhalf);
    std::vector<int> avir = d.get_alfa_vir(Nhalf);
    std::vector<int> bvir = d.get_beta_vir(Nhalf);
    REQUIRE(aocc.size() == 0);
    REQUIRE(bocc.size() == 0);
    REQUIRE(avir.size() == Nhalf);
    REQUIRE(bvir.size() == Nhalf);

    for (size_t i = 0; i < Nhalf; i++) {
        REQUIRE(d.slater_sign_a(i) == +1.0);
        REQUIRE(d.slater_sign_b(i) == +1.0);
    }
    for (size_t i = 0; i < Nhalf; i++) {
        for (size_t j = 0; j < Nhalf; j++) {
            REQUIRE(d.slater_sign_aa(i, j) == +1.0);
            REQUIRE(d.slater_sign_bb(i, j) == +1.0);
        }
    }
}

template <size_t N> void test_determinantimpl_sign_functions() {
    auto dets = generate_test_determinants<N>();
    for (auto d : dets) {
        size_t nbits = N;
        size_t nbits_half = N / 2;
        size_t beta_bit_offset = nbits_half;
        // test the slater_sign function
        for (size_t i = 0; i < nbits; i++) {
            REQUIRE(d.slater_sign(i) == d.slater_sign_safe(i));
        }
        // test the slater_sign_a function
        for (size_t i = 0; i < nbits_half; i++) {
            REQUIRE(d.slater_sign_a(i) == d.slater_sign_safe(i));
        }
        // test the slater_sign_b function
        for (size_t i = 0; i < nbits_half; i++) {
            REQUIRE(d.slater_sign_b(i) == d.slater_sign_safe(beta_bit_offset + i));
        }
        // test the slater_sign(int,int) function
        for (size_t i = 0; i < nbits; i++) {
            for (size_t j = 0; j < nbits; j++) {
                if (i < j) {
                    double test_sign = d.slater_sign_safe(i + 1) * d.slater_sign_safe(j);
                    REQUIRE(d.slater_sign(i, j) == test_sign);
                } else if (i > j) {
                    double test_sign = d.slater_sign_safe(i) * d.slater_sign_safe(j + 1);
                    REQUIRE(d.slater_sign(i, j) == test_sign);
                } else if (i == j) {
                    REQUIRE(d.slater_sign(i, j) == 1.0);
                }
            }
        }
        // test slater_sign_aa(int,int) function
        for (size_t i = 0; i < nbits_half; i++) {
            for (size_t j = 0; j < nbits_half; j++) {
                if (i < j) {
                    double test_sign = d.slater_sign_safe(i + 1) * d.slater_sign_safe(j);
                    REQUIRE(d.slater_sign_aa(i, j) == test_sign);
                } else if (i > j) {
                    double test_sign = d.slater_sign_safe(i) * d.slater_sign_safe(j + 1);
                    REQUIRE(d.slater_sign_aa(i, j) == test_sign);
                } else if (i == j) {
                    REQUIRE(d.slater_sign_aa(i, j) == 1.0);
                }
            }
        }
        // test slater_sign_bb(int,int) function
        for (size_t i = 0; i < nbits_half; i++) {
            for (size_t j = 0; j < nbits_half; j++) {
                if (i < j) {
                    double test_sign =
                        d.slater_sign_safe(nbits_half + i + 1) * d.slater_sign_safe(nbits_half + j);
                    REQUIRE(d.slater_sign_bb(i, j) == test_sign);
                } else if (i > j) {
                    double test_sign =
                        d.slater_sign_safe(nbits_half + i) * d.slater_sign_safe(nbits_half + j + 1);
                    REQUIRE(d.slater_sign_bb(i, j) == test_sign);
                } else if (i == j) {
                    REQUIRE(d.slater_sign_bb(i, j) == 1.0);
                }
            }
        }
        // test slater_sign_aaaa(int,int,int,int) function
        int ntest_max = 100;
        for (int ntest = 0; ntest < ntest_max; ntest++) {
            int i = std::rand() % nbits_half;
            int j = std::rand() % nbits_half;
            int a = std::rand() % nbits_half;
            int b = std::rand() % nbits_half;
            if (d.get_alfa_bit(i) and d.get_alfa_bit(j) and (!d.get_alfa_bit(a)) and
                (!d.get_alfa_bit(b)) and (i != j) and (a != b)) {
                double test_sign = 1.0;
                DeterminantImpl<N> d_sign = d;
                REQUIRE(d_sign == d);
                test_sign *= d_sign.slater_sign_safe(i);
                d_sign.set_bit(i, false);
                test_sign *= d_sign.slater_sign_safe(j);
                d_sign.set_bit(j, false);
                test_sign *= d_sign.slater_sign_safe(b);
                d_sign.set_bit(b, true);
                test_sign *= d_sign.slater_sign_safe(a);
                d_sign.set_bit(a, true);
                REQUIRE(d_sign.slater_sign_aaaa(i, j, a, b) == test_sign);
            }
        }

        // test slater_sign_bbbb(int,int,int,int) function
        for (int ntest = 0; ntest < ntest_max; ntest++) {
            int i = std::rand() % nbits_half;
            int j = std::rand() % nbits_half;
            int a = std::rand() % nbits_half;
            int b = std::rand() % nbits_half;
            if (d.get_beta_bit(i) and d.get_beta_bit(j) and (!d.get_beta_bit(a)) and
                (!d.get_beta_bit(b)) and (i != j) and (a != b)) {
                double test_sign = 1.0;
                DeterminantImpl<N> d_sign = d;
                REQUIRE(d_sign == d);
                test_sign *= d_sign.slater_sign_safe(nbits_half + i);
                d_sign.set_bit(nbits_half + i, false);
                test_sign *= d_sign.slater_sign_safe(nbits_half + j);
                d_sign.set_bit(nbits_half + j, false);
                test_sign *= d_sign.slater_sign_safe(nbits_half + b);
                d_sign.set_bit(nbits_half + b, true);
                test_sign *= d_sign.slater_sign_safe(nbits_half + a);
                d_sign.set_bit(nbits_half + a, true);
                REQUIRE(d_sign.slater_sign_bbbb(i, j, a, b) == test_sign);
            }
        }
    }
}

// template <size_t N> std::pair<Determinant<N>, vector<int>> generate_random_determinant()
// {
//    std::random_device rd;
//    std::mt19937 mersenne_engine(rd());            // Generates random integers
//    std::uniform_int_distribution<int> dist{0, 1}; // Possible choices are [0,1]

//    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

//    vector<int> vec(N);
//    generate(begin(vec), end(vec), gen);

//    Determinant<N> d;
//    for (size_t i = 0; i < N; i++) {
//        d.set_bit(i, vec[i]);
//    }
//    return std::make_pair(d, vec);
//}

// template <size_t N> std::pair<Determinant<N>, vector<int>> generate_random_determinant()
// {
//    std::random_device rd;
//    std::mt19937 mersenne_engine(rd());            // Generates random integers
//    std::uniform_int_distribution<int> dist{0, 1}; // Possible choices are [0,1]

//    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

//    vector<int> vec(N);
//    generate(begin(vec), end(vec), gen);

//    Determinant<N> d;
//    for (size_t i = 0; i < N; i++) {
//        d.set_bit(i, vec[i]);
//    }
//    return std::make_pair(d, vec);
//}

// template <size_t N> std::vector<Determinant<N>> generate_test_determinants() {
//    std::vector<Determinant<N>> dets;
//    size_t nbits = Determinant<N>::nbits;
//    // the "111...1" determinant
//    Determinant<N> d_all_one;
//    for (int i = 0; i < nbits; i++)
//        d_all_one.set_bit(i, 1);
//    dets.push_back(d_all_one);

//    // the "000...0" determinant
//    Determinant<N> d_all_zero;
//    for (int i = 0; i < nbits; i++)
//        d_all_zero.set_bit(i, 0);
//    dets.push_back(d_all_zero);

//    // the "1010..10" determinant
//    Determinant<N> d_p1;
//    for (int i = 0; i < nbits; i++)
//        d_p1.set_bit(i, i % 2 + 1);
//    dets.push_back(d_p1);

//    // the "0101..01" determinant
//    Determinant<N> d_p2;
//    for (int i = 0; i < nbits; i++)
//        d_p1.set_bit(i, i % 2);
//    dets.push_back(d_p2);

//    // a random determinant
//    Determinant<N> d_rand = generate_random_determinant<N>().first;
//    dets.push_back(d_rand);

//    return dets;
//}

// template <size_t N> Determinant<N> generate_determinant_with_ones_between(int i, int j) {
//    Determinant<N> d;
//    d.zero();
//    for (int k = i; k < j; k++)
//        d.set_bit(k, 1);
//    return d;
//}

// template <class D> void test_determinant_sign_functions(const D& d) {
//    size_t nbits = D::nbits;
//    size_t nbits_half = D::nbits_half;
//    size_t beta_bit_offset = D::beta_bit_offset;
//    // test the slater_sign function
//    for (int i = 0; i < nbits; i++) {
//        REQUIRE(d.slater_sign(i) == d.slater_sign_safe(i));
//    }
//    // test the slater_sign_a function
//    for (int i = 0; i < nbits_half; i++) {
//        REQUIRE(d.slater_sign_a(i) == d.slater_sign_safe(i));
//    }
//    // test the slater_sign_b function
//    for (int i = 0; i < nbits_half; i++) {
//        REQUIRE(d.slater_sign_b(i) == d.slater_sign_safe(beta_bit_offset + i));
//    }
//    // test the slater_sign(int,int) function
//    for (int i = 0; i < nbits; i++) {
//        for (int j = 0; j < nbits; j++) {
//            if (i < j) {
//                double test_sign = d.slater_sign_safe(i + 1) * d.slater_sign_safe(j);
//                REQUIRE(d.slater_sign(i, j) == test_sign);
//            } else if (i > j) {
//                double test_sign = d.slater_sign_safe(i) * d.slater_sign_safe(j + 1);
//                REQUIRE(d.slater_sign(i, j) == test_sign);
//            } else if (i == j) {
//                REQUIRE(d.slater_sign(i, j) == 1.0);
//            }
//        }
//    }
//    // test slater_sign_aa(int,int) function
//    for (int i = 0; i < nbits_half; i++) {
//        for (int j = 0; j < nbits_half; j++) {
//            if (i < j) {
//                double test_sign = d.slater_sign_safe(i + 1) * d.slater_sign_safe(j);
//                REQUIRE(d.slater_sign_aa(i, j) == test_sign);
//            } else if (i > j) {
//                double test_sign = d.slater_sign_safe(i) * d.slater_sign_safe(j + 1);
//                REQUIRE(d.slater_sign_aa(i, j) == test_sign);
//            } else if (i == j) {
//                REQUIRE(d.slater_sign_aa(i, j) == 1.0);
//            }
//        }
//    }
//    // test slater_sign_bb(int,int) function
//    for (int i = 0; i < nbits_half; i++) {
//        for (int j = 0; j < nbits_half; j++) {
//            if (i < j) {
//                double test_sign =
//                    d.slater_sign_safe(nbits_half + i + 1) * d.slater_sign_safe(nbits_half
//                    + j);
//                REQUIRE(d.slater_sign_bb(i, j) == test_sign);
//            } else if (i > j) {
//                double test_sign =
//                    d.slater_sign_safe(nbits_half + i) * d.slater_sign_safe(nbits_half + j
//                    + 1);
//                REQUIRE(d.slater_sign_bb(i, j) == test_sign);
//            } else if (i == j) {
//                REQUIRE(d.slater_sign_bb(i, j) == 1.0);
//            }
//        }
//    }
//    // test slater_sign_aaaa(int,int,int,int) function
//    int ntest_max = 100;
//    for (int ntest = 0; ntest < ntest_max; ntest++) {
//        int i = std::rand() % nbits_half;
//        int j = std::rand() % nbits_half;
//        int a = std::rand() % nbits_half;
//        int b = std::rand() % nbits_half;
//        if (d.get_alfa_bit(i) and d.get_alfa_bit(j) and (!d.get_alfa_bit(a)) and
//            (!d.get_alfa_bit(b)) and (i != j) and (a != b)) {
//            double test_sign = 1.0;
//            D d_sign = d;
//            REQUIRE(d_sign == d);
//            test_sign *= d_sign.slater_sign_safe(i);
//            d_sign.set_bit(i, false);
//            test_sign *= d_sign.slater_sign_safe(j);
//            d_sign.set_bit(j, false);
//            test_sign *= d_sign.slater_sign_safe(b);
//            d_sign.set_bit(b, true);
//            test_sign *= d_sign.slater_sign_safe(a);
//            d_sign.set_bit(a, true);
//            REQUIRE(d_sign.slater_sign_aaaa(i, j, a, b) == test_sign);
//        }
//    }

//    // test slater_sign_bbbb(int,int,int,int) function
//    for (int ntest = 0; ntest < ntest_max; ntest++) {
//        int i = std::rand() % nbits_half;
//        int j = std::rand() % nbits_half;
//        int a = std::rand() % nbits_half;
//        int b = std::rand() % nbits_half;
//        if (d.get_beta_bit(i) and d.get_beta_bit(j) and (!d.get_beta_bit(a)) and
//            (!d.get_beta_bit(b)) and (i != j) and (a != b)) {
//            double test_sign = 1.0;
//            D d_sign = d;
//            REQUIRE(d_sign == d);
//            test_sign *= d_sign.slater_sign_safe(nbits_half + i);
//            d_sign.set_bit(nbits_half + i, false);
//            test_sign *= d_sign.slater_sign_safe(nbits_half + j);
//            d_sign.set_bit(nbits_half + j, false);
//            test_sign *= d_sign.slater_sign_safe(nbits_half + b);
//            d_sign.set_bit(nbits_half + b, true);
//            test_sign *= d_sign.slater_sign_safe(nbits_half + a);
//            d_sign.set_bit(nbits_half + a, true);
//            REQUIRE(d_sign.slater_sign_bbbb(i, j, a, b) == test_sign);
//        }
//    }
//}

// template <class D> void test_determinant_count_functions(const D& d) {
//    size_t nbits = D::nbits;
//    size_t nbits_half = D::nbits_half;

//    // test the count_alfa function
//    size_t na = 0;
//    for (int i = 0; i < nbits_half; i++) {
//        na += d.get_alfa_bit(i);
//    }
//    REQUIRE(d.count_alfa() == na);

//    // test the count_beta function
//    size_t nb = 0;
//    for (int i = 0; i < nbits_half; i++) {
//        nb += d.get_beta_bit(i);
//    }
//    REQUIRE(d.count_beta() == nb);

//    // test the count function
//    REQUIRE(d.count() == na + nb);

//    // test the npair function
//    size_t npair = 0;
//    for (int i = 0; i < nbits_half; i++) {
//        if ((d.get_alfa_bit(i) == 1) and (d.get_beta_bit(i) == 1)) {
//            npair += 1;
//        }
//    }
//    REQUIRE(d.npair() == npair);
//}

// template <size_t N> void test_determinant_occ_functions() {
//    size_t nbits = N;
//    size_t nbits_half = N / 2;

//    auto d_vec = generate_random_determinant<N>();
//    auto d = d_vec.first;
//    auto vec = d_vec.second;

//    auto aocc = d.get_alfa_occ(nbits_half);
//    auto bocc = d.get_beta_occ(nbits_half);
//    auto avir = d.get_alfa_vir(nbits_half);
//    auto bvir = d.get_beta_vir(nbits_half);

//    for (int i : aocc) {
//        REQUIRE(d.get_alfa_bit(i) == 1);
//        REQUIRE(vec[i] == 1);
//    }
//    for (int i : bocc) {
//        REQUIRE(d.get_beta_bit(i) == 1);
//        REQUIRE(vec[nbits_half + i] == 1);
//    }
//    for (int i : avir) {
//        REQUIRE(d.get_alfa_bit(i) == 0);
//        REQUIRE(vec[i] == 0);
//    }
//    for (int i : bvir) {
//        REQUIRE(d.get_beta_bit(i) == 0);
//        REQUIRE(vec[nbits_half + i] == 0);
//    }

//    REQUIRE(aocc.size() + avir.size() == nbits_half);
//    REQUIRE(bocc.size() + bvir.size() == nbits_half);
//}

#endif // _test_determinant_
