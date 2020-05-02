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

template <size_t N> std::vector<DeterminantImpl<N>> generate_test_determinants(int nrandom = 10) {
    std::vector<DeterminantImpl<N>> dets;
    size_t nbits = DeterminantImpl<N>::nbits;
    size_t nbits_half = nbits / 2;
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

    // the "111...0" determinant
    DeterminantImpl<N> d_all_alpha;
    for (size_t i = 0; i < nbits_half; i++)
        d_all_alpha.set_alfa_bit(i, 0);
    dets.push_back(d_all_alpha);

    // the "000...1" determinant
    DeterminantImpl<N> d_all_beta;
    for (size_t i = 0; i < nbits_half; i++)
        d_all_beta.set_beta_bit(i, 0);
    dets.push_back(d_all_beta);

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
    for (int i = 0; i < nrandom; i++) {
        DeterminantImpl<N> d_rand = generate_random_determinant<N>().first;
        dets.push_back(d_rand);
    }

    return dets;
}

// Test that a BitArray object is initialized to zero
template <size_t N> void test_bitarray_init() {
    auto ba = forte::BitArray<N>();

    REQUIRE(ba.get_nbits() == N);

    for (size_t i = 0; i < N; i++) {
        REQUIRE(ba.get_bit(i) == 0);
    }
    REQUIRE(ba.count() == 0);
    REQUIRE(ba.find_first_one() == ~(unsigned long int)(0));
}

template <size_t N> void test_determinantimpl_init() {
    size_t Nhalf = N / 2;
    forte::DeterminantImpl<N> d;

    REQUIRE(d.get_nbits_half() == Nhalf);

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

// Test that a BitArray object is initialized to zero
template <size_t N> void test_bitarray_setget() {
    auto ba = forte::BitArray<N>();
    std::vector<bool> vals;
    for (size_t i = 0; i < N; i++) {
        vals.push_back(rand() % 2);
        ba.set_bit(i, vals[i]);
    }
    for (size_t i = 0; i < N; i++) {
        REQUIRE(ba.get_bit(i) == vals[i]);
    }
    ba.flip();
    for (size_t i = 0; i < N; i++) {
        REQUIRE(ba.get_bit(i) == !vals[i]);
    }
    ba.zero();
    for (size_t i = 0; i < N; i++) {
        REQUIRE(ba.get_bit(i) == 0);
    }
}

// Test that a BitArray object is initialized to zero
template <size_t N> void test_determinantimpl_setget() {
    auto [d, vals] = generate_random_determinant<N>();
    for (size_t i = 0; i < N; i++) {
        d.set_bit(i, vals[i]);
    }
    for (size_t i = 0; i < N; i++) {
        REQUIRE(d.get_bit(i) == vals[i]);
    }
    d.flip();
    for (size_t i = 0; i < N; i++) {
        REQUIRE(d.get_bit(i) == !vals[i]);
    }
    d.zero();
    for (size_t i = 0; i < N; i++) {
        REQUIRE(d.get_bit(i) == 0);
    }
}

template <size_t N> void test_determinantimpl_sign_functions(int ntest_max = 25000) {
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

        if (ntest_max == 0) {
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
                        double test_sign = d.slater_sign_safe(nbits_half + i + 1) *
                                           d.slater_sign_safe(nbits_half + j);
                        REQUIRE(d.slater_sign_bb(i, j) == test_sign);
                    } else if (i > j) {
                        double test_sign = d.slater_sign_safe(nbits_half + i) *
                                           d.slater_sign_safe(nbits_half + j + 1);
                        REQUIRE(d.slater_sign_bb(i, j) == test_sign);
                    } else if (i == j) {
                        REQUIRE(d.slater_sign_bb(i, j) == 1.0);
                    }
                }
            }
        } else {
            // test the slater_sign(int,int) function
            for (int ntest = 0; ntest < ntest_max; ntest++) {
                int i = std::rand() % nbits;
                int j = std::rand() % nbits;
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
            // test slater_sign_aa(int,int) function
            for (int ntest = 0; ntest < ntest_max; ntest++) {
                int i = std::rand() % nbits_half;
                int j = std::rand() % nbits_half;

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
            // test slater_sign_bb(int,int) function
            for (int ntest = 0; ntest < ntest_max; ntest++) {
                int i = std::rand() % nbits_half;
                int j = std::rand() % nbits_half;

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

template <size_t N> void test_determinantimpl_count_functions() {
    auto dets = generate_test_determinants<N>();
    for (auto d : dets) {
        size_t nbits = N;
        size_t nbits_half = N / 2;

        // test the count_alfa function
        size_t na = 0;
        for (size_t i = 0; i < nbits_half; i++) {
            na += d.get_alfa_bit(i);
        }
        REQUIRE(d.count_alfa() == na);

        // test the count_beta function
        size_t nb = 0;
        for (size_t i = 0; i < nbits_half; i++) {
            nb += d.get_beta_bit(i);
        }
        REQUIRE(d.count_beta() == nb);

        // test the count function
        REQUIRE(d.count() == na + nb);

        // test the npair function
        size_t npair = 0;
        for (size_t i = 0; i < nbits_half; i++) {
            if ((d.get_alfa_bit(i) == 1) and (d.get_beta_bit(i) == 1)) {
                npair += 1;
            }
        }
        REQUIRE(d.npair() == npair);
    }
}

#endif // _test_determinant_
