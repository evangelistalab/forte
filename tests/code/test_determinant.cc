#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file

#include <algorithm>
#include <numeric>

#include "catch.hpp"

#include "../../src/sparse_ci/determinant.h"
#include "../../src/sparse_ci/determinant.hpp"
#include "../../src/sparse_ci/bitarray.hpp"
#include "test_determinant.hpp"

using namespace forte;

Determinant make_det_from_string(std::string s) {
    Determinant d;
    size_t n = s.size() / 2;
    if (n % 2 == 0) {
        for (std::string::size_type i = 0; i < n; ++i) {
            d.set_alfa_bit(i, s[i] == '0' ? 0 : 1);
            d.set_beta_bit(i, s[i + n] == '0' ? 0 : 1);
        }
    } else {
        std::cout << "\n\n  Function make_det_from_string called with a string of odd size";
        exit(1);
    }
    return d;
}

Determinant make_det_from_string(std::string s_a, std::string s_b) {
    Determinant d;
    if (s_a.size() == s_b.size()) {
        for (std::string::size_type i = 0; i < s_a.size(); ++i) {
            d.set_alfa_bit(i, s_a[i] == '0' ? 0 : 1);
            d.set_beta_bit(i, s_b[i] == '0' ? 0 : 1);
        }
    } else {
        std::cout << "\n\n  Function make_det_from_string called with strings of different size";
        exit(1);
    }
    return d;
}

/// This function generates a complementary occupation from a given set of occupation vector
/// and the total number of orbitals. E.g.:
///
/// get_complementary_occupation({0,2},8) -> {1,3,4,5,6,7}
///
/// Used to test the aocc/bocc/avir/bvir orbitals of a determinant
std::vector<int> get_complementary_occupation(const std::vector<int>& occupation,
                                              int num_total_orbs) {
    std::vector<int> all(num_total_orbs);
    std::iota(all.begin(), all.end(), 0);
    std::vector<int> result;
    std::set_difference(all.begin(), all.end(), occupation.begin(), occupation.end(),
                        std::inserter(result, result.begin()));
    return result;
}

unsigned int Factorial(unsigned int number) {
    return number <= 1 ? number : Factorial(number - 1) * number;
}

// ==> TESTS <==

// Test that a BitArray object is initialized to zero
TEST_CASE("Initialization [BitArray]", "[BitArray]") {
    test_bitarray_init<64>();
    test_bitarray_init<128>();
    test_bitarray_init<192>();
    test_bitarray_init<256>();
    test_bitarray_init<320>();
    test_bitarray_init<384>();
    test_bitarray_init<448>();
    test_bitarray_init<512>();
    test_bitarray_init<1024>();
}

// Test that a BitArray object is initialized to zero
TEST_CASE("Initialization [DeterminantImpl]", "[DeterminantImpl]") {
    test_determinantimpl_init<128>();
    test_determinantimpl_init<256>();
    test_determinantimpl_init<384>();
    test_determinantimpl_init<512>();
    test_determinantimpl_init<640>();
    test_determinantimpl_init<768>();
    test_determinantimpl_init<896>();
    test_determinantimpl_init<1024>();
}

TEST_CASE("Set/get [BitArray]", "[BitArray]") {
    test_bitarray_setget<64>();
    test_bitarray_setget<128>();
    test_bitarray_setget<192>();
    test_bitarray_setget<256>();
    test_bitarray_setget<320>();
    test_bitarray_setget<384>();
    test_bitarray_setget<448>();
    test_bitarray_setget<512>();
    test_bitarray_setget<1024>();
}

TEST_CASE("Set/get [DeterminantImpl]", "[DeterminantImpl]") {
    test_determinantimpl_setget<128>();
    test_determinantimpl_setget<256>();
    test_determinantimpl_setget<384>();
    test_determinantimpl_setget<512>();
    test_determinantimpl_setget<1024>();
}

TEST_CASE("Determinant sign [DeterminantImpl]", "[DeterminantImpl]") {
    test_determinantimpl_sign_functions<128>();
    test_determinantimpl_sign_functions<256>();
    test_determinantimpl_sign_functions<384>();
    test_determinantimpl_sign_functions<512>();
    test_determinantimpl_sign_functions<1024>();
}

TEST_CASE("Determinant count [DeterminantImpl]", "[DeterminantImpl]") {
    test_determinantimpl_count_functions<128>();
    test_determinantimpl_count_functions<256>();
    test_determinantimpl_count_functions<384>();
    test_determinantimpl_count_functions<512>();
    test_determinantimpl_count_functions<640>();
    test_determinantimpl_count_functions<768>();
    test_determinantimpl_count_functions<896>();
    test_determinantimpl_count_functions<1024>();
}

TEST_CASE("Empty determinant", "[Determinant]") {
    Determinant det_test;
    Determinant det_ref =
        make_det_from_string("0000000000000000000000000000000000000000000000000000000000000000",
                             "0000000000000000000000000000000000000000000000000000000000000000");

    REQUIRE(det_test == det_ref); // test that default constructor gives empty determinant
    REQUIRE(det_test.count_alfa() == 0);
    REQUIRE(det_test.count_beta() == 0);
    REQUIRE(det_test.npair() == 0);

    std::vector<int> aocc = det_test.get_alfa_occ(Determinant::nbits_half);
    std::vector<int> bocc = det_test.get_beta_occ(Determinant::nbits_half);

    std::vector<int> aocc_ref{};
    std::vector<int> bocc_ref{};
    std::vector<int> avir_ref = get_complementary_occupation(aocc, Determinant::nbits_half);
    std::vector<int> bvir_ref = get_complementary_occupation(bocc, Determinant::nbits_half);

    REQUIRE(aocc == aocc_ref);
    REQUIRE(bocc == bocc_ref);

    REQUIRE(det_test.slater_sign_a(0) == 1.0);
    REQUIRE(det_test.slater_sign_a(10) == 1.0);
    REQUIRE(det_test.slater_sign_a(63) == 1.0);
    REQUIRE(det_test.slater_sign_b(0) == 1.0);
    REQUIRE(det_test.slater_sign_b(10) == 1.0);
    REQUIRE(det_test.slater_sign_b(63) == 1.0);
}

TEST_CASE("Bit counting", "[Determinant]") {
    Determinant det_test =
        make_det_from_string("1001100000000000000000000000000000000000000000000000000000010000",
                             "0001000000000000001000000000000000000000000000000000000000000001");

    REQUIRE(det_test.count_alfa() == 4);
    REQUIRE(det_test.count_beta() == 3);
    REQUIRE(det_test.npair() == 1);

    std::vector<int> aocc = det_test.get_alfa_occ(Determinant::nbits_half);
    std::vector<int> bocc = det_test.get_beta_occ(Determinant::nbits_half);
    std::vector<int> avir = det_test.get_alfa_vir(Determinant::nbits_half);
    std::vector<int> bvir = det_test.get_beta_vir(Determinant::nbits_half);

    std::vector<int> aocc_ref{0, 3, 4, 59};
    std::vector<int> bocc_ref{3, 18, 63};
    std::vector<int> avir_ref = get_complementary_occupation(aocc_ref, Determinant::nbits_half);
    std::vector<int> bvir_ref = get_complementary_occupation(bocc_ref, Determinant::nbits_half);

    REQUIRE(aocc == aocc_ref);
    REQUIRE(bocc == bocc_ref);
    REQUIRE(avir == avir_ref);
    REQUIRE(bvir == bvir_ref);

    REQUIRE(det_test.slater_sign_a(0) == 1.0);
    REQUIRE(det_test.slater_sign_a(1) == -1.0);
    REQUIRE(det_test.slater_sign_a(2) == -1.0);
    REQUIRE(det_test.slater_sign_a(3) == -1.0);
    REQUIRE(det_test.slater_sign_a(4) == 1.0);
    REQUIRE(det_test.slater_sign_a(5) == -1.0);
    REQUIRE(det_test.slater_sign_a(58) == -1.0);
    REQUIRE(det_test.slater_sign_a(59) == -1.0);
    REQUIRE(det_test.slater_sign_a(60) == 1.0);
    REQUIRE(det_test.slater_sign_a(63) == 1.0);
    REQUIRE(det_test.slater_sign_b(0) == 1.0);
    REQUIRE(det_test.slater_sign_b(3) == 1.0);
    REQUIRE(det_test.slater_sign_b(4) == -1.0);
    REQUIRE(det_test.slater_sign_b(5) == -1.0);
    REQUIRE(det_test.slater_sign_b(18) == -1.0);
    REQUIRE(det_test.slater_sign_b(19) == 1.0);
    REQUIRE(det_test.slater_sign_b(62) == 1.0);
    REQUIRE(det_test.slater_sign_b(63) == 1.0);
}

TEST_CASE("Full determinant", "[Determinant]") {
    Determinant det_test =
        make_det_from_string("1111111111111111111111111111111111111111111111111111111111111111",
                             "1111111111111111111111111111111111111111111111111111111111111111");

    REQUIRE(det_test.count_alfa() == 64);
    REQUIRE(det_test.count_beta() == 64);
    REQUIRE(det_test.npair() == 64);

    std::vector<int> aocc = det_test.get_alfa_occ(Determinant::nbits_half);
    std::vector<int> bocc = det_test.get_beta_occ(Determinant::nbits_half);
    std::vector<int> avir = det_test.get_alfa_vir(Determinant::nbits_half);
    std::vector<int> bvir = det_test.get_beta_vir(Determinant::nbits_half);

    std::vector<int> aocc_ref(64);
    std::vector<int> bocc_ref(64);
    std::iota(aocc_ref.begin(), aocc_ref.end(), 0);
    std::iota(bocc_ref.begin(), bocc_ref.end(), 0);
    std::vector<int> avir_ref = get_complementary_occupation(aocc_ref, Determinant::nbits_half);
    std::vector<int> bvir_ref = get_complementary_occupation(bocc_ref, Determinant::nbits_half);

    REQUIRE(aocc == aocc_ref);
    REQUIRE(bocc == bocc_ref);
    REQUIRE(avir == avir_ref);
    REQUIRE(bvir == bvir_ref);

    REQUIRE(det_test.slater_sign_a(0) == 1.0);
    REQUIRE(det_test.slater_sign_a(1) == -1.0);
    REQUIRE(det_test.slater_sign_a(2) == 1.0);
    REQUIRE(det_test.slater_sign_a(3) == -1.0);
    REQUIRE(det_test.slater_sign_a(4) == 1.0);
    REQUIRE(det_test.slater_sign_a(5) == -1.0);
    REQUIRE(det_test.slater_sign_a(6) == 1.0);
    REQUIRE(det_test.slater_sign_a(7) == -1.0);
    REQUIRE(det_test.slater_sign_b(0) == 1.0);
    REQUIRE(det_test.slater_sign_b(1) == -1.0);
    REQUIRE(det_test.slater_sign_b(2) == 1.0);
    REQUIRE(det_test.slater_sign_b(3) == -1.0);
    REQUIRE(det_test.slater_sign_b(4) == 1.0);
    REQUIRE(det_test.slater_sign_b(5) == -1.0);
    REQUIRE(det_test.slater_sign_b(6) == 1.0);
    REQUIRE(det_test.slater_sign_b(7) == -1.0);
}
