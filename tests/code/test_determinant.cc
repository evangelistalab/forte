#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file

#include <algorithm>
#include <numeric>

#include "catch.hpp"

#include "../../src/sparse_ci/ui64_determinant.h"

using namespace psi::forte;

UI64Determinant make_det_from_string(std::string s) {
    UI64Determinant d;
    int n = s.size() / 2;
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

UI64Determinant make_det_from_string(std::string s_a, std::string s_b) {
    UI64Determinant d;
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

TEST_CASE("Empty determinant", "[UI64Determinant]") {
    UI64Determinant det_test;
    UI64Determinant det_ref =
        make_det_from_string("0000000000000000000000000000000000000000000000000000000000000000",
                             "0000000000000000000000000000000000000000000000000000000000000000");

    REQUIRE(det_test == det_ref); // test that default constructor gives empty determinant
    REQUIRE(det_test.count_alfa() == 0);
    REQUIRE(det_test.count_beta() == 0);
    REQUIRE(det_test.npair() == 0);

    std::vector<int> aocc = det_test.get_alfa_occ(UI64Determinant::num_str_bits);
    std::vector<int> bocc = det_test.get_beta_occ(UI64Determinant::num_str_bits);

    std::vector<int> aocc_ref{};
    std::vector<int> bocc_ref{};
    std::vector<int> avir_ref = get_complementary_occupation(aocc, 64);
    std::vector<int> bvir_ref = get_complementary_occupation(bocc, 64);

    REQUIRE(aocc == aocc_ref);
    REQUIRE(bocc == bocc_ref);
}

TEST_CASE("Bit counting", "[UI64Determinant]") {
    UI64Determinant det_test =
        make_det_from_string("1001100000000000000000000000000000000000000000000000000000010000",
                             "0001000000000000001000000000000000000000000000000000000000000001");

    REQUIRE(det_test.count_alfa() == 4);
    REQUIRE(det_test.count_beta() == 3);
    REQUIRE(det_test.npair() == 1);

    std::vector<int> aocc = det_test.get_alfa_occ(UI64Determinant::num_str_bits);
    std::vector<int> bocc = det_test.get_beta_occ(UI64Determinant::num_str_bits);
    std::vector<int> avir = det_test.get_alfa_vir(UI64Determinant::num_str_bits);
    std::vector<int> bvir = det_test.get_beta_vir(UI64Determinant::num_str_bits);

    std::vector<int> aocc_ref{0, 3, 4, 59};
    std::vector<int> bocc_ref{3, 18, 63};
    std::vector<int> avir_ref = get_complementary_occupation(aocc_ref, 64);
    std::vector<int> bvir_ref = get_complementary_occupation(bocc_ref, 64);

    REQUIRE(aocc == aocc_ref);
    REQUIRE(bocc == bocc_ref);
    REQUIRE(avir == avir_ref);
    REQUIRE(bvir == bvir_ref);
}

TEST_CASE("Full determinant", "[UI64Determinant]") {
    UI64Determinant det_test =
        make_det_from_string("1111111111111111111111111111111111111111111111111111111111111111",
                             "1111111111111111111111111111111111111111111111111111111111111111");

    REQUIRE(det_test.count_alfa() == 64);
    REQUIRE(det_test.count_beta() == 64);
    REQUIRE(det_test.npair() == 64);

    std::vector<int> aocc = det_test.get_alfa_occ(UI64Determinant::num_str_bits);
    std::vector<int> bocc = det_test.get_beta_occ(UI64Determinant::num_str_bits);
    std::vector<int> avir = det_test.get_alfa_vir(UI64Determinant::num_str_bits);
    std::vector<int> bvir = det_test.get_beta_vir(UI64Determinant::num_str_bits);

    std::vector<int> avir_ref{};
    std::vector<int> bvir_ref{};
    std::vector<int> aocc_ref = get_complementary_occupation(avir_ref, 64);
    std::vector<int> bocc_ref = get_complementary_occupation(bvir_ref, 64);

    REQUIRE(aocc == aocc_ref);
    REQUIRE(bocc == bocc_ref);
    REQUIRE(avir == avir_ref);
    REQUIRE(bvir == bvir_ref);
}
