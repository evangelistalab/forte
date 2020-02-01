#include <algorithm>
#include <numeric>

#include "catch.hpp"

#include "../../src/sparse_ci/bitwise_operations.hpp"

// Convert a string of '0' and '1' into a 64-bit unsigned integer
// The bit string can also contain separators, e.g.
// "0000 1001" -> "00001001"
// Bit strings are writtent from left to right, e.g.
// string:   "00010110"
// position:  01234567
uint64_t make_uint64_from_str(const std::string s) {
    uint64_t x(0);
    //    if (s.size() <= 64) {
    size_t k = 0;
    for (auto c : s) {
        if (c == '1') {
            if (k < 64) {
                x |= (static_cast<uint64_t>(1) << k);
                k++;
            }
        }
        if (c == '0') {
            if (k < 64) {
                x &= ~(static_cast<uint64_t>(1) << k);
                k++;
            }
        }
    }
    //    }
    return x;
}

TEST_CASE("Bitwise operations", "[BitwiseOperations]") {
    auto x1 = make_uint64_from_str("0");
    REQUIRE(ui64_find_lowest_one_bit(x1) == ~0);
    REQUIRE(x1 == make_uint64_from_str("0"));

    auto x2 = make_uint64_from_str("1");
    REQUIRE(ui64_find_lowest_one_bit(x2) == 0);
    REQUIRE(x2 == make_uint64_from_str("1"));

    auto x2_clear = ui64_clear_lowest_one_bit(x2);
    REQUIRE(x2_clear == 0);
    REQUIRE(ui64_find_lowest_one_bit(x2_clear) == ~0);

    auto x3 = make_uint64_from_str("01");
    REQUIRE(ui64_find_lowest_one_bit(x3) == 1);

    auto x3_clear = ui64_clear_lowest_one_bit(x3);
    REQUIRE(x3_clear == 0);
    REQUIRE(ui64_find_lowest_one_bit(x3_clear) == ~0);

    auto x4 = make_uint64_from_str("1001");
    REQUIRE(ui64_find_lowest_one_bit(x4) == 0);

    auto x4_clear = ui64_clear_lowest_one_bit(x4);
    REQUIRE(x4_clear == make_uint64_from_str("0001"));
    REQUIRE(ui64_find_lowest_one_bit(x4_clear) == 3);

    auto x4_clear2 = ui64_clear_lowest_one_bit(x4_clear);
    REQUIRE(x4_clear2 == 0);
    REQUIRE(ui64_find_lowest_one_bit(x4_clear2) == ~0);

    auto x5 = make_uint64_from_str("10001");
    REQUIRE(ui64_find_and_clear_lowest_one_bit(x5) == 0);
    REQUIRE(x5 == make_uint64_from_str("00001"));

    auto x6 = make_uint64_from_str(
        "10000001 01000000 00000000 00000001 00000000 00000000 00000000 00000001");
    auto y1 = make_uint64_from_str(
        "00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000");
    auto y2 = make_uint64_from_str(
        "00000010 00000000 00000000 00000000 00000000 00000000 00000000 00000000");
    REQUIRE(ui64_bit_count(x6) == 5);
    REQUIRE(ui64_bit_count(y1) == 0);
    REQUIRE(ui64_bit_count(y2) == 1);
    REQUIRE(ui64_bit_parity(x6) == -1.0);
    REQUIRE(ui64_bit_parity(y1) == +1.0);
    REQUIRE(ui64_bit_parity(y2) == -1.0);
    REQUIRE(ui64_sign(x6, 0) == +1.0);
    REQUIRE(ui64_sign(x6, 1) == -1.0);
    REQUIRE(ui64_sign(x6, 6) == -1.0);
    REQUIRE(ui64_sign(x6, 7) == -1.0);
    REQUIRE(ui64_sign(x6, 8) == +1.0);
    REQUIRE(ui64_find_and_clear_lowest_one_bit(x6) == 0);
    REQUIRE(ui64_find_and_clear_lowest_one_bit(x6) == 7);
    REQUIRE(ui64_find_and_clear_lowest_one_bit(x6) == 9);
    REQUIRE(ui64_find_and_clear_lowest_one_bit(x6) == 31);
    REQUIRE(ui64_find_and_clear_lowest_one_bit(x6) == 63);
}
