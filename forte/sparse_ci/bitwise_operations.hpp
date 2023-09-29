#ifndef _bitwise_operations_hpp_
#define _bitwise_operations_hpp_

#include <bit>

#define USE_builtin_popcountll 1

// double parity(uint64_t x) { return (x & 1) ? 1.0 : -1.0; }

/// a function to accumulate hash values of 64 bit unsigned integers
/// based on boost/functional/hash/hash.hpp
inline void hash_combine_uint64(uint64_t& seed, size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/// @brief Count the number of bit set to 1 in a uint64_t
/// @param x the uint64_t integer to test
/// @return the number of bits that are set to 1
inline uint64_t ui64_bit_count(uint64_t x) { return std::popcount(x); }

/// @brief Compute the parity of a uint64_t integer (1 if odd number of bits set, -1 otherwise)
/// @param x the uint64_t integer to test
/// @return parity = (-1)^(number of bits set to 1)
inline double ui64_bit_parity(uint64_t x) { return 1 - 2 * ((std::popcount(x) & 1) == 1); }

/// @brief Bit-scan to find next set bit
/// @param x the uint64_t integer to test
/// @return the index of the least significant 1-bit of x, or if x is zero, returns ~0
inline uint64_t ui64_find_lowest_one_bit(uint64_t x) { return std::countr_zero(x); }

/// @brief Bit-scan to find next set bit after position pos
/// @param x the uint64_t integer to test
/// @param pos the position where we should start scanning (must be less than 64)
/// @return the index of the least significant 1-bit of x, or if x is zero, returns ~0
inline uint64_t ui64_find_lowest_one_bit_at_pos(uint64_t x, int pos) {
    const uint64_t mask = uint64_t(~0) << pos; // set all bits to one and shift left
    x &= mask;
    return ui64_find_lowest_one_bit(x);
}

/// @brief Clear the lowest bit set in a uint64_t word
/// @param x the uint64_t word
/// @return a modified version of x with the lowest bit set to 1 turned into a 0
inline uint64_t ui64_clear_lowest_one_bit(uint64_t x) { return x & (x - 1); }

/// @brief Find the index of the lowest bit set in a uint64_t word and clear it. A modified version
///        of x with the lowest bit set to 1 turned into a 0 is stored in x
/// @param x the uint64_t integer to test. This value is modified by the function
/// @return the index of the least significant 1-bit of x, or if x is zero, returns ~0
inline uint64_t ui64_find_and_clear_lowest_one_bit(uint64_t& x) {
    uint64_t result = ui64_find_lowest_one_bit(x);
    x = ui64_clear_lowest_one_bit(x);
    return result;
}

/// @brief Count the number of 1's from position 0 up to n - 1 and return the parity of this number.
/// @param x the uint64_t integer to test
/// @param n the end position (not counted)
/// @return the parity defined as parity = (-1)^(number of bits set to 1 between position 0 and n-1)
inline double ui64_sign(uint64_t x, int n) {
    // TODO PERF: speedup by avoiding the mask altogether
    // This implementation is 20 x times faster than one based on for loops
    // First build a mask with n bit set. Then & with string and count.
    // Example for 16 bit string
    //                 n            (n = 5)
    // want       11111000 00000000
    // mask       11111111 11111111
    // mask << 11 00000000 00011111 11 = 16 - 5
    // mask >> 11 11111000 00000000
    //
    // Note: This strategy does not work when n = 0 because ~0 << 64 = ~0 (!!!)
    // so we treat this case separately
    if (n == 0)
        return 1.0;
    uint64_t mask = ~0;
    mask = mask << (64 - n);             // make a string with n bits set
    mask = mask >> (64 - n);             // move it right by n
    mask = x & mask;                     // intersect with string
    mask = ui64_bit_count(mask);         // count bits in between
    return (mask % 2 == 0) ? 1.0 : -1.0; // compute sign
}

/// @brief Count the number of 1's from position m + 1 up to n - 1 and return the parity of this
/// number.
/// @param x the uint64_t integer to test
/// @param m the starting position (not counted)
/// @param n the end position (not counted)
/// @return the parity defined as parity = (-1)^(number of bits set to 1 between position m+1 and
/// n-1)
inline double ui64_sign(uint64_t x, int m, int n) {
    // TODO PERF: speedup by avoiding the mask altogether
    // This implementation is a bit faster than one based on for loops
    // First build a mask with bit set between positions m and n. Then & with string and count.
    // Example for 16 bit string
    //              m  n            (m = 2, n = 5)
    // want       00011000 00000000
    // mask       11111111 11111111
    // mask << 14 00000000 00000011 14 = 16 + 1 - |5-2|
    // mask >> 11 00011000 00000000 11 = 16 - |5-2| -  min(2,5)
    // the mask should have |m - n| - 1 bits turned on
    uint64_t gap = std::abs(m - n);
    if (gap < 2) { // special cases
        return 1.0;
    }
    uint64_t mask = ~0;
    mask = mask << (65 - gap);                  // make a string with |m - n| - 1 bits set
    mask = mask >> (64 - gap - std::min(m, n)); // move it right after min(m, n)
    mask = x & mask;                            // intersect with string
    mask = ui64_bit_count(mask);                // count bits in between
    return (mask % 2 == 0) ? 1.0 : -1.0;        // compute sign
}

/// @brief Count the number of 1's from position n + 1 up to 63 and return the parity of this
/// number.
/// @param x the uint64_t integer to test
/// @param n the start position (not counted)
/// @return the parity defined as parity = (-1)^(number of bits set to 1 between position n+1 and
/// 63)
inline double ui64_sign_reverse(uint64_t x, int n) {
    // TODO PERF: speedup by avoiding the mask altogether
    // This implementation is 20 x times faster than one based on for loops
    // First build a mask with n bit set. Then & with string and count.
    // Example for 16 bit string
    //                 n            (n = 5)
    // want       00000011 11111111
    // mask       11111111 11111111
    // mask << 6  00000011 11111111 6 = 5 + 1
    //
    // Note: This strategy does not work when n = 0 because ~0 << 64 = ~0 (!!!)
    // so we treat this case separately
    if (n == 63)
        return 1.0;
    uint64_t mask = ~0;
    mask = mask << (n + 1);              // make a string with 64 - n - 1 bits set
    mask = x & mask;                     // intersect with string
    mask = ui64_bit_count(mask);         // count bits in between
    return (mask % 2 == 0) ? 1.0 : -1.0; // compute sign
}

#endif // _bitwise_operations_hpp_
