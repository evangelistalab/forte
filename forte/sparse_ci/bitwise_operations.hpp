#ifndef _bitwise_operations_hpp_
#define _bitwise_operations_hpp_

#ifdef __SSE4_2__
#include <nmmintrin.h>
#endif

#define USE_builtin_popcountll 1

// double parity(uint64_t x) { return (x & 1) ? 1.0 : -1.0; }

/// a function to accumulate hash values of 64 bit unsigned integers
/// based on boost/functional/hash/hash.hpp
inline void hash_combine_uint64(uint64_t& seed, size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/**
 * @brief Count the number of bit set to 1 in a uint64_t
 * @param x the uint64_t integer to test
 * @return the number of bits that are set to 1
 *
 * If available, this function uses SSE4.2 instructions (_mm_popcnt_u64) to speed up the evaluation.
 */
inline uint64_t ui64_bit_count(uint64_t x) {
#ifdef __SSE4_2__
    // this version is 2.6 times faster than the one below
    return _mm_popcnt_u64(x);
#else
    x = (0x5555555555555555UL & x) + (0x5555555555555555UL & (x >> 1));
    x = (0x3333333333333333UL & x) + (0x3333333333333333UL & (x >> 2));
    x = (0x0f0f0f0f0f0f0f0fUL & x) + (0x0f0f0f0f0f0f0f0fUL & (x >> 4));
    x = (0x00ff00ff00ff00ffUL & x) + (0x00ff00ff00ff00ffUL & (x >> 8));
    x = (0x0000ffff0000ffffUL & x) + (0x0000ffff0000ffffUL & (x >> 16));
    x = (0x00000000ffffffffUL & x) + (0x00000000ffffffffUL & (x >> 32));
    return x;
#endif
}

/**
 * @brief Count the number of bit set to 1 in a uint64_t
 * @param x the uint64_t integer to test
 * @return the number of bits that are set to 1
 *
 * If available, this function uses SSE4.2 instructions (_mm_popcnt_u64) to speed up the evaluation.
 */
inline double ui64_bit_parity(uint64_t x) {
#ifdef __SSE4_2__
    // this version is 2.6 times faster than the one below
    return 1 - 2 * ((_mm_popcnt_u64(x) & 1) == 1);
#else
    x = (0x5555555555555555UL & x) + (0x5555555555555555UL & (x >> 1));
    x = (0x3333333333333333UL & x) + (0x3333333333333333UL & (x >> 2));
    x = (0x0f0f0f0f0f0f0f0fUL & x) + (0x0f0f0f0f0f0f0f0fUL & (x >> 4));
    x = (0x00ff00ff00ff00ffUL & x) + (0x00ff00ff00ff00ffUL & (x >> 8));
    x = (0x0000ffff0000ffffUL & x) + (0x0000ffff0000ffffUL & (x >> 16));
    x = (0x00000000ffffffffUL & x) + (0x00000000ffffffffUL & (x >> 32));
    return 1. - 2. * ((x & 1) == 1);
#endif
}

/**
 * @brief Bit-scan to find next set bit
 * @param x the uint64_t integer to test
 * @return the index of the least significant 1-bit of x, or if x is zero, returns ~0
 */
inline uint64_t ui64_find_lowest_one_bit(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    // optimized version using builtin functions
    return __builtin_ffsll(x) - 1;
#else
    // version based on bitwise operations
    if (1 >= x)
        return x - 1; // 0 if 1, ~0 if 0
    uint64_t r = 0;
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
#endif
}

/**
 * @brief Clear the lowest bit set in a uint64_t word
 * @param x the uint64_t word
 * @return a modified version of x with the lowest bit set to 1 turned into a 0
 */
inline uint64_t ui64_clear_lowest_one_bit(uint64_t x) { return x & (x - 1); }

/**
 * @brief Find the index of the lowest bit set in a uint64_t word and clear it. A modified version
 *        of x with the lowest bit set to 1 turned into a 0 is stored in x
 * @param x the uint64_t integer to test
 * @return the index of the least significant 1-bit of x, or if x is zero, returns ~0
 */
inline uint64_t ui64_find_and_clear_lowest_one_bit(uint64_t& x) {
    uint64_t result = ui64_find_lowest_one_bit(x);
    x = x & (x - 1);
    return result;
}

/**
 * @brief Count the number of 1 from position 0 up to n - 1 and return the parity of this number.
 * @param x the uint64_t integer to test
 * @param n the end position (not counted)
 * @return the parity defined as parity = (-1)^(number of bits set between position 0 and n - 1)
 */
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

/**
 * @brief Count the number of 1 from position m + 1 up to n - 1 and return the parity of this
 * number.
 * @param x the uint64_t integer to test
 * @param m the starting position (not counted)
 * @param n the end position (not counted)
 * @return the parity defined as parity = (-1)^(number of bits set between position m + 1 and n - 1)
 */
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

/**
 * @brief Count the number of 1 from position n + 1 up to 63 and return the parity of this number.
 * @param x the uint64_t integer to test
 * @param n the start position (not counted)
 * @return the parity defined as parity = (-1)^(number of bits set between position n + 1 and 63)
 */
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
