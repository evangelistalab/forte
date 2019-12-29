#ifndef _determinant_functions_hpp_
#define _determinant_functions_hpp_

#include <string>
#include <cstddef>
#include <iostream>
#include <vector>

namespace forte {

#define PERFORMANCE_OPTIMIZATION 0

double slater_rules_single_alpha(String Ib, String Ia, String Ja,
                                 const std::shared_ptr<ActiveSpaceIntegrals>& ints);
double slater_rules_single_beta(String Ia, String Ib, String Jb,
                                const std::shared_ptr<ActiveSpaceIntegrals>& ints);

double slater_rules_double_alpha_alpha(String Ia, String Ja,
                                       const std::shared_ptr<ActiveSpaceIntegrals>& ints);

double slater_rules_double_beta_beta(String Ib, String Jb,
                                     const std::shared_ptr<ActiveSpaceIntegrals>& ints);

double slater_rules_double_alpha_beta_pre(int i, int a, String Ib, String Jb,
                                          const std::shared_ptr<ActiveSpaceIntegrals>& ints);

Determinant common_occupation(const Determinant& lhs, const Determinant& rhs);

Determinant different_occupation(const Determinant& lhs, const Determinant& rhs);

/// Find the spin orbitals that are occupied only one determinant (performs a bitwise OR, |)
Determinant union_occupation(const Determinant& lhs, const Determinant& rhs);

void enforce_spin_completeness(std::vector<Determinant>& det_space, int nmo);
/*

template <size_t N>
double slater_rules_single_alpha(StringImpl<N> Ib, StringImpl<N> Ia, StringImpl<N> Ja,
                                 const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    StringImpl<N> IJa = Ia ^ Ja;
    uint64_t i = IJa.find_first_one();
    IJa.clear_first_one();
    uint64_t a = IJa.find_first_one();

    // Diagonal contribution
    double matrix_element = ints->oei_b(i, a);
    // find common bits
    IJa = Ia & Ja;
    for (int p = 0; p < N; ++p) {
        if (Ia.get_bit(p)) {
            matrix_element += ints->tei_aa(i, p, a, p);
        }
        if (Ib.get_bit(p)) {
            matrix_element += ints->tei_ab(i, p, a, p);
        }
    }
    return (Ia.slater_sign(i, a) * matrix_element);
}

template <size_t N>
double slater_rules_single_beta(StringImpl<N> Ia, StringImpl<N> Ib, StringImpl<N> Jb,
                                const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    StringImpl<N> IJb = Ib ^ Jb;
    uint64_t i = IJb.find_first_one();
    IJb.clear_first_one();
    uint64_t a = IJb.find_first_one();

    // Diagonal contribution
    double matrix_element = ints->oei_b(i, a);
    // find common bits
    IJb = Ib & Jb;
    for (int p = 0; p < N; ++p) {
        if (Ia.get_bit(p)) {
            matrix_element += ints->tei_ab(p, i, p, a);
        }
        if (Ib.get_bit(p)) {
            matrix_element += ints->tei_bb(p, i, p, a);
        }
    }
    return (Ib.slater_sign(i, a) * matrix_element);
}

template <size_t N>
double slater_rules_double_alpha_alpha(StringImpl<N> Ia, StringImpl<N> Ja,
                                       const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    StringImpl<N> IJb = Ia ^ Ja;

    StringImpl<N> Ia_sub = Ia & IJb;
    uint64_t i = Ia_sub.find_first_one();
    Ia_sub.clear_first_one();
    uint64_t j = Ia_sub.find_first_one();

    StringImpl<N> Ja_sub = Ja & IJb;
    uint64_t k = Ja_sub.find_first_one();
    Ja_sub.clear_first_one();
    uint64_t l = Ja_sub.find_first_one();

    return Ia.slater_sign(i, j) * Ja.slater_sign(k, l) * ints->tei_aa(i, j, k, l);
}

template <size_t N>
double slater_rules_double_beta_beta(StringImpl<N> Ib, StringImpl<N> Jb,
                                     const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    StringImpl<N> IJb = Ib ^ Jb;

    StringImpl<N> Ib_sub = Ib & IJb;
    uint64_t i = Ib_sub.find_first_one();
    Ib_sub.clear_first_one();
    uint64_t j = Ib_sub.find_first_one();

    StringImpl<N> Jb_sub = Jb & IJb;
    uint64_t k = Jb_sub.find_first_one();
    Jb_sub.clear_first_one();
    uint64_t l = Jb_sub.find_first_one();

    return Ib.slater_sign(i, j) * Jb.slater_sign(k, l) * ints->tei_bb(i, j, k, l);
}

template <size_t N>
double slater_rules_double_alpha_beta_pre(int i, int a, StringImpl<N> Ib, StringImpl<N> Jb,
                                          const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    //    outfile->Printf("\n %zu %zu", Ib, Jb);
    StringImpl<N> Ib_xor_Jb = Ib ^ Jb;
    uint64_t j = Ib_xor_Jb.find_first_one();
    Ib_xor_Jb.clear_first_one();
    uint64_t b = Ib_xor_Jb.find_first_one();
    //    outfile->Printf("\n  i = %d, j = %d, a = %d, b = %d", i, j, a, b);
    return Ib.slater_sign(j, b) * ints->tei_ab(i, j, a, b);
}

template <size_t N>
DeterminantImpl<N> common_occupation(const DeterminantImpl<N>& lhs, const DeterminantImpl<N>& rhs) {
    DeterminantImpl<N> result = lhs & rhs;
    return result;
}

template <size_t N>
DeterminantImpl<N> different_occupation(const DeterminantImpl<N>& lhs,
                                        const DeterminantImpl<N>& rhs) {
//    DeterminantImpl<N> result; = lhs ^ rhs;
    return lhs ^ rhs;
}

/// Find the spin orbitals that are occupied only one determinant (performs a bitwise OR, |)
template <size_t N>
DeterminantImpl<N> union_occupation(const DeterminantImpl<N>& lhs, const DeterminantImpl<N>& rhs) {
    DeterminantImpl<N> result = lhs | rhs;
    return result;
}
*/

} // namespace forte

#endif // _determinant_functions_hpp_
