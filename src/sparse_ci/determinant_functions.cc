//#include <string>
//#include <cstddef>
//#include <iostream>
//#include <vector>

#include "sparse_ci/determinant.h"
#include "integrals/active_space_integrals.h"

namespace forte {

#define PERFORMANCE_OPTIMIZATION 0

double slater_rules_single_alpha(String Ib, String Ia, String Ja,
                                 const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    size_t N = Determinant::nbits_half;
    String IJa = Ia ^ Ja;
    uint64_t i = IJa.lowest_one_index();
    IJa.clear_lowest_one();
    uint64_t a = IJa.lowest_one_index();

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

double slater_rules_single_beta(String Ia, String Ib, String Jb,
                                const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    size_t N = Determinant::nbits_half;
    String IJb = Ib ^ Jb;
    uint64_t i = IJb.lowest_one_index();
    IJb.clear_lowest_one();
    uint64_t a = IJb.lowest_one_index();

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

double slater_rules_double_alpha_alpha(String Ia, String Ja,
                                       const std::shared_ptr<ActiveSpaceIntegrals>& ints) {

    String IJb = Ia ^ Ja;

    String Ia_sub = Ia & IJb;
    uint64_t i = Ia_sub.lowest_one_index();
    Ia_sub.clear_lowest_one();
    uint64_t j = Ia_sub.lowest_one_index();

    String Ja_sub = Ja & IJb;
    uint64_t k = Ja_sub.lowest_one_index();
    Ja_sub.clear_lowest_one();
    uint64_t l = Ja_sub.lowest_one_index();

    return Ia.slater_sign(i, j) * Ja.slater_sign(k, l) * ints->tei_aa(i, j, k, l);
}

double slater_rules_double_beta_beta(String Ib, String Jb,
                                     const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    String IJb = Ib ^ Jb;

    String Ib_sub = Ib & IJb;
    uint64_t i = Ib_sub.lowest_one_index();
    Ib_sub.clear_lowest_one();
    uint64_t j = Ib_sub.lowest_one_index();

    String Jb_sub = Jb & IJb;
    uint64_t k = Jb_sub.lowest_one_index();
    Jb_sub.clear_lowest_one();
    uint64_t l = Jb_sub.lowest_one_index();

    return Ib.slater_sign(i, j) * Jb.slater_sign(k, l) * ints->tei_bb(i, j, k, l);
}

double slater_rules_double_alpha_beta_pre(int i, int a, String Ib, String Jb,
                                          const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    //    outfile->Printf("\n %zu %zu", Ib, Jb);
    String Ib_xor_Jb = Ib ^ Jb;
    uint64_t j = Ib_xor_Jb.lowest_one_index();
    Ib_xor_Jb.clear_lowest_one();
    uint64_t b = Ib_xor_Jb.lowest_one_index();
    //    outfile->Printf("\n  i = %d, j = %d, a = %d, b = %d", i, j, a, b);
    return Ib.slater_sign(j, b) * ints->tei_ab(i, j, a, b);
}

Determinant common_occupation(const Determinant& lhs, const Determinant& rhs) {
    Determinant result = lhs & rhs;
    return result;
}

Determinant different_occupation(const Determinant& lhs, const Determinant& rhs) {
    //    Determinant result; = lhs ^ rhs;
    return lhs ^ rhs;
}

/// Find the spin orbitals that are occupied only one determinant (performs a bitwise OR, |)
Determinant union_occupation(const Determinant& lhs, const Determinant& rhs) {
    Determinant result = lhs | rhs;
    return result;
}

void enforce_spin_completeness(std::vector<Determinant>& det_space, int nmo) {
    std::unordered_map<Determinant, bool, Determinant::Hash> det_map;
    // Add all determinants to the map, assume set is mostly spin complete
    for (auto& I : det_space) {
        det_map[I] = true;
    }
    // Loop over determinants
    size_t ndet_added = 0;
    std::vector<size_t> closed(nmo, 0);
    std::vector<size_t> open(nmo, 0);
    std::vector<size_t> open_bits(nmo, 0);
    for (size_t I = 0, det_size = det_space.size(); I < det_size; ++I) {
        const Determinant& det = det_space[I];
        // outfile->Printf("\n  Original determinant: %s", det.str().c_str());
        for (int i = 0; i < nmo; ++i) {
            closed[i] = open[i] = 0;
            open_bits[i] = false;
        }
        int naopen = 0;
        int nbopen = 0;
        int nclosed = 0;
        for (int i = 0; i < nmo; ++i) {
            if (det.get_alfa_bit(i) and (not det.get_beta_bit(i))) {
                open[naopen + nbopen] = i;
                naopen += 1;
            } else if ((not det.get_alfa_bit(i)) and det.get_beta_bit(i)) {
                open[naopen + nbopen] = i;
                nbopen += 1;
            } else if (det.get_alfa_bit(i) and det.get_beta_bit(i)) {
                closed[nclosed] = i;
                nclosed += 1;
            }
        }

        if (naopen + nbopen == 0)
            continue;

        // Generate the strings 1111100000
        //                      {nao}{nbo}
        for (int i = 0; i < nbopen; ++i)
            open_bits[i] = false; // 0
        for (int i = nbopen; i < naopen + nbopen; ++i)
            open_bits[i] = true; // 1
        do {
            Determinant new_det;
            for (int c = 0; c < nclosed; ++c) {
                new_det.set_alfa_bit(closed[c], true);
                new_det.set_beta_bit(closed[c], true);
            }
            for (int o = 0; o < naopen + nbopen; ++o) {
                if (open_bits[o]) { //? not
                    new_det.set_alfa_bit(open[o], true);
                } else {
                    new_det.set_beta_bit(open[o], true);
                }
            }
            if (det_map.count(new_det) == 0) {
                det_space.push_back(new_det);
                det_map[new_det] = true;
                // outfile->Printf("\n  added determinant:    %s", new_det.str().c_str());
                ndet_added++;
            }
        } while (std::next_permutation(open_bits.begin(), open_bits.begin() + naopen + nbopen));
    }
    // if( ndet_added > 0 ){
    //    outfile->Printf("\n\n  Determinant space is spin incomplete!");
    //    outfile->Printf("\n  %zu more determinants were needed.", ndet_added);
    //}else{
    //    outfile->Printf("\n\n  Determinant space is spin complete.");
    //}
}

} // namespace forte
