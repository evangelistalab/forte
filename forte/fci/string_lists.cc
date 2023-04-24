/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <algorithm>
#include <numeric>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "string_lists.h"

using namespace psi;

namespace forte {

StringLists::StringLists(RequiredLists required_lists, psi::Dimension cmopi,
                         std::vector<size_t> core_mo, std::vector<size_t> cmo_to_mo, size_t na,
                         size_t nb, int print)
    : required_lists_(required_lists), nirrep_(cmopi.n()), ncmo_(cmopi.sum()), cmopi_(cmopi),
      cmo_to_mo_(cmo_to_mo), fomo_to_mo_(core_mo), na_(na), nb_(nb), print_(print) {
    startup();
}

void StringLists::startup() {
    cmopi_offset_.push_back(0);
    for (int h = 1; h < nirrep_; ++h) {
        cmopi_offset_.push_back(cmopi_offset_[h - 1] + cmopi_[h - 1]);
    }

    std::vector<int> cmopi_int;
    for (int h = 0; h < nirrep_; ++h) {
        cmopi_int.push_back(cmopi_[h]);
    }

    for (size_t h = 0; h < nirrep_; h++) {
        fill_n(back_inserter(cmo_sym_), cmopi_int[h], h); // insert h irrep_size[h] times
    }

    // Allocate the alfa and beta graphs
    pair_graph_ = std::make_shared<StringAddress>(ncmo_, 2, cmopi_int);

    if (na_ >= 1) {
        alfa_graph_1h_ = std::make_shared<StringAddress>(ncmo_, na_ - 1, cmopi_int);
    }
    if (nb_ >= 1) {
        beta_graph_1h_ = std::make_shared<StringAddress>(ncmo_, nb_ - 1, cmopi_int);
    }

    if (na_ >= 2) {
        alfa_graph_2h_ = std::make_shared<StringAddress>(ncmo_, na_ - 2, cmopi_int);
    }
    if (nb_ >= 2) {
        beta_graph_2h_ = std::make_shared<StringAddress>(ncmo_, nb_ - 2, cmopi_int);
    }

    if (na_ >= 3) {
        alfa_graph_3h_ = std::make_shared<StringAddress>(ncmo_, na_ - 3, cmopi_int);
    }
    if (nb_ >= 3) {
        beta_graph_3h_ = std::make_shared<StringAddress>(ncmo_, nb_ - 3, cmopi_int);
    }

    nas_ = 0;
    nbs_ = 0;
    for (int h = 0; h < nirrep_; ++h) {
        nas_ += alfa_graph_->strpi(h);
        nbs_ += beta_graph_->strpi(h);
    }

    // local_timers
    double str_list_timer = 0.0;
    double vo_list_timer = 0.0;
    double nn_list_timer = 0.0;
    double oo_list_timer = 0.0;
    double h1_list_timer = 0.0;
    double h2_list_timer = 0.0;
    double h3_list_timer = 0.0;
    double vovo_list_timer = 0.0;
    double vvoo_list_timer = 0.0;

    {
        local_timer t;
        alfa_list_ = make_strings(na_);
        beta_list_ = make_strings(nb_);
        str_list_timer += t.get();
    }
    {
        local_timer t;
        make_pair_list(nn_list);
        nn_list_timer += t.get();
    }
    {
        local_timer t;
        make_vo_list(alfa_graph_, alfa_vo_list);
        make_vo_list(beta_graph_, beta_vo_list);
        vo_list_timer += t.get();
    }
    {
        local_timer t;
        make_oo_list(alfa_graph_, alfa_oo_list);
        make_oo_list(beta_graph_, beta_oo_list);
        oo_list_timer += t.get();
    }
    {
        local_timer t;
        make_1h_list(alfa_graph_, alfa_graph_1h_, alfa_1h_list);
        make_1h_list(beta_graph_, beta_graph_1h_, beta_1h_list);
        h1_list_timer += t.get();
    }
    {
        local_timer t;
        make_2h_list(alfa_graph_, alfa_graph_2h_, alfa_2h_list);
        make_2h_list(beta_graph_, beta_graph_2h_, beta_2h_list);
        h2_list_timer += t.get();
    }
    {
        local_timer t;
        make_3h_list(alfa_graph_, alfa_graph_3h_, alfa_3h_list);
        make_3h_list(beta_graph_, beta_graph_3h_, beta_3h_list);
        h3_list_timer += t.get();
    }
    if (required_lists_ == twoSubstituitionVVOO) {
        local_timer t;
        make_vvoo_list(alfa_graph_, alfa_vvoo_list);
        make_vvoo_list(beta_graph_, beta_vvoo_list);
        vvoo_list_timer += t.get();
    } else if (required_lists_ == twoSubstituitionVOVO) {
        local_timer t;
        make_vovo_list(alfa_graph_, alfa_vovo_list);
        make_vovo_list(beta_graph_, beta_vovo_list);
        vovo_list_timer += t.get();
    }

    double total_time = str_list_timer + nn_list_timer + vo_list_timer + oo_list_timer +
                        vvoo_list_timer + vovo_list_timer;

    if (print_) {
        outfile->Printf("\n\n  ==> String Lists <==\n");
        outfile->Printf("\n  Number of alpha electrons     = %zu", na_);
        outfile->Printf("\n  Number of beta electrons      = %zu", nb_);
        outfile->Printf("\n  Number of alpha strings       = %zu", nas_);
        outfile->Printf("\n  Number of beta strings        = %zu", nbs_);
        // if (na_ >= 3) {
        //     outfile->Printf("\n  Number of alpha strings (N-3) = %zu", alfa_graph_3h_->nstr());
        // }
        // if (nb_ >= 3) {
        //     outfile->Printf("\n  Number of beta strings (N-3)  = %zu", beta_graph_3h_->nstr());
        // }
        outfile->Printf("\n  Timing for strings        = %10.3f s", str_list_timer);
        outfile->Printf("\n  Timing for NN strings     = %10.3f s", nn_list_timer);
        outfile->Printf("\n  Timing for VO strings     = %10.3f s", vo_list_timer);
        outfile->Printf("\n  Timing for OO strings     = %10.3f s", oo_list_timer);
        outfile->Printf("\n  Timing for VVOO strings   = %10.3f s", vvoo_list_timer);
        outfile->Printf("\n  Timing for VOVO strings   = %10.3f s", vovo_list_timer);
        outfile->Printf("\n  Timing for 1-hole strings = %10.3f s", h1_list_timer);
        outfile->Printf("\n  Timing for 2-hole strings = %10.3f s", h2_list_timer);
        outfile->Printf("\n  Timing for 3-hole strings = %10.3f s", h3_list_timer);
        outfile->Printf("\n  Total timing              = %10.3f s", total_time);
    }
}

/**
 * Generate all the pairs p > q with pq in pq_sym
 * these are stored as pair<int,int> in pair_list[pq_sym][pairpi]
 */
void StringLists::make_pair_list(NNList& list) {
    // Loop over irreps of the pair pq
    for (int pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
        list.push_back(std::vector<std::pair<int, int>>(0));
        // Loop over irreps of p
        for (int p_sym = 0; p_sym < nirrep_; ++p_sym) {
            int q_sym = pq_sym ^ p_sym;
            for (int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel) {
                for (int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel) {
                    int p_abs = p_rel + cmopi_offset_[p_sym];
                    int q_abs = q_rel + cmopi_offset_[q_sym];
                    if (p_abs > q_abs)
                        list[pq_sym].push_back(std::make_pair(p_abs, q_abs));
                }
            }
        }
        pairpi_.push_back(list[pq_sym].size());
    }
    //  int h = 0;
    //  foreach(PairList& list_irrep,list){
    //    outfile->Printf("\n Irrep %d",h);
    //    foreach(Pair& pq, list_irrep){
    //
    //      outfile->Printf("\n Pair [%2d,%2d]",pq.first,pq.second);
    //    }
    //    h++;
    //  }
    //
}

StringList StringLists::make_strings(int ne) {
    // number of orbitals and electrons
    const int n = ncmo_;
    const int k = ne;
    auto list = StringList(nirrep_, std::vector<String>());
    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        String I;
        // Generate the strings 1111100000
        //                      { k }{n-k}
        I.zero();
        for (int i = std::max(0, n - k); i < n; ++i)
            I[i] = true;
        do {
            size_t sym_I = I.symmetry(cmo_sym_);
            list[sym_I].push_back(I);
        } while (std::next_permutation(I.begin(), I.begin() + n));
    }
    return list;
}

short StringLists::string_sign(const bool* I, size_t n) {
    short sign = 1;
    for (size_t i = 0; i < n; ++i) { // This runs up to the operator before n
        if (I[i])
            sign *= -1;
    }
    return (sign);
}

void StringLists::print_string(bool* I, size_t n) {
    // outfile->Printf();
    for (size_t i = 0; i < n; ++i) {
        outfile->Printf("%1d", I[i]);
    }
}

} // namespace forte
