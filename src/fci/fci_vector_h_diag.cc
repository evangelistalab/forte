/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <cmath>
#include <numeric>
#include <vector>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/psi4-dec.h"

#include "helpers/timer.h"
#include "base_classes/mo_space_info.h"
#include "integrals/active_space_integrals.h"
#include "fci_vector.h"
#include "binary_graph.hpp"

using namespace psi;

namespace forte {

void FCIVector::form_H_diagonal(std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    local_timer t;

    int wfn_sym = symmetry_;
    int n = ncmo_;
    int ka = alfa_graph_->nones();
    int kb = beta_graph_->nones();

    bool* Ia = new bool[n];
    bool* Ib = new bool[n];

    // Generate the alfa string 1111000000
    //                          {ka}{n-ka}
    for (int i = 0; i < n - ka; ++i)
        Ia[i] = false; // 0
    for (int i = n - ka; i < n; ++i)
        Ia[i] = true; // 1
    // Loop over all alfa strings
    do {
        // Compute irrep
        int alfa_sym = alfa_graph_->sym(Ia);
        int beta_sym = alfa_sym ^ wfn_sym;

        double** C_ha = C_[alfa_sym]->pointer();

        // Generate the beta string 1111000000
        //                          {kb}{n-kb}
        for (int i = 0; i < n - kb; ++i)
            Ib[i] = false; // 0
        for (int i = n - kb; i < n; ++i)
            Ib[i] = true; // 1
        // Loop over all beta strings
        do {
            // Check if the product of strings gives the right irrep
            if (beta_graph_->sym(Ib) == beta_sym) {
                size_t addIa = alfa_graph_->rel_add(Ia);
                size_t addIb = beta_graph_->rel_add(Ib);
                C_ha[addIa][addIb] = determinant_energy(Ia, Ib, n, fci_ints);
                //        outfile->Printf("\n |[%1d][%3d][%3d]> energy =
                //        %20.12f",alfa_sym,static_cast<int> (addIa),
                //                                                                         static_cast<int>
                //                                                                         (addIb),coefficients[alfa_sym][addIa][addIb]);
            }
        } while (std::next_permutation(Ib, Ib + n));

    } while (std::next_permutation(Ia, Ia + n));

    hdiag_timer += t.get();
    if (print_) {
        outfile->Printf("\n  Timing for Hdiag          = %10.3f s", hdiag_timer);
    }
}

double FCIVector::determinant_energy(bool*& Ia, bool*& Ib, int n,
                                     std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    double energy(fci_ints->scalar_energy() + fci_ints->frozen_core_energy() +
                  fci_ints->nuclear_repulsion_energy());

    for (int p = 0; p < n; ++p) {
        if (Ia[p])
            energy += fci_ints->oei_a(p, p);
        if (Ib[p])
            energy += fci_ints->oei_b(p, p);
        for (int q = 0; q < n; ++q) {
            if (Ia[p] && Ia[q])
                energy += 0.5 * fci_ints->tei_aa(p, q, p, q);
            if (Ib[p] && Ib[q])
                energy += 0.5 * fci_ints->tei_bb(p, q, p, q);
            if (Ia[p] && Ib[q])
                energy += fci_ints->tei_ab(p, q, p, q);
        }
    }
    return (energy);
}

std::vector<std::tuple<double, size_t, size_t, size_t>> FCIVector::min_elements(size_t num_dets) {
    num_dets = std::min(num_dets, ndet_);

    double emax = std::numeric_limits<double>::max();
    size_t added = 0;

    std::vector<std::tuple<double, size_t, size_t, size_t>> dets(num_dets);
    for (auto& det : dets) {
        std::get<0>(det) = emax;
    }

    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t maxIa = alfa_graph_->strpi(alfa_sym);
        size_t maxIb = beta_graph_->strpi(beta_sym);
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
            for (size_t Ib = 0; Ib < maxIb; ++Ib) {
                double e = C_ha[Ia][Ib];
                if ((e < emax) or (added < num_dets)) {
                    // Find where to inser this determinant
                    dets.pop_back();
                    auto it =
                        std::find_if(dets.begin(), dets.end(),
                                     [&e](const std::tuple<double, size_t, size_t, size_t>& t) {
                                         return e < std::get<0>(t);
                                     });
                    dets.insert(it, std::make_tuple(e, alfa_sym, Ia, Ib));
                    emax = std::get<0>(dets.back());
                    added++;
                }
            }
        }
    }
    return dets;
}

std::vector<std::tuple<double, double, size_t, size_t, size_t>>
FCIVector::max_abs_elements(size_t num_dets) {
    num_dets = std::min(num_dets, ndet_);

    std::vector<std::tuple<double, double, size_t, size_t, size_t>> dets(num_dets);

    double emin = 0.0;

    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t maxIa = alfa_graph_->strpi(alfa_sym);
        size_t maxIb = beta_graph_->strpi(beta_sym);
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
            for (size_t Ib = 0; Ib < maxIb; ++Ib) {
                double e = std::fabs(C_ha[Ia][Ib]);
                if (e > emin) {
                    // Find where to inser this determinant
                    dets.pop_back();
                    auto it = std::find_if(
                        dets.begin(), dets.end(),
                        [&e](const std::tuple<double, double, size_t, size_t, size_t>& t) {
                            return e > std::get<0>(t);
                        });
                    dets.insert(it, std::make_tuple(e, C_ha[Ia][Ib], alfa_sym, Ia, Ib));
                    emin = std::get<0>(dets.back());
                }
            }
        }
    }
    return dets;
}

//}

/*
 if(Ia[p]) energy += oei_aa(p,p);
        if(Ib[p]) energy += oei_bb(p,p);
//        if(Ia[p]) outfile->Printf("\n+<%2d|%2d> = %f",p,p,oei_aa(p,p));
//        if(Ib[p]) outfile->Printf("\n+<%2d|%2d> = %f",p,p,oei_bb(p,p));
        for(int q = 0; q < n; ++q){
            if(Ia[p] && Ia[q])
                energy += 0.5 * tei_aaaa(p,q,p,q);
            if(Ib[p] && Ib[q])
                energy += 0.5 * tei_bbbb(p,q,p,q);
            if(Ia[p] && Ib[q])
                energy += tei_aabb(p,q,p,q);
//            if(Ia[p] && Ia[q]) outfile->Printf("\n+<%2d,%2d|%2d,%2d> = %f
(aa)",p,q,p,q,0.5 * tei_aaaa(p,q,p,q));
//            if(Ia[p] && Ib[q]) outfile->Printf("\n+<%2d,%2d|%2d,%2d> = %f
(ab)",p,q,p,q,0.5 * tei_aabb(p,q,p,q));
//            if(Ib[p] && Ib[q]) outfile->Printf("\n+<%2d,%2d|%2d,%2d> = %f
(bb)",p,q,p,q,0.5 * tei_bbbb(p,q,p,q));
        }
    }
    return(energy);
}

double FCIVector::determinant_energy(bool*& Ia,bool*& Ib,int n)
{
//    outfile->Printf("\n  Determinant: ");
//    for(int p = 0; p < n; ++p){
//        outfile->Printf("%1d",Ia[p]);
//    }
//    for(int p = 0; p < n; ++p){
//        outfile->Printf("%1d",Ib[p]);
//    }

    double energy(scalar_energy_ + ints_->frozen_core_energy());

//    outfile->Printf("\n+E0 = %f",energy);
    for(int p = 0; p < n; ++p){
        if(Ia[p]) energy += oei_aa(p,p);
        if(Ib[p]) energy += oei_bb(p,p);
//        if(Ia[p]) outfile->Printf("\n+<%2d|%2d> = %f",p,p,oei_aa(p,p));
//        if(Ib[p]) outfile->Printf("\n+<%2d|%2d> = %f",p,p,oei_bb(p,p));
        for(int q = 0; q < n; ++q){
            if(Ia[p] && Ia[q])
                energy += 0.5 * tei_aaaa(p,q,p,q);
            if(Ib[p] && Ib[q])
                energy += 0.5 * tei_bbbb(p,q,p,q);
            if(Ia[p] && Ib[q])
                energy += tei_aabb(p,q,p,q);
//            if(Ia[p] && Ia[q]) outfile->Printf("\n+<%2d,%2d|%2d,%2d> = %f
(aa)",p,q,p,q,0.5 * tei_aaaa(p,q,p,q));
//            if(Ia[p] && Ib[q]) outfile->Printf("\n+<%2d,%2d|%2d,%2d> = %f
(ab)",p,q,p,q,0.5 * tei_aabb(p,q,p,q));
//            if(Ib[p] && Ib[q]) outfile->Printf("\n+<%2d,%2d|%2d,%2d> = %f
(bb)",p,q,p,q,0.5 * tei_bbbb(p,q,p,q));
        }
    }
    return(energy);
}

*/
} // namespace forte
