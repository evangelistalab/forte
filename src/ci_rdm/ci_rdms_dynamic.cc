/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/process.h"

#include "helpers/timer.h"
#include "ci_rdms.h"
#include "helpers.h"
#include "base_classes/reference.h"
#include "sparse_ci/determinant.h"

namespace psi {
namespace forte {

void CI_RDMS::compute_rdms_dynamic(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b,
                                   std::vector<double>& tprdm_aa, std::vector<double>& tprdm_ab,
                                   std::vector<double>& tprdm_bb, std::vector<double>& tprdm_aaa,
                                   std::vector<double>& tprdm_aab, std::vector<double>& tprdm_abb,
                                   std::vector<double>& tprdm_bbb) {

    oprdm_a.resize(ncmo2_, 0.0);
    oprdm_b.resize(ncmo2_, 0.0);

    tprdm_aa.resize(ncmo4_, 0.0);
    tprdm_ab.resize(ncmo4_, 0.0);
    tprdm_bb.resize(ncmo4_, 0.0);

    tprdm_aaa.resize(ncmo5_ * ncmo_, 0.0);
    tprdm_aab.resize(ncmo5_ * ncmo_, 0.0);
    tprdm_abb.resize(ncmo5_ * ncmo_, 0.0);
    tprdm_bbb.resize(ncmo5_ * ncmo_, 0.0);

    SortedStringList_UI64 a_sorted_string_list_(wfn_, fci_ints_, DetSpinType::Alpha);
    SortedStringList_UI64 b_sorted_string_list_(wfn_, fci_ints_, DetSpinType::Beta);
    const std::vector<UI64Determinant::bit_t>& sorted_bstr =
        b_sorted_string_list_.sorted_half_dets();
    size_t num_bstr = sorted_bstr.size();
    const auto& sorted_b_dets = b_sorted_string_list_.sorted_dets();
    const auto& sorted_a_dets = a_sorted_string_list_.sorted_dets();
    local_timer diag;
    //*-  Diagonal Contributions  -*//
    for (size_t I = 0; I < dim_space_; ++I) {
        size_t Ia = b_sorted_string_list_.add(I);
        double CIa = evecs_->get(Ia, root1_) * evecs_->get(Ia, root2_);
        UI64Determinant::bit_t det_a = sorted_b_dets[I].get_alfa_bits();
        UI64Determinant::bit_t det_b = sorted_b_dets[I].get_beta_bits();

        for (int nda = 0; nda < na_; ++nda) {
            int p = lowest_one_idx(det_a);
            oprdm_a[p * ncmo_ + p] += CIa;

            uint64_t det_ac(det_a);
            det_a = clear_lowest_one(det_a);
            for (int ndaa = nda; ndaa < na_; ++ndaa) {
                int q = lowest_one_idx(det_ac);
                // aa 2-rdm
                tprdm_aa[p * ncmo3_ + q * ncmo2_ + p * ncmo_ + q] += CIa;
                tprdm_aa[q * ncmo3_ + p * ncmo2_ + q * ncmo_ + p] += CIa;
                tprdm_aa[p * ncmo3_ + q * ncmo2_ + q * ncmo_ + p] -= CIa;
                tprdm_aa[q * ncmo3_ + p * ncmo2_ + p * ncmo_ + q] -= CIa;

                det_ac = clear_lowest_one(det_ac);
                // aaa 3rdm
                uint64_t det_acc(det_ac);
                for (int ndaaa = ndaa + 1; ndaaa < na_; ++ndaaa) {
                    int r = lowest_one_idx(det_acc);
                    fill_3rdm(tprdm_aaa, CIa, p, q, r, p, q, r, true);
                    det_acc = clear_lowest_one(det_acc);
                }

                // aab 3rdm
                uint64_t det_bc(det_b);
                for (int n = 0; n < nb_; ++n) {
                    int r = lowest_one_idx(det_bc);
                    tprdm_aab[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] +=
                        CIa;
                    tprdm_aab[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] -=
                        CIa;
                    tprdm_aab[q * ncmo5_ + p * ncmo4_ + r * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -=
                        CIa;
                    tprdm_aab[q * ncmo5_ + p * ncmo4_ + r * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] +=
                        CIa;

                    det_bc = clear_lowest_one(det_bc);
                }
            }

            uint64_t det_bc(det_b);
            for (int n = 0; n < nb_; ++n) {
                int q = lowest_one_idx(det_bc);
                tprdm_ab[p * ncmo3_ + q * ncmo2_ + p * ncmo_ + q] += CIa;
                det_bc = clear_lowest_one(det_bc);
            }
        }
        det_a = sorted_b_dets[I].get_alfa_bits();
        det_b = sorted_b_dets[I].get_beta_bits();
        size_t Ib = a_sorted_string_list_.add(I);
        double CIb = evecs_->get(Ib, root1_) * evecs_->get(Ib, root2_);
        for (int ndb = 0; ndb < nb_; ++ndb) {
            int p = lowest_one_idx(det_b);

            // b -1rdm
            oprdm_b[p * ncmo_ + p] += CIb;
            uint64_t det_bc(det_b);
            for (int ndbb = ndb; ndbb < nb_; ++ndbb) {
                int q = lowest_one_idx(det_bc);
                // bb-2rdm
                tprdm_bb[p * ncmo3_ + q * ncmo2_ + p * ncmo_ + q] += CIb;
                tprdm_bb[q * ncmo3_ + p * ncmo2_ + q * ncmo_ + p] += CIb;
                tprdm_bb[p * ncmo3_ + q * ncmo2_ + q * ncmo_ + p] -= CIb;
                tprdm_bb[q * ncmo3_ + p * ncmo2_ + p * ncmo_ + q] -= CIb;
                det_bc = clear_lowest_one(det_bc);

                // bbb-3rdm
                uint64_t det_bcc(det_bc);
                for (int ndbbb = ndbb + 1; ndbbb < nb_; ++ndbbb) {
                    int r = lowest_one_idx(det_bcc);
                    fill_3rdm(tprdm_bbb, CIa, p, q, r, p, q, r, true);
                    det_bcc = clear_lowest_one(det_bcc);
                }

                // abb - 3rdm
                uint64_t det_ac(det_a);
                for (int n = 0; n < na_; ++n) {
                    int r = lowest_one_idx(det_ac);
                    tprdm_abb[r * ncmo5_ + p * ncmo4_ + q * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] +=
                        CIb;
                    tprdm_abb[r * ncmo5_ + p * ncmo4_ + q * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] -=
                        CIb;
                    tprdm_abb[r * ncmo5_ + q * ncmo4_ + p * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -=
                        CIb;
                    tprdm_abb[r * ncmo5_ + q * ncmo4_ + p * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] +=
                        CIb;

                    det_ac = clear_lowest_one(det_ac);
                }
            }
            det_b = clear_lowest_one(det_b);
        }
    }
    outfile->Printf("\n  Diag takes %1.6f", diag.get());

    local_timer aaa;
    //-* All Alpha RDMs *-//

    // loop through all beta strings
    for (size_t bstr = 0; bstr < num_bstr; ++bstr) {
        const UI64Determinant::bit_t& Ib = sorted_bstr[bstr];
        const auto& range_I = b_sorted_string_list_.range(Ib);

        UI64Determinant::bit_t Ia;
        UI64Determinant::bit_t Ja;
        size_t first_I = range_I.first;
        size_t last_I = range_I.second;

        // Double loop through determinants with same beta string
        for (size_t I = first_I; I < last_I; ++I) {
            Ia = sorted_b_dets[I].get_alfa_bits();
            double CI = evecs_->get(b_sorted_string_list_.add(I), root1_);
            for (size_t J = I + 1; J < last_I; ++J) {
                Ja = sorted_b_dets[J].get_alfa_bits();
                UI64Determinant::bit_t IJa = Ia ^ Ja;

                int ndiff = ui64_bit_count(IJa);

                if (ndiff == 2) {
                    // 1-rdm
                    uint64_t Ia_sub = Ia & IJa;
                    uint64_t p = lowest_one_idx(Ia_sub);
                    uint64_t Ja_sub = Ja & IJa;
                    uint64_t q = lowest_one_idx(Ja_sub);

                    double Csq = CI * evecs_->get(b_sorted_string_list_.add(J), root2_);
                    double value = Csq * ui64_slater_sign(Ia, p, q);
                    oprdm_a[p * ncmo_ + q] += value;
                    oprdm_a[q * ncmo_ + p] += value;

                    // 2-rdm
                    auto Iac = Ia;
                    Iac ^= Ia_sub;
                    for (int nbit_a = 1; nbit_a < na_; nbit_a++) {
                        uint64_t m = lowest_one_idx(Iac);
                        tprdm_aa[p * ncmo3_ + m * ncmo2_ + q * ncmo_ + m] += value;
                        tprdm_aa[m * ncmo3_ + p * ncmo2_ + q * ncmo_ + m] -= value;
                        tprdm_aa[m * ncmo3_ + p * ncmo2_ + m * ncmo_ + q] += value;
                        tprdm_aa[p * ncmo3_ + m * ncmo2_ + m * ncmo_ + q] -= value;

                        tprdm_aa[q * ncmo3_ + m * ncmo2_ + p * ncmo_ + m] += value;
                        tprdm_aa[m * ncmo3_ + q * ncmo2_ + p * ncmo_ + m] -= value;
                        tprdm_aa[m * ncmo3_ + q * ncmo2_ + m * ncmo_ + p] += value;
                        tprdm_aa[q * ncmo3_ + m * ncmo2_ + m * ncmo_ + p] -= value;

                        Iac = clear_lowest_one(Iac);

                        auto Ibc = Ib;
                        for (int idx = 0; idx < nb_; ++idx) {
                            uint64_t n = lowest_one_idx(Ibc);
                            tprdm_aab[p * ncmo5_ + m * ncmo4_ + n * ncmo3_ + q * ncmo2_ +
                                      m * ncmo_ + n] += value;
                            tprdm_aab[p * ncmo5_ + m * ncmo4_ + n * ncmo3_ + m * ncmo2_ +
                                      q * ncmo_ + n] -= value;
                            tprdm_aab[m * ncmo5_ + p * ncmo4_ + n * ncmo3_ + m * ncmo2_ +
                                      q * ncmo_ + n] += value;
                            tprdm_aab[m * ncmo5_ + p * ncmo4_ + n * ncmo3_ + q * ncmo2_ +
                                      m * ncmo_ + n] -= value;

                            tprdm_aab[q * ncmo5_ + m * ncmo4_ + n * ncmo3_ + p * ncmo2_ +
                                      m * ncmo_ + n] += value;
                            tprdm_aab[q * ncmo5_ + m * ncmo4_ + n * ncmo3_ + m * ncmo2_ +
                                      p * ncmo_ + n] -= value;
                            tprdm_aab[m * ncmo5_ + q * ncmo4_ + n * ncmo3_ + m * ncmo2_ +
                                      p * ncmo_ + n] += value;
                            tprdm_aab[m * ncmo5_ + q * ncmo4_ + n * ncmo3_ + p * ncmo2_ +
                                      m * ncmo_ + n] -= value;
                            Ibc = clear_lowest_one(Ibc);
                        }
                    }
                    auto Ibc = Ib;
                    for (int nidx = 0; nidx < nb_; ++nidx) {
                        uint64_t n = lowest_one_idx(Ibc);
                        tprdm_ab[p * ncmo3_ + n * ncmo2_ + q * ncmo_ + n] += value;
                        tprdm_ab[q * ncmo3_ + n * ncmo2_ + p * ncmo_ + n] += value;
                        Ibc = clear_lowest_one(Ibc);

                        uint64_t Ibcc = Ibc;
                        for (int idx = nidx + 1; idx < nb_; ++idx) {
                            uint64_t m = lowest_one_idx(Ibcc);
                            tprdm_abb[p * ncmo5_ + m * ncmo4_ + n * ncmo3_ + q * ncmo2_ +
                                      m * ncmo_ + n] += value;
                            tprdm_abb[p * ncmo5_ + m * ncmo4_ + n * ncmo3_ + q * ncmo2_ +
                                      n * ncmo_ + m] -= value;
                            tprdm_abb[p * ncmo5_ + n * ncmo4_ + m * ncmo3_ + q * ncmo2_ +
                                      n * ncmo_ + m] += value;
                            tprdm_abb[p * ncmo5_ + n * ncmo4_ + m * ncmo3_ + q * ncmo2_ +
                                      m * ncmo_ + n] -= value;

                            tprdm_abb[q * ncmo5_ + m * ncmo4_ + n * ncmo3_ + p * ncmo2_ +
                                      m * ncmo_ + n] += value;
                            tprdm_abb[q * ncmo5_ + m * ncmo4_ + n * ncmo3_ + p * ncmo2_ +
                                      n * ncmo_ + m] -= value;
                            tprdm_abb[q * ncmo5_ + n * ncmo4_ + m * ncmo3_ + p * ncmo2_ +
                                      n * ncmo_ + m] += value;
                            tprdm_abb[q * ncmo5_ + n * ncmo4_ + m * ncmo3_ + p * ncmo2_ +
                                      m * ncmo_ + n] -= value;
                            Ibcc = clear_lowest_one(Ibcc);
                        }
                    }
                    // 3-rdm
                    uint64_t Iacc = Ia ^ Ia_sub;
                    for (int id = 1; id < na_; ++id) {
                        uint64_t n = lowest_one_idx(Iacc);
                        uint64_t I_n = clear_lowest_one(Iacc);
                        for (int idd = id + 1; idd < na_; ++idd) {
                            // while( I_n > 0 ){
                            uint64_t m = lowest_one_idx(I_n);
                            fill_3rdm(tprdm_aaa, value, p, n, m, q, n, m, false);
                            I_n = clear_lowest_one(I_n);
                        }
                        Iacc = clear_lowest_one(Iacc);
                    }

                } else if (ndiff == 4) {
                    // 2-rdm
                    uint64_t Ia_sub = Ia & IJa;
                    uint64_t p = lowest_one_idx(Ia_sub);
                    Ia_sub = clear_lowest_one(Ia_sub);
                    uint64_t q = lowest_one_idx(Ia_sub);

                    uint64_t Ja_sub = Ja & IJa;
                    uint64_t r = lowest_one_idx(Ja_sub);
                    Ja_sub = clear_lowest_one(Ja_sub);
                    uint64_t s = lowest_one_idx(Ja_sub);

                    double Csq = CI * evecs_->get(b_sorted_string_list_.add(J), root2_);
                    double value = Csq * ui64_slater_sign(Ia, p, q) * ui64_slater_sign(Ja, r, s);

                    tprdm_aa[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] += value;
                    tprdm_aa[p * ncmo3_ + q * ncmo2_ + s * ncmo_ + r] -= value;
                    tprdm_aa[q * ncmo3_ + p * ncmo2_ + r * ncmo_ + s] -= value;
                    tprdm_aa[q * ncmo3_ + p * ncmo2_ + s * ncmo_ + r] += value;

                    tprdm_aa[r * ncmo3_ + s * ncmo2_ + p * ncmo_ + q] += value;
                    tprdm_aa[s * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= value;
                    tprdm_aa[r * ncmo3_ + s * ncmo2_ + q * ncmo_ + p] -= value;
                    tprdm_aa[s * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += value;

                    // 3-rdm
                    uint64_t Iac(Ia);
                    Iac ^= Ia_sub;
                    for (int nda = 1; nda < na_; ++nda) {
                        uint64_t n = lowest_one_idx(Iac);
                        fill_3rdm(tprdm_aaa, value, p, q, n, r, s, n, false);
                        Iac = clear_lowest_one(Iac);
                    }

                    uint64_t Ibc = Ib;
                    for (int ndb = 0; ndb < nb_; ++ndb) {
                        uint64_t n = lowest_one_idx(Ibc);
                        tprdm_aab[p * ncmo5_ + q * ncmo4_ + n * ncmo3_ + r * ncmo2_ + s * ncmo_ +
                                  n] += value;
                        tprdm_aab[p * ncmo5_ + q * ncmo4_ + n * ncmo3_ + s * ncmo2_ + r * ncmo_ +
                                  n] -= value;
                        tprdm_aab[q * ncmo5_ + p * ncmo4_ + n * ncmo3_ + s * ncmo2_ + r * ncmo_ +
                                  n] += value;
                        tprdm_aab[q * ncmo5_ + p * ncmo4_ + n * ncmo3_ + r * ncmo2_ + s * ncmo_ +
                                  n] -= value;

                        tprdm_aab[r * ncmo5_ + s * ncmo4_ + n * ncmo3_ + p * ncmo2_ + q * ncmo_ +
                                  n] += value;
                        tprdm_aab[s * ncmo5_ + r * ncmo4_ + n * ncmo3_ + p * ncmo2_ + q * ncmo_ +
                                  n] -= value;
                        tprdm_aab[s * ncmo5_ + r * ncmo4_ + n * ncmo3_ + q * ncmo2_ + p * ncmo_ +
                                  n] += value;
                        tprdm_aab[r * ncmo5_ + s * ncmo4_ + n * ncmo3_ + q * ncmo2_ + p * ncmo_ +
                                  n] -= value;
                        Ibc = clear_lowest_one(Ibc);
                    }

                } else if (ndiff == 6) {
                    uint64_t Ia_sub = Ia & IJa;
                    uint64_t p = lowest_one_idx(Ia_sub);
                    Ia_sub = clear_lowest_one(Ia_sub);
                    uint64_t q = lowest_one_idx(Ia_sub);
                    Ia_sub = clear_lowest_one(Ia_sub);
                    uint64_t r = lowest_one_idx(Ia_sub);

                    uint64_t Ja_sub = Ja & IJa;
                    uint64_t s = lowest_one_idx(Ja_sub);
                    Ja_sub = clear_lowest_one(Ja_sub);
                    uint64_t t = lowest_one_idx(Ja_sub);
                    Ja_sub = clear_lowest_one(Ja_sub);
                    uint64_t u = lowest_one_idx(Ja_sub);
                    double Csq = CI * evecs_->get(b_sorted_string_list_.add(J), root2_);
                    double el = Csq * ui64_slater_sign(Ia, p, q) * ui64_slater_sign(Ia, r) *
                                ui64_slater_sign(Ja, s, t) * ui64_slater_sign(Ja, u);
                    fill_3rdm(tprdm_aaa, el, p, q, r, s, t, u, false);
                }
            }
        }
    }
    outfile->Printf("\n all alpha takes %1.6f", aaa.get());

    //- All beta RDMs -//
    local_timer bbb;
    // loop through all alpha strings
    const std::vector<UI64Determinant::bit_t>& sorted_astr =
        a_sorted_string_list_.sorted_half_dets();
    size_t num_astr = sorted_astr.size();
    for (size_t astr = 0; astr < num_astr; ++astr) {
        const UI64Determinant::bit_t& Ia = sorted_astr[astr];
        const auto& range_I = a_sorted_string_list_.range(Ia);

        UI64Determinant::bit_t Ib;
        UI64Determinant::bit_t Jb;
        UI64Determinant::bit_t IJb;
        size_t first_I = range_I.first;
        size_t last_I = range_I.second;

        // Double loop through determinants with same alpha string
        for (size_t I = first_I; I < last_I; ++I) {
            Ib = sorted_a_dets[I].get_beta_bits();
            double CI = evecs_->get(a_sorted_string_list_.add(I), root1_);
            for (size_t J = I + 1; J < last_I; ++J) {
                Jb = sorted_a_dets[J].get_beta_bits();
                IJb = Ib ^ Jb;
                int ndiff = ui64_bit_count(IJb);

                if (ndiff == 2) {
                    uint64_t Ib_sub = Ib & IJb;
                    uint64_t p = lowest_one_idx(Ib_sub);
                    uint64_t Jb_sub = Jb & IJb;
                    uint64_t q = lowest_one_idx(Jb_sub);
                    double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);

                    double value = Csq * ui64_slater_sign(Ib, p, q);
                    oprdm_b[p * ncmo_ + q] += value;
                    oprdm_b[q * ncmo_ + p] += value;
                    auto Ibc = Ib;
                    Ibc ^= Ib_sub;
                    for (int ndb = 1; ndb < nb_; ++ndb) {
                        uint64_t m = lowest_one_idx(Ibc);
                        tprdm_bb[p * ncmo3_ + m * ncmo2_ + q * ncmo_ + m] += value;
                        tprdm_bb[m * ncmo3_ + p * ncmo2_ + q * ncmo_ + m] -= value;
                        tprdm_bb[m * ncmo3_ + p * ncmo2_ + m * ncmo_ + q] += value;
                        tprdm_bb[p * ncmo3_ + m * ncmo2_ + m * ncmo_ + q] -= value;

                        tprdm_bb[q * ncmo3_ + m * ncmo2_ + p * ncmo_ + m] += value;
                        tprdm_bb[m * ncmo3_ + q * ncmo2_ + p * ncmo_ + m] -= value;
                        tprdm_bb[m * ncmo3_ + q * ncmo2_ + m * ncmo_ + p] += value;
                        tprdm_bb[q * ncmo3_ + m * ncmo2_ + m * ncmo_ + p] -= value;

                        Ibc = clear_lowest_one(Ibc);

                        uint64_t Iac = Ia;
                        for (int idx = 0; idx < na_; ++idx) {
                            uint64_t n = lowest_one_idx(Iac);
                            tprdm_abb[n * ncmo5_ + p * ncmo4_ + m * ncmo3_ + n * ncmo2_ +
                                      q * ncmo_ + m] += value;
                            tprdm_abb[n * ncmo5_ + p * ncmo4_ + m * ncmo3_ + n * ncmo2_ +
                                      m * ncmo_ + q] -= value;
                            tprdm_abb[n * ncmo5_ + m * ncmo4_ + p * ncmo3_ + n * ncmo2_ +
                                      m * ncmo_ + q] += value;
                            tprdm_abb[n * ncmo5_ + m * ncmo4_ + p * ncmo3_ + n * ncmo2_ +
                                      q * ncmo_ + m] -= value;

                            tprdm_abb[n * ncmo5_ + q * ncmo4_ + m * ncmo3_ + n * ncmo2_ +
                                      p * ncmo_ + m] += value;
                            tprdm_abb[n * ncmo5_ + q * ncmo4_ + m * ncmo3_ + n * ncmo2_ +
                                      m * ncmo_ + p] -= value;
                            tprdm_abb[n * ncmo5_ + m * ncmo4_ + q * ncmo3_ + n * ncmo2_ +
                                      m * ncmo_ + p] += value;
                            tprdm_abb[n * ncmo5_ + m * ncmo4_ + q * ncmo3_ + n * ncmo2_ +
                                      p * ncmo_ + m] -= value;
                            Iac = clear_lowest_one(Iac);
                        }
                    }
                    auto Iac = Ia;
                    for (int nidx = 0; nidx < na_; ++nidx) {
                        uint64_t n = lowest_one_idx(Iac);
                        tprdm_ab[n * ncmo3_ + p * ncmo2_ + n * ncmo_ + q] += value;
                        tprdm_ab[n * ncmo3_ + q * ncmo2_ + n * ncmo_ + p] += value;
                        Iac = clear_lowest_one(Iac);

                        auto Iacc = Iac;
                        for (int midx = nidx + 1; midx < na_; ++midx) {
                            uint64_t m = lowest_one_idx(Iacc);
                            tprdm_aab[n * ncmo5_ + m * ncmo4_ + p * ncmo3_ + n * ncmo2_ +
                                      m * ncmo_ + q] += value;
                            tprdm_aab[n * ncmo5_ + m * ncmo4_ + p * ncmo3_ + m * ncmo2_ +
                                      n * ncmo_ + q] -= value;
                            tprdm_aab[m * ncmo5_ + n * ncmo4_ + p * ncmo3_ + m * ncmo2_ +
                                      n * ncmo_ + q] += value;
                            tprdm_aab[m * ncmo5_ + n * ncmo4_ + p * ncmo3_ + n * ncmo2_ +
                                      m * ncmo_ + q] -= value;

                            tprdm_aab[n * ncmo5_ + m * ncmo4_ + q * ncmo3_ + n * ncmo2_ +
                                      m * ncmo_ + p] += value;
                            tprdm_aab[n * ncmo5_ + m * ncmo4_ + q * ncmo3_ + m * ncmo2_ +
                                      n * ncmo_ + p] -= value;
                            tprdm_aab[m * ncmo5_ + n * ncmo4_ + q * ncmo3_ + m * ncmo2_ +
                                      n * ncmo_ + p] += value;
                            tprdm_aab[m * ncmo5_ + n * ncmo4_ + q * ncmo3_ + n * ncmo2_ +
                                      m * ncmo_ + p] -= value;
                            Iacc = clear_lowest_one(Iacc);
                        }
                    }
                    // 3-rdm
                    uint64_t Ibcc(Ib);
                    Ibcc ^= Ib_sub;
                    for (int ndb = 1; ndb < nb_; ++ndb) {
                        // while(Ibcc >0){
                        uint64_t n = lowest_one_idx(Ibcc);
                        Ibcc = clear_lowest_one(Ibcc);
                        uint64_t I_n = Ibcc;
                        for (int ndbb = ndb + 1; ndbb < nb_; ++ndbb) {
                            // while( I_n > 0){
                            uint64_t m = lowest_one_idx(I_n);
                            fill_3rdm(tprdm_bbb, value, p, m, n, q, m, n, false);
                            I_n = clear_lowest_one(I_n);
                        }
                    }
                } else if (ndiff == 4) {
                    uint64_t Ib_sub = Ib & IJb;
                    uint64_t p = lowest_one_idx(Ib_sub);
                    Ib_sub = clear_lowest_one(Ib_sub);
                    uint64_t q = lowest_one_idx(Ib_sub);

                    uint64_t Jb_sub = Jb & IJb;
                    uint64_t r = lowest_one_idx(Jb_sub);
                    Jb_sub = clear_lowest_one(Jb_sub);
                    uint64_t s = lowest_one_idx(Jb_sub);

                    double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                    double value = Csq * ui64_slater_sign(Ib, p, q) * ui64_slater_sign(Jb, r, s);
                    tprdm_bb[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] += value;
                    tprdm_bb[p * ncmo3_ + q * ncmo2_ + s * ncmo_ + r] -= value;
                    tprdm_bb[q * ncmo3_ + p * ncmo2_ + r * ncmo_ + s] -= value;
                    tprdm_bb[q * ncmo3_ + p * ncmo2_ + s * ncmo_ + r] += value;

                    tprdm_bb[r * ncmo3_ + s * ncmo2_ + p * ncmo_ + q] += value;
                    tprdm_bb[s * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= value;
                    tprdm_bb[r * ncmo3_ + s * ncmo2_ + q * ncmo_ + p] -= value;
                    tprdm_bb[s * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += value;

                    // 3-rdm
                    uint64_t Ibc = Ib;
                    Ibc ^= Ib_sub;
                    for (int ndb = 1; ndb < nb_; ++ndb) {
                        uint64_t n = lowest_one_idx(Ibc);
                        fill_3rdm(tprdm_bbb, value, p, q, n, r, s, n, false);
                        Ibc = clear_lowest_one(Ibc);
                    }
                    uint64_t Iac = Ia;
                    for (int nda = 0; nda < na_; ++nda) {
                        uint64_t n = lowest_one_idx(Iac);
                        tprdm_abb[n * ncmo5_ + p * ncmo4_ + q * ncmo3_ + n * ncmo2_ + r * ncmo_ +
                                  s] += value;
                        tprdm_abb[n * ncmo5_ + p * ncmo4_ + q * ncmo3_ + n * ncmo2_ + s * ncmo_ +
                                  r] -= value;
                        tprdm_abb[n * ncmo5_ + q * ncmo4_ + p * ncmo3_ + n * ncmo2_ + s * ncmo_ +
                                  r] += value;
                        tprdm_abb[n * ncmo5_ + q * ncmo4_ + p * ncmo3_ + n * ncmo2_ + r * ncmo_ +
                                  s] -= value;

                        tprdm_abb[n * ncmo5_ + r * ncmo4_ + s * ncmo3_ + n * ncmo2_ + p * ncmo_ +
                                  q] += value;
                        tprdm_abb[n * ncmo5_ + r * ncmo4_ + s * ncmo3_ + n * ncmo2_ + q * ncmo_ +
                                  p] -= value;
                        tprdm_abb[n * ncmo5_ + s * ncmo4_ + r * ncmo3_ + n * ncmo2_ + q * ncmo_ +
                                  p] += value;
                        tprdm_abb[n * ncmo5_ + s * ncmo4_ + r * ncmo3_ + n * ncmo2_ + p * ncmo_ +
                                  q] -= value;

                        Iac = clear_lowest_one(Iac);
                    }
                } else if (ndiff == 6) {
                    uint64_t Ib_sub = Ib & IJb;
                    uint64_t p = lowest_one_idx(Ib_sub);
                    Ib_sub = clear_lowest_one(Ib_sub);
                    uint64_t q = lowest_one_idx(Ib_sub);
                    Ib_sub = clear_lowest_one(Ib_sub);
                    uint64_t r = lowest_one_idx(Ib_sub);

                    uint64_t Jb_sub = Jb & IJb;
                    uint64_t s = lowest_one_idx(Jb_sub);
                    Jb_sub = clear_lowest_one(Jb_sub);
                    uint64_t t = lowest_one_idx(Jb_sub);
                    Jb_sub = clear_lowest_one(Jb_sub);
                    uint64_t u = lowest_one_idx(Jb_sub);
                    double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                    double el = Csq * ui64_slater_sign(Ib, p, q) * ui64_slater_sign(Ib, r) *
                                ui64_slater_sign(Jb, s, t) * ui64_slater_sign(Jb, u);
                    fill_3rdm(tprdm_bbb, el, p, q, r, s, t, u, false);
                }
            }
        }
    }
    outfile->Printf("\n all beta takes %1.6f", bbb.get());
    make_ab(a_sorted_string_list_, sorted_astr, sorted_a_dets, tprdm_ab, tprdm_aab, tprdm_abb);
}
//*- Alpha/Beta  -*//
void CI_RDMS::make_ab(SortedStringList_UI64 a_sorted_string_list_,
                      const std::vector<UI64Determinant::bit_t>& sorted_astr,
                      const std::vector<UI64Determinant>& sorted_a_dets,
                      std::vector<double>& tprdm_ab, std::vector<double>& tprdm_aab,
                      std::vector<double>& tprdm_abb) {
    local_timer mix;
    double d2 = 0.0;
    double d4 = 0.0;
    for (auto& detIa : sorted_astr) {
        const auto& range_I = a_sorted_string_list_.range(detIa);
        UI64Determinant::bit_t detIJa_common;
        UI64Determinant::bit_t Ib;
        UI64Determinant::bit_t Jb;
        UI64Determinant::bit_t IJb;
        for (auto& detJa : sorted_astr) {
            detIJa_common = detIa ^ detJa;
            int ndiff = ui64_bit_count(detIJa_common);
            if (ndiff == 2) {
                local_timer t2;
                uint64_t Ia_d = detIa & detIJa_common;
                uint64_t p = lowest_one_idx(Ia_d);
                uint64_t Ja_d = detJa & detIJa_common;
                uint64_t s = lowest_one_idx(Ja_d);

                const auto& range_J = a_sorted_string_list_.range(detJa);
                size_t first_I = range_I.first;
                size_t last_I = range_I.second;
                size_t first_J = range_J.first;
                size_t last_J = range_J.second;
                double sign_Ips = ui64_slater_sign(detIa, p, s);
                double sign_IJ = ui64_slater_sign(detIa, p) * ui64_slater_sign(detJa, s);
                for (size_t I = first_I; I < last_I; ++I) {
                    Ib = sorted_a_dets[I].get_beta_bits();
                    double CI = evecs_->get(a_sorted_string_list_.add(I), root1_);
                    for (size_t J = first_J; J < last_J; ++J) {
                        Jb = sorted_a_dets[J].get_beta_bits();
                        IJb = Ib ^ Jb;
                        int nbdiff = ui64_bit_count(IJb);
                        if (nbdiff == 2) {
                            double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                            uint64_t Ib_sub = Ib & IJb;
                            uint64_t q = lowest_one_idx(Ib_sub);
                            uint64_t Jb_sub = Jb & IJb;
                            uint64_t r = lowest_one_idx(Jb_sub);

                            double value = Csq * sign_Ips *
                                           ui64_slater_sign(Ib, q, r); // * ui64_slater_sign(Jb,r);
                            tprdm_ab[p * ncmo3_ + q * ncmo2_ + s * ncmo_ + r] += value;

                            uint64_t Iac(detIa);
                            Iac ^= Ia_d;
                            for (int d = 1; d < na_; ++d) {
                                uint64_t n = lowest_one_idx(Iac);
                                tprdm_aab[p * ncmo5_ + n * ncmo4_ + q * ncmo3_ + s * ncmo2_ +
                                          n * ncmo_ + r] += value;
                                tprdm_aab[n * ncmo5_ + p * ncmo4_ + q * ncmo3_ + s * ncmo2_ +
                                          n * ncmo_ + r] -= value;
                                tprdm_aab[n * ncmo5_ + p * ncmo4_ + q * ncmo3_ + n * ncmo2_ +
                                          s * ncmo_ + r] += value;
                                tprdm_aab[p * ncmo5_ + n * ncmo4_ + q * ncmo3_ + n * ncmo2_ +
                                          s * ncmo_ + r] -= value;

                                Iac = clear_lowest_one(Iac);
                            }
                            uint64_t Ibc(Ib);
                            Ibc ^= Ib_sub;
                            for (int d = 1; d < nb_; ++d) {
                                uint64_t n = lowest_one_idx(Ibc);
                                tprdm_abb[p * ncmo5_ + q * ncmo4_ + n * ncmo3_ + s * ncmo2_ +
                                          r * ncmo_ + n] += value;
                                tprdm_abb[p * ncmo5_ + q * ncmo4_ + n * ncmo3_ + s * ncmo2_ +
                                          n * ncmo_ + r] -= value;
                                tprdm_abb[p * ncmo5_ + n * ncmo4_ + q * ncmo3_ + s * ncmo2_ +
                                          n * ncmo_ + r] += value;
                                tprdm_abb[p * ncmo5_ + n * ncmo4_ + q * ncmo3_ + s * ncmo2_ +
                                          r * ncmo_ + n] -= value;

                                Ibc = clear_lowest_one(Ibc);
                            }
                        } else if (nbdiff == 4) {
                            double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                            uint64_t Ib_sub = Ib & IJb;
                            uint64_t q = lowest_one_idx(Ib_sub);
                            Ib_sub = clear_lowest_one(Ib_sub);
                            uint64_t r = lowest_one_idx(Ib_sub);

                            uint64_t Jb_sub = Jb & IJb;
                            uint64_t t = lowest_one_idx(Jb_sub);
                            Jb_sub = clear_lowest_one(Jb_sub);
                            uint64_t u = lowest_one_idx(Jb_sub);

                            double value = Csq * sign_IJ *
                                           ui64_slater_sign(Ib, q, r) * // ui64_slater_sign(Ib,r) *
                                           ui64_slater_sign(Jb, t, u);  // * ui64_slater_sign(Jb,u);
                            tprdm_abb[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ +
                                      t * ncmo_ + u] += value;
                            tprdm_abb[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ +
                                      u * ncmo_ + t] -= value;
                            tprdm_abb[p * ncmo5_ + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ +
                                      u * ncmo_ + t] += value;
                            tprdm_abb[p * ncmo5_ + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ +
                                      t * ncmo_ + u] -= value;
                        }
                    }
                }
                d2 += t2.get();
            } else if (ndiff == 4) {
                local_timer t4;
                // Get aa-aa part of aab 3rdm
                uint64_t Ia_sub = detIa & detIJa_common;
                uint64_t p = lowest_one_idx(Ia_sub);
                Ia_sub = clear_lowest_one(Ia_sub);
                uint64_t q = lowest_one_idx(Ia_sub);

                uint64_t Ja_sub = detJa & detIJa_common;
                uint64_t s = lowest_one_idx(Ja_sub);
                Ja_sub = clear_lowest_one(Ja_sub);
                uint64_t t = lowest_one_idx(Ja_sub);

                const auto& range_J = a_sorted_string_list_.range(detJa);
                size_t first_I = range_I.first;
                size_t last_I = range_I.second;
                size_t first_J = range_J.first;
                size_t last_J = range_J.second;

                // double sign = ui64_slater_sign(detIa,p,q) * ui64_slater_sign(detJa,s,t);
                double sign = ui64_slater_sign(detIa, p, q) * // ui64_slater_sign(detIa,q) *
                              ui64_slater_sign(detJa, s, t);  // ui64_slater_sign(detJa,t);

                // Now the b-b part
                for (size_t I = first_I; I < last_I; ++I) {
                    Ib = sorted_a_dets[I].get_beta_bits();
                    double CI = evecs_->get(a_sorted_string_list_.add(I), root1_);
                    for (size_t J = first_J; J < last_J; ++J) {
                        Jb = sorted_a_dets[J].get_beta_bits();
                        IJb = Ib ^ Jb;
                        int nbdiff = ui64_bit_count(IJb);
                        if (nbdiff == 2) {
                            double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                            uint64_t Ib_sub = Ib & IJb;
                            uint64_t r = lowest_one_idx(Ib_sub);
                            uint64_t Jb_sub = Jb & IJb;
                            uint64_t u = lowest_one_idx(Jb_sub);
                            double el =
                                Csq * sign * ui64_slater_sign(Ib, r) * ui64_slater_sign(Jb, u);

                            tprdm_aab[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ +
                                      t * ncmo_ + u] += el;
                            tprdm_aab[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ +
                                      s * ncmo_ + u] -= el;
                            tprdm_aab[q * ncmo5_ + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ +
                                      t * ncmo_ + u] -= el;
                            tprdm_aab[q * ncmo5_ + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ +
                                      s * ncmo_ + u] += el;
                        }
                    }
                }
                d4 += t4.get();
            }
        }
    }
    outfile->Printf("\n  2dif: %1.6f  \n  4dif: %1.6f", d2, d4);
    outfile->Printf("\n all alpha/beta takes %1.6f", mix.get());
}

void CI_RDMS::fill_3rdm(std::vector<double>& tprdm, double el, int p, int q, int r, int s, int t,
                        int u, bool half) {

    tprdm[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
    tprdm[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
    tprdm[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
    tprdm[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
    tprdm[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
    tprdm[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

    tprdm[p * ncmo5_ + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
    tprdm[p * ncmo5_ + r * ncmo4_ + q * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
    tprdm[p * ncmo5_ + r * ncmo4_ + q * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
    tprdm[p * ncmo5_ + r * ncmo4_ + q * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
    tprdm[p * ncmo5_ + r * ncmo4_ + q * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
    tprdm[p * ncmo5_ + r * ncmo4_ + q * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

    tprdm[q * ncmo5_ + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
    tprdm[q * ncmo5_ + p * ncmo4_ + r * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
    tprdm[q * ncmo5_ + p * ncmo4_ + r * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
    tprdm[q * ncmo5_ + p * ncmo4_ + r * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
    tprdm[q * ncmo5_ + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
    tprdm[q * ncmo5_ + p * ncmo4_ + r * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

    tprdm[q * ncmo5_ + r * ncmo4_ + p * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
    tprdm[q * ncmo5_ + r * ncmo4_ + p * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
    tprdm[q * ncmo5_ + r * ncmo4_ + p * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
    tprdm[q * ncmo5_ + r * ncmo4_ + p * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
    tprdm[q * ncmo5_ + r * ncmo4_ + p * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
    tprdm[q * ncmo5_ + r * ncmo4_ + p * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

    tprdm[r * ncmo5_ + p * ncmo4_ + q * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] += el;
    tprdm[r * ncmo5_ + p * ncmo4_ + q * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] -= el;
    tprdm[r * ncmo5_ + p * ncmo4_ + q * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] -= el;
    tprdm[r * ncmo5_ + p * ncmo4_ + q * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] += el;
    tprdm[r * ncmo5_ + p * ncmo4_ + q * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] -= el;
    tprdm[r * ncmo5_ + p * ncmo4_ + q * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] += el;

    tprdm[r * ncmo5_ + q * ncmo4_ + p * ncmo3_ + s * ncmo2_ + t * ncmo_ + u] -= el;
    tprdm[r * ncmo5_ + q * ncmo4_ + p * ncmo3_ + s * ncmo2_ + u * ncmo_ + t] += el;
    tprdm[r * ncmo5_ + q * ncmo4_ + p * ncmo3_ + u * ncmo2_ + t * ncmo_ + s] += el;
    tprdm[r * ncmo5_ + q * ncmo4_ + p * ncmo3_ + u * ncmo2_ + s * ncmo_ + t] -= el;
    tprdm[r * ncmo5_ + q * ncmo4_ + p * ncmo3_ + t * ncmo2_ + s * ncmo_ + u] += el;
    tprdm[r * ncmo5_ + q * ncmo4_ + p * ncmo3_ + t * ncmo2_ + u * ncmo_ + s] -= el;

    if (!half) {
        tprdm[s * ncmo5_ + t * ncmo4_ + u * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] += el;
        tprdm[s * ncmo5_ + u * ncmo4_ + t * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -= el;
        tprdm[u * ncmo5_ + t * ncmo4_ + s * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -= el;
        tprdm[u * ncmo5_ + s * ncmo4_ + t * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] += el;
        tprdm[t * ncmo5_ + s * ncmo4_ + u * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -= el;
        tprdm[t * ncmo5_ + u * ncmo4_ + s * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] += el;

        tprdm[s * ncmo5_ + t * ncmo4_ + u * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] -= el;
        tprdm[s * ncmo5_ + u * ncmo4_ + t * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] += el;
        tprdm[u * ncmo5_ + t * ncmo4_ + s * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] += el;
        tprdm[u * ncmo5_ + s * ncmo4_ + t * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] -= el;
        tprdm[t * ncmo5_ + s * ncmo4_ + u * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] += el;
        tprdm[t * ncmo5_ + u * ncmo4_ + s * ncmo3_ + p * ncmo2_ + r * ncmo_ + q] -= el;

        tprdm[s * ncmo5_ + t * ncmo4_ + u * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] -= el;
        tprdm[s * ncmo5_ + u * ncmo4_ + t * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] += el;
        tprdm[u * ncmo5_ + t * ncmo4_ + s * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] += el;
        tprdm[u * ncmo5_ + s * ncmo4_ + t * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] -= el;
        tprdm[t * ncmo5_ + s * ncmo4_ + u * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] += el;
        tprdm[t * ncmo5_ + u * ncmo4_ + s * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] -= el;

        tprdm[s * ncmo5_ + t * ncmo4_ + u * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] += el;
        tprdm[s * ncmo5_ + u * ncmo4_ + t * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] -= el;
        tprdm[u * ncmo5_ + t * ncmo4_ + s * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] -= el;
        tprdm[u * ncmo5_ + s * ncmo4_ + t * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] += el;
        tprdm[t * ncmo5_ + s * ncmo4_ + u * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] -= el;
        tprdm[t * ncmo5_ + u * ncmo4_ + s * ncmo3_ + q * ncmo2_ + r * ncmo_ + p] += el;

        tprdm[s * ncmo5_ + t * ncmo4_ + u * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] += el;
        tprdm[s * ncmo5_ + u * ncmo4_ + t * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= el;
        tprdm[u * ncmo5_ + t * ncmo4_ + s * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= el;
        tprdm[u * ncmo5_ + s * ncmo4_ + t * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] += el;
        tprdm[t * ncmo5_ + s * ncmo4_ + u * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= el;
        tprdm[t * ncmo5_ + u * ncmo4_ + s * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] += el;

        tprdm[s * ncmo5_ + t * ncmo4_ + u * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] -= el;
        tprdm[s * ncmo5_ + u * ncmo4_ + t * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += el;
        tprdm[u * ncmo5_ + t * ncmo4_ + s * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += el;
        tprdm[u * ncmo5_ + s * ncmo4_ + t * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] -= el;
        tprdm[t * ncmo5_ + s * ncmo4_ + u * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += el;
        tprdm[t * ncmo5_ + u * ncmo4_ + s * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] -= el;
    }
}

// void CI_RDMS::compute_2rdm_dynamic(std::vector<double>& tprdm_aa,
//                                   std::vector<double>& tprdm_ab,
//                                   std::vector<double>& tprdm_bb){
//
//}
// void CI_RDMS::compute_3rdm_dynamic(std::vector<double>& tprdm_aaa,
//                                   std::vector<double>& tprdm_aab,
//                                   std::vector<double>& tprdm_abb,
//                                   std::vector<double>& tprdm_bbb){
//
//}
}
}
