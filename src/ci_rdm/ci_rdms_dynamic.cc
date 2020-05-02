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

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"

#include "psi4/libpsi4util/process.h"

#include "helpers/timer.h"
#include "ci_rdms.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/rdms.h"
#include "sparse_ci/determinant.h"

using namespace psi;

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

    SortedStringList a_sorted_string_list_(wfn_, fci_ints_, DetSpinType::Alpha);
    SortedStringList b_sorted_string_list_(wfn_, fci_ints_, DetSpinType::Beta);
    const std::vector<String>& sorted_bstr = b_sorted_string_list_.sorted_half_dets();
    size_t num_bstr = sorted_bstr.size();
    const auto& sorted_b_dets = b_sorted_string_list_.sorted_dets();
    const auto& sorted_a_dets = a_sorted_string_list_.sorted_dets();
    local_timer diag;
    //*-  Diagonal Contributions  -*//
    for (size_t I = 0; I < dim_space_; ++I) {
        size_t Ia = b_sorted_string_list_.add(I);
        double CIa = evecs_->get(Ia, root1_) * evecs_->get(Ia, root2_);
        String det_a = sorted_b_dets[I].get_alfa_bits();
        String det_b = sorted_b_dets[I].get_beta_bits();

        for (size_t nda = 0; nda < na_; ++nda) {
            int p = det_a.find_first_one();
            oprdm_a[p * ncmo_ + p] += CIa;

            String det_ac(det_a);
            det_a.clear_first_one();
            for (size_t ndaa = nda; ndaa < na_; ++ndaa) {
                int q = det_ac.find_first_one();
                // aa 2-rdm
                tprdm_aa[p * ncmo3_ + q * ncmo2_ + p * ncmo_ + q] += CIa;
                tprdm_aa[q * ncmo3_ + p * ncmo2_ + q * ncmo_ + p] += CIa;
                tprdm_aa[p * ncmo3_ + q * ncmo2_ + q * ncmo_ + p] -= CIa;
                tprdm_aa[q * ncmo3_ + p * ncmo2_ + p * ncmo_ + q] -= CIa;

                det_ac.clear_first_one();
                // aaa 3rdm
                String det_acc(det_ac);
                for (size_t ndaaa = ndaa + 1; ndaaa < na_; ++ndaaa) {
                    size_t r = det_acc.find_first_one();
                    fill_3rdm(tprdm_aaa, CIa, p, q, r, p, q, r, true);
                    det_acc.clear_first_one();
                }

                // aab 3rdm
                String det_bc(det_b);
                for (size_t n = 0; n < nb_; ++n) {
                    size_t r = det_bc.find_first_one();
                    tprdm_aab[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] +=
                        CIa;
                    tprdm_aab[p * ncmo5_ + q * ncmo4_ + r * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] -=
                        CIa;
                    tprdm_aab[q * ncmo5_ + p * ncmo4_ + r * ncmo3_ + p * ncmo2_ + q * ncmo_ + r] -=
                        CIa;
                    tprdm_aab[q * ncmo5_ + p * ncmo4_ + r * ncmo3_ + q * ncmo2_ + p * ncmo_ + r] +=
                        CIa;

                    det_bc.clear_first_one();
                }
            }

            String det_bc(det_b);
            for (size_t n = 0; n < nb_; ++n) {
                size_t q = det_bc.find_first_one();
                tprdm_ab[p * ncmo3_ + q * ncmo2_ + p * ncmo_ + q] += CIa;
                det_bc.clear_first_one();
            }
        }
        det_a = sorted_b_dets[I].get_alfa_bits();
        det_b = sorted_b_dets[I].get_beta_bits();
        size_t Ib = a_sorted_string_list_.add(I);
        double CIb = evecs_->get(Ib, root1_) * evecs_->get(Ib, root2_);
        for (size_t ndb = 0; ndb < nb_; ++ndb) {
            int p = det_b.find_first_one();

            // b -1rdm
            oprdm_b[p * ncmo_ + p] += CIb;
            String det_bc(det_b);
            for (size_t ndbb = ndb; ndbb < nb_; ++ndbb) {
                size_t q = det_bc.find_first_one();
                // bb-2rdm
                tprdm_bb[p * ncmo3_ + q * ncmo2_ + p * ncmo_ + q] += CIb;
                tprdm_bb[q * ncmo3_ + p * ncmo2_ + q * ncmo_ + p] += CIb;
                tprdm_bb[p * ncmo3_ + q * ncmo2_ + q * ncmo_ + p] -= CIb;
                tprdm_bb[q * ncmo3_ + p * ncmo2_ + p * ncmo_ + q] -= CIb;
                det_bc.clear_first_one();

                // bbb-3rdm
                String det_bcc(det_bc);
                for (size_t ndbbb = ndbb + 1; ndbbb < nb_; ++ndbbb) {
                    size_t r = det_bcc.find_first_one();
                    fill_3rdm(tprdm_bbb, CIa, p, q, r, p, q, r, true);
                    det_bcc.clear_first_one();
                }

                // abb - 3rdm
                String det_ac(det_a);
                for (size_t n = 0; n < na_; ++n) {
                    size_t r = det_ac.find_first_one();
                    tprdm_abb[r * ncmo5_ + p * ncmo4_ + q * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] +=
                        CIb;
                    tprdm_abb[r * ncmo5_ + p * ncmo4_ + q * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] -=
                        CIb;
                    tprdm_abb[r * ncmo5_ + q * ncmo4_ + p * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -=
                        CIb;
                    tprdm_abb[r * ncmo5_ + q * ncmo4_ + p * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] +=
                        CIb;

                    det_ac.clear_first_one();
                }
            }
            det_b.clear_first_one();
        }
    }
    outfile->Printf("\n  Diag takes %1.6f", diag.get());

    local_timer aaa;
    //-* All Alpha RDMs *-//

    // loop through all beta strings
    for (size_t bstr = 0; bstr < num_bstr; ++bstr) {
        const String& Ib = sorted_bstr[bstr];
        const auto& range_I = b_sorted_string_list_.range(Ib);

        String Ia;
        String Ja;
        size_t first_I = range_I.first;
        size_t last_I = range_I.second;

        // Double loop through determinants with same beta string
        for (size_t I = first_I; I < last_I; ++I) {
            Ia = sorted_b_dets[I].get_alfa_bits();
            double CI = evecs_->get(b_sorted_string_list_.add(I), root1_);
            for (size_t J = I + 1; J < last_I; ++J) {
                Ja = sorted_b_dets[J].get_alfa_bits();
                String IJa = Ia ^ Ja;

                int ndiff = IJa.count();

                if (ndiff == 2) {
                    // 1-rdm
                    String Ia_sub = Ia & IJa;
                    u_int64_t p = Ia_sub.find_first_one();
                    String Ja_sub = Ja & IJa;
                    u_int64_t q = Ja_sub.find_first_one();

                    double Csq = CI * evecs_->get(b_sorted_string_list_.add(J), root2_);
                    double value = Csq * Ia.slater_sign(p, q);
                    oprdm_a[p * ncmo_ + q] += value;
                    oprdm_a[q * ncmo_ + p] += value;

                    // 2-rdm
                    auto Iac = Ia;
                    Iac ^= Ia_sub;
                    for (size_t nbit_a = 1; nbit_a < na_; nbit_a++) {
                        uint64_t m = Iac.find_first_one();
                        tprdm_aa[p * ncmo3_ + m * ncmo2_ + q * ncmo_ + m] += value;
                        tprdm_aa[m * ncmo3_ + p * ncmo2_ + q * ncmo_ + m] -= value;
                        tprdm_aa[m * ncmo3_ + p * ncmo2_ + m * ncmo_ + q] += value;
                        tprdm_aa[p * ncmo3_ + m * ncmo2_ + m * ncmo_ + q] -= value;

                        tprdm_aa[q * ncmo3_ + m * ncmo2_ + p * ncmo_ + m] += value;
                        tprdm_aa[m * ncmo3_ + q * ncmo2_ + p * ncmo_ + m] -= value;
                        tprdm_aa[m * ncmo3_ + q * ncmo2_ + m * ncmo_ + p] += value;
                        tprdm_aa[q * ncmo3_ + m * ncmo2_ + m * ncmo_ + p] -= value;

                        Iac.clear_first_one();

                        auto Ibc = Ib;
                        for (size_t idx = 0; idx < nb_; ++idx) {
                            uint64_t n = Ibc.find_first_one();
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
                            Ibc.clear_first_one();
                        }
                    }
                    auto Ibc = Ib;
                    for (size_t nidx = 0; nidx < nb_; ++nidx) {
                        uint64_t n = Ibc.find_first_one();
                        tprdm_ab[p * ncmo3_ + n * ncmo2_ + q * ncmo_ + n] += value;
                        tprdm_ab[q * ncmo3_ + n * ncmo2_ + p * ncmo_ + n] += value;
                        Ibc.clear_first_one();

                        String Ibcc = Ibc;
                        for (size_t idx = nidx + 1; idx < nb_; ++idx) {
                            uint64_t m = Ibcc.find_first_one();
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
                            Ibcc.clear_first_one();
                        }
                    }
                    // 3-rdm
                    String Iacc = Ia ^ Ia_sub;
                    for (size_t id = 1; id < na_; ++id) {
                        uint64_t n = Iacc.find_first_one();
                        String I_n(Iacc);
                        I_n.clear_first_one(); // TODO: not clear what is going on here (Francesco)
                        for (size_t idd = id + 1; idd < na_; ++idd) {
                            // while( I_n > 0 ){
                            uint64_t m = I_n.find_first_one();
                            fill_3rdm(tprdm_aaa, value, p, n, m, q, n, m, false);
                            I_n.clear_first_one();
                        }
                        Iacc.clear_first_one();
                    }

                } else if (ndiff == 4) {
                    // 2-rdm
                    auto Ia_sub = Ia & IJa;
                    uint64_t p = Ia_sub.find_first_one();
                    Ia_sub.clear_first_one();
                    uint64_t q = Ia_sub.find_first_one();

                    auto Ja_sub = Ja & IJa;
                    uint64_t r = Ja_sub.find_first_one();
                    Ja_sub.clear_first_one();
                    uint64_t s = Ja_sub.find_first_one();

                    double Csq = CI * evecs_->get(b_sorted_string_list_.add(J), root2_);
                    double value = Csq * Ia.slater_sign(p, q) * Ja.slater_sign(r, s);

                    tprdm_aa[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] += value;
                    tprdm_aa[p * ncmo3_ + q * ncmo2_ + s * ncmo_ + r] -= value;
                    tprdm_aa[q * ncmo3_ + p * ncmo2_ + r * ncmo_ + s] -= value;
                    tprdm_aa[q * ncmo3_ + p * ncmo2_ + s * ncmo_ + r] += value;

                    tprdm_aa[r * ncmo3_ + s * ncmo2_ + p * ncmo_ + q] += value;
                    tprdm_aa[s * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= value;
                    tprdm_aa[r * ncmo3_ + s * ncmo2_ + q * ncmo_ + p] -= value;
                    tprdm_aa[s * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += value;

                    // 3-rdm
                    String Iac(Ia);
                    Iac ^= Ia_sub;
                    for (size_t nda = 1; nda < na_; ++nda) {
                        uint64_t n = Iac.find_first_one();
                        fill_3rdm(tprdm_aaa, value, p, q, n, r, s, n, false);
                        Iac.clear_first_one();
                    }

                    String Ibc = Ib;
                    for (size_t ndb = 0; ndb < nb_; ++ndb) {
                        uint64_t n = Ibc.find_first_one();
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
                        Ibc.clear_first_one();
                    }

                } else if (ndiff == 6) {
                    auto Ia_sub = Ia & IJa;
                    uint64_t p = Ia_sub.find_first_one();
                    Ia_sub.clear_first_one();
                    uint64_t q = Ia_sub.find_first_one();
                    Ia_sub.clear_first_one();
                    uint64_t r = Ia_sub.find_first_one();

                    auto Ja_sub = Ja & IJa;
                    uint64_t s = Ja_sub.find_first_one();
                    Ja_sub.clear_first_one();
                    uint64_t t = Ja_sub.find_first_one();
                    Ja_sub.clear_first_one();
                    uint64_t u = Ja_sub.find_first_one();
                    double Csq = CI * evecs_->get(b_sorted_string_list_.add(J), root2_);
                    double el = Csq * Ia.slater_sign(p, q) * Ia.slater_sign(r) *
                                Ja.slater_sign(s, t) * Ja.slater_sign(u);
                    fill_3rdm(tprdm_aaa, el, p, q, r, s, t, u, false);
                }
            }
        }
    }
    outfile->Printf("\n all alpha takes %1.6f", aaa.get());

    //- All beta RDMs -//
    local_timer bbb;
    // loop through all alpha strings
    const std::vector<String>& sorted_astr = a_sorted_string_list_.sorted_half_dets();
    size_t num_astr = sorted_astr.size();
    for (size_t astr = 0; astr < num_astr; ++astr) {
        const String& Ia = sorted_astr[astr];
        const auto& range_I = a_sorted_string_list_.range(Ia);

        String Ib;
        String Jb;
        String IJb;
        size_t first_I = range_I.first;
        size_t last_I = range_I.second;

        // Double loop through determinants with same alpha string
        for (size_t I = first_I; I < last_I; ++I) {
            Ib = sorted_a_dets[I].get_beta_bits();
            double CI = evecs_->get(a_sorted_string_list_.add(I), root1_);
            for (size_t J = I + 1; J < last_I; ++J) {
                Jb = sorted_a_dets[J].get_beta_bits();
                IJb = Ib ^ Jb;
                int ndiff = IJb.count();

                if (ndiff == 2) {
                    auto Ib_sub = Ib & IJb;
                    uint64_t p = Ib_sub.find_first_one();
                    auto Jb_sub = Jb & IJb;
                    uint64_t q = Jb_sub.find_first_one();
                    double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);

                    double value = Csq * Ib.slater_sign(p, q);
                    oprdm_b[p * ncmo_ + q] += value;
                    oprdm_b[q * ncmo_ + p] += value;
                    auto Ibc = Ib;
                    Ibc ^= Ib_sub;
                    for (size_t ndb = 1; ndb < nb_; ++ndb) {
                        uint64_t m = Ibc.find_first_one();
                        tprdm_bb[p * ncmo3_ + m * ncmo2_ + q * ncmo_ + m] += value;
                        tprdm_bb[m * ncmo3_ + p * ncmo2_ + q * ncmo_ + m] -= value;
                        tprdm_bb[m * ncmo3_ + p * ncmo2_ + m * ncmo_ + q] += value;
                        tprdm_bb[p * ncmo3_ + m * ncmo2_ + m * ncmo_ + q] -= value;

                        tprdm_bb[q * ncmo3_ + m * ncmo2_ + p * ncmo_ + m] += value;
                        tprdm_bb[m * ncmo3_ + q * ncmo2_ + p * ncmo_ + m] -= value;
                        tprdm_bb[m * ncmo3_ + q * ncmo2_ + m * ncmo_ + p] += value;
                        tprdm_bb[q * ncmo3_ + m * ncmo2_ + m * ncmo_ + p] -= value;

                        Ibc.clear_first_one();

                        String Iac = Ia;
                        for (size_t idx = 0; idx < na_; ++idx) {
                            uint64_t n = Iac.find_first_one();
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
                            Iac.clear_first_one();
                        }
                    }
                    auto Iac = Ia;
                    for (size_t nidx = 0; nidx < na_; ++nidx) {
                        uint64_t n = Iac.find_first_one();
                        tprdm_ab[n * ncmo3_ + p * ncmo2_ + n * ncmo_ + q] += value;
                        tprdm_ab[n * ncmo3_ + q * ncmo2_ + n * ncmo_ + p] += value;
                        Iac.clear_first_one();

                        auto Iacc = Iac;
                        for (size_t midx = nidx + 1; midx < na_; ++midx) {
                            uint64_t m = Iacc.find_first_one();
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
                            Iacc.clear_first_one();
                        }
                    }
                    // 3-rdm
                    String Ibcc(Ib);
                    Ibcc ^= Ib_sub;
                    for (size_t ndb = 1; ndb < nb_; ++ndb) {
                        // while(Ibcc >0){
                        uint64_t n = Ibcc.find_first_one();
                        Ibcc.clear_first_one();
                        String I_n = Ibcc;
                        for (size_t ndbb = ndb + 1; ndbb < nb_; ++ndbb) {
                            // while( I_n > 0){
                            uint64_t m = I_n.find_first_one();
                            fill_3rdm(tprdm_bbb, value, p, m, n, q, m, n, false);
                            I_n.clear_first_one();
                        }
                    }
                } else if (ndiff == 4) {
                    auto Ib_sub = Ib & IJb;
                    uint64_t p = Ib_sub.find_first_one();
                    Ib_sub.clear_first_one();
                    uint64_t q = Ib_sub.find_first_one();

                    auto Jb_sub = Jb & IJb;
                    uint64_t r = Jb_sub.find_first_one();
                    Jb_sub.clear_first_one();
                    uint64_t s = Jb_sub.find_first_one();

                    double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                    double value = Csq * Ib.slater_sign(p, q) * Jb.slater_sign(r, s);
                    tprdm_bb[p * ncmo3_ + q * ncmo2_ + r * ncmo_ + s] += value;
                    tprdm_bb[p * ncmo3_ + q * ncmo2_ + s * ncmo_ + r] -= value;
                    tprdm_bb[q * ncmo3_ + p * ncmo2_ + r * ncmo_ + s] -= value;
                    tprdm_bb[q * ncmo3_ + p * ncmo2_ + s * ncmo_ + r] += value;

                    tprdm_bb[r * ncmo3_ + s * ncmo2_ + p * ncmo_ + q] += value;
                    tprdm_bb[s * ncmo3_ + r * ncmo2_ + p * ncmo_ + q] -= value;
                    tprdm_bb[r * ncmo3_ + s * ncmo2_ + q * ncmo_ + p] -= value;
                    tprdm_bb[s * ncmo3_ + r * ncmo2_ + q * ncmo_ + p] += value;

                    // 3-rdm
                    auto Ibc = Ib;
                    Ibc ^= Ib_sub;
                    for (size_t ndb = 1; ndb < nb_; ++ndb) {
                        uint64_t n = Ibc.find_first_one();
                        fill_3rdm(tprdm_bbb, value, p, q, n, r, s, n, false);
                        Ibc.clear_first_one();
                    }
                    auto Iac = Ia;
                    for (size_t nda = 0; nda < na_; ++nda) {
                        uint64_t n = Iac.find_first_one();
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

                        Iac.clear_first_one();
                    }
                } else if (ndiff == 6) {
                    auto Ib_sub = Ib & IJb;
                    uint64_t p = Ib_sub.find_first_one();
                    Ib_sub.clear_first_one();
                    uint64_t q = Ib_sub.find_first_one();
                    Ib_sub.clear_first_one();
                    uint64_t r = Ib_sub.find_first_one();

                    auto Jb_sub = Jb & IJb;
                    uint64_t s = Jb_sub.find_first_one();
                    Jb_sub.clear_first_one();
                    uint64_t t = Jb_sub.find_first_one();
                    Jb_sub.clear_first_one();
                    uint64_t u = Jb_sub.find_first_one();
                    double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                    double el = Csq * Ib.slater_sign(p, q) * Ib.slater_sign(r) *
                                Jb.slater_sign(s, t) * Jb.slater_sign(u);
                    fill_3rdm(tprdm_bbb, el, p, q, r, s, t, u, false);
                }
            }
        }
    }
    outfile->Printf("\n all beta takes %1.6f", bbb.get());
    make_ab(a_sorted_string_list_, sorted_astr, sorted_a_dets, tprdm_ab, tprdm_aab, tprdm_abb);
}
//*- Alpha/Beta  -*//
void CI_RDMS::make_ab(SortedStringList a_sorted_string_list_,
                      const std::vector<String>& sorted_astr,
                      const std::vector<Determinant>& sorted_a_dets, std::vector<double>& tprdm_ab,
                      std::vector<double>& tprdm_aab, std::vector<double>& tprdm_abb) {
    local_timer mix;
    double d2 = 0.0;
    double d4 = 0.0;
    for (auto& detIa : sorted_astr) {
        const auto& range_I = a_sorted_string_list_.range(detIa);
        String detIJa_common;
        String Ib;
        String Jb;
        String IJb;
        for (auto& detJa : sorted_astr) {
            detIJa_common = detIa ^ detJa;
            int ndiff = detIJa_common.count();
            if (ndiff == 2) {
                local_timer t2;
                auto Ia_d = detIa & detIJa_common;
                uint64_t p = Ia_d.find_first_one();
                auto Ja_d = detJa & detIJa_common;
                uint64_t s = Ja_d.find_first_one();

                const auto& range_J = a_sorted_string_list_.range(detJa);
                size_t first_I = range_I.first;
                size_t last_I = range_I.second;
                size_t first_J = range_J.first;
                size_t last_J = range_J.second;
                double sign_Ips = detIa.slater_sign(p, s);
                double sign_IJ = detIa.slater_sign(p) * detJa.slater_sign(s);
                for (size_t I = first_I; I < last_I; ++I) {
                    Ib = sorted_a_dets[I].get_beta_bits();
                    double CI = evecs_->get(a_sorted_string_list_.add(I), root1_);
                    for (size_t J = first_J; J < last_J; ++J) {
                        Jb = sorted_a_dets[J].get_beta_bits();
                        IJb = Ib ^ Jb;
                        int nbdiff = IJb.count();
                        if (nbdiff == 2) {
                            double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                            auto Ib_sub = Ib & IJb;
                            uint64_t q = Ib_sub.find_first_one();
                            auto Jb_sub = Jb & IJb;
                            uint64_t r = Jb_sub.find_first_one();

                            double value =
                                Csq * sign_Ips * Ib.slater_sign(q, r); // * ui64_slater_sign(Jb,r);
                            tprdm_ab[p * ncmo3_ + q * ncmo2_ + s * ncmo_ + r] += value;

                            auto Iac(detIa);
                            Iac ^= Ia_d;
                            for (size_t d = 1; d < na_; ++d) {
                                uint64_t n = Iac.find_first_one();
                                tprdm_aab[p * ncmo5_ + n * ncmo4_ + q * ncmo3_ + s * ncmo2_ +
                                          n * ncmo_ + r] += value;
                                tprdm_aab[n * ncmo5_ + p * ncmo4_ + q * ncmo3_ + s * ncmo2_ +
                                          n * ncmo_ + r] -= value;
                                tprdm_aab[n * ncmo5_ + p * ncmo4_ + q * ncmo3_ + n * ncmo2_ +
                                          s * ncmo_ + r] += value;
                                tprdm_aab[p * ncmo5_ + n * ncmo4_ + q * ncmo3_ + n * ncmo2_ +
                                          s * ncmo_ + r] -= value;

                                Iac.clear_first_one();
                            }
                            auto Ibc(Ib);
                            Ibc ^= Ib_sub;
                            for (size_t d = 1; d < nb_; ++d) {
                                uint64_t n = Ibc.find_first_one();
                                tprdm_abb[p * ncmo5_ + q * ncmo4_ + n * ncmo3_ + s * ncmo2_ +
                                          r * ncmo_ + n] += value;
                                tprdm_abb[p * ncmo5_ + q * ncmo4_ + n * ncmo3_ + s * ncmo2_ +
                                          n * ncmo_ + r] -= value;
                                tprdm_abb[p * ncmo5_ + n * ncmo4_ + q * ncmo3_ + s * ncmo2_ +
                                          n * ncmo_ + r] += value;
                                tprdm_abb[p * ncmo5_ + n * ncmo4_ + q * ncmo3_ + s * ncmo2_ +
                                          r * ncmo_ + n] -= value;

                                Ibc.clear_first_one();
                            }
                        } else if (nbdiff == 4) {
                            double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                            auto Ib_sub = Ib & IJb;
                            uint64_t q = Ib_sub.find_first_one();
                            Ib_sub.clear_first_one();
                            uint64_t r = Ib_sub.find_first_one();

                            auto Jb_sub = Jb & IJb;
                            uint64_t t = Jb_sub.find_first_one();
                            Jb_sub.clear_first_one();
                            uint64_t u = Jb_sub.find_first_one();

                            double value = Csq * sign_IJ *
                                           Ib.slater_sign(q, r) * // ui64_slater_sign(Ib,r) *
                                           Jb.slater_sign(t, u);  // * ui64_slater_sign(Jb,u);
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
                auto Ia_sub = detIa & detIJa_common;
                uint64_t p = Ia_sub.find_first_one();
                Ia_sub.clear_first_one();
                uint64_t q = Ia_sub.find_first_one();

                auto Ja_sub = detJa & detIJa_common;
                uint64_t s = Ja_sub.find_first_one();
                Ja_sub.clear_first_one();
                uint64_t t = Ja_sub.find_first_one();

                const auto& range_J = a_sorted_string_list_.range(detJa);
                size_t first_I = range_I.first;
                size_t last_I = range_I.second;
                size_t first_J = range_J.first;
                size_t last_J = range_J.second;

                // double sign = ui64_slater_sign(detIa,p,q) * ui64_slater_sign(detJa,s,t);
                double sign = detIa.slater_sign(p, q) * // ui64_slater_sign(detIa,q) *
                              detJa.slater_sign(s, t);  // ui64_slater_sign(detJa,t);

                // Now the b-b part
                for (size_t I = first_I; I < last_I; ++I) {
                    Ib = sorted_a_dets[I].get_beta_bits();
                    double CI = evecs_->get(a_sorted_string_list_.add(I), root1_);
                    for (size_t J = first_J; J < last_J; ++J) {
                        Jb = sorted_a_dets[J].get_beta_bits();
                        IJb = Ib ^ Jb;
                        int nbdiff = IJb.count();
                        if (nbdiff == 2) {
                            double Csq = CI * evecs_->get(a_sorted_string_list_.add(J), root2_);
                            auto Ib_sub = Ib & IJb;
                            uint64_t r = Ib_sub.find_first_one();
                            auto Jb_sub = Jb & IJb;
                            uint64_t u = Jb_sub.find_first_one();
                            double el = Csq * sign * Ib.slater_sign(r) * Jb.slater_sign(u);

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
} // namespace forte
