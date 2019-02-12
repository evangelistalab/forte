/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <cmath>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/vector.h"

#include "avas.h"

using namespace psi;

namespace forte {

psi::SharedMatrix semicanonicalize_block(psi::SharedWavefunction ref_wfn, psi::SharedMatrix C_tilde,
                                         std::vector<int>& mos, int offset);

void make_avas(psi::SharedWavefunction ref_wfn, psi::Options& options, psi::SharedMatrix Ps) {
    if (Ps) {
        outfile->Printf("\n  Generating AVAS orbitals\n");

        // Allocate a matrix for the occupied block
        int nocc = ref_wfn->nalpha();
        int nso = ref_wfn->nso();
        int nmo = ref_wfn->nmo();
        int nvir = nmo - nocc;
        outfile->Printf("\n  Number of occupied MOs: %6d", nocc);
        outfile->Printf("\n  Number of virtual MOs:  %6d", nvir);
        outfile->Printf("\n");

        auto Socc = std::make_shared<psi::Matrix>("S occupied block", nocc, nocc);
        auto Svir = std::make_shared<psi::Matrix>("S virtual block", nvir, nvir);

        psi::SharedMatrix CPsC = Ps->clone();
        CPsC->transform(ref_wfn->Ca());
        // No diagonalization Socc and Svir
        bool diagonalize_s = options.get_bool("AVAS_DIAGONALIZE");

        auto Uocc = std::make_shared<psi::Matrix>("U occupied block", nocc, nocc);
        auto sigmaocc = std::make_shared<Vector>("sigma occupied block", nocc);

        auto Uvir = std::make_shared<psi::Matrix>("U virtual block", nvir, nvir);
        auto sigmavir = std::make_shared<Vector>("sigma virtual block", nvir);

        auto U = std::make_shared<psi::Matrix>("U", nmo, nmo);

        // diagnolize S
        if (diagonalize_s) {
            // Grab the occupied block and diagonalize it
            for (int i = 0; i < nocc; i++) {
                for (int j = 0; j < nocc; j++) {
                    double value = CPsC->get(i, j);
                    Socc->set(i, j, value);
                }
            }

            Socc->diagonalize(Uocc, sigmaocc, descending);

            // Grab the virtual block and diagonalize it
            for (int a = 0; a < nvir; a++) {
                for (int b = 0; b < nvir; b++) {
                    double value = CPsC->get(nocc + a, nocc + b);
                    Svir->set(a, b, value);
                }
            }

            Svir->diagonalize(Uvir, sigmavir, descending);

            // Form the full matrix U
            for (int i = 0; i < nocc; i++) {
                for (int j = 0; j < nocc; j++) {
                    double value = Uocc->get(i, j);
                    U->set(i, j, value);
                }
            }
            for (int a = 0; a < nvir; a++) {
                for (int b = 0; b < nvir; b++) {
                    double value = Uvir->get(a, b);
                    U->set(a + nocc, b + nocc, value);
                }
            }
        } else {
            // Socc
            for (int i = 0; i < nocc; i++) {
                for (int j = 0; j < nocc; j++) {
                    double value = CPsC->get(i, j);
                    Socc->set(i, j, value);
                }
            }

            for (int i = 0; i < nocc; i++) {
                double value = Socc->get(i, i);
                sigmaocc->set(i, value);
            }
            // Svir
            for (int a = 0; a < nvir; a++) {
                for (int b = 0; b < nvir; b++) {
                    double value = CPsC->get(nocc + a, nocc + b);
                    Svir->set(a, b, value);
                }
            }

            for (int a = 0; a < nvir; a++) {
                double value = Svir->get(a, a);
                sigmavir->set(a, value);
            }
            for (int p = 0; p < nmo; p++) {
                for (int q = 0; q < nmo; q++) {
                    U->set(p, q, p == q ? 1.0 : 0.0);
                }
            }
        } // end options of dia

        auto Ca_tilde = psi::Matrix::doublet(ref_wfn->Ca(), U);

        // sum of the eigenvalues (occ + vir)
        double s_sum = 0.0;
        for (int i = 0; i < nocc; i++) {
            s_sum += sigmaocc->get(i);
        }
        for (int a = 0; a < nvir; a++) {
            s_sum += sigmavir->get(a);
        }
        outfile->Printf("\n  Sum of eigenvalues: %f\n", s_sum);

        std::vector<std::tuple<double, bool, int>> sorted_mos;
        for (int i = 0; i < nocc; i++) {
            sorted_mos.push_back(std::make_tuple(sigmaocc->get(i), true, i));
        }
        for (int a = 0; a < nvir; a++) {
            sorted_mos.push_back(std::make_tuple(sigmavir->get(a), false, a));
        }
        std::sort(sorted_mos.rbegin(), sorted_mos.rend());

        std::vector<int> occ_inact, occ_act, vir_inact, vir_act;

        int avas_num_active = options.get_int("AVAS_NUM_ACTIVE");
        size_t avas_num_active_occ = options.get_int("AVAS_NUM_ACTIVE_OCC");
        size_t avas_num_active_vir = options.get_int("AVAS_NUM_ACTIVE_VIR");
        double avas_sigma = options.get_double("AVAS_SIGMA");

        if (avas_num_active_occ + avas_num_active_vir > 0) {
            for (const auto& mo_tuple : sorted_mos) {
                bool is_occ = std::get<1>(mo_tuple);
                int p = std::get<2>(mo_tuple);
                if (is_occ) {
                    if (occ_act.size() < avas_num_active_occ) {
                        occ_act.push_back(p);
                    } else {
                        occ_inact.push_back(p);
                    }
                } else {
                    if (vir_act.size() < avas_num_active_vir) {
                        vir_act.push_back(p);
                    } else {
                        vir_inact.push_back(p);
                    }
                }
            }
        } else if (avas_num_active > 0) {
            for (int n = 0; n < avas_num_active; ++n) {
                bool is_occ = std::get<1>(sorted_mos[n]);
                int p = std::get<2>(sorted_mos[n]);
                if (is_occ) {
                    occ_act.push_back(p);
                } else {
                    vir_act.push_back(p);
                }
            }
            for (int n = avas_num_active; n < nmo; ++n) {
                bool is_occ = std::get<1>(sorted_mos[n]);
                int p = std::get<2>(sorted_mos[n]);
                if (is_occ) {
                    occ_inact.push_back(p);
                } else {
                    vir_inact.push_back(p);
                }
            }
        } else {
            double s_act_sum = 0.0;
            for (const auto& mo_tuple : sorted_mos) {
                double sigma = std::get<0>(mo_tuple);
                bool is_occ = std::get<1>(mo_tuple);
                int p = std::get<2>(mo_tuple);

                s_act_sum += sigma;
                double fraction = s_act_sum / s_sum;

                // decide if this is orbital is active depending on the ratio of
                // the
                // partial sum of singular values and the total sum of singular
                // values
                if ((fraction < avas_sigma) and (std::fabs(sigma) > 1.0e-6)) {
                    if (is_occ) {
                        occ_act.push_back(p);
                    } else {
                        vir_act.push_back(p);
                    }
                } else {
                    if (is_occ) {
                        occ_inact.push_back(p);
                    } else {
                        vir_inact.push_back(p);
                    }
                }
            }
        }

        outfile->Printf("\n  Number of inactive occupied MOs: %6d", occ_inact.size());
        outfile->Printf("\n  Number of active occupied MOs:   %6d", occ_act.size());
        outfile->Printf("\n  Number of active virtual MOs:    %6d", vir_act.size());
        outfile->Printf("\n  Number of inactive virtual MOs:  %6d", vir_inact.size());
        outfile->Printf("\n");
        outfile->Printf("\n  restricted_docc = [%d]", occ_inact.size());
        outfile->Printf("\n  active          = [%d]", occ_act.size() + vir_act.size());
        outfile->Printf("\n");

        outfile->Printf("\n  Atomic Valence MOs:\n");
        outfile->Printf("    ============================\n");
        outfile->Printf("    Occupation  MO   <phi|P|phi>\n");
        outfile->Printf("    ----------------------------\n");
        for (int i : occ_act) {
            outfile->Printf("      %1d       %4d    %.6f\n", 2, i + 1, sigmaocc->get(i));
        }
        for (int i : vir_act) {
            outfile->Printf("      %1d       %4d    %.6f\n", 0, nocc + i + 1, sigmavir->get(i));
        }
        outfile->Printf("    ============================\n");

        // occupied inactive
        auto Coi = semicanonicalize_block(ref_wfn, Ca_tilde, occ_inact, 0);
        auto Coa = semicanonicalize_block(ref_wfn, Ca_tilde, occ_act, 0);
        auto Cvi = semicanonicalize_block(ref_wfn, Ca_tilde, vir_inact, nocc);
        auto Cva = semicanonicalize_block(ref_wfn, Ca_tilde, vir_act, nocc);

        auto Ca_tilde_prime = std::make_shared<psi::Matrix>("C tilde prime", nso, nmo);

        int offset = 0;
        for (auto& C_block : {Coi, Coa, Cva, Cvi}) {
            int nmo_block = C_block->ncol();
            for (int i = 0; i < nmo_block; ++i) {
                for (int mu = 0; mu < nso; ++mu) {
                    double value = C_block->get(mu, i);
                    Ca_tilde_prime->set(mu, offset, value);
                }
                offset += 1;
            }
        }

        psi::SharedMatrix Fa = ref_wfn->Fa(); // get Fock matrix
        psi::SharedMatrix Fa_mo = Fa->clone();
        Fa_mo->transform(Ca_tilde_prime);

        // Update both the alpha and beta orbitals
        // This assumes a restricted MO set
        // TODO: generalize to unrestricted references
        ref_wfn->Ca()->copy(Ca_tilde_prime);
        ref_wfn->Cb()->copy(Ca_tilde_prime);
    }
}

psi::SharedMatrix semicanonicalize_block(psi::SharedWavefunction ref_wfn, psi::SharedMatrix C_tilde,
                                         std::vector<int>& mos, int offset) {
    int nso = ref_wfn->nso();
    int nmo_block = mos.size();
    auto C_block = std::make_shared<psi::Matrix>("C block", nso, nmo_block);

    int mo_count = 0;
    for (int i : mos) {
        for (int mu = 0; mu < nso; ++mu) {
            double value = C_tilde->get(mu, i + offset);
            C_block->set(mu, mo_count, value);
        }
        mo_count += 1;
    }
    // compute (C_block)^T F C_block
    auto Foi = psi::Matrix::triplet(C_block, ref_wfn->Fa(), C_block, true, false, false);

    auto U_block = std::make_shared<psi::Matrix>("U block", nmo_block, nmo_block);
    auto epsilon_block = std::make_shared<Vector>("epsilon block", nmo_block);
    Foi->diagonalize(U_block, epsilon_block);
    auto C_block_prime = psi::Matrix::doublet(C_block, U_block);
    return C_block_prime;
}
} // namespace forte

// outfile->Printf("\n  Orbital overlap with ao subspace:\n");
// outfile->Printf("    ========================\n");
// outfile->Printf("    Irrep   MO   <phi|P|phi>\n");
// outfile->Printf("    ------------------------\n");
// for (int i = 0; i < nocc; i++) {
//    outfile->Printf("      %1d   %4d    %.6f\n", 2, i + 1,
//                    sigmaocc->get(i));
//}
// for (int i = 0; i < nvir; i++) {
//    outfile->Printf("      %1d   %4d    %.6f\n", 0, nocc + i + 1,
//                    sigmavir->get(i));
//}
// outfile->Printf("    ========================\n");
