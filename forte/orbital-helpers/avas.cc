/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/wavefunction.h"

#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.hpp"

#include "base_classes/scf_info.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"

#include "helpers/helpers.h"
#include "helpers/printing.h"

#include "avas.h"
#include "pao_builder.h"

using namespace psi;

namespace forte {

void make_avas(std::shared_ptr<psi::Wavefunction> ref_wfn, std::shared_ptr<ForteOptions> options,
               std::shared_ptr<psi::Matrix> Ps) {
    print_method_banner({"Atomic Valence Active Space (AVAS)", "Chenxi Cai and Chenyang Li"});

    if (Ps == nullptr) {
        outfile->Printf("Warning: AVAS projector not available!!! Skip AVAS.");
        return;
    }

    int nirrep = Ps->nirrep();
    if (nirrep != ref_wfn->nirrep()) {
        outfile->Printf("\n  Inconsistent number of irreps in AVAS projector", nirrep);
        outfile->Printf(" and reference wavefunction (%d)", ref_wfn->nirrep());
        outfile->Printf("\n  Please try C1 symmetry.");
        throw std::runtime_error("Inconsistent number of irreps in AVAS. Try C1 symmetry?");
    }

    auto nmopi = ref_wfn->nmopi();
    auto doccpi = ref_wfn->doccpi();
    auto soccpi = ref_wfn->soccpi();
    auto uoccpi = nmopi - doccpi - soccpi;
    auto dim_zero = nmopi - nmopi;

    // read options
    bool diagonalize_s = options->get_bool("AVAS_DIAGONALIZE");
    int avas_num_active = options->get_int("AVAS_NUM_ACTIVE");
    int avas_num_active_occ = options->get_int("AVAS_NUM_ACTIVE_OCC");
    int avas_num_active_vir = options->get_int("AVAS_NUM_ACTIVE_VIR");
    double avas_sigma = options->get_double("AVAS_SIGMA");
    double avas_cutoff = options->get_double("AVAS_CUTOFF");
    double nonzero_threshold = options->get_double("AVAS_EVALS_THRESHOLD");

    // AVAS selection scheme for active/inactive orbitals
    enum class AVAS_SELECTION { CUMULATIVE, CUTOFF, ACTV_TOTAL, ACTV_SEPARATE };
    auto avas_selection = AVAS_SELECTION::CUMULATIVE;
    if (avas_num_active_occ + avas_num_active_vir > 0) {
        avas_selection = AVAS_SELECTION::ACTV_SEPARATE;
    } else if (avas_num_active > 0) {
        avas_selection = AVAS_SELECTION::ACTV_TOTAL;
    } else if (1.0 - avas_cutoff > nonzero_threshold) {
        avas_selection = AVAS_SELECTION::CUTOFF;
    }
    std::map<AVAS_SELECTION, std::string> avas_selection_to_string;
    avas_selection_to_string[AVAS_SELECTION::CUMULATIVE] = "SIGMA";
    avas_selection_to_string[AVAS_SELECTION::CUTOFF] = "CUTOFF";
    avas_selection_to_string[AVAS_SELECTION::ACTV_SEPARATE] = "# DOCC/UOCC MOS";
    avas_selection_to_string[AVAS_SELECTION::ACTV_TOTAL] = "# ACTIVE MOS";

    // print options
    table_printer printer;
    printer.add_int_data({{"Number of doubly occupied MOs", doccpi.sum()},
                          {"Number of singly occupied MOs", soccpi.sum()},
                          {"Number of unoccupied MOs", uoccpi.sum()},
                          {"# Active AVAS MOs requested", avas_num_active},
                          {"# Active occupied AVAS MOs requested", avas_num_active_occ},
                          {"# Active virtual AVAS MOs requested", avas_num_active_vir}});
    printer.add_double_data({{"AVAS sigma threshold (cumulative)", avas_sigma},
                             {"AVAS sigma direct cutoff", avas_cutoff},
                             {"Nonzero eigenvalue threshold", nonzero_threshold}});
    printer.add_string_data({{"AVAS selection scheme", avas_selection_to_string[avas_selection]}});
    printer.add_bool_data({{"Diagonalize projected overlap matrices", diagonalize_s}});
    std::string table = printer.get_table("AVAS Options");
    psi::outfile->Printf("%s", table.c_str());

    // compute projected overlap matrix
    auto Ca = ref_wfn->Ca()->clone();
    auto CPsC = psi::linalg::triplet(Ca, Ps, Ca, true, false, false);
    CPsC->set_name("Projected Overlap ( C^+ Ps C )");

    // build orbital rotation matrix
    auto U = std::make_shared<psi::Matrix>("AVAS U", nmopi, nmopi);

    auto sdocc = std::make_shared<psi::Vector>("AVAS sigma docc", doccpi);
    psi::Slice slice_docc(dim_zero, doccpi);

    auto suocc = std::make_shared<psi::Vector>("AVAS sigma uocc", uoccpi);
    psi::Slice slice_uocc(doccpi + soccpi, nmopi);

    if (diagonalize_s) {
        outfile->Printf("\n  Diagonalizing the doubly occupied projected overlap matrix ...");
        auto Udocc = std::make_shared<psi::Matrix>("AVAS Udocc", doccpi, doccpi);
        auto Sdocc = CPsC->get_block(slice_docc, slice_docc);
        Sdocc->diagonalize(Udocc, sdocc, descending);
        U->set_block(slice_docc, slice_docc, Udocc);
        outfile->Printf(" Done");

        outfile->Printf("\n  Diagonalizing the unoccupied projected overlap matrix ........");
        auto Uuocc = std::make_shared<psi::Matrix>("AVAS Uuocc", uoccpi, uoccpi);
        auto Suocc = CPsC->get_block(slice_uocc, slice_uocc);
        Suocc->diagonalize(Uuocc, suocc, descending);
        U->set_block(slice_uocc, slice_uocc, Uuocc);
        outfile->Printf(" Done");
    } else {
        outfile->Printf("\n  Skipping diagonalization of the projector matrix.");
        outfile->Printf("\n  Orbitals will be sorted instead of being rotated.");

        for (int h = 0; h < nirrep; ++h) {
            for (int i = 0; i < doccpi[h]; ++i) {
                sdocc->set(h, i, CPsC->get(h, i, i));
            }

            auto offset = doccpi[h] + soccpi[h];
            for (int a = 0; a < uoccpi[h]; ++a) {
                auto na = a + offset;
                suocc->set(h, a, CPsC->get(h, na, na));
            }
        }

        U->identity();
    }

    // sort MOs according to eigen values of projected overlaps
    // tuple of <sigma, is_occ, irrep, relative_index>
    std::vector<std::tuple<double, bool, int, int>> sorted_mos;

    double s_sum = 0.0; // sum of the eigenvalues (occ + vir)

    for (int h = 0; h < nirrep; ++h) {
        for (int i = 0; i < doccpi[h]; ++i) {
            s_sum += sdocc->get(h, i);
            sorted_mos.push_back({sdocc->get(h, i), true, h, i});
        }

        auto a_offset = doccpi[h] + soccpi[h];
        for (int a = 0; a < uoccpi[h]; ++a) {
            s_sum += suocc->get(h, a);
            sorted_mos.push_back({suocc->get(h, a), false, h, a + a_offset});
        }
    }
    outfile->Printf("\n  Sum of eigenvalues: %.8f", s_sum);

    std::sort(sorted_mos.rbegin(), sorted_mos.rend()); // in descending order

    // partition orbitals to active and inactive
    std::vector<std::vector<int>> Amos_docc(nirrep), Imos_docc(nirrep);
    std::vector<std::vector<int>> Amos_uocc(nirrep), Imos_uocc(nirrep);

    double s_act_sum = 0.0;

    if (avas_selection == AVAS_SELECTION::ACTV_SEPARATE) {
        int counter_Adocc = 0, counter_Auocc = 0;

        for (const auto& mo_tuple : sorted_mos) {
            double sigma;
            bool is_occ;
            int h, idx;
            std::tie(sigma, is_occ, h, idx) = mo_tuple;

            if (is_occ) {
                if (counter_Adocc < avas_num_active_occ) {
                    Amos_docc[h].push_back(idx);
                    counter_Adocc += 1;
                    s_act_sum += sigma;
                } else {
                    Imos_docc[h].push_back(idx);
                }
            } else {
                if (counter_Auocc < avas_num_active_vir) {
                    Amos_uocc[h].push_back(idx);
                    counter_Auocc += 1;
                    s_act_sum += sigma;
                } else {
                    Imos_uocc[h].push_back(idx);
                }
            }
        }
    } else if (avas_selection == AVAS_SELECTION::ACTV_TOTAL) {
        for (int n = 0; n < avas_num_active; ++n) {
            double sigma;
            bool is_occ;
            int h, idx;
            std::tie(sigma, is_occ, h, idx) = sorted_mos[n];

            if (is_occ) {
                Amos_docc[h].push_back(idx);
            } else {
                Amos_uocc[h].push_back(idx);
            }

            s_act_sum += sigma;
        }

        for (int n = avas_num_active, size = sorted_mos.size(); n < size; ++n) {
            bool is_occ;
            int h, idx;
            std::tie(std::ignore, is_occ, h, idx) = sorted_mos[n];

            if (is_occ) {
                Imos_docc[h].push_back(idx);
            } else {
                Imos_uocc[h].push_back(idx);
            }
        }
    } else if (avas_selection == AVAS_SELECTION::CUTOFF) {
        for (const auto& mo_tuple : sorted_mos) {
            double sigma;
            bool is_occ;
            int h, idx;
            std::tie(sigma, is_occ, h, idx) = mo_tuple;

            if (sigma > avas_cutoff and sigma >= nonzero_threshold) {
                if (is_occ) {
                    Amos_docc[h].push_back(idx);
                } else {
                    Amos_uocc[h].push_back(idx);
                }
                s_act_sum += sigma;
            } else {
                if (is_occ) {
                    Imos_docc[h].push_back(idx);
                } else {
                    Imos_uocc[h].push_back(idx);
                }
            }
        }
    } else {
        for (const auto& mo_tuple : sorted_mos) {
            double sigma;
            bool is_occ;
            int h, idx;
            std::tie(sigma, is_occ, h, idx) = mo_tuple;

            // decide if this orbital is active depending on the ratio of the
            // partial sum of singular values and the total sum of singular values
            if ((s_act_sum / s_sum - avas_sigma <= 1.0e-10) and (sigma >= nonzero_threshold)) {
                if (is_occ) {
                    Amos_docc[h].push_back(idx);
                } else {
                    Amos_uocc[h].push_back(idx);
                }
                s_act_sum += sigma;
            } else {
                if (is_occ) {
                    Imos_docc[h].push_back(idx);
                } else {
                    Imos_uocc[h].push_back(idx);
                }
            }
        }
    }
    outfile->Printf("\n  AVAS covers %.2f%% of the subspace.", 100.0 * s_act_sum / s_sum);

    // dimensions of subsets of orbitals
    std::map<std::string, psi::Dimension> avas_dims;
    auto to_dimension = [&](const std::vector<std::vector<int>>& mos) {
        psi::Dimension dim(nirrep);
        for (int h = 0; h < nirrep; ++h) {
            dim[h] = mos[h].size();
        }
        return dim;
    };

    avas_dims["DOCC ACTIVE"] = to_dimension(Amos_docc);
    avas_dims["DOCC INACTIVE"] = to_dimension(Imos_docc);
    avas_dims["UOCC ACTIVE"] = to_dimension(Amos_uocc);
    avas_dims["UOCC INACTIVE"] = to_dimension(Imos_uocc);
    avas_dims["SOCC ACTIVE"] = soccpi;

    avas_dims["RESTRICTED_DOCC"] = avas_dims["DOCC INACTIVE"];
    avas_dims["RESTRICTED_UOCC"] = avas_dims["UOCC INACTIVE"];
    avas_dims["ACTIVE"] = avas_dims["DOCC ACTIVE"] + avas_dims["UOCC ACTIVE"] + soccpi;

    // printing AVAS MO summary
    print_h2("AVAS MOs Information");

    std::string dash(15 + 6 * nirrep, '-');
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    %15s", " ");
    auto irrep_labels = ref_wfn->molecule()->irrep_labels();
    for (int h = 0; h < nirrep; ++h)
        outfile->Printf(" %5s", irrep_labels[h].c_str());
    outfile->Printf("\n    %s", dash.c_str());

    for (const std::string& name :
         {"DOCC INACTIVE", "DOCC ACTIVE", "SOCC ACTIVE", "UOCC ACTIVE", "UOCC INACTIVE"}) {
        outfile->Printf("\n    %-15s", name.c_str());
        for (int h = 0; h < nirrep; ++h) {
            outfile->Printf(" %5d", avas_dims[name][h]);
        }
    }
    outfile->Printf("\n    %s", dash.c_str());

    for (const std::string& name : {"RESTRICTED_DOCC", "ACTIVE", "RESTRICTED_UOCC"}) {
        outfile->Printf("\n    %-15s", name.c_str());
        for (int h = 0; h < nirrep; ++h) {
            outfile->Printf(" %5d", avas_dims[name][h]);
        }
    }
    outfile->Printf("\n    %s", dash.c_str());

    // print all MOs with nonzero overlap
    print_h2("Atomic Valence MOs (Active Marked by *)");

    outfile->Printf("\n    ===============================");
    outfile->Printf("\n     Irrep    MO  Occ.  <phi|P|phi>");
    outfile->Printf("\n    -------------------------------");
    for (int h = 0; h < nirrep; ++h) {
        auto label = irrep_labels[h].c_str();

        std::unordered_set<int> Adocc(Amos_docc[h].begin(), Amos_docc[h].end());
        for (int i = 0; i < doccpi[h]; ++i) {
            double s_value = sdocc->get(h, i);
            if (s_value < nonzero_threshold)
                continue;
            char chosen = Adocc.count(i) ? '*' : ' ';
            outfile->Printf("\n    %c%4s  %5d  %3d  %12.6f", chosen, label, i, 2, s_value);
        }

        auto offset = doccpi[h] + soccpi[h];
        std::unordered_set<int> Auocc(Amos_uocc[h].begin(), Amos_uocc[h].end());
        for (int a = 0; a < uoccpi[h]; ++a) {
            double s_value = suocc->get(h, a);
            if (s_value < nonzero_threshold)
                continue;
            auto na = a + offset;
            char chosen = Auocc.count(na) ? '*' : ' ';
            outfile->Printf("\n    %c%4s  %5d  %3d  %12.6f", chosen, label, na, 0, s_value);
        }
    }
    outfile->Printf("\n    ===============================");

    // rotate orbitals
    auto Ca_tilde = psi::linalg::doublet(Ca, U);

    // canonicalize orbitals and pass to Ca
    auto nsopi = ref_wfn->nsopi();
    psi::Slice so_slice(dim_zero, nsopi);

    auto canonicalize_block = [&](const std::string& title,
                                  const std::vector<std::vector<int>>& mos,
                                  const psi::Slice& slice) {
        outfile->Printf("\n  Canonicalizing orbital block %s %s", title.c_str(),
                        std::string(20 - title.size(), '.').c_str());
        psi::Dimension dim_mo(nirrep);
        for (int h = 0; h < nirrep; ++h) {
            dim_mo[h] = mos[h].size();
        }

        // build sub block of Ca_tilde
        auto Csub = std::make_shared<psi::Matrix>("Csub", nsopi, dim_mo);
        for (int h = 0; h < nirrep; ++h) {
            for (int p = 0; p < dim_mo[h]; ++p) {
                Csub->set_column(h, p, Ca_tilde->get_column(h, mos[h][p]));
            }
        }

        // diagonalize Fock matrix
        auto Fsub = psi::linalg::triplet(Csub, ref_wfn->Fa(), Csub, true, false, false);
        auto Usub = std::make_shared<psi::Matrix>("Usub", dim_mo, dim_mo);
        auto Esub = std::make_shared<psi::Vector>("Esub", dim_mo);
        Fsub->diagonalize(Usub, Esub);

        // rotated orbitals
        auto Csub_r = psi::linalg::doublet(Csub, Usub);

        // pass to Ca
        Ca->set_block(so_slice, slice, Csub_r);
        outfile->Printf(" Done");
    };

    print_h2("Semicanonicalize Subsets of Orbitals");

    psi::Slice slice_Idocc(dim_zero, avas_dims["DOCC INACTIVE"]);
    canonicalize_block("INACTIVE DOCC", Imos_docc, slice_Idocc);

    psi::Slice slice_Adocc(slice_Idocc.end(),
                           avas_dims["DOCC INACTIVE"] + avas_dims["DOCC ACTIVE"]);
    canonicalize_block("ACTIVE DOCC", Amos_docc, slice_Adocc);

    psi::Slice slice_Auocc(slice_Adocc.end() + soccpi,
                           avas_dims["RESTRICTED_DOCC"] + avas_dims["ACTIVE"]);
    canonicalize_block("ACTIVE UOCC", Amos_uocc, slice_Auocc);

    psi::Slice slice_Iuocc(slice_Auocc.end(), nmopi);
    canonicalize_block("INACTIVE UOCC", Imos_uocc, slice_Iuocc);

    // Update both the alpha and beta orbitals assuming restricted orbitals
    ref_wfn->Ca()->copy(Ca);
    ref_wfn->Cb()->copy(Ca);

    // Push to Psi4 environment
    for (const std::string& name : {"RESTRICTED_DOCC", "ACTIVE", "RESTRICTED_UOCC"}) {
        auto array = std::make_shared<psi::Matrix>("AVAS " + name, 1, nirrep);
        for (int h = 0; h < nirrep; ++h) {
            array->set(0, h, avas_dims[name][h]);
        }
        Process::environment.arrays[array->name()] = array;
        ref_wfn->set_array_variable(array->name(), array);
    }
}

} // namespace forte
