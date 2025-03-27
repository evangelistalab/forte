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

#include "aset.h"
#include "pao_builder.h"

using namespace psi;

namespace forte {

std::shared_ptr<psi::Matrix> semicanonicalize_block(psi::SharedWavefunction ref_wfn,
                                                    std::shared_ptr<psi::Matrix> C_tilde,
                                                    std::vector<int>& mos, int offset,
                                                    bool prevent_rotate = false);

std::shared_ptr<MOSpaceInfo> make_embedding(psi::SharedWavefunction ref_wfn,
                                            std::shared_ptr<SCFInfo> scf_info,
                                            std::shared_ptr<ForteOptions> options,
                                            std::shared_ptr<psi::Matrix> Pf, int nbf_A,
                                            std::shared_ptr<MOSpaceInfo> mo_space_info) {

    // 1. Get necessary information, print method initialization information and exceptions
    double thresh = options->get_double("EMBEDDING_THRESHOLD");
    if (thresh > 1.0 || thresh < 0.0) {
        throw std::invalid_argument(
            "make_embedding: Embedding threshold must be between 0.0 and 1.0 !");
    }

    int A_docc = 0;
    int A_uocc = 0;

    if (options->get_str("EMBEDDING_CUTOFF_METHOD") == "THRESHOLD") {
        print_h2("Orbital partition done according to simple threshold");
        outfile->Printf("\n  Simple threshold t = %8.8f", thresh);
    } else if (options->get_str("EMBEDDING_CUTOFF_METHOD") == "CUM_THRESHOLD") {
        print_h2("Orbital partition done according to cumulative threshold");
        outfile->Printf("\n  Cumulative threshold t = %8.8f", thresh);
    } else if (options->get_str("EMBEDDING_CUTOFF_METHOD") == "NUM_OF_ORBITALS") {
        print_h2(
            "Orbital partition done according to fixed number of occupied and virtual orbitals");
        A_docc = options->get_int("NUM_A_DOCC");
        A_uocc = options->get_int("NUM_A_UOCC");
        outfile->Printf("\n  Number of A occupied/virtual MOs set to %d and %d\n", A_docc, A_uocc);
    } else {
        throw std::invalid_argument("make_embedding: Impossible embedding cutoff method!");
    }

    if (not ref_wfn) {
        throw std::runtime_error("make_embedding: SCF has not been run yet!");
    }

    if (not Pf) {
        throw std::runtime_error("make_embedding: No projector (matrix) found!");
    }

    int nirrep = ref_wfn->nirrep();
    if (nirrep > 1) {
        throw std::invalid_argument(
            "Fragment projection works only without symmetry! (symmetry C1)");
    }

    // Additional input parameters used to control numbers of orbitals in A/B space
    int adj_sys_docc = options->get_int("EMBEDDING_ADJUST_B_DOCC");
    int adj_sys_uocc = options->get_int("EMBEDDING_ADJUST_B_UOCC");

    std::shared_ptr<PSIO> psio(_default_psio_lib_);

    const Dimension nmopi = ref_wfn->nmopi();
    Dimension zeropi(nirrep);

    // 2. Apply projector to rotate the orbitals

    // Get information of rocc, actv and rvir from MOSpaceInfo
    Dimension frzopi = mo_space_info->dimension("FROZEN_DOCC");
    Dimension nroccpi = mo_space_info->dimension("RESTRICTED_DOCC");
    Dimension actv_a = mo_space_info->dimension("ACTIVE");
    Dimension nrvirpi = mo_space_info->dimension("RESTRICTED_UOCC");
    Dimension frzvpi = mo_space_info->dimension("FROZEN_UOCC");

    // When doing single-reference, put actv to 0
    Dimension doc = ref_wfn->doccpi();
    int diff = doc[0] - frzopi[0] - nroccpi[0];
    int diff2 = actv_a[0] - diff;

    if (options->get_str("EMBEDDING_REFERENCE") == "HF") {
        nroccpi[0] += diff;
        nrvirpi[0] += diff2;
        actv_a[0] = 0;
    }

    // Define corresponding blocks (slices), occ slick will start at frzopi
    Slice occ(frzopi, nroccpi + frzopi);
    Slice vir(frzopi + nroccpi + actv_a, nmopi - frzvpi);
    Slice actv(frzopi + nroccpi, frzopi + nroccpi + actv_a);
    Slice mo(zeropi, nmopi);

    // Save original orbitals for frozen and active orbital reconstruction
    SharedMatrix Ca_ori = ref_wfn->Ca();
    SharedMatrix Ca_save = Ca_ori->clone();

    // Transform Pf to MO basis
    Pf->transform(Ca_ori);

    // Diagonalize Pf_pq for occ and vir space, respectively.
    SharedMatrix P_oo = Pf->get_block(occ, occ);
    auto Uo = std::make_shared<psi::Matrix>("Uo", nirrep, nroccpi, nroccpi);
    auto lo = std::make_shared<Vector>("lo", nroccpi);
    P_oo->diagonalize(Uo, lo, descending);

    auto Uv = std::make_shared<psi::Matrix>("Uv", nirrep, nrvirpi, nrvirpi);
    auto lv = std::make_shared<Vector>("lv", nrvirpi);
    SharedMatrix P_vv = Pf->get_block(vir, vir);
    P_vv->diagonalize(Uv, lv, descending);

    auto U_all = std::make_shared<psi::Matrix>("U with Pab", nirrep, nmopi, nmopi);
    U_all->set_block(occ, occ, Uo);
    U_all->set_block(vir, vir, Uv);

    // Rotate MOs (This rotation will zero frozen and active space)
    ref_wfn->Ca()->copy(psi::linalg::doublet(Ca_ori, U_all, false, false));

    // Based on threshold or num_occ/num_vir, decide the partition
    std::vector<int> index_frozen_core = {};
    std::vector<int> index_frozen_virtual = {};

    std::vector<int> index_A_occ = {};
    std::vector<int> index_A_vir = {};
    std::vector<int> index_B_occ = {};
    std::vector<int> index_B_vir = {};
    std::vector<int> index_actv = {};

    // Create the active orbital index vector (for any reference)
    for (int i = 0; i < actv_a[0]; ++i) {
        index_actv.push_back(frzopi[0] + nroccpi[0] + i);
    }

    int offset_vec = frzopi[0] + nroccpi[0] + actv_a[0];

    // Create frozen orbital index vectors
    if (frzopi[0] != 0) {
        for (int i = 0; i < frzopi[0]; ++i) {
            index_frozen_core.push_back(i);
        }
    }

    if (frzvpi[0] != 0) {
        for (int i = offset_vec + nrvirpi[0]; i < nmopi[0]; ++i) {
            index_frozen_virtual.push_back(i);
        }
    }

    // Create ro and rv orbital index vectors
    if (options->get_str("EMBEDDING_CUTOFF_METHOD") == "THRESHOLD") {
        for (int i = 0; i < nroccpi[0]; i++) {
            if (lo->get(0, i) > thresh) {
                index_A_occ.push_back(i + frzopi[0]);
            } else {
                index_B_occ.push_back(i + frzopi[0]);
            }
        }
        for (int i = 0; i < nrvirpi[0]; i++) {
            if (lv->get(0, i) > thresh) {
                index_A_vir.push_back(i + offset_vec);
            } else {
                index_B_vir.push_back(i + offset_vec);
            }
        }
    }

    if (options->get_str("EMBEDDING_CUTOFF_METHOD") == "NUM_OF_ORBITALS") {
        for (int i = 0; i < nroccpi[0]; i++) {
            if (i < A_docc) {
                index_A_occ.push_back(i + frzopi[0]);
            } else {
                index_B_occ.push_back(i + frzopi[0]);
                if (lo->get(0, i) > 0.5) {
                    outfile->Printf("\n Warning! Occupied orbital %d have eigenvalue (%8.8f) "
                                    "larger than 0.5 are partitioned to B!",
                                    i + frzopi[0], lo->get(0, i));
                }
            }
        }
        for (int i = 0; i < nrvirpi[0]; i++) {
            if (i < A_uocc) {
                index_A_vir.push_back(i + offset_vec);
            } else {
                index_B_vir.push_back(i + offset_vec);
                if (lv->get(0, i) > 0.5) {
                    outfile->Printf("\n  Warning! Virtual orbital %d has a large overlap with the "
                                    "system (%8.8f) but was assigned to the environment (B).\n",
                                    i + offset_vec, lv->get(0, i));
                }
            }
        }
    }

    if (options->get_str("EMBEDDING_CUTOFF_METHOD") == "CUM_THRESHOLD") {
        double tmp = 0.0;
        double sum_lo = 0.0;
        double sum_lv = 0.0;
        for (int i = 0; i < nroccpi[0]; i++) {
            sum_lo += lo->get(0, i);
        }
        for (int i = 0; i < nrvirpi[0]; i++) {
            sum_lv += lv->get(0, i);
        }

        double cum_l_o = 0.0;
        for (int i = 0; i < nroccpi[0]; i++) {
            tmp += lo->get(0, i);
            cum_l_o = tmp / sum_lo;
            if (cum_l_o < thresh) {
                index_A_occ.push_back(i + frzopi[0]);
            } else {
                index_B_occ.push_back(i + frzopi[0]);
                if (lo->get(0, i) > 0.5) {
                    outfile->Printf("\n Warning! Occupied orbital %d have eigenvalue (%8.8f) "
                                    "larger than 0.5 are partitioned to B!",
                                    i + frzopi[0], lo->get(0, i));
                }
            }
        }
        tmp = 0.0;
        double cum_l_v = 0.0;
        for (int i = 0; i < nrvirpi[0]; i++) {
            tmp += lv->get(0, i);
            cum_l_v = tmp / sum_lv;
            if (cum_l_v < thresh) {
                index_A_vir.push_back(i + offset_vec);
            } else {
                index_B_vir.push_back(i + offset_vec);
                if (lv->get(0, i) > 0.5) {
                    outfile->Printf("\n Warning! Virtual orbital %d have eigenvalue (%8.8f) "
                                    "larger than 0.5 are partitioned to B!",
                                    i + offset_vec, lv->get(0, i));
                }
            }
        }
    }

    if (options->get_str("EMBEDDING_VIRTUAL_SPACE") == "PAO") {
        outfile->Printf("\n ****** Build PAOs for virtual space ******");

        // Call build_PAOs
        double tau = options->get_double("PAO_THRESHOLD");
        PAObuilder pao(Ca_save, frzopi + nroccpi + actv_a, ref_wfn->basisset());

        bool fix_number = options->get_bool("PAO_FIX_VIRTUAL_NUMBER");

        // ref_wfn->Ca()->print();
        outfile->Printf("\n ****** Update C_vir ******");
        // Write Ca
        SharedMatrix C_pao = pao.build_A_virtual(nbf_A, tau);

        int n_pao = 0;
        if (!fix_number) {
            n_pao = C_pao->ncol();
        } else {
            n_pao = index_A_vir.size();
            ; // Fix virtual space to be equal to ASET partitioned virtual
        }

        Dimension nshortpi = nmopi;
        nshortpi[0] = offset_vec + n_pao;
        Dimension vir_start = nmopi;
        vir_start[0] = offset_vec;

        Slice dvir(vir_start, nshortpi);

        if (!fix_number) {
            ref_wfn->Ca()->set_block(mo, dvir, C_pao);
        } else {
            Dimension npao_fix = nmopi;
            npao_fix[0] = n_pao;
            Slice pao_fix(zeropi, npao_fix);
            SharedMatrix C_pao_fix = C_pao->get_block(mo, pao_fix);
            ref_wfn->Ca()->set_block(mo, dvir, C_pao_fix);
        }

        // B_vir ignored for now

        outfile->Printf("\n ****** Create index lists ******");
        // Form new index_A_vir and index_B_vir
        index_A_vir.clear();
        index_B_vir.clear();

        // Number of virtual PAOs should be PAOs - active_virtual (?)
        for (int i = 0; i < nrvirpi[0]; i++) {
            if (i < n_pao) {
                index_A_vir.push_back(i + offset_vec);
                outfile->Printf("\n Orbital %d built as PAO", i + offset_vec + 1);
            } else {
                index_B_vir.push_back(i + offset_vec);
            }
        }

        outfile->Printf("\n ****** PAOs done ******");
    }

    // Collect the size of each space
    int num_Fo = index_frozen_core.size();
    int num_Ao = index_A_occ.size();
    int num_Bo = index_B_occ.size();
    int num_Av = index_A_vir.size();
    int num_Bv = index_B_vir.size();
    int num_Fv = index_frozen_virtual.size();

    // Print system orbital information
    outfile->Printf("\n    Frozen-orbital Embedding MOs (System A)\n");
    outfile->Printf("    ============================\n");
    outfile->Printf("      MO     Type    <phi|P|phi>\n");
    outfile->Printf("    ----------------------------\n");
    for (int i : index_A_occ) {
        outfile->Printf("    %4d   %8s   %.6f\n", i + 1, "Occupied", lo->get(i - frzopi[0]));
    }
    for (int i : index_actv) {
        outfile->Printf("    %4d   %8s      --\n", i + 1, "Active");
    }
    if (options->get_str("EMBEDDING_VIRTUAL_SPACE") == "ASET") {
        for (int i : index_A_vir) {
            outfile->Printf("    %4d   %8s   %.6f\n", i + 1, "Virtual", lv->get(i - offset_vec));
        }
    }
    outfile->Printf("    ============================\n");

    // If less than 50 frozen environment orbitals, print the environment orbital
    if (num_Bo + num_Bv < 50) {
        outfile->Printf("\n    Frozen-orbital Embedding MOs (Environment B)\n");
        outfile->Printf("    ============================\n");
        outfile->Printf("      MO     Type    <phi|P|phi>\n");
        outfile->Printf("    ----------------------------\n");
        for (int i : index_B_occ) {
            outfile->Printf("    %4d   %8s   %.6f\n", i + 1, "Occupied", lo->get(i - frzopi[0]));
        }
        for (int i : index_B_vir) {
            outfile->Printf("    %4d   %8s   %.6f\n", i + 1, "Virtual", lv->get(i - offset_vec));
        }
        outfile->Printf("    ============================\n");
    } else {
        outfile->Printf(
            "\n    Frozen-orbital Embedding MOs (Environment B) more than 50, no printing. \n");
    }

    outfile->Printf("\n  Summary: ");
    if (options->get_str("EMBEDDING_VIRTUAL_SPACE") == "ASET") {
        outfile->Printf("\n    System (A): %d Occupied MOs, %d Active MOs, %d Virtual MOs", num_Ao,
                        actv_a[0], num_Av);
    }
    if (options->get_str("EMBEDDING_VIRTUAL_SPACE") == "PAO") {
        outfile->Printf("\n    System (A): %d Occupied MOs, %d Active MOs, %d orthogonalized PAOs",
                        num_Ao, actv_a[0], num_Av);
    }
    outfile->Printf("\n    Environment (B): %d Occupied MOs, %d Virtual MOs", num_Bo, num_Bv);
    outfile->Printf("\n    Frozen Orbitals: %d Core MOs, %d Virtual MOs\n", num_Fo, num_Fv);

    SharedMatrix Ca_tilde(ref_wfn->Ca()->clone());
    // outfile->Printf("fock matrix")
    // Fa()->print();

    bool semi_f = options->get_bool("EMBEDDING_SEMICANONICALIZE_FROZEN");
    bool semi_a = options->get_bool("EMBEDDING_SEMICANONICALIZE_ACTIVE");

    // Build and semi-canonicalize BO, AO, AV and BV blocks from rotated Ca()
    auto C_bo = semicanonicalize_block(ref_wfn, Ca_tilde, index_B_occ, 0, !semi_f);
    auto C_ao = semicanonicalize_block(ref_wfn, Ca_tilde, index_A_occ, 0, false);

    bool preserve_virtual = false;
    if (options->get_str("EMBEDDING_VIRTUAL_SPACE") == "PAO") {
        preserve_virtual = true;
    }

    auto C_av = semicanonicalize_block(ref_wfn, Ca_tilde, index_A_vir, 0, preserve_virtual);
    auto C_bv = semicanonicalize_block(ref_wfn, Ca_tilde, index_B_vir, 0, !semi_f);

    // Copy the active block (if any) from original Ca_save
    auto C_A = std::make_shared<psi::Matrix>("Active_coeff_block", nirrep, nmopi, actv_a);
    if (options->get_str("EMBEDDING_REFERENCE") == "CASSCF") {
        C_A->copy(semicanonicalize_block(ref_wfn, Ca_save, index_actv, 0, !semi_a));
        if (semi_a) {
            outfile->Printf("\n  Semi-canonicalizing active orbitals");
        }
    }

    // Copy the frozen blocks (if any) from original Ca_save without any changes
    auto C_Fo = std::make_shared<psi::Matrix>("Fo_coeff_block", nirrep, nmopi, frzopi);
    auto C_Fv = std::make_shared<psi::Matrix>("Fv_coeff_block", nirrep, nmopi, frzvpi);

    C_Fo->copy(semicanonicalize_block(ref_wfn, Ca_save, index_frozen_core, 0, !semi_f));
    C_Fv->copy(semicanonicalize_block(ref_wfn, Ca_save, index_frozen_virtual, 0, !semi_f));

    // Form new C matrix: Frozen-core, B_occ, A_occ, Active, A_vir, B_vir, Frozen-virtual
    auto Ca_Rt = std::make_shared<psi::Matrix>("Ca rotated tilde", nirrep, nmopi, nmopi);

    int offset = 0;
    for (auto& C_block : {C_Fo, C_bo, C_ao, C_A, C_av, C_bv, C_Fv}) {
        int nmo_block = C_block->ncol();
        for (int i = 0; i < nmo_block; ++i) {
            for (int mu = 0; mu < nmopi[0]; ++mu) {
                double value = C_block->get(mu, i);
                Ca_Rt->set(mu, offset, value);
            }
            offset += 1;
        }
    }

    // Update both the alpha and beta orbitals
    scf_info->update_orbitals(Ca_Rt, Ca_Rt);

    // Write a new MOSpaceInfo:
    std::map<std::string, std::vector<size_t>> mo_space_map;

    // Frozen docc space
    size_t freeze_o =
        static_cast<size_t>(num_Fo + num_Bo + adj_sys_docc); // Add the additional frozen core to Bo
    mo_space_map["FROZEN_DOCC"] = {freeze_o};

    // Restricted docc space
    size_t ro = static_cast<size_t>(num_Ao - adj_sys_docc);
    if (options->get_str("EMBEDDING_REFERENCE") == "HF") {
        ro -= diff;
    }
    mo_space_map["RESTRICTED_DOCC"] = {ro};

    // Active space
    size_t a = static_cast<size_t>(actv_a[0]);
    if (options->get_str("EMBEDDING_REFERENCE") == "HF") {
        a += diff;
        a += diff2;
    }
    mo_space_map["ACTIVE"] = {a};

    // Restricted uocc space
    size_t rv = static_cast<size_t>(num_Av - adj_sys_uocc);
    if (options->get_str("EMBEDDING_REFERENCE") == "HF") {
        rv -= diff2;
    }
    mo_space_map["RESTRICTED_UOCC"] = {rv};

    // Frozen uocc space
    size_t freeze_v = static_cast<size_t>(num_Fv + num_Bv +
                                          adj_sys_uocc); // Add the additional frozen virtual to Bv
    mo_space_map["FROZEN_UOCC"] = {freeze_v};

    // Write new MOSpaceInfo
    outfile->Printf("\n  Updating MOSpaceInfo");
    std::vector<size_t> reorder;
    std::string point_group = ref_wfn->molecule()->point_group()->symbol();
    std::shared_ptr<MOSpaceInfo> mo_space_info_emb =
        make_mo_space_info_from_map(nmopi, point_group, mo_space_map);

    // Return the new embedding MOSpaceInfo to pymodule
    outfile->Printf("\n\n  --------------- End of Frozen-orbital Embedding --------------- ");
    return mo_space_info_emb;
}

std::shared_ptr<psi::Matrix> semicanonicalize_block(psi::SharedWavefunction ref_wfn,
                                                    std::shared_ptr<psi::Matrix> C_tilde,
                                                    std::vector<int>& mos, int offset,
                                                    bool prevent_rotate) {
    int nso = ref_wfn->nso();
    int nmo_block = mos.size();
    auto C_block = std::make_shared<psi::Matrix>("C block", nso, nmo_block);

    // Extract corresponding elements from C with respect to the index vector
    int mo_count = 0;
    for (int i : mos) {
        for (int mu = 0; mu < nso; ++mu) {
            double value = C_tilde->get(mu, i + offset);
            C_block->set(mu, mo_count, value);
        }
        mo_count += 1;
    }
    if (!prevent_rotate) {
        // compute (C_block)^T F C_block
        auto Foi = psi::linalg::triplet(C_block, ref_wfn->Fa(), C_block, true, false, false);

        auto U_block = std::make_shared<psi::Matrix>("U block", nmo_block, nmo_block);
        auto epsilon_block = std::make_shared<Vector>("epsilon block", nmo_block);
        Foi->diagonalize(U_block, epsilon_block);
        auto C_block_prime = psi::linalg::doublet(C_block, U_block);
        return C_block_prime;
    } else { // If true, only form the block matrix but avoid diagonalization
        return C_block;
    }
}
} // namespace forte
