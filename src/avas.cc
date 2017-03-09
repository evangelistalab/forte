#include "psi4/psi4-dec.h"

#include "avas.h"
#include "psi4/libmints/vector.h"

namespace psi {
namespace forte {

SharedMatrix semicanonicalize_block(SharedWavefunction ref_wfn,
                                    SharedMatrix C_tilde, std::vector<int>& mos,
                                    int offset);

void set_AVAS_options(ForteOptions& foptions) {
    foptions.add_double("AVAS_SIGMA", 0.90,
                        "Threshold that controls the size of the active space");
    foptions.add_int("AVAS_NUM_ACTIVE", 0, "Allows the user to specify the "
                                           "total number of active orbitals. "
                                           "It takes priority over the "
                                           "threshold based selection.");
}

void make_avas(SharedWavefunction ref_wfn, Options& options, SharedMatrix Ps) {
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

        auto Socc = std::make_shared<Matrix>("S occupied block", nocc, nocc);
        auto Svir = std::make_shared<Matrix>("S virtual block", nvir, nvir);

        SharedMatrix CPsC = Ps->clone();
        CPsC->transform(ref_wfn->Ca());

        // Grab the occupied block and diagonalize it
        for (int i = 0; i < nocc; i++) {
            for (int j = 0; j < nocc; j++) {
                double value = CPsC->get(i, j);
                Socc->set(i, j, value);
            }
        }

        auto Uocc = std::make_shared<Matrix>("U occupied block", nocc, nocc);
        auto sigmaocc = std::make_shared<Vector>("sigma occupied block", nocc);

        Socc->diagonalize(Uocc, sigmaocc, descending);

        //        Uocc->print();
        //        sigmaocc->print();

        // Grab the virtual block and diagonalize it
        for (int a = 0; a < nvir; a++) {
            for (int b = 0; b < nvir; b++) {
                double value = CPsC->get(nocc + a, nocc + b);
                Svir->set(a, b, value);
            }
        }
        //        Svir->print();

        auto Uvir = std::make_shared<Matrix>("U virtual block", nvir, nvir);
        auto sigmavir = std::make_shared<Vector>("sigma virtual block", nvir);

        Svir->diagonalize(Uvir, sigmavir, descending);

        // Form the full matrix U
        auto U = std::make_shared<Matrix>("U", nmo, nmo);

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

        auto Ca_tilde = Matrix::doublet(ref_wfn->Ca(), U);

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
        double avas_sigma = options.get_double("AVAS_SIGMA");

        if (avas_num_active > 0) {
            for (int n = 0; n < avas_num_active; ++n){
                bool is_occ = std::get<1>(sorted_mos[n]);
                int p = std::get<2>(sorted_mos[n]);
                if (is_occ) {
                    occ_act.push_back(p);
                } else {
                    vir_act.push_back(p);
                }
            }
            for (int n = avas_num_active; n < nmo; ++n){
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

        outfile->Printf("\n  Number of inactive occupied MOs: %6d",
                        occ_inact.size());
        outfile->Printf("\n  Number of active occupied MOs:   %6d",
                        occ_act.size());
        outfile->Printf("\n  Number of active virtual MOs:    %6d",
                        vir_act.size());
        outfile->Printf("\n  Number of inactive virtual MOs:  %6d",
                        vir_inact.size());
        outfile->Printf("\n");
        outfile->Printf("\n  restricted_docc = [%d]", occ_inact.size());
        outfile->Printf("\n  active          = [%d]",
                        occ_act.size() + vir_act.size());
        outfile->Printf("\n");

        outfile->Printf("\n  Atomic Valence MOs:\n");
        outfile->Printf("    ============================\n");
        outfile->Printf("    Occupation  MO   <phi|P|phi>\n");
        outfile->Printf("    ----------------------------\n");
        for (int i : occ_act) {
            outfile->Printf("      %1d       %4d    %.6f\n", 2, i + 1,
                            sigmaocc->get(i));
        }
        for (int i : vir_act) {
            outfile->Printf("      %1d       %4d    %.6f\n", 0, nocc + i + 1,
                            sigmavir->get(i));
        }
        outfile->Printf("    ============================\n");

        // occupied inactive
        auto Coi = semicanonicalize_block(ref_wfn, Ca_tilde, occ_inact, 0);
        auto Coa = semicanonicalize_block(ref_wfn, Ca_tilde, occ_act, 0);
        auto Cvi = semicanonicalize_block(ref_wfn, Ca_tilde, vir_inact, nocc);
        auto Cva = semicanonicalize_block(ref_wfn, Ca_tilde, vir_act, nocc);

        auto Ca_tilde_prime =
            std::make_shared<Matrix>("C tilde prime", nso, nmo);

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

        SharedMatrix Fa = ref_wfn->Fa(); // get Fock matrix
        SharedMatrix Fa_mo = Fa->clone();
        Fa_mo->transform(Ca_tilde_prime);

        // Update both the alpha and beta orbitals
        // This assumes a restricted MO set
        // TODO: generalize to unrestricted references
        ref_wfn->Ca()->copy(Ca_tilde_prime);
        ref_wfn->Cb()->copy(Ca_tilde_prime);
    }
}

SharedMatrix semicanonicalize_block(SharedWavefunction ref_wfn,
                                    SharedMatrix C_tilde, std::vector<int>& mos,
                                    int offset) {
    int nso = ref_wfn->nso();
    int nmo_block = mos.size();
    auto C_block = std::make_shared<Matrix>("C block", nso, nmo_block);

    int mo_count = 0;
    for (int i : mos) {
        for (int mu = 0; mu < nso; ++mu) {
            double value = C_tilde->get(mu, i + offset);
            C_block->set(mu, mo_count, value);
        }
        mo_count += 1;
    }
    // compute (C_block)^T F C_block
    auto Foi =
        Matrix::triplet(C_block, ref_wfn->Fa(), C_block, true, false, false);

    auto U_block = std::make_shared<Matrix>("U block", nmo_block, nmo_block);
    auto epsilon_block = std::make_shared<Vector>("epsilon block", nmo_block);
    Foi->diagonalize(U_block, epsilon_block);
    auto C_block_prime = Matrix::doublet(C_block, U_block);
    return C_block_prime;
}
}
}

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
