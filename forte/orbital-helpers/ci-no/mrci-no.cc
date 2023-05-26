/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"

#include "helpers/disk_io.h"
#include "helpers/timer.h"
#include "ci_rdm/ci_rdms.h"
#include "base_classes/mo_space_info.h"
#include "integrals/active_space_integrals.h"

#include "sparse_ci/sparse_ci_solver.h"
#include "sparse_ci/determinant.h"
#include "mrci-no.h"
#include "ci-no.h"
#include "sparse_ci/sigma_vector.h"

using namespace psi;

namespace forte {

#ifdef _OPENMP
#include <omp.h>
#include <unordered_set>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

// std::string dimension_to_string(psi::Dimension dim) {
//    std::string s = "[";
//    int nirrep = dim.n();
//    for (int h = 0; h < nirrep; h++) {
//        s += (h == 0) ? "" : ",";
//        s += std::to_string(dim[h]);
//    }
//    s += "]";
//    return s;
//}

MRCINO::MRCINO(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : OrbitalTransform(ints, mo_space_info), scf_info_(scf_info), options_(options) {
    // Copy the wavefunction information

    std::vector<size_t> active_mo(mo_space_info_->size("CORRELATED"));
    std::iota(active_mo.begin(), active_mo.end(), 0);

    auto active_mo_symmetry = mo_space_info_->symmetry("CORRELATED");

    fci_ints_ = std::make_shared<ActiveSpaceIntegrals>(ints, active_mo, active_mo_symmetry,
                                                       std::vector<size_t>());

    ambit::Tensor tei_active_aa = ints->aptei_aa_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_ab = ints->aptei_ab_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_bb = ints->aptei_bb_block(active_mo, active_mo, active_mo, active_mo);

    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();

    startup();
}

MRCINO::~MRCINO() {}

void MRCINO::compute_transformation() {
    outfile->Printf("\n\n  Computing CIS natural orbitals\n");

    auto Density_a = std::make_shared<psi::Matrix>(corrpi_, corrpi_);
    auto Density_b = std::make_shared<psi::Matrix>(corrpi_, corrpi_);
    int sum = 0;

    // Build CAS determinants
    std::vector<std::vector<Determinant>> dets_cas = build_dets_cas();

    std::vector<int> rootspi = options_->get_int_list("MRCINO_ROOTS_PER_IRREP");
    for (int h = 0; h < nirrep_; ++h) {
        int nsolutions = rootspi[h];
        sum += nsolutions;
        if (nsolutions > 0) {
            outfile->Printf("\n  ==> Irrep %s: %d solutions <==\n", h, nsolutions);

            // 1. Build the space of determinants
            std::vector<Determinant> dets = build_dets(h, dets_cas);
            //           // std::vector<Determinant> dets = build_dets(h);

            // 2. Diagonalize the Hamiltonian in this basis
            std::pair<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>> evals_evecs =
                diagonalize_hamiltonian(dets, nsolutions);

            // 3. Build the density matrix
            std::pair<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>> gamma =
                build_density_matrix(dets, evals_evecs.second, nsolutions);

            // Add density matrix to avg_gamma;
            gamma.first->scale(static_cast<double>(nsolutions));
            gamma.second->scale(static_cast<double>(nsolutions));
            Density_a->add(gamma.first);
            Density_b->add(gamma.second);
        }
    }

    Density_a->scale(1.0 / static_cast<double>(sum));
    Density_b->scale(1.0 / static_cast<double>(sum));

    // Density_a->print();

    std::pair<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>> avg_gamma =
        std::make_pair(Density_a, Density_b);

    // 4. Diagonalize the density matrix
    std::tuple<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>,
               std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
        no_U = diagonalize_density_matrix(avg_gamma);

    //    // 5. Find optimal active space and transform the orbitals
    find_active_space_and_transform(no_U);
}

void MRCINO::startup() {
    wavefunction_multiplicity_ = 1;
    if (options_->get_int("MULTIPLICITY") >= 1) {
        wavefunction_multiplicity_ = options_->get_int("MULTIPLICITY");
    }

    nirrep_ = ints_->nirrep();

    // Read Options
    rdm_level_ = options_->get_int("ACI_MAX_RDM");
    nactv_ = mo_space_info_->size("ACTIVE");
    corr_ = mo_space_info_->size("CORRELATED");

    actvpi_ = mo_space_info_->dimension("ACTIVE");
    fdoccpi_ = mo_space_info_->dimension("FROZEN_DOCC");
    rdoccpi_ = mo_space_info_->dimension("RESTRICTED_DOCC");
    fuoccpi_ = mo_space_info_->dimension("FROZEN_UOCC");
    ruoccpi_ = mo_space_info_->dimension("RESTRICTED_UOCC");
    corrpi_ = mo_space_info_->dimension("CORRELATED");

    ncmo2_ = corr_ * corr_;

    aoccpi_ = ints_->wfn()->nalphapi() - fdoccpi_;
    boccpi_ = ints_->wfn()->nbetapi() - fdoccpi_;

    mrcino_auto = options_->get_bool("MRCINO_AUTO");
}

std::vector<std::vector<Determinant>> MRCINO::build_dets_cas() {

    // Build vector of all irrep determinants
    std::vector<std::vector<Determinant>> dets_cas(nirrep_, std::vector<Determinant>());

    // Compute subspace vectors
    std::vector<bool> tmp_det_a(nactv_, false);
    std::vector<bool> tmp_det_b(nactv_, false);

    int offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        int aocc_h = aoccpi_[h];
        for (int i = 0; i < aocc_h; i++) {
            tmp_det_a[i + offset] = true;
        }
        int bocc_h = boccpi_[h];
        for (int i = 0; i < bocc_h; i++) {
            tmp_det_b[i + offset] = true;
        }
        offset += actvpi_[h];
    }
    // Make sure we start with the first permutation
    std::sort(begin(tmp_det_a), end(tmp_det_a));
    std::sort(begin(tmp_det_b), end(tmp_det_b));

    std::vector<bool> occupation_a(corr_);
    std::vector<bool> occupation_b(corr_);

    // add the reference determinant
    offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        int aocc_h = aoccpi_[h] + rdoccpi_[h];
        for (int i = 0; i < aocc_h; i++) {
            occupation_a[i + offset] = true;
        }
        int bocc_h = boccpi_[h] + rdoccpi_[h];
        for (int i = 0; i < bocc_h; i++) {
            occupation_b[i + offset] = true;
        }
        offset += corrpi_[h];
        //        outfile->Printf("\n corrpi_ is : %d \n", corrpi_[h]);
    }

    Determinant ref(occupation_a, occupation_b);

    // Generate all permutations, add the correct ones
    do {
        do {
            // Build determinant
            Determinant det(ref);
            int offset_corr = 0;
            int offset_act = 0;
            int sym = 0;
            for (int h = 0; h < nirrep_; h++) {
                int core_irrep = rdoccpi_[h];
                for (int i = 0; i < actvpi_[h]; ++i) {
                    int corr_p = i + offset_corr + core_irrep;
                    int active_p = offset_act + i;
                    det.set_alfa_bit(corr_p, tmp_det_a[active_p]);
                    det.set_beta_bit(corr_p, tmp_det_b[active_p]);
                    if (tmp_det_a[active_p]) {
                        sym ^= h;
                    }
                    if (tmp_det_b[active_p]) {
                        sym ^= h;
                    }
                }

                offset_corr += corrpi_[h];
                offset_act += actvpi_[h];
            }
            // Check Symmetry and assign to right vector
            for (int irrep = 0; irrep < nirrep_; irrep++) {
                if (sym == irrep)
                    dets_cas[irrep].push_back(det);
            }

        } while (std::next_permutation(tmp_det_b.begin(), tmp_det_b.begin() + nactv_));
    } while (std::next_permutation(tmp_det_a.begin(), tmp_det_a.begin() + nactv_));

    //   for(auto& i : dets_cas[0]){
    //       i.print();
    //   }
    return dets_cas;
}

std::vector<Determinant> MRCINO::build_dets(int irrep,
                                            const std::vector<std::vector<Determinant>>& dets_cas) {
    std::vector<Determinant> dets_irrep = dets_cas[irrep];

    // Build unordered set of irrep determinant;
    std::unordered_set<Determinant, Determinant::Hash> dets_set;
    for (auto dets : dets_cas[irrep]) {
        dets_set.emplace(dets);
    }

    // alpha excitation
    for (auto dets : dets_cas[irrep]) {
        int offset = 0;
        for (int irrep_i = 0; irrep_i < nirrep_; irrep_i++) {
            // i and a orbitals should have same sym

            // core -> virtual
            // loop over core orbitals in irrep
            // loop over virtual orbitals in irrep
            int core_irrep = rdoccpi_[irrep_i];
            int unvir_irrep = rdoccpi_[irrep_i] + actvpi_[irrep_i];
            for (int i = 0; i < core_irrep; ++i) {
                for (int a = unvir_irrep; a < corrpi_[irrep_i]; ++a) {
                    Determinant single_ia(dets);
                    single_ia.set_alfa_bit(i + offset, false);
                    single_ia.set_alfa_bit(a + offset, true);
                    if (dets_set.count(single_ia) == 0) {
                        dets_irrep.push_back(single_ia);
                    }
                }
            }

            // core -> active
            // loop over core orbitals in irrep
            // loop over active orbitals in irrep
            for (int a = core_irrep; a < unvir_irrep; ++a) {
                if (not dets.get_alfa_bit(a + offset)) {
                    for (int i = 0; i < core_irrep; ++i) {
                        Determinant single_ia(dets);
                        single_ia.set_alfa_bit(i + offset, false);
                        single_ia.set_alfa_bit(a + offset, true);

                        if (dets_set.count(single_ia) == 0) {
                            dets_irrep.push_back(single_ia);
                        }
                    }
                }
            }

            // active -> virtual
            // loop over active orbitals in irrep
            // loop over virtual orbitals in irrep
            for (int i = core_irrep; i < unvir_irrep; ++i) {
                if (dets.get_alfa_bit(i + offset)) {
                    for (int a = unvir_irrep; a < corrpi_[irrep_i]; ++a) {
                        Determinant single_ia(dets);
                        single_ia.set_alfa_bit(i + offset, false);
                        single_ia.set_alfa_bit(a + offset, true);
                        if (dets_set.count(single_ia) == 0) {
                            dets_irrep.push_back(single_ia);
                        }
                    }
                }
            }

            offset += corrpi_[irrep_i];
        }

        // beta excitation
        offset = 0;
        for (int irrep_i = 0; irrep_i < nirrep_; irrep_i++) {

            // core -> virtual
            // loop over core orbitals in irrep
            // loop over virtual orbitals in irrep
            int core_irrep = rdoccpi_[irrep_i];
            int unvir_irrep = rdoccpi_[irrep_i] + actvpi_[irrep_i];
            for (int i = 0; i < core_irrep; ++i) {
                for (int a = unvir_irrep; a < corrpi_[irrep_i]; ++a) {
                    Determinant single_ib(dets);
                    single_ib.set_beta_bit(i + offset, false);
                    single_ib.set_beta_bit(a + offset, true);
                    if (dets_set.count(single_ib) == 0) {
                        dets_irrep.push_back(single_ib);
                    }
                }
            }

            // core -> active
            // loop over core orbitals in irrep
            // loop over active orbitals in irrep
            for (int a = core_irrep; a < unvir_irrep; ++a) {
                if (not dets.get_beta_bit(a + offset)) {
                    for (int i = 0; i < core_irrep; ++i) {
                        Determinant single_ib(dets);
                        single_ib.set_beta_bit(i + offset, false);
                        single_ib.set_beta_bit(a + offset, true);
                        if (dets_set.count(single_ib) == 0) {
                            dets_irrep.push_back(single_ib);
                        }
                    }
                }
            }

            // active -> virtual
            // loop over active orbitals in irrep
            // loop over virtual orbitals in irrep
            for (int i = core_irrep; i < unvir_irrep; ++i) {
                if (dets.get_beta_bit(i + offset)) {
                    for (int a = unvir_irrep; a < corrpi_[irrep_i]; ++a) {
                        Determinant single_ib(dets);
                        single_ib.set_beta_bit(i + offset, false);
                        single_ib.set_beta_bit(a + offset, true);
                        if (dets_set.count(single_ib) == 0) {
                            dets_irrep.push_back(single_ib);
                        }
                    }
                }
            }

            offset += corrpi_[irrep_i];
        }
    }

    if (options_->get_str("MRCINO_TYPE") == "CISD") {
        // alpha-alpha double excitation
        //        for (int i = 0; i < naocc_; ++i) {
        //            for (int j = i + 1; j < naocc_; ++j) {
        //                for (int a = naocc_; a < navir_; ++a) {
        //                    for (int b = a + 1; b < navir_; ++b) {
        //                        Determinant double_ia(ref);
        //                        double_ia.set_alfa_bit(i, false);
        //                        double_ia.set_alfa_bit(j, false);
        //                        double_ia.set_alfa_bit(a, true);
        //                        double_ia.set_alfa_bit(b, true);
        //                        dets.push_back(double_ia);
        //                    }
        //                }
        //            }
        //        }
        // beta-beta double excitation
        //        for (int i = 0; i < nbocc_; ++i) {
        //            for (int j = i + 1; j < nbocc_; ++j) {
        //                for (int a = nbocc_; a < nbvir_; ++a) {
        //                    for (int b = a + 1; b < nbvir_; ++b) {
        //                        Determinant double_ib(ref);
        //                        double_ib.set_beta_bit(i, false);
        //                        double_ib.set_beta_bit(j, false);
        //                        double_ib.set_beta_bit(a, true);
        //                        double_ib.set_beta_bit(b, true);
        //                        dets.push_back(double_ib);
        //                    }
        //                }
        //            }
        //        }
        // alpha-beta double excitation
        //        for (int i = 0; i < naocc_; ++i) {
        //            for (int j = 0; j < nbocc_; ++j) {
        //                for (int a = naocc_; a < navir_; ++a) {
        //                    for (int b = nbocc_; b < nbvir_; ++b) {
        //                        Determinant double_iab(ref);
        //                        double_iab.set_alfa_bit(i, false);
        //                        double_iab.set_beta_bit(j, false);
        //                        double_iab.set_alfa_bit(a, true);
        //                        double_iab.set_beta_bit(b, true);
        //                        dets.push_back(double_iab);
        //                    }
        //                }
        //            }
        //        }
    }

    outfile->Printf("\n size is %d\n", dets_irrep.size());
    //    outfile->Printf("\n cis is :\n");
    //    for (auto& d: dets_irrep) {
    //        d.print();
    //    }

    return dets_irrep;
}
/// Diagonalize the Hamiltonian in this basis
std::pair<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
MRCINO::diagonalize_hamiltonian(const std::vector<Determinant>& dets, int nsolutions) {

    /// TODO: remove
    //    for (auto& d: dets) {
    //        d.print();
    //        outfile->Printf("  Energy: %20.15f", fci_ints_->energy(d));
    //    }

    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);
    sparse_solver.set_e_convergence(options_->get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_->get_int("DL_MAXITER"));
    sparse_solver.set_spin_project(project_out_spin_contaminants_);
    sparse_solver.set_guess_dimension(options_->get_int("DL_GUESS_SIZE"));
    sparse_solver.set_spin_project_full(true);
    sparse_solver.set_print_details(true);

    outfile->Printf("\n size is %d\n", dets.size());

    // Here we use the SparseList algorithm to diagonalize the Hamiltonian
    DeterminantHashVec detmap(dets);
    auto sigma_vector = make_sigma_vector(detmap, fci_ints_, 0, SigmaVectorType::SparseList);
    auto evals_evecs = sparse_solver.diagonalize_hamiltonian(detmap, sigma_vector, nsolutions,
                                                             wavefunction_multiplicity_);

    outfile->Printf("\n\n    STATE      CI ENERGY");
    outfile->Printf("\n  ----------------------------");
    for (int i = 0; i < nsolutions; ++i) {
        double energy = evals_evecs.first->get(i) + fci_ints_->scalar_energy() +
                        ints_->nuclear_repulsion_energy();
        outfile->Printf("\n    %3d %20.10f", i, energy);
    }
    outfile->Printf("\n  ------------------------------\n");
    return evals_evecs;
}
/// Build the density matrix
std::pair<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>>
MRCINO::build_density_matrix(const std::vector<Determinant>& dets,
                             std::shared_ptr<psi::Matrix> evecs, int n) {
    std::vector<double> average_a_(ncmo2_);
    std::vector<double> average_b_(ncmo2_);
    std::vector<double> template_a_;
    std::vector<double> template_b_;
    std::vector<double> ordm_a_(ncmo2_);
    std::vector<double> ordm_b_(ncmo2_);

    for (int i = 0; i < n; ++i) {
        template_a_.clear();
        template_b_.clear();

        CI_RDMS ci_rdms_(fci_ints_->active_mo_symmetry(), dets, evecs, i, i);
        ci_rdms_.set_max_rdm(rdm_level_);
        if (rdm_level_ >= 1) {
            local_timer one_r;
            ci_rdms_.compute_1rdm(template_a_, template_b_);
            outfile->Printf("\n  1-RDM  took %2.6f s (determinant)", one_r.get());
        }
        // Add template value to average vector
        for (size_t i = 0; i < ncmo2_; ++i) {
            average_a_[i] += template_a_[i];
        }
        for (size_t i = 0; i < ncmo2_; ++i) {
            average_b_[i] += template_b_[i];
        }
    }
    // Divided by the number of solutions
    for (size_t i = 0; i < ncmo2_; ++i) {
        ordm_a_[i] = average_a_[i] / n;
    }
    for (size_t i = 0; i < ncmo2_; ++i) {
        ordm_b_[i] = average_b_[i] / n;
    }
    // Invert vector to matrix
    //    psi::Dimension nmopi = reference_wavefunction_->nmopi();
    //    psi::Dimension ncmopi = mo_space_info_->dimension("CORRELATED");

    auto opdm_a = std::make_shared<psi::Matrix>("OPDM_A", corrpi_, corrpi_);
    auto opdm_b = std::make_shared<psi::Matrix>("OPDM_B", corrpi_, corrpi_);

    int offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        for (int u = 0; u < corrpi_[h]; u++) {
            for (int v = 0; v < corrpi_[h]; v++) {
                opdm_a->set(h, u, v, ordm_a_[(u + offset) * corr_ + v + offset]);
                opdm_b->set(h, u, v, ordm_b_[(u + offset) * corr_ + v + offset]);
            }
        }
        offset += corrpi_[h];
    }

    return std::make_pair(opdm_a, opdm_b);
}

/// Diagonalize the density matrix
std::tuple<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Vector>,
           std::shared_ptr<psi::Matrix>>
MRCINO::diagonalize_density_matrix(
    std::pair<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>> gamma) {
    std::pair<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>> no_U;

    auto OCC_A = std::make_shared<psi::Vector>("ALPHA OCCUPATION", corrpi_);
    auto OCC_B = std::make_shared<psi::Vector>("BETA OCCUPATION", corrpi_);
    auto NO_A = std::make_shared<psi::Matrix>(corrpi_, corrpi_);
    auto NO_B = std::make_shared<psi::Matrix>(corrpi_, corrpi_);

    psi::Dimension zero_dim(nirrep_);
    psi::Dimension aoccpi = ints_->wfn()->nalphapi() - fdoccpi_;
    psi::Dimension avirpi = corrpi_ - aoccpi;

    // Grab the alpha occupied/virtual block of the density matrix
    Slice aocc_slice(zero_dim, aoccpi);
    auto gamma_a_occ = gamma.first->get_block(aocc_slice, aocc_slice);
    gamma_a_occ->set_name("Gamma alpha occupied");

    Slice avir_slice(aoccpi, corrpi_);
    auto gamma_a_vir = gamma.first->get_block(avir_slice, avir_slice);
    gamma_a_vir->set_name("Gamma alpha virtual");

    // Diagonalize alpha density matrix
    auto NO_A_occ = std::make_shared<psi::Matrix>(aoccpi, aoccpi);
    auto NO_A_vir = std::make_shared<psi::Matrix>(avirpi, avirpi);
    auto OCC_A_occ = std::make_shared<Vector>("Occupied ALPHA OCCUPATION", aoccpi);
    auto OCC_A_vir = std::make_shared<Vector>("Virtual ALPHA OCCUPATION", avirpi);
    gamma_a_occ->diagonalize(NO_A_occ, OCC_A_occ, descending);
    gamma_a_vir->diagonalize(NO_A_vir, OCC_A_vir, descending);
    //    OCC_A_occ->print();
    //    OCC_A_vir->print();

    OCC_A->set_block(aocc_slice, *OCC_A_occ);
    NO_A->set_block(aocc_slice, aocc_slice, *NO_A_occ);
    OCC_A->set_block(avir_slice, *OCC_A_vir);
    NO_A->set_block(avir_slice, avir_slice, *NO_A_vir);

    /// Diagonalize Beta density matrix
    psi::Dimension boccpi = ints_->wfn()->nbetapi() - fdoccpi_;
    psi::Dimension bvirpi = corrpi_ - boccpi;

    // Grab the beta occupied/virtual block of the density matrix
    Slice bocc_slice(zero_dim, boccpi);
    auto gamma_b_occ = gamma.second->get_block(bocc_slice, bocc_slice);
    gamma_b_occ->set_name("Gamma beta occupied");

    Slice bvir_slice(boccpi, corrpi_);
    auto gamma_b_vir = gamma.second->get_block(bvir_slice, bvir_slice);
    gamma_b_vir->set_name("Gamma beta virtual");

    // Diagonalize beta density matrix
    auto NO_B_occ = std::make_shared<psi::Matrix>(boccpi, boccpi);
    auto NO_B_vir = std::make_shared<psi::Matrix>(bvirpi, bvirpi);
    auto OCC_B_occ = std::make_shared<Vector>("Occupied BETA OCCUPATION", aoccpi);
    auto OCC_B_vir = std::make_shared<Vector>("Virtual BETA OCCUPATION", avirpi);
    gamma_b_occ->diagonalize(NO_B_occ, OCC_B_occ, descending);
    gamma_b_vir->diagonalize(NO_B_vir, OCC_B_vir, descending);
    //        OCC_B_occ->print();
    //        OCC_B_vir->print();

    OCC_B->set_block(bocc_slice, *OCC_B_occ);
    NO_B->set_block(bocc_slice, bocc_slice, *NO_B_occ);
    OCC_B->set_block(bvir_slice, *OCC_B_vir);
    NO_B->set_block(bvir_slice, bvir_slice, *NO_B_vir);

    //        gamma.first->diagonalize(NO_A, OCC_A, descending);
    //        gamma.second->diagonalize(NO_B, OCC_B, descending);
    //        OCC_A->print();
    //        OCC_B->print();
    return std::make_tuple(OCC_A, NO_A, OCC_B, NO_B);
}

// Find optimal active space and transform the orbitals
void MRCINO::find_active_space_and_transform(
    std::tuple<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>,
               std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
        no_U) {

    auto nmopi = mo_space_info_->dimension("ALL");
    Ua_ = std::make_shared<psi::Matrix>("U", nmopi, nmopi);
    Ub_ = std::make_shared<psi::Matrix>("U", nmopi, nmopi);
    auto NO_A = std::get<1>(no_U);
    for (int h = 0; h < nirrep_; h++) {
        for (int p = 0; p < nmopi[h]; p++) {
            Ua_->set(h, p, p, 1.0);
        }
    }
    Slice corr_slice(fdoccpi_, fdoccpi_ + corrpi_);
    Ua_->set_block(corr_slice, corr_slice, NO_A);
    Ub_->copy(Ua_->clone());

    auto OCC_A = std::get<0>(no_U);
    auto OCC_B = std::get<2>(no_U);

    std::vector<std::tuple<double, int, int>> sorted_aocc; // (non,irrep,index)
    double sum_o = 0.0;
    for (int h = 0; h < nirrep_; h++) {
        for (int i = 0; i < aoccpi_[h]; i++) {
            sum_o += 1.0 - OCC_A->get(h, i);
            sorted_aocc.push_back(std::make_tuple(1.0 - OCC_A->get(h, i), h, i));
        }
    }

    std::vector<std::tuple<double, int, int>> sorted_avir; // (non,irrep,index)
    double sum_v = 0.0;
    for (int h = 0; h < nirrep_; h++) {
        for (int a = aoccpi_[h]; a < corrpi_[h]; a++) {
            sum_v += OCC_A->get(h, a);
            sorted_avir.push_back(std::make_tuple(OCC_A->get(h, a), h, a));
        }
    }

    // here we use the reverse iterators to sort in descending order
    std::sort(sorted_aocc.rbegin(), sorted_aocc.rend());
    std::sort(sorted_avir.rbegin(), sorted_avir.rend());

    double mrcino_threshold = options_->get_double("MRCINO_THRESHOLD");

    psi::Dimension nactv_occ(nirrep_);
    double partial_sum_o = 0.0;
    for (auto& non_h_p : sorted_aocc) {
        double w = std::get<0>(non_h_p);
        int h = std::get<1>(non_h_p);
        partial_sum_o += w;
        nactv_occ[h] += 1;
        if (partial_sum_o / sum_o > mrcino_threshold)
            break;
    }

    psi::Dimension nactv_vir(nirrep_);
    double partial_sum_v = 0.0;
    for (auto& non_h_p : sorted_avir) {
        double w = std::get<0>(non_h_p);
        int h = std::get<1>(non_h_p);
        partial_sum_v += w;
        nactv_vir[h] += 1;
        if (partial_sum_v / sum_v > mrcino_threshold)
            break;
    }

    outfile->Printf("\n  Number of active occupied MOs per irrep: %s",
                    dimension_to_string(nactv_occ).c_str());
    outfile->Printf("\n  Number of active virtual MOs per irrep:  %s",
                    dimension_to_string(nactv_vir).c_str());

    psi::Dimension noci_fdocc = fdoccpi_;
    psi::Dimension noci_actv = nactv_occ + nactv_vir;
    psi::Dimension noci_rdocc = aoccpi_ - nactv_occ;
    // psi::Dimension noci_rducc = corrpi_ - aoccpi_ - nactv_vir;

    outfile->Printf("\n  FROZEN_DOCC     = %s", dimension_to_string(noci_fdocc).c_str());
    outfile->Printf("\n  RESTRICTED_DOCC = %s", dimension_to_string(noci_rdocc).c_str());
    outfile->Printf("\n  ACTIVE          = %s", dimension_to_string(noci_actv).c_str());
    // outfile->Printf("\n  RESTRICTED_UOCC = %s", dimension_to_string(noci_rducc).c_str());

    dump_occupations(
        "mrci_nos_occ",
        {{"FROZEN_DOCC", noci_fdocc}, {"RESTRICTED_DOCC", noci_rdocc}, {"ACTIVE", noci_actv}});

    // Pass the MOSpaceInfo
    //   if (mrcino_auto) {
    //       for (int h = 0; h < nirrep_; h++) {
    //           // options_["RESTRICTED_DOCC"].add(h);
    //           // options_["ACTIVE"].add(h);
    //           options_["RESTRICTED_DOCC"][h].assign(noci_rdocc[h]);
    //           options_["ACTIVE"][h].assign(noci_actv[h]);
    //       }
    //   }
}
} // namespace forte
