/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER,
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
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/vector.h"

#include "base_classes/mo_space_info.h"
#include "sparse_ci/sigma_vector.h"

#include "ci-no.h"

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

std::string dimension_to_string(psi::Dimension dim) {
    std::string s = "[";
    int nirrep = dim.n();
    for (int h = 0; h < nirrep; h++) {
        s += (h == 0) ? "" : ",";
        s += std::to_string(dim[h]);
    }
    s += "]";
    return s;
}

CINO::CINO(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
           std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : OrbitalTransform(ints, mo_space_info), options_(options) {
    fci_ints_ = std::make_shared<ActiveSpaceIntegrals>(
        ints, mo_space_info_->corr_absolute_mo("ACTIVE"), mo_space_info_->symmetry("ACTIVE"),
        mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC"));

    auto active_mo = mo_space_info_->corr_absolute_mo("ACTIVE");
    ambit::Tensor tei_active_aa = ints->aptei_aa_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_ab = ints->aptei_ab_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_bb = ints->aptei_bb_block(active_mo, active_mo, active_mo, active_mo);

    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();

    nirrep_ = ints->nirrep();

    startup();
}

CINO::~CINO() {}

psi::SharedMatrix CINO::get_Ua() { return Ua_; }
psi::SharedMatrix CINO::get_Ub() { return Ub_; }

void CINO::compute_transformation() {
    outfile->Printf("\n\n  Computing CIS natural orbitals\n");

    psi::SharedMatrix Density_a(new psi::Matrix(actvpi_, actvpi_));
    psi::SharedMatrix Density_b(new psi::Matrix(actvpi_, actvpi_));
    int sum = 0;

    // Build CAS determinants
    // std::vector<std::vector<Determinant> > dets_cas = build_dets_cas();

    std::vector<int> root_pi = options_->get_int_list("CINO_ROOTS_PER_IRREP");
    for (int h = 0; h < nirrep_; ++h) {
        int nsolutions = root_pi[h];
        sum += nsolutions;
        if (nsolutions > 0) {
            outfile->Printf("\n  ==> Irrep %s: %d solutions <==\n", h, nsolutions);

            // 1. Build the space of determinants

            std::vector<Determinant> dets = build_dets(h);

            // 2. Diagonalize the Hamiltonian in this basis
            std::pair<psi::SharedVector, psi::SharedMatrix> evals_evecs =
                diagonalize_hamiltonian(dets, nsolutions);

            // 3. Build the density matrix
            std::pair<psi::SharedMatrix, psi::SharedMatrix> gamma =
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

    std::pair<psi::SharedMatrix, psi::SharedMatrix> avg_gamma =
        std::make_pair(Density_a, Density_b);

    // 4. Diagonalize the density matrix
    std::tuple<psi::SharedVector, psi::SharedMatrix, psi::SharedVector, psi::SharedMatrix> no_U =
        diagonalize_density_matrix(avg_gamma);

    // 5. Find optimal active space and transform the orbitals
    find_active_space_and_transform(no_U);
}

void CINO::startup() {
    wavefunction_multiplicity_ = 1;
    if (options_->get_int("MULTIPLICITY") >= 1) {
        wavefunction_multiplicity_ = options_->get_int("MULTIPLICITY");
        // TODO: should read default from somewhere (Francesco)
    }

    // Read Options
    rdm_level_ = options_->get_int("ACI_MAX_RDM");
    nactv_ = mo_space_info_->size("ACTIVE");
    nmo_ = mo_space_info_->size("FROZEN_DOCC") + mo_space_info_->size("RESTRICTED_DOCC") +
           mo_space_info_->size("ACTIVE") + mo_space_info_->size("FROZEN_UOCC") +
           mo_space_info_->size("RESTRICTED_UOCC");

    actvpi_ = mo_space_info_->dimension("ACTIVE");
    fdoccpi_ = mo_space_info_->dimension("FROZEN_DOCC");
    rdoccpi_ = mo_space_info_->dimension("RESTRICTED_DOCC");
    fuoccpi_ = mo_space_info_->dimension("FROZEN_UOCC");
    ruoccpi_ = mo_space_info_->dimension("RESTRICTED_UOCC");

    ncmo2_ = nactv_ * nactv_;

    aoccpi_ = ints_->wfn()->nalphapi() - rdoccpi_ - fdoccpi_;
    boccpi_ = ints_->wfn()->nbetapi() - rdoccpi_ - fdoccpi_;
    cino_auto = options_->get_bool("CINO_AUTO");
}

std::vector<Determinant> CINO::build_dets(int irrep) {

    // build the reference determinant

    std::vector<Determinant> dets;

    std::vector<bool> occupation_a(nactv_);
    std::vector<bool> occupation_b(nactv_);

    // add the reference determinant
    int offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        int aocc_h = aoccpi_[h];
        for (int i = 0; i < aocc_h; i++) {
            occupation_a[i + offset] = true;
        }
        int bocc_h = boccpi_[h];
        for (int i = 0; i < bocc_h; i++) {
            occupation_b[i + offset] = true;
        }
        offset += actvpi_[h];
    }

    Determinant ref(occupation_a, occupation_b);

    // add the reference only to the total symmetric irrep
    if (irrep == 0) {
        dets.push_back(ref);
    }

    // alpha single excitation
    offset = 0;
    for (int irrep_i = 0; irrep_i < nirrep_; irrep_i++) {
        // loop over i in irrep h
        int irrep_a = irrep_i ^ irrep;
        int offset_vir = 0;
        for (int h = 0; h < irrep_a; h++) {
            offset_vir += actvpi_[h];
        }
        // loop over occupied orbitals in irrep_i
        // loop over virtual orbitals in irrep_a
        int occ_irrep_i = aoccpi_[irrep_i];
        int occ_irrep_a = aoccpi_[irrep_a];
        for (int i = 0; i < occ_irrep_i; ++i) {
            for (int a = occ_irrep_a; a < actvpi_[irrep_a]; ++a) {
                Determinant single_ia(ref);
                single_ia.set_alfa_bit(i + offset, false);
                single_ia.set_alfa_bit(a + offset_vir, true);
                dets.push_back(single_ia);
                //                single_ia.print();
            }
        }
        offset += actvpi_[irrep_i];
    }

    // beta single excitation
    offset = 0;
    for (int irrep_i = 0; irrep_i < nirrep_; irrep_i++) {
        // loop over i in irrep h
        int irrep_a = irrep_i ^ irrep;
        int offset_vir = 0;
        for (int h = 0; h < irrep_a; h++) {
            offset_vir += actvpi_[h];
        }
        // loop over occupied orbitals in irrep_i
        // loop over virtual orbitals in irrep_a
        int occ_irrep_i = boccpi_[irrep_i];
        int occ_irrep_a = boccpi_[irrep_a];
        for (int i = 0; i < occ_irrep_i; ++i) {
            for (int a = occ_irrep_a; a < actvpi_[irrep_a]; ++a) {
                Determinant single_ib(ref);
                single_ib.set_beta_bit(i + offset, false);
                single_ib.set_beta_bit(a + offset_vir, true);
                dets.push_back(single_ib);
                //                single_ib.print();
            }
        }
        offset += actvpi_[irrep_i];
    }

    if (options_->get_str("CINO_TYPE") == "CISD") {
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

    return dets;
}
/// Diagonalize the Hamiltonian in this basis
std::pair<psi::SharedVector, psi::SharedMatrix>
CINO::diagonalize_hamiltonian(const std::vector<Determinant>& dets, int nsolutions) {

    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);
    sparse_solver.set_e_convergence(options_->get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_->get_int("DL_MAXITER"));
    sparse_solver.set_spin_project(project_out_spin_contaminants_);
    sparse_solver.set_guess_dimension(options_->get_int("DL_GUESS_SIZE"));
    sparse_solver.set_spin_project_full(true);
    sparse_solver.set_print_details(true);

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
std::pair<psi::SharedMatrix, psi::SharedMatrix>
CINO::build_density_matrix(const std::vector<Determinant>& dets, psi::SharedMatrix evecs, int n) {
    std::vector<double> average_a_(ncmo2_);
    std::vector<double> average_b_(ncmo2_);
    std::vector<double> template_a_;
    std::vector<double> template_b_;
    std::vector<double> ordm_a_(ncmo2_);
    std::vector<double> ordm_b_(ncmo2_);

    for (int i = 0; i < n; ++i) {
        template_a_.clear();
        template_b_.clear();

        CI_RDMS ci_rdms_(fci_ints_, dets, evecs, i, i);
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

    std::shared_ptr<psi::Matrix> opdm_a(new psi::Matrix("OPDM_A", actvpi_, actvpi_));
    std::shared_ptr<psi::Matrix> opdm_b(new psi::Matrix("OPDM_B", actvpi_, actvpi_));

    int offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        for (int u = 0; u < actvpi_[h]; u++) {
            for (int v = 0; v < actvpi_[h]; v++) {
                opdm_a->set(h, u, v, ordm_a_[(u + offset) * nactv_ + v + offset]);
                opdm_b->set(h, u, v, ordm_b_[(u + offset) * nactv_ + v + offset]);
            }
        }
        offset += actvpi_[h];
    }
    // opdm_a->print();
    return std::make_pair(opdm_a, opdm_b);
}

/// Diagonalize the density matrix
std::tuple<psi::SharedVector, psi::SharedMatrix, psi::SharedVector, psi::SharedMatrix>
CINO::diagonalize_density_matrix(std::pair<psi::SharedMatrix, psi::SharedMatrix> gamma) {
    std::pair<psi::SharedVector, psi::SharedMatrix> no_U;

    psi::SharedVector OCC_A(new Vector("ALPHA OCCUPATION", actvpi_));
    psi::SharedVector OCC_B(new Vector("BETA OCCUPATION", actvpi_));
    psi::SharedMatrix NO_A(new psi::Matrix(actvpi_, actvpi_));
    psi::SharedMatrix NO_B(new psi::Matrix(actvpi_, actvpi_));

    psi::Dimension zero_dim(nirrep_);
    psi::Dimension avirpi = actvpi_ - aoccpi_;

    // Grab the alpha occupied/virtual block of the density matrix
    Slice aocc_slice(zero_dim, aoccpi_);
    psi::SharedMatrix gamma_a_occ = gamma.first->get_block(aocc_slice, aocc_slice);
    gamma_a_occ->set_name("Gamma alpha occupied");

    Slice avir_slice(aoccpi_, actvpi_);
    psi::SharedMatrix gamma_a_vir = gamma.first->get_block(avir_slice, avir_slice);
    gamma_a_vir->set_name("Gamma alpha virtual");

    // Diagonalize alpha density matrix
    psi::SharedMatrix NO_A_occ(new psi::Matrix(aoccpi_, aoccpi_));
    psi::SharedMatrix NO_A_vir(new psi::Matrix(avirpi, avirpi));
    psi::SharedVector OCC_A_occ(new Vector("Occupied ALPHA OCCUPATION", aoccpi_));
    psi::SharedVector OCC_A_vir(new Vector("Virtual ALPHA OCCUPATION", avirpi));
    gamma_a_occ->diagonalize(NO_A_occ, OCC_A_occ, descending);
    gamma_a_vir->diagonalize(NO_A_vir, OCC_A_vir, descending);
    //        OCC_A_occ->print();
    //        OCC_A_vir->print();

    OCC_A->set_block(aocc_slice, OCC_A_occ);
    NO_A->set_block(aocc_slice, aocc_slice, NO_A_occ);
    OCC_A->set_block(avir_slice, OCC_A_vir);
    NO_A->set_block(avir_slice, avir_slice, NO_A_vir);

    /// Diagonalize Beta density matrix
    psi::Dimension bvirpi = actvpi_ - boccpi_;

    // Grab the beta occupied/virtual block of the density matrix
    Slice bocc_slice(zero_dim, boccpi_);
    psi::SharedMatrix gamma_b_occ = gamma.second->get_block(bocc_slice, bocc_slice);
    gamma_b_occ->set_name("Gamma beta occupied");

    Slice bvir_slice(boccpi_, actvpi_);
    psi::SharedMatrix gamma_b_vir = gamma.second->get_block(bvir_slice, bvir_slice);
    gamma_b_vir->set_name("Gamma beta virtual");

    //    for (int h = 0; h < nirrep_; h++) {
    //        for (int i = 0; i < boccpi[h]; i++) {
    //            for (int j = 0; j < boccpi[h]; j++) {
    //                gamma_b_occ->set(h, i, j, gamma.second->get(h, i, j));
    //            }
    //        }
    //    }
    //    for (int h = 0; h < nirrep_; h++) {
    //        for (int a = 0; a < bvirpi[h]; a++) {
    //            for (int b = 0; b < bvirpi[h]; b++) {
    //                gamma_b_vir->set(h, a, b, gamma.second->get(h, a + boccpi[h], b + boccpi[h]));
    //            }
    //        }
    //    }

    // Diagonalize beta density matrix
    psi::SharedMatrix NO_B_occ(new psi::Matrix(boccpi_, boccpi_));
    psi::SharedMatrix NO_B_vir(new psi::Matrix(bvirpi, bvirpi));
    psi::SharedVector OCC_B_occ(new Vector("Occupied BETA OCCUPATION", boccpi_));
    psi::SharedVector OCC_B_vir(new Vector("Virtual BETA OCCUPATION", bvirpi));
    gamma_b_occ->diagonalize(NO_B_occ, OCC_B_occ, descending);
    gamma_b_vir->diagonalize(NO_B_vir, OCC_B_vir, descending);
    //    OCC_B_occ->print();
    //    OCC_B_vir->print();

    OCC_B->set_block(bocc_slice, OCC_B_occ);
    NO_B->set_block(bocc_slice, bocc_slice, NO_B_occ);
    OCC_B->set_block(bvir_slice, OCC_B_vir);
    NO_B->set_block(bvir_slice, bvir_slice, NO_B_vir);

    //    for (int h = 0; h < nirrep_; h++) {
    //        for (int i = 0; i < boccpi[h]; i++) {
    //            OCC_B->set(h, i, OCC_B_occ->get(h, i));
    //            for (int j = 0; j < boccpi[h]; j++) {
    //                NO_B->set(h, i, j, NO_B_occ->get(h, i, j));
    //            }
    //        }
    //    }
    //    for (int h = 0; h < nirrep_; h++) {
    //        for (int a = 0; a < bvirpi[h]; a++) {
    //            OCC_B->set(h, a + boccpi[h], OCC_B_vir->get(h, a));
    //            for (int b = 0; b < bvirpi[h]; b++) {
    //                NO_B->set(h, a + boccpi[h], b + boccpi[h], NO_B_vir->get(h, a, b));
    //            }
    //        }
    //    }

    //    gamma.first->diagonalize(NO_A, OCC_A, descending);
    //    gamma.second->diagonalize(NO_B, OCC_B, descending);

    return std::make_tuple(OCC_A, NO_A, OCC_B, NO_B);
}

// Find optimal active space and transform the orbitals
void CINO::find_active_space_and_transform(
    std::tuple<psi::SharedVector, psi::SharedMatrix, psi::SharedVector, psi::SharedMatrix> no_U) {

    auto nmopi = mo_space_info_->dimension("ALL");

    Ua_.reset(new psi::Matrix("U", nmopi, nmopi));
    psi::SharedMatrix NO_A = std::get<1>(no_U);
    for (int h = 0; h < nirrep_; h++) {
        for (int p = 0; p < nmopi[h]; p++) {
            Ua_->set(h, p, p, 1.0);
        }
    }
    Slice actv_slice(fdoccpi_ + rdoccpi_, fdoccpi_ + rdoccpi_ + actvpi_);
    Ua_->set_block(actv_slice, actv_slice, NO_A);

    Ub_->copy(Ua_->clone());
    // psi::SharedMatrix Ca_new = psi::linalg::doublet(Ca_, Ua);
    // Ca_->copy(Ca_new);
    // Cb_ = Ca_; // Fix this for unrestricted case

    psi::SharedVector OCC_A = std::get<0>(no_U);
    psi::SharedVector OCC_B = std::get<2>(no_U);

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
        for (int a = aoccpi_[h]; a < actvpi_[h]; a++) {
            sum_v += OCC_A->get(h, a);
            sorted_avir.push_back(std::make_tuple(OCC_A->get(h, a), h, a));
        }
    }

    // here we use the reverse iterators to sort in descending order
    std::sort(sorted_aocc.rbegin(), sorted_aocc.rend());
    std::sort(sorted_avir.rbegin(), sorted_avir.rend());

    double cino_threshold = options_->get_double("CINO_THRESHOLD");

    psi::Dimension nactv_occ(nirrep_);
    double partial_sum_o = 0.0;
    for (auto& non_h_p : sorted_aocc) {
        double w = std::get<0>(non_h_p);
        int h = std::get<1>(non_h_p);
        partial_sum_o += w;
        nactv_occ[h] += 1;
        if (partial_sum_o / sum_o > cino_threshold)
            break;
    }

    psi::Dimension nactv_vir(nirrep_);
    double partial_sum_v = 0.0;
    for (auto& non_h_p : sorted_avir) {
        double w = std::get<0>(non_h_p);
        int h = std::get<1>(non_h_p);
        partial_sum_v += w;
        nactv_vir[h] += 1;
        if (partial_sum_v / sum_v > cino_threshold)
            break;
    }

    outfile->Printf("\n  Number of active occupied MOs per irrep: %s",
                    dimension_to_string(nactv_occ).c_str());
    outfile->Printf("\n  Number of active virtual MOs per irrep:  %s",
                    dimension_to_string(nactv_vir).c_str());

    psi::Dimension noci_fdocc = fdoccpi_;
    psi::Dimension noci_actv = nactv_occ + nactv_vir;
    psi::Dimension noci_rdocc = rdoccpi_ + aoccpi_ - nactv_occ;

    outfile->Printf("\n  FROZEN_DOCC     = %s", dimension_to_string(noci_fdocc).c_str());
    outfile->Printf("\n  RESTRICTED_DOCC = %s", dimension_to_string(noci_rdocc).c_str());
    outfile->Printf("\n  ACTIVE          = %s", dimension_to_string(noci_actv).c_str());

    // Pass the MOSpaceInfo
    // TODO: Re-enable this
    //    if (cino_auto) {
    //        for (int h = 0; h < nirrep_; h++) {
    //            options_["RESTRICTED_DOCC"].add(h);
    //            options_["ACTIVE"].add(h);
    //            options_["RESTRICTED_DOCC"][h].assign(noci_rdocc[h]);
    //            options_["active"][h].assign(noci_actv[h]);
    //        }
    //    }
}
} // namespace forte
