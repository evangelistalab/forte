/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
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

#include "psi4/libmints/molecule.h"
#include "psi4/psi4-dec.h"
//#include "psi4/libmints/pointgrp.h"
//#include "psi4/libpsio/psio.hpp"

#include "../ci_rdms.h"
#include "../fci/fci_integrals.h"
#include "../sparse_ci_solver.h"
#include "../stl_bitset_determinant.h"
#include "ci-no.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

void set_CINO_options(ForteOptions& foptions) {
    foptions.add_bool("CINO", false, "Do a CINO computation?");
    foptions.add_str("CINO_TYPE", "CIS", {"CIS", "CISD"},
                     "The type of wave function.");
    foptions.add_int("CINO_NROOT", 1, "The number of roots computed");
    foptions.add_double("CINO_THRESHOLD", 0.99,
                        "The fraction of NOs to include in the active space");
    foptions.add_int("ACI_MAX_RDM", 1, "Order of RDM to compute");
    /*- Type of spin projection
     * 0 - None
     * 1 - Project initial P spaces at each iteration
     * 2 - Project only after converged PQ space
     * 3 - Do 1 and 2 -*/
}

CINO::CINO(SharedWavefunction ref_wfn, Options& options,
           std::shared_ptr<ForteIntegrals> ints,
           std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info) {
    // Copy the wavefunction information
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    fci_ints_ = std::make_shared<FCIIntegrals>(
        ints, mo_space_info_->get_corr_abs_mo("ACTIVE"),
        mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));

    auto active_mo = mo_space_info_->get_corr_abs_mo("ACTIVE");
    ambit::Tensor tei_active_aa =
        ints->aptei_aa_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_ab =
        ints->aptei_ab_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_bb =
        ints->aptei_bb_block(active_mo, active_mo, active_mo, active_mo);

    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab,
                                    tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();

    STLBitsetDeterminant::set_ints(fci_ints_);
    startup();
}

CINO::~CINO() {}

double CINO::compute_energy() {
    outfile->Printf("\n\n  Computing CIS natural orbitals\n");

    // 1. Build the space of determinants
    std::vector<Determinant> dets = build_dets();

    // 2. Diagonalize the Hamiltonian in this basis
    std::pair<SharedVector, SharedMatrix> evals_evecs =
        diagonalize_hamiltonian(dets);

    // 3. Build the density matrix
    std::pair<SharedMatrix, SharedMatrix> gamma =
        build_density_matrix(dets, evals_evecs.second, nroot_);

    // 4. Diagonalize the density matrix
    std::tuple<SharedVector, SharedMatrix, SharedVector, SharedMatrix> no_U =
        diagonalize_density_matrix(gamma);

    // 5. Find optimal active space and transform the orbitals
    find_active_space_and_transform(no_U);

    return 0.0;
}

void CINO::startup() {
    wavefunction_multiplicity_ = 1;
    if (options_["MULTIPLICITY"].has_changed()) {
        wavefunction_multiplicity_ = options_.get_int("MULTIPLICITY");
    }
    diag_method_ = DLSolver;
    if (options_["DIAG_ALGORITHM"].has_changed()) {
        if (options_.get_str("DIAG_ALGORITHM") == "FULL") {
            diag_method_ = Full;
        } else if (options_.get_str("DIAG_ALGORITHM") == "DLSTRING") {
            diag_method_ = DLString;
        } else if (options_.get_str("DIAG_ALGORITHM") == "DLDISK") {
            diag_method_ = DLDisk;
        }
    }
    // Read Options
    nroot_ = options_.get_int("CINO_NROOT");
    rdm_level_ = options_.get_int("ACI_MAX_RDM");
    nact_ = mo_space_info_->size("ACTIVE");
    ncmo2_ = nact_*nact_;
    nactpi_ = mo_space_info_->get_dimension("ACTIVE");
}

std::vector<Determinant> CINO::build_dets() {
    std::vector<Determinant> dets;

    // build the reference determinant
    size_t nactv = mo_space_info_->size("ACTIVE");
    size_t nrdocc = mo_space_info_->size("RESTRICTED_DOCC");
    int naocc = nalpha_ - nrdocc;
    int nbocc = nbeta_ - nrdocc;
    int navir = nactv - naocc;
    int nbvir = nactv - nbocc;
    std::vector<bool> occupation_a(nactv);
    std::vector<bool> occupation_b(nactv);
    for (int i = 0; i < naocc; i++) {
        occupation_a[i] = true;
    }
    for (int i = 0; i < nbocc; i++) {
        occupation_b[i] = true;
    }
    Determinant ref(occupation_a, occupation_b);
    ref.print();

    // add the reference determinant
    dets.push_back(ref);

    // alpha-alpha single excitation
    for (int i = 0; i < naocc; ++i) {
        for (int a = naocc; a < nactv; ++a) {
            Determinant single_ia(ref);
            single_ia.set_alfa_bit(i, false);
            single_ia.set_alfa_bit(a, true);
            single_ia.print();
            dets.push_back(single_ia);
        }
    }
    // beta-beta single excitation
    for (int i = 0; i < nbocc; ++i) {
        for (int b = nbocc; b < nactv; ++b) {
            Determinant single_ib(ref);
            single_ib.set_beta_bit(i, false);
            single_ib.set_beta_bit(b, true);
            single_ib.print();
            dets.push_back(single_ib);
        }
    }
    return dets;
}

std::pair<SharedVector, SharedMatrix>
CINO::diagonalize_hamiltonian(const std::vector<Determinant>& dets) {
    std::pair<SharedVector, SharedMatrix> evals_evecs;

    SparseCISolver sparse_solver;
    sparse_solver.set_parallel(true);
    sparse_solver.set_e_convergence(options_.get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_.get_int("DL_MAXITER"));
    sparse_solver.set_spin_project(project_out_spin_contaminants_);
    sparse_solver.set_guess_dimension(options_.get_int("DL_GUESS_SIZE"));
    sparse_solver.set_spin_project_full(false);
    sparse_solver.set_print_details(true);

    sparse_solver.diagonalize_hamiltonian(dets, evals_evecs.first,
                                          evals_evecs.second, nroot_,
                                          wavefunction_multiplicity_, DLSolver);

    // CiCi: print first 10 excited state energies and check with CIS results
    // (York?)
    for (int i = 0; i < nroot_; ++i) {
        outfile->Printf("\n%12f", evals_evecs.first->get(i) +
                                      fci_ints_->scalar_energy() +
                                      molecule_->nuclear_repulsion_energy());
    }

    return evals_evecs;
}

std::pair<SharedMatrix, SharedMatrix>
CINO::build_density_matrix(const std::vector<Determinant>& dets,
                           SharedMatrix evecs, int n) {
    std::vector<double> average_a_(ncmo2_);
    std::vector<double> average_b_(ncmo2_);
    std::vector<double> template_a_;
    std::vector<double> template_b_;
    std::vector<double> ordm_a_(ncmo2_);
    std::vector<double> ordm_b_(ncmo2_);

    for (int i = 0; i < n; ++i) {
        template_a_.clear();
        template_b_.clear();

        CI_RDMS ci_rdms_(options_, fci_ints_, dets, evecs, i, i);
        ci_rdms_.set_max_rdm(rdm_level_);
        if (rdm_level_ >= 1) {
            Timer one_r;
            ci_rdms_.compute_1rdm(template_a_, template_b_);
            outfile->Printf("\n  1-RDM  took %2.6f s (determinant)",
                            one_r.get());
        }
        // Add template value to average vector
        for (int i = 0; i < ncmo2_; ++i) {
            average_a_[i] += template_a_[i];
        }
        for (int i = 0; i < ncmo2_; ++i) {
            average_b_[i] += template_b_[i];
        }
    }
    // Divided by the number of root
    for (int i = 0; i < ncmo2_; ++i) {
        ordm_a_[i] = average_a_[i] / n;
    }
    for (int i = 0; i < ncmo2_; ++i) {
        ordm_b_[i] = average_b_[i] / n;
    }
    // Invert vector to matrix
    Dimension nmopi = reference_wavefunction_->nmopi();
    Dimension ncmopi = mo_space_info_->get_dimension("CORRELATED");
    Dimension fdocc = mo_space_info_->get_dimension("FROZEN_DOCC");
    Dimension rdocc = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    Dimension ruocc = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    std::shared_ptr<Matrix> opdm_a(
        new Matrix("OPDM_A", nirrep_, nactpi_, nactpi_));
    std::shared_ptr<Matrix> opdm_b(
        new Matrix("OPDM_B", nirrep_, nactpi_, nactpi_));

    int offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            for (int v = 0; v < nactpi_[h]; v++) {
                opdm_a->set(h, u, v,
                            ordm_a_[(u + offset) * nact_ + v + offset]);
                opdm_b->set(h, u, v,
                            ordm_b_[(u + offset) * nact_ + v + offset]);
            }
        }
        offset += nactpi_[h];
    }
    return std::make_pair(opdm_a, opdm_b);
}

/// Diagonalize the density matrix
std::tuple<SharedVector, SharedMatrix, SharedVector, SharedMatrix>
CINO::diagonalize_density_matrix(std::pair<SharedMatrix, SharedMatrix> gamma) {
    std::pair<SharedVector, SharedMatrix> no_U;

    SharedVector OCC_A(new Vector("ALPHA OCCUPATION", nactpi_));
    SharedVector OCC_B(new Vector("BETA OCCUPATION", nactpi_));
    SharedMatrix NO_A(new Matrix(nactpi_, nactpi_));
    SharedMatrix NO_B(new Matrix(nactpi_, nactpi_));

    gamma.first->diagonalize(NO_A, OCC_A, descending);
    gamma.second->diagonalize(NO_B, OCC_B, descending);
    OCC_A->print();
    OCC_B->print();
    return std::make_tuple(OCC_A, NO_A, OCC_B, NO_B);
}

/// Find optimal active space and transform the orbitals
void CINO::find_active_space_and_transform(
    std::tuple<SharedVector, SharedMatrix, SharedVector, SharedMatrix> no_U) {

    Dimension fdocc = mo_space_info_->get_dimension("FROZEN_DOCC");
    Dimension rdocc = mo_space_info_->get_dimension("RESTRICTED_DOCC");

    SharedMatrix Ua = std::make_shared<Matrix>("U", nmopi_, nmopi_);
    SharedMatrix NO_A = std::get<1>(no_U);
    for (int h = 0; h < nirrep_; h++) {
        int offset = fdocc[h] + rdocc[h];

        for (int p = 0; p < nmopi_[h]; p++) {
            Ua->set(h, p, p, 1.0);
        }

        for (int u = 0; u < nactpi_[h]; u++) {
            for (int v = 0; v < nactpi_[h]; v++) {
                double value = NO_A->get(h, u, v);
                Ua->set(h, u + offset, v + offset, value);
            }
        }
    }
    SharedMatrix Ca_new = Matrix::doublet(Ca_, Ua);
    Ca_->copy(Ca_new);
    Cb_ = Ca_; // Fix this for unrestricted case

    size_t nactv = mo_space_info_->size("ACTIVE");
    size_t nrdocc = mo_space_info_->size("RESTRICTED_DOCC");
    int naocc = nalpha_ - nrdocc;
    int nbocc = nbeta_ - nrdocc;
    int navir = nactv - naocc;
    int nbvir = nactv - nbocc;

    SharedVector OCC_A = std::get<0>(no_U);
    SharedVector OCC_B = std::get<2>(no_U);

    double sum_o = 0.0;
    for (int i = 0; i < naocc; i++) {
        sum_o += 1.0 - OCC_A->get(i);
        sum_o += 1.0 - OCC_B->get(i);
    }
    double sum_v = 0.0;
    for (int a = naocc; a < nactv; a++) {
        sum_v += OCC_A->get(a);
        sum_v += OCC_B->get(a);
    }

    double cino_threshold = options_.get_double("CINO_THRESHOLD");
    int nactv_o = 0;
    double partial_sum_o = 0.0;
    for (int i = 0; i < naocc; i++) {
        double w = 2.0 - OCC_A->get(naocc - 1 - i)-OCC_B->get(naocc - 1 - i);
        partial_sum_o += w;
        nactv_o += 1;
        if (partial_sum_o / sum_o > cino_threshold)
            break;
    }

    int nactv_v = 0;
    double partial_sum_v = 0.0;
    for (int a = naocc; a < nactv; a++) {
        double w = OCC_A->get(a)+OCC_B->get(a);
        partial_sum_v += w;
        nactv_v += 1;
        if (partial_sum_v / sum_v > cino_threshold)
            break;
    }

    outfile->Printf("\n  Number of active occupied MOs: %d",nactv_o);
    outfile->Printf("\n  Number of active virtual MOs:   %d",nactv_v);
}
}
} // EndNamespaces
