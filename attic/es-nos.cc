
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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"

#include "helpers/timer.h"
#include "helpers/printing.h"
#include "es-nos.h"

using namespace psi;

namespace forte {

ESNO::ESNO(psi::SharedWavefunction ref_wfn, psi::Options& options,
           std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info,
           DeterminantHashVec& reference)
    : Wavefunction(options), ints_(ints), reference_(reference), ref_wfn_(ref_wfn),
      mo_space_info_(mo_space_info) {
    shallow_copy(ref_wfn);
    print_method_banner({"External Singles Natural Orbitals", "Jeff Schriber"});
    startup();
}

ESNO::~ESNO() {}

void ESNO::startup() {
    mo_symmetry_ = mo_space_info_->symmetry("GENERALIZED PARTICLE");

    nirrep_ = mo_space_info_->nirrep();

    // Define the correlated space
    auto correlated_mo = mo_space_info_->corr_absolute_mo("GENERALIZED PARTICLE");
    std::sort(correlated_mo.begin(), correlated_mo.end());

    fci_ints_ = std::make_shared<ActiveSpaceIntegrals>(
        ints_, correlated_mo, mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC"));

    // Set the integrals
    outfile->Printf("\n  Resetting FCI integrals");
    ambit::Tensor tei_active_aa =
        ints_->aptei_aa_block(correlated_mo, correlated_mo, correlated_mo, correlated_mo);
    ambit::Tensor tei_active_ab =
        ints_->aptei_ab_block(correlated_mo, correlated_mo, correlated_mo, correlated_mo);
    ambit::Tensor tei_active_bb =
        ints_->aptei_bb_block(correlated_mo, correlated_mo, correlated_mo, correlated_mo);

    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);

    fci_ints_->compute_restricted_one_body_operator();

    nroot_ = options_.get_int("NROOT");
    multiplicity_ = options_.get_int("MULTIPLICITY");

    diag_method_ = DLSolver;
}

void ESNO::compute_nos() {

    outfile->Printf("\n Casting determinants into full correlated basis");
    upcast_reference();
    WFNOperator op(mo_symmetry_, fci_ints_);

    outfile->Printf("\n  Adding single excitations ...");
    local_timer add;
    get_excited_determinants();
    outfile->Printf("\n  Excitations took %1.5f s", add.get());
    outfile->Printf("\n  psi::Dimension of full space: %zu", reference_.size());

    std::string sigma_alg = options_.get_str("SIGMA_BUILD_TYPE");

    if (sigma_alg == "HZ") {
        op.op_lists(reference_);
        op.tp_lists(reference_);
    } else {
        op.build_strings(reference_);
        op.op_s_lists(reference_);
        op.tp_s_lists(reference_);
    }

    // Diagonalize Hamiltonian
    psi::SharedMatrix evecs;
    psi::SharedVector evals;

    SparseCISolver sparse_solver(fci_ints_);

    // set options
    sparse_solver.set_sigma_method(sigma_alg);
    sparse_solver.set_parallel(true);
    sparse_solver.set_e_convergence(options_.get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_.get_int("DL_MAXITER"));
    sparse_solver.set_spin_project(true);
    sparse_solver.set_guess_dimension(options_.get_int("DL_GUESS_SIZE"));
    sparse_solver.set_spin_project_full(false);

    sparse_solver.diagonalize_hamiltonian_map(reference_, op, evals, evecs, nroot_, multiplicity_,
                                              diag_method_);

    // build 1 rdm

    outfile->Printf("\n  Computing 1RDM");
    CI_RDMS ci_rdms(reference_, fci_ints_, evecs, 0, 0);
    ci_rdms.set_max_rdm(1);

    size_t ncmo = mo_space_info_->size("GENERALIZED PARTICLE");
    psi::Dimension ncmopi = mo_space_info_->dimension("GENERALIZED PARTICLE");
    psi::Dimension fdocc = mo_space_info_->dimension("FROZEN_DOCC");
    psi::Dimension nmopi = mo_space_info_->dimension("ALL");

    std::vector<double> ordm_a(ncmo * ncmo, 0.0);
    std::vector<double> ordm_b(ncmo * ncmo, 0.0);

    ci_rdms.compute_1rdm(ordm_a, ordm_b, op);

    psi::SharedMatrix ordm_a_mat(new psi::Matrix("OPDM_A", nirrep_, ncmopi, ncmopi));
    psi::SharedMatrix ordm_b_mat(new psi::Matrix("OPDM_B", nirrep_, ncmopi, ncmopi));
    int offset = 0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int u = 0; u < ncmopi[h]; ++u) {
            for (int v = 0; v < ncmopi[h]; ++v) {
                ordm_a_mat->set(h, u, v, ordm_a[(u + offset) * ncmo + (v + offset)]);
                ordm_b_mat->set(h, u, v, ordm_b[(u + offset) * ncmo + (v + offset)]);
            }
        }
        offset += ncmopi[h];
    }
    // diagonalize ordm
    outfile->Printf("\n  Diagonalizing 1RDM");
    psi::SharedVector OCC_A(new Vector("ALPHA NO OCC", nirrep_, ncmopi));
    psi::SharedVector OCC_B(new Vector("BETA NO OCC", nirrep_, ncmopi));
    psi::SharedMatrix NO_A(new psi::Matrix(nirrep_, ncmopi, ncmopi));
    psi::SharedMatrix NO_B(new psi::Matrix(nirrep_, ncmopi, ncmopi));

    ordm_a_mat->diagonalize(NO_A, OCC_A, descending);
    ordm_b_mat->diagonalize(NO_B, OCC_B, descending);

    // Build the transformation matrix
    psi::SharedMatrix Ua(new psi::Matrix("Ua", nmopi, nmopi));
    psi::SharedMatrix Ub(new psi::Matrix("Ub", nmopi, nmopi));

    Ua->identity();
    Ub->identity();

    for (int h = 0; h < nirrep_; ++h) {
        size_t irrep_offset = 0;

        // Skip frozen core
        irrep_offset += fdocc[h];

        for (int p = 0; p < ncmopi[h]; ++p) {
            for (int q = 0; q < ncmopi[h]; ++q) {
                Ua->set(h, p + irrep_offset, q + irrep_offset, NO_A->get(h, p, q));
                Ub->set(h, p + irrep_offset, q + irrep_offset, NO_B->get(h, p, q));
            }
        }
    }

    // Transform C matrix
    ints_->rotate_orbitals(Ua, Ub);
}

void ESNO::get_excited_determinants() {
    // Only excite into the restricted uocc

    auto ruocc = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");
    psi::Dimension rdoccpi = mo_space_info_->dimension("RESTRICTED_DOCC");
    size_t nrdo = mo_space_info_->size("RESTRICTED_DOCC");
    size_t nact = mo_space_info_->size("ACTIVE");

    std::vector<size_t> external_mo = get_excitation_space();
    int n_ext = external_mo.size();

    DeterminantHashVec external;
    external.clear();

    const auto& internal = reference_.wfn_hash();
    for (const auto& det : internal) {
        outfile->Printf("\n  %s", det.str().c_str());
        std::vector<int> aocc = det.get_alfa_occ(nrdo + nact); // TODO check this
        std::vector<int> bocc = det.get_beta_occ(nrdo + nact); // TODO check this

        int noalfa = aocc.size();
        int nobeta = bocc.size();

        Determinant new_det(det);

        // Single Alpha
        for (int i = 0; i < noalfa; ++i) {
            int ii = aocc[i];
            for (int a = 0; a < n_ext; ++a) {
                int aa = external_mo[a];
                outfile->Printf("\n  aa: %d", aa);
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    new_det = det;
                    new_det.set_alfa_bit(ii, false);
                    new_det.set_alfa_bit(aa, true);
                    external.add(new_det);
                    outfile->Printf("\n  %s", new_det.str().c_str());
                }
            }
        }
        // Single Beta
        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int a = 0; a < n_ext; ++a) {
                int aa = external_mo[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    new_det = det;
                    new_det.set_beta_bit(ii, false);
                    new_det.set_beta_bit(aa, true);
                    external.add(new_det);
                }
            }
        }
    }
    // const auto dets = external.determinants();
    //    for( auto& det : dets) outfile->Printf("\n  %s", det.str().c_str());

    outfile->Printf("\n  Added %zu determinants from external space", external.size());
    reference_.merge(external);
}

void ESNO::upcast_reference() {
    auto mo_sym = mo_space_info_->symmetry("GENERALIZED PARTICLE");

    psi::Dimension old_dim = mo_space_info_->dimension("ACTIVE");
    psi::Dimension new_dim = mo_space_info_->dimension("GENERALIZED PARTICLE");
    size_t nact = mo_space_info_->size("ACTIVE");
    int n_irrep = old_dim.n();

    const det_hashvec& ref_dets = reference_.wfn_hash();
    reference_.clear();

    // Compute shifts
    std::vector<int> shift(n_irrep, 0);
    if (n_irrep > 1) {
        for (int n = 1; n < n_irrep; ++n) {
            shift[n] += new_dim[n - 1] - old_dim[n - 1] + shift[n - 1];
        }
    }
    for (size_t I = 0, max = ref_dets.size(); I < max; ++I) {
        Determinant det = ref_dets[I];

        // First beta
        for (int n = n_irrep - 1; n >= 0; --n) {
            int min = 0;
            for (int m = 0; m < n; ++m) {
                min += old_dim[m];
            }
            for (size_t pos = nact + min + old_dim[n] - 1; pos >= min + nact; --pos) {
                det.set_beta_bit(pos + shift[n], det.get_beta_bit(pos));
                det.set_beta_bit(pos, false);
            }
        }
        // Then alpha
        for (int n = n_irrep - 1; n >= 0; --n) {
            int min = 0;
            for (int m = 0; m < n; ++m) {
                min += old_dim[m];
            }
            for (int pos = min + old_dim[n] - 1; pos >= min; --pos) {
                det.set_alfa_bit(pos + shift[n], det.get_alfa_bit(pos));
                if (n > 0)
                    det.set_alfa_bit(pos, false);
            }
        }
        reference_.add(det);
        outfile->Printf("\n  %s", det.str().c_str());
    }
}

std::vector<size_t> ESNO::get_excitation_space() {
    std::vector<size_t> ex_space;

    // First get a list of absolute position of RUOCC
    std::vector<size_t> ruocc = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");
    psi::Dimension rdocc_dim = mo_space_info_->dimension("RESTRICTED_DOCC");
    std::vector<int> c_sym = mo_space_info_->symmetry("CORRELATED");

    int max_n = ruocc.size();
    int n_kept = options_.get_int("ESNO_MAX_SIZE");

    if (!options_["ESNO_MAX_SIZE"].has_changed()) {
        n_kept = max_n;
    }

    if ((n_kept <= max_n) and (nirrep_ == 1)) {
        max_n = n_kept;
    }

    for (int n = 0; n < max_n; ++n) {
        ex_space.push_back(ruocc[n] - rdocc_dim[c_sym[ruocc[n]]]);
        //   ex_space.push_back( ruocc[n]);
        //        outfile->Printf("\n idx: %d", ruocc[n]- rdocc_dim[ c_sym[ruocc[n]]] );
    }
    /*

        // Create a vector of orbital energy and index pairs
        std::vector<std::tuple<double, int, int>> labeled_orb;
        std::vector<std::pair<double, int>> orb_e;

        int cumidx = 0;
        for (int h = 0; h < nirrep_; ++h) {
            for (int a = 0; a < nactpi_[h]; ++a) {
                orb_e.push_back(std::make_pair(epsilon_a_->get(h, frzcpi_[h] + a), a + cumidx));
            }
            cumidx += nactpi_[h];
        }
        // Create a vector that stores the orbital energy, symmetry, and idx
        for (size_t a = 0; a < nact_; ++a) {
            labeled_orb.push_back(std::make_tuple(orb_e[a].first, mo_symmetry_[a],orb_e[a].second));
        }
        // Order by energy, low to high
        std::sort(labeled_orb.begin(), labeled_orb.end());
    */
    return ex_space;
}
} // namespace forte
