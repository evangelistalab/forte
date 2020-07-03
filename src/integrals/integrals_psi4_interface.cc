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

//#include <algorithm>
//#include <cmath>
//#include <numeric>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "psi4/libfock/jk.h"

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"

#include "psi4/libpsi4util/process.h"
////#include "psi4/libmints/factory.h"
////#include "psi4/libmints/mintshelper.h"
////#include "psi4/libqt/qt.h"
////#include "psi4/libtrans/integraltransform.h"
////#include "psi4/psifiles.h"

//#include "helpers/blockedtensorfactory.h"
//#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "integrals/integrals.h"
//#include "memory.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

using namespace psi;
// using namespace ambit;

namespace forte {

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif

void Psi4Integrals::base_initialize_psi4() {
    setup_psi4_ints();
    allocate();
    transform_one_electron_integrals();
    build_dipole_ints_ao();
}

void Psi4Integrals::setup_psi4_ints() {
    if (not wfn_) {
        outfile->Printf("\n  No wave function object found!  Run a scf calculation first!\n");
        exit(1);
    }

    // Grab the MO coefficients from psi and enforce spin restriction if necessary
    Ca_ = wfn_->Ca()->clone();
    Cb_ = (spin_restriction_ == IntegralSpinRestriction::Restricted ? wfn_->Ca()->clone()
                                                                    : wfn_->Cb()->clone());

    nucrep_ = wfn_->molecule()->nuclear_repulsion_energy(wfn_->get_dipole_field_strength());

    nirrep_ = wfn_->nirrep();
    nso_ = wfn_->nso();
    nmo_ = wfn_->nmo();
    nsopi_ = wfn_->nsopi();
    nmopi_ = wfn_->nmopi();
    frzcpi_ = mo_space_info_->dimension("FROZEN_DOCC");
    frzvpi_ = mo_space_info_->dimension("FROZEN_UOCC");
    ncmopi_ = mo_space_info_->dimension("CORRELATED");

    ncmo_ = ncmopi_.sum();

    // Create an array that maps the CMOs to the MOs (cmotomo_).
    for (int h = 0, q = 0; h < nirrep_; ++h) {
        q += frzcpi_[h]; // skip the frozen core
        for (int r = 0; r < ncmopi_[h]; ++r) {
            cmotomo_.push_back(q);
            q++;
        }
        q += frzvpi_[h]; // skip the frozen virtual
    }

    // Indexing
    // This is important!  Set the indexing to work using the number of
    // molecular integrals
    aptei_idx_ = nmo_;
    num_tei_ = INDEX4(nmo_ - 1, nmo_ - 1, nmo_ - 1, nmo_ - 1) + 1;
    num_aptei_ = nmo_ * nmo_ * nmo_ * nmo_;
    num_threads_ = omp_get_max_threads();
    print_ = options_->get_int("PRINT");
    /// If MO_ROTATE is set in option, call rotate_mos.
    /// Wasn't really sure where to put this function, but since, integrals is
    /// always called, this seems like a good spot.
    auto rotate_mos_list = options_->get_int_vec("ROTATE_MOS");
    if (rotate_mos_list.size() > 0) {
        rotate_mos();
    }
}

void Psi4Integrals::allocate() {
    // full one-electron integrals
    full_one_electron_integrals_a_.assign(nmo_ * nmo_, 0.0);
    full_one_electron_integrals_b_.assign(nmo_ * nmo_, 0.0);

    // these will hold only the correlated part
    one_electron_integrals_a_.assign(ncmo_ * ncmo_, 0.0);
    one_electron_integrals_b_.assign(ncmo_ * ncmo_, 0.0);
    fock_matrix_a_.assign(ncmo_ * ncmo_, 0.0);
    fock_matrix_b_.assign(ncmo_ * ncmo_, 0.0);
}

void Psi4Integrals::transform_one_electron_integrals() {
    // Grab the one-electron integrals from psi4's wave function object
    std::shared_ptr<psi::Matrix> Ha = wfn_->H()->clone();
    std::shared_ptr<psi::Matrix> Hb = wfn_->H()->clone();

    OneIntsAO_ = Ha->clone();

    Ha->transform(Ca_);
    Hb->transform(Cb_);

    OneBody_symm_ = Ha;

    // zero these vectors
    std::fill(full_one_electron_integrals_a_.begin(), full_one_electron_integrals_a_.end(), 0.0);
    std::fill(full_one_electron_integrals_b_.begin(), full_one_electron_integrals_b_.end(), 0.0);

    // Read the one-electron integrals (T + V, restricted)
    int offset = 0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < nmopi_[h]; ++p) {
            for (int q = 0; q < nmopi_[h]; ++q) {
                full_one_electron_integrals_a_[(p + offset) * nmo_ + q + offset] = Ha->get(h, p, q);
                full_one_electron_integrals_b_[(p + offset) * nmo_ + q + offset] = Hb->get(h, p, q);
            }
        }
        offset += nmopi_[h];
    }

    // Copy the correlated part into one_electron_integrals_a/one_electron_integrals_b
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            one_electron_integrals_a_[p * ncmo_ + q] =
                full_one_electron_integrals_a_[cmotomo_[p] * nmo_ + cmotomo_[q]];
            one_electron_integrals_b_[p * ncmo_ + q] =
                full_one_electron_integrals_b_[cmotomo_[p] * nmo_ + cmotomo_[q]];
        }
    }
}

void Psi4Integrals::compute_frozen_one_body_operator() {
    local_timer timer_frozen_one_body;

    psi::Dimension frozen_dim = mo_space_info_->dimension("FROZEN_DOCC");
    psi::Dimension nmopi = mo_space_info_->dimension("ALL");
    // Need to get the inactive block of the C matrix
    psi::Dimension nsopi = wfn_->nsopi();
    std::shared_ptr<psi::Matrix> C_core(new psi::Matrix("C_core", nirrep_, nsopi, frozen_dim));

    for (int h = 0; h < nirrep_; h++) {
        for (int mu = 0; mu < nsopi[h]; mu++) {
            for (int i = 0; i < frozen_dim[h]; i++) {
                C_core->set(h, mu, i, Ca_->get(h, mu, i));
            }
        }
    }

    std::shared_ptr<JK> JK_core;
    if (options_->get_str("SCF_TYPE") == "GTFOCK") {
#ifdef HAVE_JK_FACTORY
        psi::Process::environment.set_legacy_molecule(wfn_->molecule());
        JK_core = std::shared_ptr<JK>(new GTFockJK(wfn_->basisset()));
#else
        throw psi::PSIEXCEPTION("GTFock was not compiled in this version");
#endif
    } else {
        if (options_->get_str("SCF_TYPE") == "DF") {
            if ((integral_type_ == DF) or (integral_type_ == DiskDF)) {
                JK_core = JK::build_JK(wfn_->basisset(), wfn_->get_basisset("DF_BASIS_MP2"),
                                       psi::Process::environment.options, "MEM_DF");
            } else {
                throw psi::PSIEXCEPTION(
                    "Trying to compute the frozen one-body operator with MEM_DF but "
                    "using a non-DF integral type");
            }
        } else {
            JK_core = JK::build_JK(wfn_->basisset(), psi::BasisSet::zero_ao_basis_set(),
                                   psi::Process::environment.options);
        }
    }

    JK_core->set_memory(psi::Process::environment.get_memory() * 0.8);
    /// Already transform everything to C1 so make sure JK does not do this.

    // JK_core->set_cutoff(options_->get_double("INTEGRAL_SCREENING"));
    JK_core->set_cutoff(options_->get_double("INTEGRAL_SCREENING"));
    JK_core->initialize();
    JK_core->set_do_J(true);
    // JK_core->set_allow_desymmetrization(true);
    JK_core->set_do_K(true);

    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_core->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_core->C_right();

    Cl.clear();
    Cr.clear();
    Cl.push_back(C_core);
    Cr.push_back(C_core);

    JK_core->compute();

    std::shared_ptr<psi::Matrix> F_core = JK_core->J()[0];
    std::shared_ptr<psi::Matrix> K_core = JK_core->K()[0];

    F_core->scale(2.0);
    F_core->subtract(K_core);
    F_core->transform(Ca_);

    // This loop grabs only the correlated part of the correction
    int full_offset = 0;
    int corr_offset = 0;
    //    int full_offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        for (int p = 0; p < ncmopi_[h]; ++p) {
            for (int q = 0; q < ncmopi_[h]; ++q) {
                // the index of p and q in the full block of irrep h
                size_t p_full = cmotomo_[p + corr_offset] - full_offset;
                size_t q_full = cmotomo_[q + corr_offset] - full_offset;
                one_electron_integrals_a_[(p + corr_offset) * ncmo_ + (q + corr_offset)] +=
                    F_core->get(h, p_full, q_full);
                one_electron_integrals_b_[(p + corr_offset) * ncmo_ + (q + corr_offset)] +=
                    F_core->get(h, p_full, q_full);
            }
        }
        full_offset += nmopi_[h];
        corr_offset += ncmopi_[h];
    }

    F_core->add(OneBody_symm_);

    frozen_core_energy_ = 0.0;
    double E_frozen = 0.0;
    for (int h = 0; h < nirrep_; h++) {
        for (int fr = 0; fr < frozen_dim[h]; fr++) {
            E_frozen += OneBody_symm_->get(h, fr, fr) + F_core->get(h, fr, fr);
        }
    }

    OneBody_symm_ = F_core;
    frozen_core_energy_ = E_frozen;

    if (print_ > 0) {
        outfile->Printf("\n  Frozen-core energy        %20.12f a.u.", frozen_core_energy_);
        print_timing("frozen one-body operator", timer_frozen_one_body.get());
    }
}

void Psi4Integrals::rotate_orbitals(std::shared_ptr<psi::Matrix> Ua,
                                    std::shared_ptr<psi::Matrix> Ub) {
    // 1. Rotate the orbital coefficients and store them in the ForteIntegral object
    auto Ca_rotated = psi::linalg::doublet(Ca_, Ua);
    auto Cb_rotated = psi::linalg::doublet(Cb_, Ub);

    update_orbitals(Ca_rotated, Cb_rotated);
}

void Psi4Integrals::update_orbitals(std::shared_ptr<psi::Matrix> Ca,
                                    std::shared_ptr<psi::Matrix> Cb) {

    // 1. Copy orbitals and, if necessary, test they meet the spin restriction condition
    Ca_->copy(Ca);
    Cb_->copy(Cb);

    if (spin_restriction_ == IntegralSpinRestriction::Restricted) {
        if (not test_orbital_spin_restriction(Ca, Cb)) {
            Ca->print();
            Cb->print();
            auto msg = "ForteIntegrals::update_orbitals was passed two different sets of orbitals"
                       "\n  but the integral object assumes restricted orbitals";
            throw std::runtime_error(msg);
        }
    }

    // 2. Send a copy to psi::Wavefunction
    wfn_->Ca()->copy(Ca_);
    wfn_->Cb()->copy(Cb_);

    // 3. Re-transform the integrals
    aptei_idx_ = nmo_;
    transform_one_electron_integrals();
    int my_proc = 0;
#ifdef HAVE_GA
    my_proc = GA_Nodeid();
#endif
    if (my_proc == 0) {
        outfile->Printf("\n Integrals are about to be computed.");
        gather_integrals();
        outfile->Printf("\n Integrals are about to be updated.");
        freeze_core_orbitals();
    }
}

void Psi4Integrals::freeze_core_orbitals() {
    local_timer freeze_timer;
    if (ncmo_ < nmo_) {
        compute_frozen_one_body_operator();
        resort_integrals_after_freezing();
        aptei_idx_ = ncmo_;
    }
    if (print_) {
        print_timing("freezing core and virtual orbitals", freeze_timer.get());
    }
}

void Psi4Integrals::rotate_mos() {
    auto rotate_mos_list = options_->get_int_vec("ROTATE_MOS");
    int size_mo_rotate = rotate_mos_list.size();
    outfile->Printf("\n\n\n  ==> ROTATING MOS <==");
    if (size_mo_rotate % 3 != 0) {
        outfile->Printf("\n Check ROTATE_MOS array");
        outfile->Printf("\nFormat should be in group of 3s");
        outfile->Printf("\n Irrep, rotate_1, rotate_2, irrep, rotate_3, rotate_4");
        throw psi::PSIEXCEPTION("User specifed ROTATE_MOS incorrectly.  Check output for notes");
    }
    int orbital_rotate_group = (size_mo_rotate / 3);
    std::vector<std::vector<int>> rotate_mo_list;
    outfile->Printf("\n\n  IRREP  MO_1  MO_2\n");
    for (int a = 0; a < orbital_rotate_group; a++) {
        std::vector<int> rotate_mo_group(3);
        int offset_a = 3 * a;
        rotate_mo_group[0] = rotate_mos_list[offset_a] - 1;
        if (rotate_mo_group[0] > nirrep_) {
            outfile->Printf("\n Irrep:%d does not match wfn_ symmetry:%d", rotate_mo_group[0],
                            nirrep_);
            throw psi::PSIEXCEPTION("Irrep does not match wavefunction symmetry");
        }

        rotate_mo_group[1] = rotate_mos_list[offset_a + 1] - 1;
        rotate_mo_group[2] = rotate_mos_list[offset_a + 2] - 1;
        rotate_mo_list.push_back(rotate_mo_group);

        outfile->Printf("   %d   %d   %d\n", rotate_mo_group[0], rotate_mo_group[1],
                        rotate_mo_group[2]);
    }
    // std::shared_ptr<psi::Matrix> C_old = wfn_->Ca();
    std::shared_ptr<psi::Matrix> C_old = Ca_;
    std::shared_ptr<psi::Matrix> C_new(C_old->clone());

    for (auto mo_group : rotate_mo_list) {
        psi::SharedVector C_mo1 = C_old->get_column(mo_group[0], mo_group[1]);
        psi::SharedVector C_mo2 = C_old->get_column(mo_group[0], mo_group[2]);
        C_new->set_column(mo_group[0], mo_group[2], C_mo1);
        C_new->set_column(mo_group[0], mo_group[1], C_mo2);
    }
    C_old->copy(C_new);

    // std::shared_ptr<psi::Matrix> Cb_old = wfn_->Cb();
    std::shared_ptr<psi::Matrix> Cb_old = Cb_;
    Cb_old->copy(C_new);

    // Send a copy to psi::Wavefunction
    wfn_->Ca()->copy(Ca_);
    wfn_->Cb()->copy(Cb_);
}

void Psi4Integrals::build_dipole_ints_ao() {
    std::shared_ptr<psi::BasisSet> basisset = wfn_->basisset();
    std::shared_ptr<IntegralFactory> ints_fac = std::make_shared<IntegralFactory>(basisset);
    int nbf = basisset->nbf();

    dipole_ints_ao_.clear();
    for (const std::string& direction : {"X", "Y", "Z"}) {
        std::string name = "AO Dipole " + direction;
        dipole_ints_ao_.push_back(std::make_shared<psi::Matrix>(name, nbf, nbf));
    }
    std::shared_ptr<OneBodyAOInt> aodOBI(ints_fac->ao_dipole());
    aodOBI->compute(dipole_ints_ao_);
}

std::vector<std::shared_ptr<psi::Matrix>> Psi4Integrals::compute_MOdipole_ints(const bool& alpha,
                                                                               const bool& resort) {
    if (alpha) {
        return dipole_ints_mo_helper(wfn_->Ca_subset("AO"), wfn_->epsilon_a(), resort);
    } else {
        return dipole_ints_mo_helper(wfn_->Cb_subset("AO"), wfn_->epsilon_b(), resort);
    }
}

std::vector<std::shared_ptr<psi::Matrix>>
Psi4Integrals::dipole_ints_mo_helper(std::shared_ptr<psi::Matrix> Cao, psi::SharedVector epsilon,
                                     const bool& resort) {
    std::vector<std::shared_ptr<psi::Matrix>> MOdipole_ints;
    std::vector<std::string> names{"X", "Y", "Z"};
    for (int i = 0; i < 3; ++i) {
        std::shared_ptr<psi::Matrix> modipole(dipole_ints_ao_[i]->clone());
        modipole->set_name("MO Dipole " + names[i]);
        modipole->transform(Cao);
        MOdipole_ints.push_back(modipole);
    }

    if (resort) {
        // figure out the correspondance between C1 and Pitzer
        std::vector<std::tuple<double, int, int>> order;
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < nmopi_[h]; ++i) {
                order.push_back(std::tuple<double, int, int>(epsilon->get(h, i), i, h));
            }
        }
        std::sort(order.begin(), order.end(), std::less<std::tuple<double, int, int>>());

        std::vector<int> irrep_offset(nirrep_, 0);
        for (int h = 1, sum = 0; h < nirrep_; ++h) {
            sum += nmopi_[h - 1];
            irrep_offset[h] = sum;
        }

        std::vector<int> indices;
        for (int iC1 = 0; iC1 < (int)nmo_; ++iC1) {
            int i = std::get<1>(order[iC1]);
            int h = std::get<2>(order[iC1]);
            indices.push_back(irrep_offset[h] + i);
        }

        for (int i = 0; i < 3; ++i) {
            std::shared_ptr<psi::Matrix> modipole(
                new psi::Matrix("MO Dipole " + names[i], (int)nmo_, (int)nmo_));
            for (int p = 0; p < (int)nmo_; ++p) {
                int np = indices[p];
                for (int q = 0; q < (int)nmo_; ++q) {
                    int nq = indices[q];
                    modipole->set(np, nq, MOdipole_ints[i]->get(p, q));
                }
            }
            MOdipole_ints[i] = modipole;
        }
    }

    return MOdipole_ints;
}
} // namespace forte
