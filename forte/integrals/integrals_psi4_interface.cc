/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/vector.h"

#include "psi4/libfock/jk.h"

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsi4util/process.h"

#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "integrals/integrals.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

using namespace psi;

namespace forte {

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif

Psi4Integrals::Psi4Integrals(std::shared_ptr<ForteOptions> options,
                             std::shared_ptr<psi::Wavefunction> ref_wfn,
                             std::shared_ptr<MOSpaceInfo> mo_space_info, IntegralType integral_type,
                             IntegralSpinRestriction restricted)
    : ForteIntegrals(options, ref_wfn, mo_space_info, integral_type, restricted) {
    base_initialize_psi4();
}

void Psi4Integrals::base_initialize_psi4() {
    schwarz_cutoff_ = options_->get_double("INTS_TOLERANCE");
    df_fitting_cutoff_ = options_->get_double("DF_FITTING_CONDITION");

    setup_psi4_ints();
    build_dipole_ints_ao();

    if (not skip_build_) {
        transform_one_electron_integrals();
    }
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
    nso_ = wfn_->nso();
    nsopi_ = wfn_->nsopi();
    nucrep_ = wfn_->molecule()->nuclear_repulsion_energy(wfn_->get_dipole_field_strength());

    /// If MO_ROTATE is set in option, call rotate_mos.
    /// Wasn't really sure where to put this function, but since, integrals is
    /// always called, this seems like a good spot.
    auto rotate_mos_list = options_->get_int_list("ROTATE_MOS");
    if (!rotate_mos_list.empty()) {
        rotate_mos();
    }

    make_psi4_JK();
}

void Psi4Integrals::transform_one_electron_integrals() {
    // Grab the one-electron integrals from psi4's wave function object
    std::shared_ptr<psi::Matrix> Ha = wfn_->H()->clone();
    std::shared_ptr<psi::Matrix> Hb = wfn_->H()->clone();

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

void Psi4Integrals::make_psi4_JK() {
    auto& psi4_options = psi::Process::environment.options;
    auto basis = wfn_->basisset();

    outfile->Printf("\n\n  ==> Primary Basis Set Summary <==\n\n");
    basis->print();

    if (integral_type_ == Conventional) {
        outfile->Printf("\n  JK created using conventional PK integrals\n");
        JK_ = JK::build_JK(basis, psi::BasisSet::zero_ao_basis_set(), psi4_options, "PK");
    } else if (integral_type_ == Cholesky) {
        if (spin_restriction_ == IntegralSpinRestriction::Unrestricted) {
            throw psi::PSIEXCEPTION("Unrestricted orbitals not supported for CD integrals");
        }
        outfile->Printf("\n  JK created using Cholesky integrals\n");

        // push Forte option "CHOLESKY_TOLERANCE" to Psi4 environment
        auto psi4_cd = psi4_options.get_double("CHOLESKY_TOLERANCE");
        auto forte_cd = options_->get_double("CHOLESKY_TOLERANCE");
        if (psi4_cd != forte_cd) {
            print_h1("Warning from Forte JK Builder (CD)");
            outfile->Printf("\n  Inconsistent Cholesky tolerance between Psi4 and Forte");
            outfile->Printf("\n  Psi4: %.3E, Forte: %3E", psi4_cd, forte_cd);
            outfile->Printf("\n  Forte threshold pushed to Psi4 global options!");
            psi4_options.set_global_double("CHOLESKY_TOLERANCE", forte_cd);
        }

        JK_ = JK::build_JK(basis, psi::BasisSet::zero_ao_basis_set(), psi4_options, "CD");
    } else if ((integral_type_ == DF) or (integral_type_ == DiskDF) or (integral_type_ == DistDF)) {
        if (spin_restriction_ == IntegralSpinRestriction::Unrestricted) {
            throw psi::PSIEXCEPTION("Unrestricted orbitals not supported for DF integrals");
        }

        if (not options_->is_none("SCF_TYPE"))
            if (options_->get_str("SCF_TYPE").find("DF") == std::string::npos) {
                print_h1("Vital Warning from Forte JK Builder (DF)");
                outfile->Printf("\n  Inconsistent integrals used in Psi4 and Forte!");
                outfile->Printf("\n  This can be fixed by setting SCF_TYPE to DF or DISK_DF.");
            }

        auto basis_aux = wfn_->get_basisset("DF_BASIS_MP2");
        auto job_type = options_->get_str("JOB_TYPE");
        if (job_type == "CASSCF" or job_type == "MCSCF_TWO_STEP")
            basis_aux = wfn_->get_basisset("DF_BASIS_SCF");

        if (integral_type_ == DiskDF) {
            outfile->Printf("\n  JK created using DiskDF integrals\n");
            JK_ = JK::build_JK(basis, basis_aux, psi4_options, "DISK_DF");
        } else {
            outfile->Printf("\n  JK created using MemDF integrals\n");
            JK_ = JK::build_JK(basis, basis_aux, psi4_options, "MEM_DF");
        }
    } else {
        throw psi::PSIEXCEPTION("Unknown Pis4 integral type to initialize JK in Forte");
    }

    JK_->set_cutoff(schwarz_cutoff_);
    jk_initialize();
    JK_->print_header();
}

void Psi4Integrals::jk_initialize(double mem_percentage, int print_level) {
    if (mem_percentage > 1.0 or mem_percentage <= 0.0) {
        throw std::runtime_error("Invalid mem_percentage: must be 0 < value < 1.");
    }
    JK_->set_print(print_level);
    JK_->set_memory(psi::Process::environment.get_memory() * mem_percentage / sizeof(double));
    JK_->initialize();
    JK_status_ = JKStatus::initialized;
}

void Psi4Integrals::compute_frozen_one_body_operator() {
    local_timer timer_frozen_one_body;

    // compute frozen-core contribution using closed-shell Fock build
    auto nfrzcpi = mo_space_info_->dimension("FROZEN_DOCC");
    auto f = make_fock_inactive(psi::Dimension(nirrep_), nfrzcpi);
    auto Fock_a = std::get<0>(f);
    auto Fock_b = std::get<1>(f);
    frozen_core_energy_ = std::get<2>(f);

    // This loop grabs only the correlated part of the correction
    for (int h = 0, corr_offset = 0, full_offset = 0; h < nirrep_; h++) {
        for (int p = 0; p < ncmopi_[h]; ++p) {
            auto p_corr = p + corr_offset;
            auto p_full = cmotomo_[p + corr_offset] - full_offset;

            for (int q = 0; q < ncmopi_[h]; ++q) {
                auto q_corr = q + corr_offset;
                auto q_full = cmotomo_[q + corr_offset] - full_offset;

                one_electron_integrals_a_[p_corr * ncmo_ + q_corr] = Fock_a->get(h, p_full, q_full);
                one_electron_integrals_b_[p_corr * ncmo_ + q_corr] = Fock_b->get(h, p_full, q_full);
            }
        }

        full_offset += nmopi_[h];
        corr_offset += ncmopi_[h];
    }

    if (print_ > 0) {
        outfile->Printf("\n  Frozen-core energy        %20.15f a.u.", frozen_core_energy_);
        print_timing("frozen one-body operator", timer_frozen_one_body.get());
    }
    if (print_ > 2) {
        print_h1("One-body Hamiltonian elements dressed by frozen-core orbitals");
        if (Fock_a == Fock_b) {
            Fock_a->set_name("Frozen One Body");
            Fock_a->print();
        } else {
            Fock_a->set_name("Frozen One Body (alpha)");
            Fock_a->print();
            Fock_b->set_name("Frozen One Body (beta)");
            Fock_b->print();
        }
    }
}

void Psi4Integrals::update_orbitals(std::shared_ptr<psi::Matrix> Ca,
                                    std::shared_ptr<psi::Matrix> Cb, bool re_transform) {

    // 1. Copy orbitals and, if necessary, test they meet the spin restriction condition
    Ca_->copy(Ca);
    Cb_->copy(Cb);

    if (spin_restriction_ == IntegralSpinRestriction::Restricted) {
        if (not test_orbital_spin_restriction(Ca, Cb)) {
            Ca->print();
            Cb->print();
            auto msg = "Psi4Integrals::update_orbitals was passed two different sets of orbitals"
                       "\n  but the integral object assumes restricted orbitals";
            throw std::runtime_error(msg);
        }
    }

    // 2. Send a copy to psi::Wavefunction
    wfn_->Ca()->copy(Ca_);
    wfn_->Cb()->copy(Cb_);

    // 3. Re-transform the integrals
    if (re_transform) {
        aptei_idx_ = nmo_;
        transform_one_electron_integrals();
        int my_proc = 0;
#ifdef HAVE_GA
        my_proc = GA_Nodeid();
#endif
        if (my_proc == 0) {
            local_timer int_timer;
            outfile->Printf("\n  Integrals are about to be updated.");
            gather_integrals();
            freeze_core_orbitals();
            outfile->Printf("\n  Integrals update took %9.3f s.", int_timer.get());
        }
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
    auto rotate_mos_list = options_->get_int_list("ROTATE_MOS");
    int size_mo_rotate = rotate_mos_list.size();
    outfile->Printf("\n\n\n  ==> ROTATING MOS <==");
    if (size_mo_rotate % 3 != 0) {
        outfile->Printf("\n Check ROTATE_MOS array");
        outfile->Printf("\nFormat should be in group of 3s");
        outfile->Printf("\n Irrep, rotate_1, rotate_2, irrep, rotate_3, rotate_4");
        throw psi::PSIEXCEPTION("User specified ROTATE_MOS incorrectly.  Check output for notes");
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
            throw psi::PSIEXCEPTION("Irrep does not match wave function symmetry");
        }

        rotate_mo_group[1] = rotate_mos_list[offset_a + 1] - 1;
        rotate_mo_group[2] = rotate_mos_list[offset_a + 2] - 1;
        rotate_mo_list.push_back(rotate_mo_group);
        outfile->Printf("   %d   %d   %d\n", rotate_mo_group[0], rotate_mo_group[1],
                        rotate_mo_group[2]);
    }

    std::shared_ptr<psi::Matrix> C_old = Ca_;
    std::shared_ptr<psi::Matrix> C_new(C_old->clone());

    const auto& eps_a_old = *wfn_->epsilon_a();
    auto eps_a_new = *eps_a_old.clone();

    for (auto mo_group : rotate_mo_list) {
        auto C_mo1 = C_old->get_column(mo_group[0], mo_group[1]);
        auto C_mo2 = C_old->get_column(mo_group[0], mo_group[2]);
        auto epsilon_mo1 = eps_a_old.get(mo_group[0], mo_group[1]);
        auto epsilon_mo2 = eps_a_old.get(mo_group[0], mo_group[2]);
        C_new->set_column(mo_group[0], mo_group[2], C_mo1);
        C_new->set_column(mo_group[0], mo_group[1], C_mo2);
        eps_a_new.set(mo_group[0], mo_group[2], epsilon_mo1);
        eps_a_new.set(mo_group[0], mo_group[1], epsilon_mo2);
    }
    // Update local copy of the orbitals
    Ca_->copy(C_new);
    Cb_->copy(C_new);

    // Copy to psi::Wavefunction
    wfn_->Ca()->copy(C_new);
    wfn_->Cb()->copy(C_new);
    wfn_->epsilon_a()->copy(eps_a_new);
    wfn_->epsilon_b()->copy(eps_a_new);
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

std::vector<std::shared_ptr<psi::Matrix>> Psi4Integrals::mo_dipole_ints(const bool& alpha,
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
        // figure out the correspondence between C1 and Pitzer
        std::vector<std::tuple<double, int, int>> order;
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < nmopi_[h]; ++i) {
                order.emplace_back(epsilon->get(h, i), i, h);
            }
        }
        std::sort(order.begin(), order.end());

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

void Psi4Integrals::make_fock_matrix(ambit::Tensor gamma_a, ambit::Tensor gamma_b) {
    // build inactive Fock
    auto rdoccpi = mo_space_info_->dimension("INACTIVE_DOCC");
    auto fock_closed = make_fock_inactive(psi::Dimension(nirrep_), rdoccpi);

    // build active Fock
    auto fock_active = make_fock_active(gamma_a, gamma_b);

    if (std::get<0>(fock_active) == std::get<1>(fock_active)) {
        // restricted orbitals and ms-averaged RDMs
        fock_a_ = std::get<0>(fock_closed)->clone();
        fock_a_->add(std::get<0>(fock_active));
        fock_a_->set_name("Fock");
        fock_b_ = fock_a_;
    } else {
        // unrestricted orbitals or non-singlet RDMs
        fock_a_ = std::get<0>(fock_closed)->clone();
        fock_a_->add(std::get<0>(fock_active));
        fock_a_->set_name("Fock alpha");

        fock_b_ = std::get<1>(fock_closed)->clone();
        fock_b_->add(std::get<1>(fock_active));
        fock_b_->set_name("Fock beta");
    }
}

std::tuple<psi::SharedMatrix, psi::SharedMatrix, double>
Psi4Integrals::make_fock_inactive(psi::Dimension dim_start, psi::Dimension dim_end) {
    /* F_closed = Hcore + Vclosed in AO basis
     *
     * Vclosed = D_{uv}^{inactive docc} * (2 * (uv|rs) - (us|rv))
     * D_{uv}^{inactive docc} = \sum_{i}^{inactive docc} C_{ui} * C_{vi}
     *
     * u,v,r,s: AO indices; i: MO indices
     */
    if (JK_status_ == JKStatus::finalized) {
        outfile->Printf("\n  JK object had been finalized. JK is about to be initialized.\n");
        jk_initialize(0.7);
    }

    auto dim = dim_end - dim_start;

    if (spin_restriction_ == IntegralSpinRestriction::Restricted) {
        auto Csub = std::make_shared<psi::Matrix>("Ca_sub", nsopi_, dim);

        for (int h = 0; h < nirrep_; ++h) {
            for (int p = 0, offset = dim_start[h]; p < dim[h]; ++p) {
                Csub->set_column(h, p, Ca_->get_column(h, p + offset));
            }
        }

        // JK build
        JK_->set_do_K(true);
        std::vector<std::shared_ptr<psi::Matrix>>& Cls = JK_->C_left();
        std::vector<std::shared_ptr<psi::Matrix>>& Crs = JK_->C_right();
        Cls.clear();
        Crs.clear();

        Cls.push_back(Csub);

        JK_->compute();

        auto J = JK_->J()[0];
        J->scale(2.0);
        J->subtract(JK_->K()[0]);
        J->add(wfn_->H());

        // transform to MO
        auto F_closed = psi::linalg::triplet(Ca_, J, Ca_, true, false, false);
        F_closed->set_name("Fock_closed");

        // compute closed-shell energy
        J->add(wfn_->H());
        double e_closed = J->vector_dot(psi::linalg::doublet(Csub, Csub, false, true));

        // pass AO fock to psi4 Wavefunction
        if (wfn_->Fa() != nullptr) {
            wfn_->Fa()->copy(J);
            wfn_->Fb() = wfn_->Fa();
            fock_ao_level_ = FockAOStatus::inactive;
        }

        return std::make_tuple(F_closed, F_closed, e_closed);
    } else {
        auto Ca_sub = std::make_shared<psi::Matrix>("Ca_sub", nsopi_, dim);
        auto Cb_sub = std::make_shared<psi::Matrix>("Cb_sub", nsopi_, dim);

        for (int h = 0; h < nirrep_; ++h) {
            for (int p = 0, offset = dim_start[h]; p < dim[h]; ++p) {
                Ca_sub->set_column(h, p, Ca_->get_column(h, p + offset));
                Cb_sub->set_column(h, p, Cb_->get_column(h, p + offset));
            }
        }

        auto Fa_closed = wfn_->H()->clone();
        auto Fb_closed = wfn_->H()->clone();

        // JK build
        JK_->set_do_K(true);
        std::vector<std::shared_ptr<psi::Matrix>>& Cls = JK_->C_left();
        std::vector<std::shared_ptr<psi::Matrix>>& Crs = JK_->C_right();
        Crs.clear();

        Cls.clear();
        Cls.push_back(Ca_sub);
        Cls.push_back(Cb_sub);

        JK_->compute();

        // some algebra
        Fa_closed->add(JK_->J()[0]);
        Fa_closed->subtract(JK_->K()[0]);
        Fa_closed->add(JK_->J()[1]);

        Fb_closed->add(JK_->J()[0]);
        Fb_closed->add(JK_->J()[1]);
        Fb_closed->subtract(JK_->K()[1]);

        // save a copy of AO Fock
        auto J = JK_->J()[0];
        auto K = JK_->K()[0];
        J->copy(Fa_closed);
        K->copy(Fb_closed);

        // transform to MO basis
        Fa_closed = psi::linalg::triplet(Ca_, Fa_closed, Ca_, true, false, false);
        Fa_closed->set_name("Fock_closed alpha");
        Fb_closed = psi::linalg::triplet(Cb_, Fb_closed, Cb_, true, false, false);
        Fb_closed->set_name("Fock_closed beta");

        // compute closed-shell energy using unrestricted equation
        J->add(wfn_->H());
        K->add(wfn_->H());
        double e_closed = 0.5 * J->vector_dot(psi::linalg::doublet(Ca_sub, Ca_sub, false, true));
        e_closed += 0.5 * K->vector_dot(psi::linalg::doublet(Cb_sub, Cb_sub, false, true));

        // pass AO fock to psi4 Wavefunction
        if (wfn_->Fa() != nullptr) {
            wfn_->Fa()->copy(J);
            wfn_->Fb()->copy(K);
            fock_ao_level_ = FockAOStatus::inactive;
        }

        return std::make_tuple(Fa_closed, Fb_closed, e_closed);
    }
}

std::tuple<psi::SharedMatrix, psi::SharedMatrix> Psi4Integrals::make_fock_active(ambit::Tensor Da,
                                                                                 ambit::Tensor Db) {
    // Implementation Notes (in AO basis)
    // F_active = D_{uv}^{active} * ( (uv|rs) - 0.5 * (us|rv) )
    // D_{uv}^{active} = \sum_{xy}^{active} C_{ux} * C_{vy} * Gamma1_{xy}

    if (Da.dims() != Db.dims()) {
        throw std::runtime_error("Different dimensions of alpha and beta 1RDM!");
    }
    if (mo_space_info_->size("ACTIVE") != Da.dim(0)) {
        throw std::runtime_error("Inconsistent number of active orbitals");
    }

    // test if spin equivalence between 1RDM
    bool rdm_eq_spin = true;
    auto gamma = Da.clone();
    gamma("pq") -= Db("pq");
    double diff_max = gamma.norm(0);
    if (diff_max > options_->get_double("R_CONVERGENCE") or
        diff_max > options_->get_double("D_CONVERGENCE")) {
        print_h1("Warning from Forte Fock build (active)");
        outfile->Printf("\n  Unequivalent alpha and beta 1RDMs.");
        outfile->Printf("\n  Largest difference between alpha and beta: %.15f", diff_max);
        outfile->Printf("\n  Use unrestricted formalism to build Fock martix!\n");
        rdm_eq_spin = false;
    }

    // general setup
    auto nactvpi = mo_space_info_->dimension("ACTIVE");
    auto nactv = mo_space_info_->size("ACTIVE");
    auto& Da_data = Da.data();
    auto& Db_data = Db.data();

    if (rdm_eq_spin and spin_restriction_ == IntegralSpinRestriction::Restricted) {
        // fill in density (spin-summed)
        auto g1 = std::make_shared<psi::Matrix>("1RDM", nactvpi, nactvpi);
        for (int h = 0, offset = 0; h < nirrep_; ++h) {
            for (int i = 0; i < nactvpi[h]; ++i) {
                auto ni = i + offset;
                for (int j = 0; j < nactvpi[h]; ++j) {
                    auto nj = j + offset;
                    double v = Da_data[ni * nactv + nj] + Db_data[ni * nactv + nj];
                    g1->set(h, i, j, v);
                }
            }
            offset += nactvpi[h];
        }

        auto F_active = make_fock_active_restricted(g1);

        return {F_active, F_active};
    } else {
        // fill in density
        auto g1a = std::make_shared<psi::Matrix>("1RDMa", nactvpi, nactvpi);
        auto g1b = std::make_shared<psi::Matrix>("1RDMb", nactvpi, nactvpi);
        for (int h = 0, offset = 0; h < nirrep_; ++h) {
            for (int i = 0; i < nactvpi[h]; ++i) {
                auto ni = i + offset;
                for (int j = 0; j < nactvpi[h]; ++j) {
                    auto nj = j + offset;
                    g1a->set(h, i, j, Da_data[ni * nactv + nj]);
                    g1b->set(h, i, j, Db_data[ni * nactv + nj]);
                }
            }
            offset += nactvpi[h];
        }

        return make_fock_active_unrestricted(g1a, g1b);
    }
}

psi::SharedMatrix Psi4Integrals::make_fock_active_restricted(psi::SharedMatrix g1) {
    if (JK_status_ == JKStatus::finalized) {
        outfile->Printf("\n  JK object had been finalized. JK is about to be initialized.\n");
        jk_initialize(0.7);
    }

    auto nactvpi = mo_space_info_->dimension("ACTIVE");
    auto ndoccpi = mo_space_info_->dimension("INACTIVE_DOCC");

    // grab sub-block of Ca
    auto Cactv = std::make_shared<psi::Matrix>("Ca_sub", nsopi_, nactvpi);

    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0, offset = ndoccpi[h]; p < nactvpi[h]; ++p) {
            Cactv->set_column(h, p, Ca_->get_column(h, p + offset));
        }
    }

    // dress Cactv by one-density, which will the C_right for JK
    auto Cactv_dressed = psi::linalg::doublet(Cactv, g1, false, false);

    // JK build
    JK_->set_do_K(true);
    std::vector<std::shared_ptr<psi::Matrix>>& Cls = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Crs = JK_->C_right();
    Cls.clear();
    Crs.clear();

    Cls.push_back(Cactv);
    Crs.push_back(Cactv_dressed);

    JK_->compute();

    auto K = JK_->K()[0];
    K->scale(-0.5);
    K->add(JK_->J()[0]);

    // transform to MO
    auto F_active = psi::linalg::triplet(Ca_, K, Ca_, true, false, false);
    F_active->set_name("Fock_active");

    // pass AO fock to psi4 Wavefunction
    if (fock_ao_level_ == FockAOStatus::inactive) {
        wfn_->Fa()->add(K);
        fock_ao_level_ = FockAOStatus::generalized;
    }

    return F_active;
}

std::tuple<psi::SharedMatrix, psi::SharedMatrix>
Psi4Integrals::make_fock_active_unrestricted(psi::SharedMatrix g1a, psi::SharedMatrix g1b) {
    if (JK_status_ == JKStatus::finalized) {
        outfile->Printf("\n  JK object had been finalized. JK is about to be initialized.\n");
        jk_initialize(0.7);
    }

    auto nactvpi = mo_space_info_->dimension("ACTIVE");
    auto ndoccpi = mo_space_info_->dimension("INACTIVE_DOCC");

    // grab sub-block of Ca and Cb
    auto Ca_actv = std::make_shared<psi::Matrix>("Ca active", nsopi_, nactvpi);
    auto Cb_actv = std::make_shared<psi::Matrix>("Cb active", nsopi_, nactvpi);

    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0, offset = ndoccpi[h]; p < nactvpi[h]; ++p) {
            Ca_actv->set_column(h, p, Ca_->get_column(h, p + offset));
            Cb_actv->set_column(h, p, Cb_->get_column(h, p + offset));
        }
    }

    // dress Cactv by one-density, which will the C_right for JK
    auto Ca_actv_dressed = psi::linalg::doublet(Ca_actv, g1a, false, false);
    auto Cb_actv_dressed = psi::linalg::doublet(Cb_actv, g1b, false, false);

    // JK build
    JK_->set_do_K(true);
    std::vector<std::shared_ptr<psi::Matrix>>& Cls = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Crs = JK_->C_right();
    Cls.clear();
    Crs.clear();

    Cls.push_back(Ca_actv);
    Crs.push_back(Ca_actv_dressed);
    Cls.push_back(Cb_actv);
    Crs.push_back(Cb_actv_dressed);

    JK_->compute();

    // some algebra
    auto Ka = JK_->K()[0];
    Ka->scale(-1.0);
    Ka->add(JK_->J()[0]);
    Ka->add(JK_->J()[1]);

    auto Kb = JK_->K()[1];
    Kb->scale(-1.0);
    Kb->add(JK_->J()[0]);
    Kb->add(JK_->J()[1]);

    // transform to MO
    auto Fa_active = psi::linalg::triplet(Ca_, Ka, Ca_, true, false, false);
    Fa_active->set_name("Fock_active alpha");
    auto Fb_active = psi::linalg::triplet(Cb_, Kb, Cb_, true, false, false);
    Fb_active->set_name("Fock_active beta");

    // pass AO fock to psi4 Wavefunction
    if (fock_ao_level_ == FockAOStatus::inactive) {
        if (wfn_->Fa() == wfn_->Fb()) {
            Ka->add(Kb);
            Ka->scale(0.5);
            wfn_->Fa()->add(Ka);
        } else {
            wfn_->Fa()->add(Ka);
            wfn_->Fb()->add(Kb);
        }
        fock_ao_level_ = FockAOStatus::generalized;
    }

    return {Fa_active, Fb_active};
}
} // namespace forte
