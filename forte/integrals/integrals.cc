/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <numeric>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/matrix.h"

#include "helpers/blockedtensorfactory.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "integrals.h"
#include "memory.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

using namespace psi;
using namespace ambit;

namespace forte {

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif

std::map<IntegralType, std::string> int_type_label{
    {Conventional, "Conventional"},          {DF, "Density fitting"},
    {Cholesky, "Cholesky decomposition"},    {DiskDF, "Disk-based density fitting"},
    {DistDF, "Distributed density fitting"}, {Custom, "Custom"}};

ForteIntegrals::ForteIntegrals(std::shared_ptr<ForteOptions> options,
                               std::shared_ptr<psi::Wavefunction> ref_wfn,
                               std::shared_ptr<MOSpaceInfo> mo_space_info,
                               IntegralType integral_type, IntegralSpinRestriction restricted)
    : options_(options), mo_space_info_(mo_space_info), wfn_(ref_wfn),
      integral_type_(integral_type), spin_restriction_(restricted), frozen_core_energy_(0.0),
      scalar_energy_(0.0) {
    common_initialize();
}

ForteIntegrals::ForteIntegrals(std::shared_ptr<ForteOptions> options,
                               std::shared_ptr<MOSpaceInfo> mo_space_info,
                               IntegralType integral_type, IntegralSpinRestriction restricted)
    : options_(options), mo_space_info_(mo_space_info), integral_type_(integral_type),
      spin_restriction_(restricted) {
    common_initialize();
}

void ForteIntegrals::common_initialize() {
    read_information();

    if (not skip_build_) {
        allocate();
    }
}
void ForteIntegrals::read_information() {
    // Extract information from options
    print_ = options_->get_int("PRINT");

    nirrep_ = mo_space_info_->nirrep();
    nmopi_ = mo_space_info_->dimension("ALL");
    frzcpi_ = mo_space_info_->dimension("FROZEN_DOCC");
    frzvpi_ = mo_space_info_->dimension("FROZEN_UOCC");
    ncmopi_ = mo_space_info_->dimension("CORRELATED");

    nmo_ = nmopi_.sum();
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

    mo_to_relmo_ = mo_space_info_->relative_mo("ALL");

    // Set the indexing to work using the number of molecular integrals
    aptei_idx_ = nmo_;
    num_tei_ = INDEX4(nmo_ - 1, nmo_ - 1, nmo_ - 1, nmo_ - 1) + 1;
    num_aptei_ = nmo_ * nmo_ * nmo_ * nmo_;
    num_threads_ = omp_get_max_threads();

    // skip integral allocation and transformation if doing CASSCF
    auto job_type = options_->get_str("JOB_TYPE");
    skip_build_ = (job_type == "MCSCF_TWO_STEP") and (integral_type_ != Custom);
}

void ForteIntegrals::allocate() {
    // full one-electron integrals
    full_one_electron_integrals_a_.assign(nmo_ * nmo_, 0.0);
    full_one_electron_integrals_b_.assign(nmo_ * nmo_, 0.0);

    // these will hold only the correlated part
    one_electron_integrals_a_.assign(ncmo_ * ncmo_, 0.0);
    one_electron_integrals_b_.assign(ncmo_ * ncmo_, 0.0);

    if ((integral_type_ == Conventional) or (integral_type_ == Custom)) {
        // Allocate the memory required to store the two-electron integrals
        aphys_tei_aa_.assign(num_aptei_, 0.0);
        aphys_tei_ab_.assign(num_aptei_, 0.0);
        aphys_tei_bb_.assign(num_aptei_, 0.0);

        int_mem_ = sizeof(double) * 3 * 8 * num_aptei_ / 1073741824.0;
    }
}

std::shared_ptr<psi::Matrix> ForteIntegrals::Ca() const { return Ca_; }

std::shared_ptr<psi::Matrix> ForteIntegrals::Cb() const { return Cb_; }

double ForteIntegrals::nuclear_repulsion_energy() const { return nucrep_; }

std::shared_ptr<psi::Wavefunction> ForteIntegrals::wfn() { return wfn_; }

std::shared_ptr<psi::JK> ForteIntegrals::jk() { return JK_; }

ForteIntegrals::JKStatus ForteIntegrals::jk_status() { return JK_status_; }

void ForteIntegrals::jk_finalize() {
    if (JK_status_ == JKStatus::initialized) {
        JK_->finalize();
        // nothing done in finalize() for PKJK and MemDFJK
        if (integral_type_ == DiskDF or integral_type_ == Cholesky)
            JK_status_ = JKStatus::finalized;
    }
}

size_t ForteIntegrals::nso() const { return nso_; }

size_t ForteIntegrals::nmo() const { return nmo_; }

int ForteIntegrals::nirrep() const { return nirrep_; }

const psi::Dimension& ForteIntegrals::frzcpi() const { return frzcpi_; }

const psi::Dimension& ForteIntegrals::frzvpi() const { return frzvpi_; }

const psi::Dimension& ForteIntegrals::nsopi() const { return nsopi_; }

const psi::Dimension& ForteIntegrals::ncmopi() const { return ncmopi_; }

size_t ForteIntegrals::ncmo() const { return ncmo_; }

const std::vector<size_t>& ForteIntegrals::cmotomo() const { return cmotomo_; }

void ForteIntegrals::set_print(int print) { print_ = print; }

double ForteIntegrals::frozen_core_energy() { return frozen_core_energy_; }

double ForteIntegrals::scalar() const { return scalar_energy_; }

double ForteIntegrals::oei_a(size_t p, size_t q) const {
    return one_electron_integrals_a_[p * aptei_idx_ + q];
}

double ForteIntegrals::oei_b(size_t p, size_t q) const {
    return one_electron_integrals_b_[p * aptei_idx_ + q];
}

ambit::Tensor ForteIntegrals::oei_a_block(const std::vector<size_t>& p,
                                          const std::vector<size_t>& q) {
    ambit::Tensor t = ambit::Tensor::build(ambit::CoreTensor, "oei_a", {p.size(), q.size()});
    t.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = oei_a(p[i[0]], q[i[1]]); });
    return t;
}

ambit::Tensor ForteIntegrals::oei_b_block(const std::vector<size_t>& p,
                                          const std::vector<size_t>& q) {
    ambit::Tensor t = ambit::Tensor::build(ambit::CoreTensor, "oei_b", {p.size(), q.size()});
    t.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = oei_b(p[i[0]], q[i[1]]); });
    return t;
}

void ForteIntegrals::set_fock_matrix(std::shared_ptr<psi::Matrix> fa,
                                     std::shared_ptr<psi::Matrix> fb) {
    fock_a_ = fa;
    fock_b_ = fb;
}

double ForteIntegrals::get_fock_a(size_t p, size_t q, bool corr) const {
    auto p_full = p, q_full = q;
    if (corr) {
        p_full = cmotomo_[p], q_full = cmotomo_[q];
    }
    auto p_pair = mo_to_relmo_[p_full], q_pair = mo_to_relmo_[q_full];
    if (p_pair.first == q_pair.first) {
        return fock_a_->get(p_pair.first, p_pair.second, q_pair.second);
    } else {
        return 0.0;
    }
}

double ForteIntegrals::get_fock_b(size_t p, size_t q, bool corr) const {
    auto p_full = p, q_full = q;
    if (corr) {
        p_full = cmotomo_[p], q_full = cmotomo_[q];
    }
    auto p_pair = mo_to_relmo_[p_full], q_pair = mo_to_relmo_[q_full];
    if (p_pair.first == q_pair.first) {
        return fock_b_->get(p_pair.first, p_pair.second, q_pair.second);
    } else {
        return 0.0;
    }
}

std::shared_ptr<psi::Matrix> ForteIntegrals::get_fock_a(bool corr) const {
    if (corr) {
        auto dim_frzc = mo_space_info_->dimension("FROZEN_DOCC");
        auto dim_corr = mo_space_info_->dimension("CORRELATED");
        psi::Slice slice_corr(dim_frzc, dim_frzc + dim_corr);
        return fock_a_->get_block(slice_corr, slice_corr);
    } else {
        return fock_a_;
    }
}

std::shared_ptr<psi::Matrix> ForteIntegrals::get_fock_b(bool corr) const {
    if (corr) {
        auto dim_frzc = mo_space_info_->dimension("FROZEN_DOCC");
        auto dim_corr = mo_space_info_->dimension("CORRELATED");
        psi::Slice slice_corr(dim_frzc, dim_frzc + dim_corr);
        return fock_b_->get_block(slice_corr, slice_corr);
    } else {
        return fock_b_;
    }
}

void ForteIntegrals::set_nuclear_repulsion(double value) { nucrep_ = value; }

void ForteIntegrals::set_scalar(double value) { scalar_energy_ = value; }

void ForteIntegrals::set_oei_all(const std::vector<double>& oei_a,
                                 const std::vector<double>& oei_b) {
    full_one_electron_integrals_a_ = oei_a;
    full_one_electron_integrals_b_ = oei_b;
}

void ForteIntegrals::set_tei_all(const std::vector<double>& tei_aa,
                                 const std::vector<double>& tei_ab,
                                 const std::vector<double>& tei_bb) {
    aphys_tei_aa_ = tei_aa;
    aphys_tei_ab_ = tei_ab;
    aphys_tei_bb_ = tei_bb;
}

IntegralSpinRestriction ForteIntegrals::spin_restriction() const { return spin_restriction_; }

IntegralType ForteIntegrals::integral_type() const { return integral_type_; }

int ForteIntegrals::ga_handle() { return 0; }

std::vector<std::shared_ptr<psi::Matrix>> ForteIntegrals::ao_dipole_ints() const {
    return dipole_ints_ao_;
}

std::vector<std::shared_ptr<psi::Matrix>> ForteIntegrals::ao_quadrupole_ints() const {
    return quadrupole_ints_ao_;
}

// void ForteIntegrals::set_oei(double** ints, bool alpha) {
//    std::vector<double>& p_oei = alpha ? one_electron_integrals_a_ : one_electron_integrals_b_;
//    for (size_t p = 0; p < aptei_idx_; ++p) {
//        for (size_t q = 0; q < aptei_idx_; ++q) {
//            p_oei[p * aptei_idx_ + q] = ints[p][q];
//        }
//    }
//}

void ForteIntegrals::set_oei(size_t p, size_t q, double value, bool alpha) {
    std::vector<double>& p_oei = alpha ? one_electron_integrals_a_ : one_electron_integrals_b_;
    p_oei[p * aptei_idx_ + q] = value;
}

bool ForteIntegrals::fix_orbital_phases(std::shared_ptr<psi::Matrix> U, bool is_alpha, bool debug) {
    if (integral_type_ == Custom) {
        outfile->Printf("\n  Warning: Cannot fix orbital phases (%s) for CustomIntegrals.",
                        is_alpha ? "Ca" : "Cb");
        return false;
    }

    // MO overlap (old by new)
    // S_MO = Cold^T S_AO Cnew = Cold^T S_AO Cold U = U

    // transformation matrix
    auto T = std::make_shared<psi::Matrix>("Reordering matrix", U->rowspi(), U->colspi());

    for (int h = 0; h < nirrep_; ++h) {
        auto ncol = T->coldim(h);
        auto nrow = T->rowdim(h);
        for (int q = 0; q < ncol; ++q) {
            double max = 0.0, sign = 1.0;
            int p_temp = q;

            for (int p = 0; p < nrow; ++p) {
                double v = U->get(h, p, q);
                if (std::fabs(v) > max) {
                    max = std::fabs(v);
                    p_temp = p;
                    sign = v < 0 ? -1.0 : 1.0;
                }
            }

            T->set(h, p_temp, q, sign);
        }
    }

    // test transformation matrix
    bool trans_ok = true;
    for (int h = 0; h < nirrep_; ++h) {
        auto nrow = T->rowdim(h);
        auto ncol = T->coldim(h);

        for (int i = 0; i < nrow; ++i) {
            double sum = 0.0;
            for (int j = 0; j < ncol; ++j) {
                sum += std::fabs(T->get(h, i, j));
            }
            if (sum - 1.0 > 1.0e-3) {
                trans_ok = false;
                break;
            }
        }

        if (not trans_ok) {
            break;
        }
    }

    // transform Ua
    if (trans_ok) {
        auto Unew = psi::linalg::doublet(U, T, false, false);
        U->copy(Unew);
        return true;
    } else {
        psi::outfile->Printf("\n  Warning: Failed to fix orbital phase and order.");
        if (debug) {
            psi::outfile->Printf("\n  Printing the MO overlap and transformation matrix.\n");
            U->print();
            T->print();
        }
        return false;
    }
}

bool ForteIntegrals::test_orbital_spin_restriction(std::shared_ptr<psi::Matrix> A,
                                                   std::shared_ptr<psi::Matrix> B) const {
    auto A_minus_B = A->clone();
    A_minus_B->subtract(B);
    return A_minus_B->absmax() < 1.0e-7;
}

void ForteIntegrals::freeze_core_orbitals() {
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

void ForteIntegrals::print_info() {
    outfile->Printf("\n\n  ==> Integral Transformation <==\n");
    outfile->Printf("\n  Number of molecular orbitals:            %15d", nmopi_.sum());
    outfile->Printf("\n  Number of correlated molecular orbitals: %15zu", ncmo_);
    outfile->Printf("\n  Number of frozen occupied orbitals:      %15d", frzcpi_.sum());
    outfile->Printf("\n  Number of frozen unoccupied orbitals:    %15d", frzvpi_.sum());
    outfile->Printf("\n  Two-electron integral type:              %15s\n\n",
                    int_type_label[integral_type()].c_str());
    if (skip_build_) {
        outfile->Printf("\n  Skip integral allocation and transformation for AO-driven CASSCF.");
    }
}

void ForteIntegrals::print_ints() {
    //    Ca_->print();
    //    Cb_->print();
    outfile->Printf("\n  nmo_ = %zu", nmo_);

    outfile->Printf("\n  Nuclear repulsion energy: %20.12f", nucrep_);
    outfile->Printf("\n  Scalar energy:            %20.12f", scalar_energy_);
    outfile->Printf("\n  Frozen-core energt:       %20.12f", frozen_core_energy_);
    outfile->Printf("\n  Alpha one-electron integrals (T + V_{en})");
    Matrix ha(" Alpha one-electron integrals (T + V_{en})", nmo_, nmo_);
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            if (std::abs(oei_a(p, q)) >= 1e-14)
                outfile->Printf("\n  h[%6d][%6d] = %20.12f", p, q, oei_a(p, q));
        }
    }

    outfile->Printf("\n  Beta one-electron integrals (T + V_{en})");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            if (std::abs(oei_b(p, q)) >= 1e-14)
                outfile->Printf("\n  h[%6d][%6d] = %20.12f", p, q, oei_b(p, q));
        }
    }

    outfile->Printf("\n  Alpha-alpha two-electron integrals <pq||rs>");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            for (size_t r = 0; r < nmo_; ++r) {
                for (size_t s = 0; s < nmo_; ++s) {
                    if (std::abs(aptei_aa(p, q, r, s)) >= 1e-14)
                        outfile->Printf("\n  v[%6d][%6d][%6d][%6d] = %20.12f", p, q, r, s,
                                        aptei_aa(p, q, r, s));
                }
            }
        }
    }

    outfile->Printf("\n  Alpha-beta two-electron integrals <pq||rs>");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            for (size_t r = 0; r < nmo_; ++r) {
                for (size_t s = 0; s < nmo_; ++s) {
                    if (std::abs(aptei_ab(p, q, r, s)) >= 1e-14)
                        outfile->Printf("\n  v[%6d][%6d][%6d][%6d] = %20.12f", p, q, r, s,
                                        aptei_ab(p, q, r, s));
                }
            }
        }
    }
    outfile->Printf("\n  Beta-beta two-electron integrals <pq||rs>");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            for (size_t r = 0; r < nmo_; ++r) {
                for (size_t s = 0; s < nmo_; ++s) {
                    if (std::abs(aptei_bb(p, q, r, s)) >= 1e-14)
                        outfile->Printf("\n  v[%6d][%6d][%6d][%6d] = %20.12f", p, q, r, s,
                                        aptei_bb(p, q, r, s));
                }
            }
        }
    }
}

void ForteIntegrals::rotate_orbitals(std::shared_ptr<psi::Matrix> Ua,
                                     std::shared_ptr<psi::Matrix> Ub, bool re_transform) {
    // 1. Rotate the orbital coefficients and store them in the ForteIntegral object
    auto Ca_rotated = psi::linalg::doublet(Ca_, Ua);
    auto Cb_rotated = psi::linalg::doublet(Cb_, Ub);

    update_orbitals(Ca_rotated, Cb_rotated, re_transform);
}

// The following functions throw an error by default

void ForteIntegrals::update_orbitals(std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>,
                                     bool) {
    _undefined_function("update_orbitals");
}

void ForteIntegrals::compute_frozen_one_body_operator() {
    _undefined_function("compute_frozen_one_body_operator");
}

size_t ForteIntegrals::nthree() const {
    _undefined_function("nthree");
    return 0;
}

ambit::Tensor ForteIntegrals::three_integral_block(const std::vector<size_t>&,
                                                   const std::vector<size_t>&,
                                                   const std::vector<size_t>&,
                                                   ThreeIntsBlockOrder) {
    _undefined_function("three_integral_block");
    return ambit::Tensor();
}

ambit::Tensor ForteIntegrals::three_integral_block_two_index(const std::vector<size_t>&, size_t,
                                                             const std::vector<size_t>&) {
    _undefined_function("three_integral_block_two_index");
    return ambit::Tensor();
}

double** ForteIntegrals::three_integral_pointer() {
    _undefined_function("three_integral_pointer");
    return nullptr;
}

void ForteIntegrals::rotate_mos() { _undefined_function("rotate_mos"); }

void ForteIntegrals::build_multipole_ints_ao() { _undefined_function("build_multipole_ints_ao"); }

std::vector<std::shared_ptr<psi::Matrix>> ForteIntegrals::mo_dipole_ints() const {
    _undefined_function("mo_dipole_ints");
    return std::vector<std::shared_ptr<psi::Matrix>>();
}

std::vector<std::shared_ptr<psi::Matrix>> ForteIntegrals::mo_quadrupole_ints() const {
    _undefined_function("mo_quadrupole_ints");
    return std::vector<std::shared_ptr<psi::Matrix>>();
}

void ForteIntegrals::_undefined_function(const std::string& method) const {
    outfile->Printf("\n  ForteIntegrals::" + method + "not supported for integral type " +
                    std::to_string(integral_type()));
    throw std::runtime_error("ForteIntegrals::" + method + " not supported for integral type " +
                             std::to_string(integral_type()));
}

} // namespace forte
