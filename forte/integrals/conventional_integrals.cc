/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <cmath>

#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"
#include "psi4/libdpd/dpd.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libtrans/integraltransform.h"

#include "base_classes/mo_space_info.h"

#include "helpers/blockedtensorfactory.h"
#include "helpers/timer.h"
#include "helpers/printing.h"

#include "conventional_integrals.h"

#define ID(x) integral_transform->DPD_ID(x)

using namespace psi;

namespace forte {

ConventionalIntegrals::ConventionalIntegrals(std::shared_ptr<ForteOptions> options,
                                             std::shared_ptr<psi::Wavefunction> ref_wfn,
                                             std::shared_ptr<MOSpaceInfo> mo_space_info,
                                             IntegralSpinRestriction restricted)
    : Psi4Integrals(options, ref_wfn, mo_space_info, Conventional, restricted) {
    initialize();
}

void ConventionalIntegrals::initialize() {
    print_info();

    if (not skip_build_) {
        local_timer ConvTime;
        gather_integrals();
        freeze_core_orbitals();
        print_timing("computing conventional integrals", ConvTime.get());
    }
}

std::shared_ptr<psi::IntegralTransform> ConventionalIntegrals::transform_integrals() {

    // For now, we'll just transform for closed shells and generate all integrals
    std::vector<std::shared_ptr<MOSpace>> spaces;
    spaces.push_back(MOSpace::all);

    std::shared_ptr<psi::IntegralTransform> integral_transform;

    if (spin_restriction_ == IntegralSpinRestriction::Restricted) {
        integral_transform = std::make_shared<psi::IntegralTransform>(
            wfn_, spaces, psi::IntegralTransform::TransformationType::Restricted,
            psi::IntegralTransform::OutputType::DPDOnly,
            psi::IntegralTransform::MOOrdering::PitzerOrder,
            psi::IntegralTransform::FrozenOrbitals::None);
    } else {
        outfile->Printf("\n  Unrestricted orbitals are currently disabled");
        throw psi::PSIEXCEPTION("Unrestricted orbitals are currently disabled in "
                                "ConventionalIntegrals");
        integral_transform = std::make_shared<psi::IntegralTransform>(
            wfn_, spaces, psi::IntegralTransform::TransformationType::Unrestricted,
            psi::IntegralTransform::OutputType::DPDOnly,
            psi::IntegralTransform::MOOrdering::PitzerOrder,
            psi::IntegralTransform::FrozenOrbitals::None);
    }

    // Keep the SO integrals on disk in case we want to retransform them
    integral_transform->set_keep_iwl_so_ints(true);
    local_timer int_timer;
    integral_transform->transform_tei(MOSpace::all, MOSpace::all, MOSpace::all, MOSpace::all);

    dpd_set_default(integral_transform->get_dpd_id());
    if (print_ > 0) {
        outfile->Printf("\n  Integral transformation done. %8.8f s", int_timer.get());
    }
    return integral_transform;
}

double ConventionalIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_aa_[aptei_index(p, q, r, s)];
}

double ConventionalIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_ab_[aptei_index(p, q, r, s)];
}

double ConventionalIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_bb_[aptei_index(p, q, r, s)];
}

ambit::Tensor ConventionalIntegrals::aptei_aa_block(const std::vector<size_t>& p,
                                                    const std::vector<size_t>& q,
                                                    const std::vector<size_t>& r,
                                                    const std::vector<size_t>& s) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
        value = aptei_aa(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor ConventionalIntegrals::aptei_ab_block(const std::vector<size_t>& p,
                                                    const std::vector<size_t>& q,
                                                    const std::vector<size_t>& r,
                                                    const std::vector<size_t>& s) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
        value = aptei_ab(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor ConventionalIntegrals::aptei_bb_block(const std::vector<size_t>& p,
                                                    const std::vector<size_t>& q,
                                                    const std::vector<size_t>& r,
                                                    const std::vector<size_t>& s) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
        value = aptei_bb(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

void ConventionalIntegrals::set_tei(size_t p, size_t q, size_t r, size_t s, double value,
                                    bool alpha1, bool alpha2) {
    size_t index = aptei_index(p, q, r, s);
    if (alpha1 == true and alpha2 == true)
        aphys_tei_aa_[index] = value;
    if (alpha1 == true and alpha2 == false)
        aphys_tei_ab_[index] = value;
    if (alpha1 == false and alpha2 == false)
        aphys_tei_bb_[index] = value;
}

void ConventionalIntegrals::gather_integrals() {
    if (print_) {
        outfile->Printf("\n  Computing Conventional Integrals");
    }
    local_timer timer;
    MintsHelper mints = MintsHelper(wfn_->basisset());
    mints.integrals();
    auto integral_transform = transform_integrals();

    if (print_ > 0) {
        outfile->Printf("\n  Reading the two-electron integrals from disk");
        outfile->Printf("\n  Size of two-electron integrals: %10.6f GB",
                        double(3 * 8 * num_aptei_) / 1073741824.0);
    }
    int_mem_ = sizeof(double) * 3 * 8 * num_aptei_ / 1073741824.0;

    std::fill(aphys_tei_aa_.begin(), aphys_tei_aa_.end(), 0.0);
    std::fill(aphys_tei_ab_.begin(), aphys_tei_ab_.end(), 0.0);
    std::fill(aphys_tei_bb_.begin(), aphys_tei_bb_.end(), 0.0);

    if (spin_restriction_ == IntegralSpinRestriction::Restricted) {
        std::vector<double> two_electron_integrals(num_tei_, 0.0);

        // Read the integrals
        dpdbuf4 K;
        std::shared_ptr<PSIO> psio(_default_psio_lib_);
        psio->open(PSIF_LIBTRANS_DPD, PSIO_OPEN_OLD);
        // To only process the permutationally unique integrals, change the
        // ID("[A,A]") to ID("[A>=A]+")
        global_dpd_->buf4_init(&K, PSIF_LIBTRANS_DPD, 0, ID("[A,A]"), ID("[A,A]"), ID("[A>=A]+"),
                               ID("[A>=A]+"), 0, "MO Ints (AA|AA)");
        for (int h = 0; h < nirrep_; ++h) {
            global_dpd_->buf4_mat_irrep_init(&K, h);
            global_dpd_->buf4_mat_irrep_rd(&K, h);
            for (int pq = 0; pq < K.params->rowtot[h]; ++pq) {
                int p = K.params->roworb[h][pq][0];
                int q = K.params->roworb[h][pq][1];
                for (int rs = 0; rs < K.params->coltot[h]; ++rs) {
                    int r = K.params->colorb[h][rs][0];
                    int s = K.params->colorb[h][rs][1];
                    two_electron_integrals[INDEX4(p, q, r, s)] = K.matrix[h][pq][rs];
                }
            }
            global_dpd_->buf4_mat_irrep_close(&K, h);
        }
        global_dpd_->buf4_close(&K);
        psio->close(PSIF_LIBTRANS_DPD, PSIO_OPEN_OLD);

        // Store the integrals
        for (size_t p = 0; p < nmo_; ++p) {
            for (size_t q = 0; q < nmo_; ++q) {
                for (size_t r = 0; r < nmo_; ++r) {
                    for (size_t s = 0; s < nmo_; ++s) {
                        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
                        double direct = two_electron_integrals[INDEX4(p, r, q, s)];
                        double exchange = two_electron_integrals[INDEX4(p, s, q, r)];
                        size_t index = aptei_index(p, q, r, s);
                        aphys_tei_aa_[index] = direct - exchange;
                        aphys_tei_ab_[index] = direct;
                        aphys_tei_bb_[index] = direct - exchange;
                    }
                }
            }
        }
    } else {
        outfile->Printf("\n  Unrestricted orbitals are currently disabled");
        throw psi::PSIEXCEPTION("Unrestricted orbitals are currently disabled in "
                                "ConventionalIntegrals");
    }
    if (print_) {
        print_timing("conventional integral transformation", timer.get());
    }
}

void ConventionalIntegrals::resort_integrals_after_freezing() {
    if (print_ > 0) {
        outfile->Printf("\n  Resorting integrals after freezing core.");
    }
    // Resort the four-index integrals
    resort_four(aphys_tei_aa_, cmotomo_);
    resort_four(aphys_tei_ab_, cmotomo_);
    resort_four(aphys_tei_bb_, cmotomo_);
}

void ConventionalIntegrals::resort_four(std::vector<double>& tei, std::vector<size_t>& map) {
    // Store the integrals in a temporary array
    size_t num_aptei_corr = nmo_ * nmo_ * nmo_ * nmo_;
    std::vector<double> temp_ints(num_aptei_corr, 0.0);
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
                    size_t pqrs_cmo = ncmo_ * ncmo_ * ncmo_ * p + ncmo_ * ncmo_ * q + ncmo_ * r + s;
                    size_t pqrs_mo =
                        nmo_ * nmo_ * nmo_ * map[p] + nmo_ * nmo_ * map[q] + nmo_ * map[r] + map[s];
                    temp_ints[pqrs_cmo] = tei[pqrs_mo];
                }
            }
        }
    }
    temp_ints.swap(tei);
}
} // namespace forte
