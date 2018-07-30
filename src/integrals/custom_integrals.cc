/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libdpd/dpd.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"

#include "../blockedtensorfactory.h"

#include "custom_integrals.h"

#define ID(x) ints_->DPD_ID(x)

namespace psi {
namespace forte {

/**
 * @brief CustomIntegrals::CustomIntegrals
 * @param options - psi options class
 * @param restricted - type of integral transformation
 * @param resort_frozen_core -
 */
CustomIntegrals::CustomIntegrals(psi::Options& options, SharedWavefunction ref_wfn,
                                 IntegralSpinRestriction restricted,
                                 std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ForteIntegrals(options, ref_wfn, restricted, mo_space_info) {
    integral_type_ = Custom;

    outfile->Printf("\n  Using Custom integrals\n\n");

    allocate();

    gather_integrals();

    if (ncmo_ < nmo_) {
        freeze_core_orbitals();
        // Set the new value of the number of orbitals to be used in indexing
        // routines
        aptei_idx_ = ncmo_;
    }
}

CustomIntegrals::~CustomIntegrals() { deallocate(); }

void CustomIntegrals::allocate() {
    // Allocate the memory required to store the one-electron integrals

    // Allocate the memory required to store the two-electron integrals
    aphys_tei_aa.resize(num_aptei);
    aphys_tei_ab.resize(num_aptei);
    aphys_tei_bb.resize(num_aptei);
}

void CustomIntegrals::deallocate() {
    // nothing to deallocate
}

double CustomIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_aa[aptei_index(p, q, r, s)];
}

double CustomIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_ab[aptei_index(p, q, r, s)];
}

double CustomIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_bb[aptei_index(p, q, r, s)];
}

ambit::Tensor CustomIntegrals::aptei_aa_block(const std::vector<size_t>& p,
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

ambit::Tensor CustomIntegrals::aptei_ab_block(const std::vector<size_t>& p,
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

ambit::Tensor CustomIntegrals::aptei_bb_block(const std::vector<size_t>& p,
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

void CustomIntegrals::set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                              bool alpha2) {
    size_t index = aptei_index(p, q, r, s);
    if (alpha1 == true and alpha2 == true)
        aphys_tei_aa[index] = value;
    if (alpha1 == true and alpha2 == false)
        aphys_tei_ab[index] = value;
    if (alpha1 == false and alpha2 == false)
        aphys_tei_bb[index] = value;
}

void CustomIntegrals::gather_integrals() {

    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            one_electron_integrals_a[p * nmo_ + q] = 0.0;
            one_electron_integrals_b[p * nmo_ + q] = 0.0;
        }
    }

    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs)
        aphys_tei_aa[pqrs] = 0.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs)
        aphys_tei_ab[pqrs] = 0.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs)
        aphys_tei_bb[pqrs] = 0.0;

    // Store the integrals
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            for (size_t r = 0; r < nmo_; ++r) {
                for (size_t s = 0; s < nmo_; ++s) {
                    // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
                    double direct = 0.0;
                    double exchange = 0.0;
                    size_t index = aptei_index(p, q, r, s);
                    aphys_tei_aa[index] = direct - exchange;
                    aphys_tei_ab[index] = direct;
                    aphys_tei_bb[index] = direct - exchange;
                }
            }
        }
    }
}

void CustomIntegrals::resort_integrals_after_freezing() {
    if (print_ > 0) {
        outfile->Printf("\n  Resorting integrals after freezing core.");
    }
    // Resort the four-index integrals
    resort_four(aphys_tei_aa, cmotomo_);
    resort_four(aphys_tei_ab, cmotomo_);
    resort_four(aphys_tei_bb, cmotomo_);
}

void CustomIntegrals::resort_four(double*& tei, std::vector<size_t>& map) {
    // Store the integrals in a temporary array
    double* temp_ints = new double[num_aptei];
    for (size_t p = 0; p < num_aptei; ++p) {
        temp_ints[p] = 0.0;
    }
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
    // Delete old integrals and assign the pointer
    delete[] tei;
    tei = temp_ints;
}

void CustomIntegrals::resort_four(std::vector<double>& tei, std::vector<size_t>& map) {
    // Store the integrals in a temporary array
    std::vector<double> temp_ints(num_aptei, 0.0);
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
    // Swap old integrals with new
    tei.swap(temp_ints);
}

void CustomIntegrals::make_fock_matrix(SharedMatrix gamma_a, SharedMatrix gamma_b) {
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            fock_matrix_a[p * ncmo_ + q] = oei_a(p, q);
            fock_matrix_b[p * ncmo_ + q] = oei_b(p, q);
        }
    }
    double zero = 1e-12;
    /// TODO: Either use ambit or use structure of gamma.
    for (size_t r = 0; r < ncmo_; ++r) {
        for (size_t s = 0; s < ncmo_; ++s) {
            double gamma_a_rs = gamma_a->get(r, s);
            if (std::fabs(gamma_a_rs) > zero) {
                for (size_t p = 0; p < ncmo_; ++p) {
                    for (size_t q = 0; q < ncmo_; ++q) {
                        fock_matrix_a[p * ncmo_ + q] += aptei_aa(p, r, q, s) * gamma_a_rs;
                        fock_matrix_b[p * ncmo_ + q] += aptei_ab(r, p, s, q) * gamma_a_rs;
                    }
                }
            }
        }
    }
    for (size_t r = 0; r < ncmo_; ++r) {
        for (size_t s = 0; s < ncmo_; ++s) {
            double gamma_b_rs = gamma_b->get(r, s);
            if (std::fabs(gamma_b_rs) > zero) {
                for (size_t p = 0; p < ncmo_; ++p) {
                    for (size_t q = 0; q < ncmo_; ++q) {
                        fock_matrix_a[p * ncmo_ + q] += aptei_ab(p, r, q, s) * gamma_b_rs;
                        fock_matrix_b[p * ncmo_ + q] += aptei_bb(p, r, q, s) * gamma_b_rs;
                    }
                }
            }
        }
    }
}
} // namespace forte
} // namespace psi
