/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "psi4/libpsio/psio.hpp"
#include "psi4/libtrans/integraltransform.h"

#include "helpers/blockedtensorfactory.h"
#include "base_classes/mo_space_info.h"
#include "helpers/timer.h"

#include "conventional_integrals.h"
#include "integrals/active_space_integrals.h"

#define ID(x) integral_transform->DPD_ID(x)

using namespace psi;

namespace forte {

/**
 * @brief ConventionalIntegrals::ConventionalIntegrals
 * @param options - psi options class
 * @param restricted - type of integral transformation
 * @param resort_frozen_core -
 */
ConventionalIntegrals::ConventionalIntegrals(psi::Options& options,
                                             std::shared_ptr<psi::Wavefunction> ref_wfn,
                                             std::shared_ptr<MOSpaceInfo> mo_space_info,
                                             IntegralSpinRestriction restricted)
    : ForteIntegrals(options, ref_wfn, mo_space_info, restricted) {

    integral_type_ = Conventional;
    print_info();
    outfile->Printf("\n  Overall Conventional Integrals timings\n\n");
    local_timer ConvTime;

    // Allocate the memory required to store the two-electron integrals
    aphys_tei_aa.assign(num_aptei_, 0.0);
    aphys_tei_ab.assign(num_aptei_, 0.0);
    aphys_tei_bb.assign(num_aptei_, 0.0);

    gather_integrals();
    freeze_core_orbitals();

    outfile->Printf("\n  Conventional integrals take %8.8f s", ConvTime.get());
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
    return aphys_tei_aa[aptei_index(p, q, r, s)];
}

double ConventionalIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_ab[aptei_index(p, q, r, s)];
}

double ConventionalIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_bb[aptei_index(p, q, r, s)];
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

ambit::Tensor ConventionalIntegrals::three_integral_block(const std::vector<size_t>&,
                                                          const std::vector<size_t>&,
                                                          const std::vector<size_t>&) {
    outfile->Printf("\n Oh no!, you tried to grab a ThreeIntegral but this "
                    "is not there!!");
    throw psi::PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral");
}

ambit::Tensor ConventionalIntegrals::three_integral_block_two_index(const std::vector<size_t>&,
                                                                    size_t,
                                                                    const std::vector<size_t>&) {
    outfile->Printf("\n Oh no! this isn't here");
    throw psi::PSIEXCEPTION("INT_TYPE=DISKDF");
}

double** ConventionalIntegrals::three_integral_pointer() {
    outfile->Printf("\n Doh! There is no Three_integral here.  Use DF/CD");
    throw psi::PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral!");
}

size_t ConventionalIntegrals::nthree() const { throw psi::PSIEXCEPTION("Wrong Int_Type"); }

void ConventionalIntegrals::set_tei(size_t p, size_t q, size_t r, size_t s, double value,
                                    bool alpha1, bool alpha2) {
    size_t index = aptei_index(p, q, r, s);
    if (alpha1 == true and alpha2 == true)
        aphys_tei_aa[index] = value;
    if (alpha1 == true and alpha2 == false)
        aphys_tei_ab[index] = value;
    if (alpha1 == false and alpha2 == false)
        aphys_tei_bb[index] = value;
}

void ConventionalIntegrals::set_tei_from_asints(std::shared_ptr<ActiveSpaceIntegrals> as_ints, bool alpha1, bool alpha2) {
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
                    size_t index = aptei_index(p, q, r, s);
                    if (alpha1 == true and alpha2 == true)
                        aphys_tei_aa[index] = as_ints->tei_aa(p, q, r, s);
                    if (alpha1 == true and alpha2 == false)
                        aphys_tei_ab[index] = as_ints->tei_ab(p, q, r, s);
                    if (alpha1 == false and alpha2 == false)
                        aphys_tei_bb[index] = as_ints->tei_bb(p, q, r, s);
                }
            }
        }
    }
}

void ConventionalIntegrals::set_tei_from_another_ints(std::shared_ptr<ForteIntegrals> ints_b, bool alpha1, bool alpha2, int ncmo_star) {
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
                    if (p > ncmo_star - 1 && q > ncmo_star - 1) {
                        size_t index = aptei_index(p, q, r, s);
                        if (alpha1 == true and alpha2 == true)
                            aphys_tei_aa[index] = ints_b->aptei_aa(p - ncmo_star, q - ncmo_star, r - ncmo_star, s - ncmo_star);
                        if (alpha1 == true and alpha2 == false)
                            aphys_tei_ab[index] = ints_b->aptei_ab(p - ncmo_star, q - ncmo_star, r - ncmo_star, s - ncmo_star);
                        if (alpha1 == false and alpha2 == false)
                            aphys_tei_bb[index] = ints_b->aptei_bb(p - ncmo_star, q - ncmo_star, r - ncmo_star, s - ncmo_star);
                    }
                }
            }
        }
    }
}

void ConventionalIntegrals::build_from_asints(std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    outfile->Printf("\n  Updating one-electron integrals from Hbar");
    set_oei_from_asints(as_ints, true);
    set_oei_from_asints(as_ints, false);
    outfile->Printf("\n  Updating two-electron integrals from Hbar");
    set_tei_from_asints(as_ints, true, true);
    set_tei_from_asints(as_ints, true, false);
    set_tei_from_asints(as_ints, false, false);
}

void ConventionalIntegrals::build_from_another_ints(std::shared_ptr<ForteIntegrals> ints_b, int ncmo_star) {
    outfile->Printf("\n  Updating one-electron integrals from new ints");
    set_oei_from_another_ints(ints_b, true, ncmo_star);
    set_oei_from_another_ints(ints_b, false, ncmo_star);
    outfile->Printf("\n  Updating two-electron integrals from new ints");
    set_tei_from_another_ints(ints_b, true, true, ncmo_star);
    set_tei_from_another_ints(ints_b, true, false, ncmo_star);
    set_tei_from_another_ints(ints_b, false, false, ncmo_star);
}

void ConventionalIntegrals::gather_integrals() {
    MintsHelper mints = MintsHelper(wfn_->basisset());
    mints.integrals();
    auto integral_transform = transform_integrals();

    if (print_ > 0) {
        outfile->Printf("\n  Reading the two-electron integrals from disk");
        outfile->Printf("\n  Size of two-electron integrals: %10.6f GB",
                        double(3 * 8 * num_aptei_) / 1073741824.0);
    }
    int_mem_ = sizeof(double) * 3 * 8 * num_aptei_ / 1073741824.0;
    for (size_t pqrs = 0; pqrs < num_aptei_; ++pqrs)
        aphys_tei_aa[pqrs] = 0.0;
    for (size_t pqrs = 0; pqrs < num_aptei_; ++pqrs)
        aphys_tei_ab[pqrs] = 0.0;
    for (size_t pqrs = 0; pqrs < num_aptei_; ++pqrs)
        aphys_tei_bb[pqrs] = 0.0;

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
                        aphys_tei_aa[index] = direct - exchange;
                        aphys_tei_ab[index] = direct;
                        aphys_tei_bb[index] = direct - exchange;
                    }
                }
            }
        }
    } else {
        outfile->Printf("\n  Unrestricted orbitals are currently disabled");
        throw psi::PSIEXCEPTION("Unrestricted orbitals are currently disabled in "
                                "ConventionalIntegrals");

        //        std::vector<double> two_electron_integrals(num_tei_, 0.0);
        //        // Alpha-alpha integrals

        //        // Read the integrals
        //        struct iwlbuf V_AAAA;
        //        iwl_buf_init(&V_AAAA, PSIF_MO_AA_TEI, 0.0, 1, 1);
        //        iwl_buf_rd_all(&V_AAAA, two_electron_integrals, myioff, myioff, 0, myioff, 0,
        //        "outfile"); iwl_buf_close(&V_AAAA, 1);

        //        for (size_t p = 0; p < nmo_; ++p) {
        //            for (size_t q = 0; q < nmo_; ++q) {
        //                for (size_t r = 0; r < nmo_; ++r) {
        //                    for (size_t s = 0; s < nmo_; ++s) {
        //                        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) -
        //                        (ps | qr) double direct = two_electron_integrals[INDEX4(p, r, q,
        //                        s)]; double exchange = two_electron_integrals[INDEX4(p, s, q, r)];
        //                        size_t index = aptei_index(p, q, r, s);
        //                        aphys_tei_aa[index] = direct - exchange;
        //                    }
        //                }
        //            }
        //        }

        //        // Beta-beta integrals
        //        // Zero the memory, because iwl_buf_rd_all copies only the nonzero entries
        //        std::fill(two_electron_integrals.begin(), two_electron_integrals.end(), 0.0);

        //        // Read the integrals
        //        struct iwlbuf V_BBBB;
        //        iwl_buf_init(&V_BBBB, PSIF_MO_BB_TEI, 0.0, 1, 1);
        //        iwl_buf_rd_all(&V_BBBB, two_electron_integrals, myioff, myioff, 0, myioff, 0,
        //        "outfile"); iwl_buf_close(&V_BBBB, 1);

        //        for (size_t p = 0; p < nmo_; ++p) {
        //            for (size_t q = 0; q < nmo_; ++q) {
        //                for (size_t r = 0; r < nmo_; ++r) {
        //                    for (size_t s = 0; s < nmo_; ++s) {
        //                        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) -
        //                        (ps | qr) double direct = two_electron_integrals[INDEX4(p, r, q,
        //                        s)]; double exchange = two_electron_integrals[INDEX4(p, s, q, r)];
        //                        size_t index = aptei_index(p, q, r, s);
        //                        aphys_tei_bb[index] = direct - exchange;
        //                    }
        //                }
        //            }
        //        }

        //        // Alpha-beta integrals
        //        Matrix Tei(num_oei, num_oei);
        //        double** two_electron_integrals_ab = Tei.pointer();
        //        // Zero the memory, because iwl_buf_rd_all copies only the
        //        nonzero entries for (size_t pq = 0; pq < num_oei; ++pq) {
        //            for (size_t rs = 0; rs < num_oei; ++rs) {
        //                two_electron_integrals_ab[pq][rs] = 0.0;
        //            }
        //        }

        //        // Read the integrals
        //        struct iwlbuf V_AABB;
        //        iwl_buf_init(&V_AABB, PSIF_MO_AB_TEI, 0.0, 1, 1);
        //        iwl_buf_rd_all2(&V_AABB, two_electron_integrals_ab, myioff, myioff, 0, myioff, 0,
        //                        "outfile");
        //        iwl_buf_close(&V_AABB, 1);

        //        for (size_t p = 0; p < nmo_; ++p) {
        //            for (size_t q = 0; q < nmo_; ++q) {
        //                for (size_t r = 0; r < nmo_; ++r) {
        //                    for (size_t s = 0; s < nmo_; ++s) {
        //                        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) -
        //                        (ps | qr) double direct =
        //                            two_electron_integrals_ab[INDEX2(p, r)][INDEX2(q, s)];
        //                        size_t index = aptei_index(p, q, r, s);
        //                        aphys_tei_ab[index] = direct;
        //                    }
        //                }
        //            }
        //        }
    }
}

void ConventionalIntegrals::resort_integrals_after_freezing() {
    if (print_ > 0) {
        outfile->Printf("\n  Resorting integrals after freezing core.");
    }
    // Resort the four-index integrals
    resort_four(aphys_tei_aa, cmotomo_);
    resort_four(aphys_tei_ab, cmotomo_);
    resort_four(aphys_tei_bb, cmotomo_);
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

void ConventionalIntegrals::make_fock_matrix(std::shared_ptr<psi::Matrix> gamma_a,
                                             std::shared_ptr<psi::Matrix> gamma_b) {
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            fock_matrix_a_[p * ncmo_ + q] = oei_a(p, q);
            fock_matrix_b_[p * ncmo_ + q] = oei_b(p, q);
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
                        fock_matrix_a_[p * ncmo_ + q] += aptei_aa(p, r, q, s) * gamma_a_rs;
                        fock_matrix_b_[p * ncmo_ + q] += aptei_ab(r, p, s, q) * gamma_a_rs;
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
                        fock_matrix_a_[p * ncmo_ + q] += aptei_ab(p, r, q, s) * gamma_b_rs;
                        fock_matrix_b_[p * ncmo_ + q] += aptei_bb(p, r, q, s) * gamma_b_rs;
                    }
                }
            }
        }
    }
}
} // namespace forte
