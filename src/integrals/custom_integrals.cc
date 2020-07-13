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

#include <cmath>
#include <fstream>

#include "psi4/libdpd/dpd.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"

#include "base_classes/mo_space_info.h"
#include "helpers/blockedtensorfactory.h"
#include "helpers/string_algorithms.h"
#include "helpers/timer.h"
#include "helpers/printing.h"

#include "custom_integrals.h"

#define IOFFINDEX(i) (i * (i + 1) / 2)
#define PAIRINDEX(i, j) ((i > j) ? (IOFFINDEX(i) + (j)) : (IOFFINDEX(j) + (i)))
#define four(i, j, k, l) PAIRINDEX(PAIRINDEX(i, j), PAIRINDEX(k, l))

using namespace psi;

namespace forte {

/**
 * @brief CustomIntegrals::CustomIntegrals
 * @param options - psi options class
 * @param restricted - type of integral transformation
 * @param resort_frozen_core -
 */
CustomIntegrals::CustomIntegrals(std::shared_ptr<ForteOptions> options,
                                 std::shared_ptr<MOSpaceInfo> mo_space_info,
                                 IntegralSpinRestriction restricted, double scalar,
                                 const std::vector<double>& oei_a, const std::vector<double>& oei_b,
                                 const std::vector<double>& tei_aa,
                                 const std::vector<double>& tei_ab,
                                 const std::vector<double>& tei_bb)
    : ForteIntegrals(options, mo_space_info, Custom, restricted), full_aphys_tei_aa_(tei_aa),
      full_aphys_tei_ab_(tei_ab), full_aphys_tei_bb_(tei_bb) {
    set_nuclear_repulsion(scalar);
    set_oei_all(oei_a, oei_b);
    initialize();
}

void CustomIntegrals::initialize() {
    Ca_ = std::make_shared<psi::Matrix>(nmopi_, nmopi_);
    Cb_ = std::make_shared<psi::Matrix>(nmopi_, nmopi_);
    Ca_->identity();
    Cb_->identity();
    nsopi_ = nmopi_;
    nso_ = nmo_;

    print_info();
    outfile->Printf("\n  Using Custom integrals\n\n");
    gather_integrals();

    freeze_core_orbitals();
}

double CustomIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_aa_[aptei_index(p, q, r, s)];
}

double CustomIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_ab_[aptei_index(p, q, r, s)];
}

double CustomIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_bb_[aptei_index(p, q, r, s)];
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
        aphys_tei_aa_[index] = value;
    if (alpha1 == true and alpha2 == false)
        aphys_tei_ab_[index] = value;
    if (alpha1 == false and alpha2 == false)
        aphys_tei_bb_[index] = value;
}

void CustomIntegrals::gather_integrals() {
    // Copy the correlated part into one_electron_integrals_a/one_electron_integrals_b
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            one_electron_integrals_a_[p * ncmo_ + q] =
                full_one_electron_integrals_a_[cmotomo_[p] * nmo_ + cmotomo_[q]];
            one_electron_integrals_b_[p * ncmo_ + q] =
                full_one_electron_integrals_b_[cmotomo_[p] * nmo_ + cmotomo_[q]];
        }
    }
    aphys_tei_aa_ = full_aphys_tei_aa_;
    aphys_tei_ab_ = full_aphys_tei_ab_;
    aphys_tei_bb_ = full_aphys_tei_bb_;
}

void CustomIntegrals::resort_integrals_after_freezing() {
    if (print_ > 0) {
        outfile->Printf("\n  Resorting integrals after freezing core.");
    }
    // Resort the four-index integrals
    resort_four(aphys_tei_aa_, cmotomo_);
    resort_four(aphys_tei_ab_, cmotomo_);
    resort_four(aphys_tei_bb_, cmotomo_);
}

void CustomIntegrals::resort_four(std::vector<double>& tei, std::vector<size_t>& map) {
    // Store the integrals in a temporary array
    size_t num_aptei_corr = ncmo_ * ncmo_ * ncmo_ * ncmo_;
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

void CustomIntegrals::make_fock_matrix(std::shared_ptr<psi::Matrix> gamma_a,
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

void CustomIntegrals::compute_frozen_one_body_operator() {
    local_timer timer_frozen_one_body;

    //    local_timer timer_frozen_one_body;
    psi::Dimension frozen_dim = mo_space_info_->dimension("FROZEN_DOCC");
    psi::Dimension nmopi = mo_space_info_->dimension("ALL");

    std::vector<size_t> frozen_mos = mo_space_info_->absolute_mo("FROZEN_DOCC");

    // This loop grabs only the correlated part of the correction
    int full_offset = 0;
    int corr_offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        for (int p = 0; p < ncmopi_[h]; ++p) {
            for (int q = 0; q < ncmopi_[h]; ++q) {
                // the index of p and q in the full block of irrep h
                size_t p_full = cmotomo_[p + corr_offset]; // - full_offset;
                size_t q_full = cmotomo_[q + corr_offset]; // - full_offset;
                for (size_t m_full : frozen_mos) {
                    one_electron_integrals_a_[(p + corr_offset) * ncmo_ + (q + corr_offset)] +=
                        aptei_aa(p_full, m_full, q_full, m_full) +
                        aptei_ab(p_full, m_full, q_full, m_full);

                    one_electron_integrals_b_[(p + corr_offset) * ncmo_ + (q + corr_offset)] +=
                        aptei_bb(p_full, m_full, q_full, m_full) +
                        aptei_ab(m_full, p_full, m_full, q_full);
                }
            }
        }
        full_offset += nmopi_[h];
        corr_offset += ncmopi_[h];
    }

    frozen_core_energy_ = 0.0;
    for (int m : frozen_mos) {
        frozen_core_energy_ += full_one_electron_integrals_a_[m * nmo_ + m] +
                               full_one_electron_integrals_b_[m * nmo_ + m];
        for (int n : frozen_mos) {
            frozen_core_energy_ += 0.5 * aptei_aa(m, n, m, n) + 1.0 * aptei_ab(m, n, m, n) +
                                   0.5 * aptei_bb(m, n, m, n);
        }
    }

    if (print_ > 0) {
        outfile->Printf("\n  Frozen-core energy        %20.12f a.u.", frozen_core_energy_);
        print_timing("frozen one-body operator", timer_frozen_one_body.get());
    }
}

void CustomIntegrals::transform_one_electron_integrals() {
    // the first time we transform, we keep a copy of the original integrals
    if (original_full_one_electron_integrals_a_.size() == 0) {
        original_full_one_electron_integrals_a_ = full_one_electron_integrals_a_;
        original_full_one_electron_integrals_b_ = full_one_electron_integrals_b_;
    }

    // Grab the one-electron integrals from psi4's wave function object
    auto Ha = std::make_shared<psi::Matrix>(nmopi_, nmopi_);
    auto Hb = std::make_shared<psi::Matrix>(nmopi_, nmopi_);

    // Read the one-electron integrals (T + V)
    int offset = 0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < nmopi_[h]; ++p) {
            for (int q = 0; q < nmopi_[h]; ++q) {
                Ha->set(h, p, q,
                        original_full_one_electron_integrals_a_[(p + offset) * nmo_ + q + offset]);
                Hb->set(h, p, q,
                        original_full_one_electron_integrals_b_[(p + offset) * nmo_ + q + offset]);
            }
        }
        offset += nmopi_[h];
    }

    // transform the one-electron integrals
    Ha->transform(Ca_);
    Hb->transform(Cb_);

    OneBody_symm_ = Ha;

    // zero these vectors
    std::fill(full_one_electron_integrals_a_.begin(), full_one_electron_integrals_a_.end(), 0.0);
    std::fill(full_one_electron_integrals_b_.begin(), full_one_electron_integrals_b_.end(), 0.0);

    // Read the one-electron integrals (T + V, restricted)
    offset = 0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < nmopi_[h]; ++p) {
            for (int q = 0; q < nmopi_[h]; ++q) {
                full_one_electron_integrals_a_[(p + offset) * nmo_ + q + offset] = Ha->get(h, p, q);
                full_one_electron_integrals_b_[(p + offset) * nmo_ + q + offset] = Hb->get(h, p, q);
            }
        }
        offset += nmopi_[h];
    }
}

void CustomIntegrals::transform_two_electron_integrals() {
    if (not save_original_tei_) {
        original_V_aa_ = ambit::Tensor::build(tensor_type_, "V_aa", {nmo_, nmo_, nmo_, nmo_});
        original_V_ab_ = ambit::Tensor::build(tensor_type_, "V_ab", {nmo_, nmo_, nmo_, nmo_});
        original_V_bb_ = ambit::Tensor::build(tensor_type_, "V_bb", {nmo_, nmo_, nmo_, nmo_});

        original_V_aa_.data() = full_aphys_tei_aa_;
        original_V_ab_.data() = full_aphys_tei_ab_;
        original_V_bb_.data() = full_aphys_tei_bb_;

        save_original_tei_ = true;
    }

    auto Ca = ambit::Tensor::build(tensor_type_, "Ca", {nmo_, nmo_});
    auto Cb = ambit::Tensor::build(tensor_type_, "Cb", {nmo_, nmo_});

    auto& Ca_data = Ca.data();
    auto& Cb_data = Cb.data();

    int offset = 0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < nmopi_[h]; ++p) {
            for (int q = 0; q < nmopi_[h]; ++q) {
                int p_full = p + offset;
                int q_full = q + offset;
                Ca_data[p_full * nmo_ + q_full] = Ca_->get(h, p, q);
                Cb_data[p_full * nmo_ + q_full] = Cb_->get(h, p, q);
            }
        }
        offset += nmopi_[h];
    }

    auto T = ambit::Tensor::build(tensor_type_, "temp", {nmo_, nmo_, nmo_, nmo_});

    T("ijkl") = original_V_aa_("pqrs") * Ca("pi") * Ca("qj") * Ca("rk") * Ca("sl");
    full_aphys_tei_aa_ = T.data();

    T("ijkl") = original_V_ab_("pqrs") * Ca("pi") * Cb("qj") * Ca("rk") * Cb("sl");
    full_aphys_tei_ab_ = T.data();

    T("ijkl") = original_V_bb_("pqrs") * Cb("pi") * Cb("qj") * Cb("rk") * Cb("sl");
    full_aphys_tei_bb_ = T.data();
}

void CustomIntegrals::update_orbitals(std::shared_ptr<psi::Matrix> Ca,
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

    // 2. Re-transform the integrals
    aptei_idx_ = nmo_;
    transform_one_electron_integrals();
    transform_two_electron_integrals();
    gather_integrals();
    outfile->Printf("\n  Integrals are about to be updated.");
    freeze_core_orbitals();
}

// void CustomIntegrals::resort_integrals_after_freezing() {}

//    // Read the integrals from a file
//    std::string filename("INTDUMP");
//    std::ifstream file(filename);

//    if (not file.is_open()) {
//    }
//    std::string str((std::istreambuf_iterator<char>(file)),
//    std::istreambuf_iterator<char>());

//    std::vector<std::string> lines = split_string(str, "\n");

//    std::string open_tag("&FCI");
//    std::string close_tag("&END");

//    int nelec = 0;
//    int norb = 0;
//    int ms2 = 0;
//    std::vector<int> orbsym;
//    std::vector<double> two_electron_integrals_chemist;

//    bool parsing_section = false;
//    for (const auto& line : lines) {
//        outfile->Printf("\n%s", line.c_str());
//        if (line.find(close_tag) != std::string::npos) {
//            parsing_section = false;
//            // now we know how many orbitals are there and we can allocate memory for the
//            one- and
//            // two-electron integrals
//            custom_integrals_allocate(norb, orbsym);
//            two_electron_integrals_chemist.assign(num_tei_, 0.0);
//        } else if (line.find(open_tag) != std::string::npos) {
//            parsing_section = true;
//        } else {
//            if (parsing_section) {
//                std::vector<std::string> split_line = split_string(line, "=");
//                if (split_line[0] == "NORB") {
//                    split_line[1].pop_back();
//                    norb = stoi(split_line[1]);
//                }
//                if (split_line[0] == "NELEC") {
//                    split_line[1].pop_back();
//                    nelec = stoi(split_line[1]);
//                }
//                if (split_line[0] == "MS2") {
//                    split_line[1].pop_back();
//                    ms2 = stoi(split_line[1]);
//                }
//                if (split_line[0] == "ORBSYM") {
//                    split_line[1].pop_back();
//                    std::vector<std::string> vals = split_string(split_line[1], ",");
//                    for (const auto& val : vals) {
//                        orbsym.push_back(stoi(val));
//                    }
//                }
//            } else {
//                if (line.size() > 10) {
//                    std::vector<std::string> split_line = split_string(line, " ");
//                    double integral = stoi(split_line[0]);
//                    int p = stoi(split_line[1]);
//                    int q = stoi(split_line[2]);
//                    int r = stoi(split_line[3]);
//                    int s = stoi(split_line[4]);

//                    if (q == 0) {
//                        // orbital energies, skip them
//                    } else if ((r == 0) and (s == 0)) {
//                        // one-electron integrals
//                        full_one_electron_integrals_a_[(p - 1) * aptei_idx_ + q - 1] =
//                        integral; full_one_electron_integrals_b_[(p - 1) * aptei_idx_ + q
//                        - 1] = integral; full_one_electron_integrals_a_[(q - 1) *
//                        aptei_idx_ + p - 1] = integral; full_one_electron_integrals_b_[(q
//                        - 1) * aptei_idx_ + p - 1] = integral;
//                        one_electron_integrals_a_[(p - 1) * aptei_idx_ + q - 1] =
//                        integral; one_electron_integrals_b_[(p - 1) * aptei_idx_ + q - 1]
//                        = integral; one_electron_integrals_a_[(q - 1) * aptei_idx_ + p -
//                        1] = integral; one_electron_integrals_b_[(q - 1) * aptei_idx_ + p
//                        - 1] = integral;
//                    } else {
//                        // two-electron integrals
//                        two_electron_integrals_chemist[four(p, q, r, s)] = integral;
//                    }
//                }
//            }
//        }
//    }

//    // Store the integrals
//    for (size_t p = 0; p < nmo_; ++p) {
//        for (size_t q = 0; q < nmo_; ++q) {
//            for (size_t r = 0; r < nmo_; ++r) {
//                for (size_t s = 0; s < nmo_; ++s) {
//                    // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
//                    double direct = two_electron_integrals_chemist[INDEX4(p, r, q, s)];
//                    double exchange = two_electron_integrals_chemist[INDEX4(p, s, q, r)];
//                    size_t index = aptei_index(p, q, r, s);
//                    aphys_tei_aa[index] = direct - exchange;
//                    aphys_tei_ab[index] = direct;
//                    aphys_tei_bb[index] = direct - exchange;
//                }
//            }
//        }
//    }

//    std::string s(std::istreambuf_iterator<char>(file >> std::skipws),
//                   std::istreambuf_iterator<char>());

//    std::copy(std::istream_iterator<std::string>(file),
//              std::istream_iterator<std::string>(),
//              std::back_inserter(lines));

//    outfile->Printf("%s",s.c_str());

//    for (size_t p = 0; p < nmo_; ++p) {
//        for (size_t q = 0; q < nmo_; ++q) {
//            one_electron_integrals_a_[p * nmo_ + q] = 0.0;
//            one_electron_integrals_b_[p * nmo_ + q] = 0.0;
//        }
//    }

//    for (size_t pqrs = 0; pqrs < num_aptei_; ++pqrs)
//        aphys_tei_aa[pqrs] = 0.0;
//    for (size_t pqrs = 0; pqrs < num_aptei_; ++pqrs)
//        aphys_tei_ab[pqrs] = 0.0;
//    for (size_t pqrs = 0; pqrs < num_aptei_; ++pqrs)
//        aphys_tei_bb[pqrs] = 0.0;

//    // Store the integrals
//    for (size_t p = 0; p < nmo_; ++p) {
//        for (size_t q = 0; q < nmo_; ++q) {
//            for (size_t r = 0; r < nmo_; ++r) {
//                for (size_t s = 0; s < nmo_; ++s) {
//                    // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
//                    double direct = 0.0;
//                    double exchange = 0.0;
//                    size_t index = aptei_index(p, q, r, s);
//                    aphys_tei_aa[index] = direct - exchange;
//                    aphys_tei_ab[index] = direct;
//                    aphys_tei_bb[index] = direct - exchange;
//                }
//            }
//        }
//    }
} // namespace forte
